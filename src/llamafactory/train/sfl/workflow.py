# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import TYPE_CHECKING, Any, Optional

from ...data import get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments


logger = get_logger(__name__)


def _prepare_sfl_dataset(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    tokenizer,
):
    r"""Load raw pairwise data and tokenize for SfL.

    Each example is expected to have ``chosen`` and ``rejected`` fields
    containing raw user-message text. The chat template is applied to each
    with ``add_generation_prompt=True``, then ``{"score":`` prefix tokens
    are appended for the model to score.
    """
    from datasets import Dataset, load_dataset

    # Load raw dataset(s) — supports json, jsonl, parquet, etc.
    # dataset may be a str like "data/a.json" or a list like ["data/a.json"]
    raw = data_args.dataset
    if isinstance(raw, list):
        dataset_names = raw
    elif isinstance(raw, str):
        dataset_names = [s.strip() for s in raw.split(",") if s.strip()]
    else:
        dataset_names = []
    if not dataset_names:
        raise ValueError("No dataset specified for SfL training.")

    all_datasets = []
    for name in dataset_names:
        name = name.strip()
        # Resolve path: try raw name first, then dataset_dir/name
        if os.path.isfile(name):
            path = name
        elif data_args.dataset_dir and not name.startswith(data_args.dataset_dir):
            path = os.path.join(data_args.dataset_dir, name)
        else:
            path = name  # hope for the best

        # Determine format from file extension and load
        lower = name.lower()
        if lower.endswith(".json") or lower.endswith(".jsonl"):
            ds = load_dataset("json", data_files=path, split="train")
        elif lower.endswith(".parquet"):
            ds = load_dataset("parquet", data_files=path, split="train")
        elif lower.endswith(".csv"):
            ds = load_dataset("csv", data_files=path, split="train")
        else:
            raise ValueError(
                f"Unsupported dataset format for SfL: '{name}'. "
                f"Use .json, .jsonl, .parquet, or .csv files."
            )
        all_datasets.append(ds)

    from datasets import concatenate_datasets

    dataset = concatenate_datasets(all_datasets) if len(all_datasets) > 1 else all_datasets[0]

    # Shuffle
    dataset = dataset.shuffle(seed=training_args.seed)

    # Subsample
    if data_args.max_samples is not None:
        dataset = dataset.select(range(min(data_args.max_samples, len(dataset))))

    # Train/val split
    if data_args.val_size > 1e-6:
        split = dataset.train_test_split(test_size=data_args.val_size, seed=training_args.seed)
        train_data = split["train"]
        eval_data = split["test"]
    else:
        train_data = dataset
        eval_data = None

    # Cache processed datasets to disk for fast reload
    with training_args.main_process_first(desc="tokenize sfl dataset"):
        # Encode score prefix tokens once
        prefix_token_ids = tokenizer.encode(
            finetuning_args.sfl_prefix_str, add_special_tokens=False
        )
        if not prefix_token_ids:
            raise ValueError(f"sfl_prefix_str '{finetuning_args.sfl_prefix_str}' encodes to 0 tokens.")

        def tokenize_sfl_example(example: dict[str, Any]) -> dict[str, Any]:
            chosen_text = str(example["chosen"])
            rejected_text = str(example["rejected"])

            # Apply chat template as user turn with add_generation_prompt=True
            # → <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
            chosen_enc = tokenizer.apply_chat_template(
                [{"role": "user", "content": chosen_text}],
                add_generation_prompt=True,
                tokenize=True,
            )
            rejected_enc = tokenizer.apply_chat_template(
                [{"role": "user", "content": rejected_text}],
                add_generation_prompt=True,
                tokenize=True,
            )
            chosen_tokens = chosen_enc if isinstance(chosen_enc, list) else list(chosen_enc["input_ids"])
            rejected_tokens = rejected_enc if isinstance(rejected_enc, list) else list(rejected_enc["input_ids"])

            # Append {"score": prefix — score digit follows immediately
            chosen_input_ids = chosen_tokens + prefix_token_ids
            rejected_input_ids = rejected_tokens + prefix_token_ids

            # Truncate if needed
            max_len = data_args.cutoff_len
            chosen_input_ids = chosen_input_ids[:max_len]
            rejected_input_ids = rejected_input_ids[:max_len]

            # All labels IGNORE — SfL computes loss on score position only
            chosen_labels = [IGNORE_INDEX] * len(chosen_input_ids)
            rejected_labels = [IGNORE_INDEX] * len(rejected_input_ids)

            return {
                "chosen_input_ids": chosen_input_ids,
                "chosen_attention_mask": [1] * len(chosen_input_ids),
                "chosen_labels": chosen_labels,
                "rejected_input_ids": rejected_input_ids,
                "rejected_attention_mask": [1] * len(rejected_input_ids),
                "rejected_labels": rejected_labels,
                "images": [],
                "videos": [],
                "audios": [],
            }

        train_dataset = train_data.map(tokenize_sfl_example, remove_columns=train_data.column_names)
        eval_dataset = eval_data.map(tokenize_sfl_example, remove_columns=eval_data.column_names) if eval_data else None

    return {"train_dataset": train_dataset, "eval_dataset": eval_dataset}


def run_sfl(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
) -> None:
    r"""Run Score-from-Logits (SfL) training.

    Data format: each example has ``chosen`` and ``rejected`` fields as raw
    user-message strings. The chat template is applied with ``add_generation_prompt``,
    then ``{"score":`` tokens are appended. The model predicts the score digit
    from the logits at the rightmost position.
    """
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # SfL-specific data pipeline: raw chosen/rejected text → tokenized pairs
    dataset_module = _prepare_sfl_dataset(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
    )

    from ...data import PairwiseDataCollatorWithPadding

    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )

    from .trainer import CustomSfLTrainer

    trainer = CustomSfLTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    if training_args.do_train:
        # Pre-training evaluation (baseline accuracy)
        if training_args.do_eval and dataset_module.get("eval_dataset") is not None:
            logger.info_rank0("***** Pre-training eval *****")
            metrics = trainer.evaluate(metric_key_prefix="eval")
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sfl"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss", "accuracy", "score_diff"]
            if dataset_module.get("eval_dataset") is not None:
                keys += ["eval_loss", "eval_accuracy", "eval_score_diff"]
            plot_loss(training_args.output_dir, keys=keys)

    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
