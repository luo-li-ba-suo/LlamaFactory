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

from typing import TYPE_CHECKING, Optional

from ...data import get_dataset, get_template_and_fix_tokenizer
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

    # SfL data pipeline: converter → processor → collator
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sfl", **tokenizer_module)

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
