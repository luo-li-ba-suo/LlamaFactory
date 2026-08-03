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

from typing import TYPE_CHECKING, Literal, Optional, Union

import torch
from typing_extensions import override

from ...extras import logging
from ..dpo.trainer import CustomDPOTrainer
from ..trainer_utils import compute_tag_metrics, patch_rewrite_logs
from .sfl_loss import extract_score_logits, sfl_binary_loss, sfl_multiclass_loss


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSfLTrainer(CustomDPOTrainer):
    r"""Trainer for Score-from-Logits (SfL) preference training.

    Inherits from ``CustomDPOTrainer`` to reuse the pairwise data pipeline
    (``PairwiseDataCollatorWithPadding``, ``concatenated_forward``, etc.).
    Replaces DPO loss computation with token-level score logit extraction
    and pairwise ranking loss.
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        # SfL does not use a reference model
        finetuning_args.use_ref_model = False

        super().__init__(model, None, finetuning_args, processor, disable_dropout, **kwargs)

        self.sfl_temperature = finetuning_args.sfl_temperature

        # Encode score token strings to token ids (each must be exactly 1 token)
        token_strs = [t.strip() for t in finetuning_args.sfl_score_tokens.split(",")]
        self.sfl_score_token_ids: list[int] = []
        for t in token_strs:
            if not t:
                continue
            ids = self.processing_class.encode(t, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    f"Score token '{t}' encodes to {len(ids)} tokens ({ids}). "
                    f"Each score value must be exactly 1 token."
                )
            self.sfl_score_token_ids.append(ids[0])

        if not self.sfl_score_token_ids:
            raise ValueError("`sfl_score_tokens` must contain at least one token string.")

        patch_rewrite_logs()  # prevent HF from mangling per-tag wandb keys
        self._eval_scores: list[tuple[float, float, list[str]]] = []

        logger.info_rank0(f"SfL score token ids: {self.sfl_score_token_ids}")
        logger.info_rank0(f"SfL temperature: {self.sfl_temperature}")

    def _sfl_forward(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"]
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        r"""Return score-token logits for chosen and rejected responses.

        Qwen2 and Qwen3 expose their decoder backbone and language-model head separately.
        SfL only needs the score-token logits following the final non-padding
        token, so avoid materializing the otherwise enormous ``[B, L, V]``
        logits tensor for Qwen models. The regular model path is retained for other
        architectures.
        """
        _ = batch.pop("labels")
        batch_size = batch["input_ids"].size(0) // 2

        # During multi-GPU training, ``model`` is wrapped by DDP. Inspect and
        # access the underlying PEFT model for the Qwen-specific fast path.
        model_to_unwrap = model.module if hasattr(model, "module") else model
        if getattr(model_to_unwrap.config, "model_type", None) in {"qwen2", "qwen3"}:
            base_model = (
                model_to_unwrap.get_base_model() if hasattr(model_to_unwrap, "get_base_model") else model_to_unwrap
            )
            backbone = getattr(base_model, "model", None)
            lm_head = getattr(base_model, "lm_head", None)
            if backbone is not None and lm_head is not None:
                outputs = backbone(**batch, return_dict=True, use_cache=False)
                last_hidden_state = outputs.last_hidden_state
                last_valid_pos = batch["attention_mask"].sum(-1) - 1
                if (last_valid_pos < 0).any():
                    raise RuntimeError("No valid tokens found (all pad). Check cutoff_len or data.")

                batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
                last_hidden_state = last_hidden_state[batch_indices, last_valid_pos]
                score_token_ids = torch.tensor(
                    self.sfl_score_token_ids, device=last_hidden_state.device, dtype=torch.long
                )
                score_logits = lm_head(last_hidden_state)[:, score_token_ids]
                chosen_logits, rejected_logits = score_logits.split(batch_size, dim=0)
                return chosen_logits, rejected_logits

        all_logits: torch.Tensor = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        return chosen_logits, rejected_logits

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        r"""Compute SfL loss: extract score logits from chosen/rejected, apply pairwise ranking loss."""
        tags = batch.pop("tags", None)  # pop before model forward (model rejects unknown kwargs)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        chosen_logits, rejected_logits = self._sfl_forward(model, batch)

        if chosen_logits.size(-1) == len(self.sfl_score_token_ids):
            chosen_scores, rejected_scores = chosen_logits, rejected_logits
        else:
            B = input_ids.size(0) // 2
            chosen_input_ids = input_ids[:B]
            rejected_input_ids = input_ids[B:]
            chosen_attention_mask = attention_mask[:B]
            rejected_attention_mask = attention_mask[B:]

            chosen_scores = extract_score_logits(
                chosen_logits, chosen_input_ids, chosen_attention_mask, self.sfl_score_token_ids
            )
            rejected_scores = extract_score_logits(
                rejected_logits, rejected_input_ids, rejected_attention_mask, self.sfl_score_token_ids
            )

        if len(self.sfl_score_token_ids) == 2:
            losses = sfl_binary_loss(chosen_scores, rejected_scores)
            chosen_score = chosen_scores[:, 1] - chosen_scores[:, 0]
            rejected_score = rejected_scores[:, 1] - rejected_scores[:, 0]
            score_diff = chosen_score - rejected_score
            accuracy = (chosen_score > rejected_score).float().mean()
        else:
            losses = sfl_multiclass_loss(chosen_scores, rejected_scores, self.sfl_temperature)
            # Expected score: sum_i i * p_i
            p_chosen = (chosen_scores / self.sfl_temperature).softmax(dim=-1)
            p_rejected = (rejected_scores / self.sfl_temperature).softmax(dim=-1)
            weights = torch.arange(
                len(self.sfl_score_token_ids), device=chosen_scores.device, dtype=chosen_scores.dtype
            )
            expected_chosen = (p_chosen * weights).sum(dim=-1)
            expected_rejected = (p_rejected * weights).sum(dim=-1)
            score_diff = expected_chosen - expected_rejected
            accuracy = (expected_chosen > expected_rejected).float().mean()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics: dict[str, float] = {
            f"{prefix}chosen_score": chosen_scores.mean().item(),
            f"{prefix}rejected_score": rejected_scores.mean().item(),
            f"{prefix}score_diff": score_diff.mean().item(),
            f"{prefix}accuracy": accuracy.item(),
        }

        # Store per-sample scores AND tags for per-tag eval
        if not self.model.training:
            if len(self.sfl_score_token_ids) == 2:
                cs, rs = chosen_score, rejected_score
            else:
                cs, rs = expected_chosen, expected_rejected
            for i in range(len(cs)):
                tag_i = tags[i] if tags is not None else []
                self._eval_scores.append((cs[i].item(), rs[i].item(), tag_i))

        return losses.mean(), metrics

    @override
    def evaluate(
        self,
        eval_dataset: Optional["Dataset"] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        import torch.distributed as dist

        self._eval_scores = []
        metrics = super().evaluate(
            eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )

        all_scores = self._eval_scores  # list[tuple[float, float, list[str]]]
        self._eval_scores = []

        if len(all_scores) > 0:
            if self.accelerator.num_processes > 1:
                world_size = dist.get_world_size()
                gathered: list | None = [None] * world_size if dist.get_rank() == 0 else None
                dist.gather_object(all_scores, gathered, dst=0)
                if dist.get_rank() == 0 and gathered is not None:
                    all_gathered: list = []
                    for g in gathered:
                        if g is not None:
                            all_gathered.extend(g)
                    chosen = [s[0] for s in all_gathered]
                    rejected = [s[1] for s in all_gathered]
                    tags_list = [s[2] for s in all_gathered]
                    tag_metrics = compute_tag_metrics(chosen, rejected, tags_list)
                    metrics.update(tag_metrics)
                else:
                    tag_metrics = {}
            else:
                chosen = [s[0] for s in all_scores]
                rejected = [s[1] for s in all_scores]
                tags_list = [s[2] for s in all_scores]
                tag_metrics = compute_tag_metrics(chosen, rejected, tags_list)
                metrics.update(tag_metrics)

            # Log to wandb — self.log() uses accelerator.reduce() which is a collective op
            self.log(tag_metrics if self.accelerator.is_main_process else {})

        return metrics

    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"]
    ) -> tuple[None, None]:
        r"""SfL does not require reference log probabilities."""
        return None, None
