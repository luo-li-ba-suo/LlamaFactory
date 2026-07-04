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

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor


if TYPE_CHECKING:
    pass


logger = logging.get_logger(__name__)


class SfLDatasetProcessor(DatasetProcessor):
    r"""Dataset processor for SfL (Score-from-Logits) training.

    Expects ``_chosen_messages`` and ``_rejected_messages`` from the
    SfL converter — each is a list of standardized role/content dicts
    representing a full multi-turn conversation ending with a user turn.

    Applies the chat template with ``add_generation_prompt=True``, appends
    the score prefix tokens, and sets all labels to ``IGNORE_INDEX``
    (SfL computes loss only at the score-prediction position).
    """

    def __init__(self, template, tokenizer, processor, data_args):
        super().__init__(template, tokenizer, processor, data_args)
        self.prefix_token_ids = tokenizer.encode(data_args.sfl_prefix_str, add_special_tokens=False)
        if not self.prefix_token_ids:
            raise ValueError(f"sfl_prefix_str '{data_args.sfl_prefix_str}' encodes to 0 tokens.")

    def _encode_sfl_example(self, messages: list[dict[str, str]], cutoff_len: int):
        r"""Apply chat template to a full conversation and append score prefix."""
        encoded = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        tokens = list(encoded) if isinstance(encoded, list) else list(encoded["input_ids"])
        input_ids = tokens + self.prefix_token_ids
        input_ids = input_ids[:cutoff_len]
        labels = [IGNORE_INDEX] * len(input_ids)
        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        r"""Build model inputs from SfL examples."""
        model_inputs = defaultdict(list)
        for i in range(len(examples["_chosen_messages"])):
            if i >= len(examples["_rejected_messages"]):
                break

            chosen_ids, chosen_labels = self._encode_sfl_example(
                examples["_chosen_messages"][i], self.data_args.cutoff_len
            )
            rejected_ids, rejected_labels = self._encode_sfl_example(
                examples["_rejected_messages"][i], self.data_args.cutoff_len
            )

            model_inputs["chosen_input_ids"].append(chosen_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            model_inputs["images"].append(examples.get("_images", [None] * len(examples["_chosen_messages"]))[i] or [])
            model_inputs["videos"].append(examples.get("_videos", [None] * len(examples["_chosen_messages"]))[i] or [])
            model_inputs["audios"].append(examples.get("_audios", [None] * len(examples["_chosen_messages"]))[i] or [])
            model_inputs["tags"].append(examples.get("_tags", [None] * len(examples["_chosen_messages"]))[i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        r"""Print a data example to stdout."""
        print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
        print(
            "chosen_inputs:\n{}".format(self.tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False))
        )
        print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
        print(
            "rejected_inputs:\n{}".format(
                self.tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)
            )
        )
