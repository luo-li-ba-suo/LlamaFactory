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


import torch
import torch.nn.functional as F


def extract_score_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    score_token_ids: list[int],
) -> torch.Tensor:
    r"""Extract logits for score tokens at the rightmost non-pad position.

    Assumes left-padded data where ``{"score":`` prefix tokens are at the end
    of the input. The model's logit at the last valid position predicts the
    score digit that follows the prefix.

    Args:
        logits: [B, L, V] model output logits (before shift).
        input_ids: [B, L] input token ids (left-padded).
        attention_mask: [B, L] attention mask (1 = real token, 0 = pad).
        score_token_ids: list of token ids representing valid score values.

    Returns:
        [B, num_scores] logits for score tokens at the predicted position.
    """
    device = logits.device
    batch_size, seq_len = input_ids.shape

    # Find rightmost non-pad position per sample (left-pad → data at end)
    # The logit at this position predicts the token immediately after the prefix
    last_valid_pos = attention_mask.sum(-1) - 1  # [B]

    if (last_valid_pos < 0).any():
        raise RuntimeError("No valid tokens found (all pad). Check cutoff_len or data.")

    # Gather: logits[b, last_valid_pos, :] → [B, V], index by score_token_ids
    score_logits = logits[torch.arange(batch_size, device=device), last_valid_pos]  # [B, V]
    score_token_ids_t = torch.tensor(score_token_ids, device=device, dtype=torch.long)
    return score_logits[:, score_token_ids_t]


def sfl_binary_loss(
    chosen_scores: torch.Tensor,
    rejected_scores: torch.Tensor,
) -> torch.Tensor:
    r"""Binary SfL loss for 0-1 scoring.

    Treats score = logit[1] - logit[0] as a continuous preference signal.
    Loss = -log_sigmoid(score_chosen - score_rejected).

    Args:
        chosen_scores: [B, 2] logits for tokens [0, 1] in chosen response.
        rejected_scores: [B, 2] logits for tokens [0, 1] in rejected response.

    Returns:
        [B] per-sample losses.
    """
    score_chosen = chosen_scores[:, 1] - chosen_scores[:, 0]  # [B]
    score_rejected = rejected_scores[:, 1] - rejected_scores[:, 0]  # [B]
    return -F.logsigmoid(score_chosen - score_rejected)


def sfl_multiclass_loss(
    chosen_scores: torch.Tensor,
    rejected_scores: torch.Tensor,
    temperature: float = 5.0,
) -> torch.Tensor:
    r"""Multi-class SfL loss for 1-5 (or any ordered) scoring.

    Computes softmax over score tokens for chosen and rejected separately,
    then penalizes configurations where rejected gets a higher score than chosen.

        loss = sum_{i} sum_{j>i} p_chosen[:, j] * p_rejected[:, i]

    Args:
        chosen_scores: [B, K] logits for score tokens in chosen response.
        rejected_scores: [B, K] logits for score tokens in rejected response.
        temperature: softmax temperature.

    Returns:
        [B] per-sample losses.
    """
    p_chosen = F.softmax(chosen_scores / temperature, dim=-1)  # [B, K]
    p_rejected = F.softmax(rejected_scores / temperature, dim=-1)  # [B, K]

    K = p_chosen.size(-1)
    loss = torch.zeros(p_chosen.size(0), device=p_chosen.device)
    for i in range(K):
        for j in range(i + 1, K):
            loss = loss + p_chosen[:, j] * p_rejected[:, i]

    return loss
