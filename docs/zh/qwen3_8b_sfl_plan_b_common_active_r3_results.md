# Qwen3-8B SfL Plan B：G2_half + G3_2 + common_active_r3

更新时间：2026-08-11。该实验已完成 2 个 epoch（364 steps），结果来自 `saves/curriculum/qwen3-8b-sfl-plan-b-g2-half-plus-g3-2-plus-common-active-r3/all_results.json`。

## 训练设置

- 基座：`Qwen3-8B`；SfL LoRA（`rank=16`、`alpha=32`、`target=all`）。
- 训练数据：`curriculum_qwen3_sfl_plan_b_g2_half_plus_g3_2_plus_common_active_r3`。
- 验证数据：Plan C 统一的带标签验证集（1,350 条）：B1–B5 为 IID（750 条），Chat/Code/Math/Safety 为 RM Bench（600 条）。
- 8 卡 DDP，`per_device_train_batch_size=4`、`gradient_accumulation_steps=4`，全局有效 batch 为 128；2 epochs，学习率 `5e-5`、cosine schedule、warmup `0.1`、`cutoff_len=4096`。
- Qwen3 使用 `enable_thinking: false`。

## 最终评估

| 聚合指标 | 准确率 |
| --- | ---: |
| 总准确率 | 70.93% |
| IID（B1–B5 宏平均） | 69.33% |
| RM Bench（Chat/Code/Math/Safety 宏平均） | 72.83% |

| IID 标签 | B1 | B2 | B3 | B4 | B5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 准确率 | 100.00% | 94.67% | 76.00% | 46.67% | 29.33% |

| RM Bench 领域 | Chat | Code | Math | Safety |
| --- | ---: | ---: | ---: | ---: |
| 准确率 | 75.33% | 64.00% | 66.00% | 86.00% |

| RM Bench 难度 | Easy | Normal | Hard |
| --- | ---: | ---: | ---: |
| 准确率 | 87.50% | 72.50% | 58.50% |

## 损失与训练吞吐

| Eval loss | Score diff | Train loss | 训练耗时 | 样本/秒 | step/秒 |
| ---: | ---: | ---: | --- | ---: | ---: |
| 0.5570 | 1.9975 | 0.4842 | 3:27:54 | 3.734 | 0.029 |

最终的 `score_diff` 在 B1–B4 为正（依次 6.0658、2.5167、0.7267、0.0475），但 B5 为 -0.7200；这与 B5 的低准确率（29.33%）一致。领域中 Safety 的 score diff 最高（5.6525），Code 最低（0.5825）。

## 与 G2 + G3_2 基线的对比

共同基线为 `G2 + G3_2`（IID 76.40%、RM Bench 72.58%、总准确率 74.70%）。以下变化均为“本实验 − 基线”，单位为百分点。

| 指标 | G2 + G3_2 基线 | common_active_r3 | 变化 |
| --- | ---: | ---: | ---: |
| 总准确率 | 74.70 | 70.93 | -3.77 |
| IID（B1–B5） | 76.40 | 69.33 | -7.07 |
| RM Bench | 72.58 | 72.83 | **+0.25** |
| Easy | 88.50 | 87.50 | -1.00 |
| Normal | 71.00 | 72.50 | +1.50 |
| Hard | 58.00 | 58.50 | +0.50 |

| 标签 | B1 | B2 | B3 | B4 | B5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 相对基线变化 | +0.66 | 0.00 | -6.00 | -13.33 | -16.67 |

| RM Bench 领域 | Chat | Code | Math | Safety |
| --- | ---: | ---: | ---: | ---: |
| 相对基线变化 | **+1.33** | +0.67 | -1.33 | +0.67 |

它只换来 0.25 个百分点的 RM Bench 提升，却使 IID 降 7.07，且损失集中在 B4/B5；因此相对 `G2 + G3_2`，该附加 `common_active_r3` 的配方不是有效折中。

## 最后一次中间评估与最终结果

最后一次周期评估发生在 step 320、epoch 1.758；之后训练至 step 364。最终相较 step 320：总准确率从 71.45% 降至 70.93%，IID 的 B3/B4/B5 分别变化 -0.67/+1.33/+0.67，RM Bench 的 Chat/Code/Math/Safety 分别变化 -1.33/-3.33/-0.67/-0.67。该尾段未带来总体评估提升。

## 可追溯性

- 训练配置：`examples/train_sfl/curriculum_qwen3_8b_sfl_plan_b_g2_half_plus_g3_2_plus_common_active_r3.yaml`
- LoRA 输出：`saves/curriculum/qwen3-8b-sfl-plan-b-g2-half-plus-g3-2-plus-common-active-r3`
- 运行日志：`logs/rm_qwen3_8b_sfl_plan_b_common_active_r3_20260810_retry.log`
