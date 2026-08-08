# Qwen3-8B SfL Plan C Round 2 实验记录

更新时间：2026-08-08。Round 2 将 Round 1 的 G2 或 G3 数据与第二轮数据集（`G2_2` / `G3_2`）混合训练，用于检验不同起始数据分布与第二轮数据的组合效果。

> 四组训练均已完成，以下数值均来自各自最终 `eval_results.json` 与 `train_results.json`。

## 共同训练设置

- 基座模型：`../../model/Qwen3-8B`
- 方法：SfL LoRA（`rank=16`、`alpha=32`、`target=all`）
- 模板：`qwen3`，`enable_thinking: false`
- 评分 token：`0,1`；评分前缀：`{"score": `；温度：`5.0`
- 训练：2 epochs，8 卡 DDP，`per_device_train_batch_size=1`，`gradient_accumulation_steps=16`，全局有效 batch size 为 128
- 学习率：`5e-5`，cosine scheduler，warmup ratio `0.1`，`cutoff_len=4096`
- 验证：与 Round 1 相同的 1,350 条带标签验证集；其中 `B1`--`B5` 为 IID 子集（750 条），其余标签为 RM Bench 子集（600 条）。

## 实验组合与状态

| Group | 训练数据组合 | 状态 |
| --- | --- | --- |
| G2 + G2_2 | Teacher-filtered random + Round 2 G2 | 已完成 |
| G3 + G3_2 | Teacher-filtered active + Round 2 G3 | 已完成 |
| G2 + G3_2 | Teacher-filtered random + Round 2 G3 | 已完成 |
| G3 + G2_2 | Teacher-filtered active + Round 2 G2 | 已完成 |

## 未训练基座与已完成实验的最终结果

`Qwen3-8B（未训练）`取自 G2、G3 run 开始训练前的同一次 eval；日志只保留了四位小数的总准确率，因此其 RM Bench 准确率按总准确率与 IID 准确率反推，标记为近似值。

| Group | IID 准确率 | RM Bench 准确率 | Eval loss | Score diff | Train loss | 训练耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-8B（未训练） | 49.20% | ≈75.46% | 1.8681 | — | — | — |
| G2 + G2_2 | 83.47% | 71.40% | 0.5214 | 2.8011 | 0.4377 | 3:28:21 |
| G3 + G3_2 | **86.01%** | 68.73% | **0.4804** | 2.5703 | 0.5745 | 3:19:24 |
| G2 + G3_2 | 76.40% | **72.58%** | 0.5629 | **3.1004** | **0.3638** | 3:27:11 |
| G3 + G2_2 | 83.88% | 60.40% | 0.5444 | 1.7595 | 0.6237 | 3:20:40 |

## 与 Round 1 基线的对比

下表中的变化为“Round 2 组合 − 对应 Round 1 首段数据”的绝对百分点变化。IID 为 B1--B5（750 条），RM Bench 为其余验证样本（600 条）。

| Round 2 组合 | 对比的 Round 1 基线 | IID：基线 → Round 2 | IID 变化 | RM Bench：基线 → Round 2 | RM Bench 变化 |
| --- | --- | ---: | ---: | ---: | ---: |
| G2 + G2_2 | G2 | 79.87% → 83.47% | **+3.60** | 73.07% → 71.40% | -1.67 |
| G3 + G3_2 | G3 | 84.02% → 86.01% | **+1.99** | 58.90% → 68.73% | **+9.83** |
| G2 + G3_2 | G2 | 79.87% → 76.40% | -3.47 | 73.07% → 72.58% | -0.49 |
| G3 + G2_2 | G3 | 84.02% → 83.88% | -0.13 | 58.90% → 60.40% | +1.50 |

难度标签的相对变化（Round 2 − 对应 Round 1 基线，单位：百分点）：

| Round 2 组合 | Easy 变化 | Normal 变化 | Hard 变化 |
| --- | ---: | ---: | ---: |
| G2 + G2_2 | +1.00 | +2.00 | -8.00 |
| G3 + G3_2 | -4.00 | +11.00 | **+22.50** |
| G2 + G3_2 | -3.50 | -1.50 | +3.50 |
| G3 + G2_2 | -2.50 | +3.50 | +3.50 |

关键解读：

- **G2 + G3_2 值得直接与 G2 对比**：RM Bench 仅下降 **0.49** 个百分点（73.07% → 72.58%），但 IID 下降 **3.47** 个百分点（79.87% → 76.40%）。因此，G3_2 没有带来可见的 RM Bench 泛化收益，反而牺牲了 G2 原有的 IID 能力。
- **G2 + G2_2 是更有效的 G2 延续**：IID 提升 3.60 个百分点，RM Bench 只下降 1.67 个百分点。
- **G3 + G3_2 是最显著的跨分布改进**：在保持并提升 IID 的同时，RM Bench 提升 9.83 个百分点，特别是 Hard 提升 22.50 个百分点，明显缓解了 Round 1 G3 的 RM Bench 偏弱问题。

最终验证的分标签准确率：

| Group | B1 | B2 | B3 | B4 | B5 | Chat | Code | Easy | Hard | Math | Normal | Safety |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B（未训练） | 100.00% | 100.00% | 46.00% | 0.00% | 0.00% | 79.33% | 68.67% | 83.50% | 68.50% | 66.67% | 74.00% | 86.67% |
| G2 + G2_2 | 98.68% | 92.67% | **88.00%** | 74.00% | 64.00% | 68.67% | 64.67% | 93.00% | 46.50% | **67.33%** | **74.50%** | 84.67% |
| G3 + G3_2 | 98.03% | **95.33%** | 84.67% | **78.67%** | **73.33%** | 62.67% | 60.00% | 91.00% | 45.50% | 64.67% | 69.50% | **87.33%** |
| G2 + G3_2 | **99.34%** | 94.67% | 82.00% | 60.00% | 46.00% | **74.00%** | 63.33% | 88.50% | **58.00%** | **67.33%** | 71.00% | 85.33% |
| G3 + G2_2 | 93.42% | 90.67% | 78.67% | 82.67% | 74.00% | 58.67% | 56.67% | 92.50% | 26.50% | 62.00% | 62.00% | 64.00% |

## 当前结论

- **G3 + G3_2 的 IID 最强**，并在 B4、B5 与 Safety 上最佳；它更适合偏重 IID curriculum 分桶的候选。
- **G2 + G2_2 是 IID 与 RM Bench 的折中选择**：IID 仅比最佳低 2.54 个百分点，RM Bench 也保持 71.40%，并在 B3、Math、Normal 最优。
- **G2 + G3_2 是已完成 Round 2 组合中 RM Bench 最高者**（72.58%）：Chat、Hard、整体 score diff 也最佳，但仍低于 Round 1 的 **G2（73.07%）** 0.49 个百分点，且 IID 的 B4/B5 明显较弱。因此，当前所有已训练 Plan C 实验中，RM Bench 的最佳候选仍是 **G2**。
- **G3 + G2_2 没有复现 G3 + G3_2 的跨分布改善**：相对 G3，RM Bench 仅提升 1.50 个百分点、IID 微降 0.13 个百分点；其 RM Bench 显著低于 G3 + G3_2（68.73%）。

## 可追溯性

| Group | 训练配置 | LoRA 输出目录 | W&B run name |
| --- | --- | --- | --- |
| G2 + G2_2 | `examples/train_sfl/curriculum_qwen3_8b_sfl_plan_c_g2_plus_g2_2.yaml` | `saves/curriculum/qwen3-8b-sfl-plan-c-g2-plus-g2-2` | `curriculum_qwen3_8b_sfl_plan_c_g2_plus_g2_2` |
| G3 + G3_2 | `examples/train_sfl/curriculum_qwen3_8b_sfl_plan_c_g3_plus_g3_2.yaml` | `saves/curriculum/qwen3-8b-sfl-plan-c-g3-plus-g3-2` | `curriculum_qwen3_8b_sfl_plan_c_g3_plus_g3_2` |
| G2 + G3_2 | `examples/train_sfl/curriculum_qwen3_8b_sfl_plan_c_g2_plus_g3_2.yaml` | `saves/curriculum/qwen3-8b-sfl-plan-c-g2-plus-g3-2` | `curriculum_qwen3_8b_sfl_plan_c_g2_plus_g3_2` |
| G3 + G2_2 | `examples/train_sfl/curriculum_qwen3_8b_sfl_plan_c_g3_plus_g2_2.yaml` | `saves/curriculum/qwen3-8b-sfl-plan-c-g3-plus-g2-2` | `curriculum_qwen3_8b_sfl_plan_c_g3_plus_g2_2` |

数值来源：各实验输出目录中的 `eval_results.json` 与 `train_results.json`；串行训练日志为 `logs/rm_qwen3_8b_sfl_plan_c_round2_8gpu_20260807.log`。
