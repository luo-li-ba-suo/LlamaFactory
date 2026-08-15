# HelpSteer3 multilingual 训练结果汇总

更新时间：2026-08-10。本页汇总本地已保存最终 `all_results.json` 的 multilingual 实验，包括 `saves/helpsteer3/` 中的 SfL 结果与 `/root/model/ckpt/reward_model/` 中的 BT 结果。

## 评估集与可比性

两种 SfL 目标使用不同评估集，不能跨表直接排序：

| 目标 | 评估集 | 样本数 | 可比较的完成模型 |
| --- | --- | ---: | --- |
| BT、SfL-binary | `helpsteer3_bt_eval_multilingual`、`helpsteer3_sfl_binary_eval_multilingual` | 1,267 | 见各自结果表 |
| SfL 1–5 | `helpsteer3_sfl_1_5_eval_multilingual` | 1,267 | Qwen2.5-7B、Qwen3-8B |

每份 multilingual 验证集由 HelpSteer2、HelpSteer3 validation 与 600 条 RM Bench 组成。RM Bench 的 Chat、Code、Math、Safety 各 150 条；因此下文的「RM 四项均值」为这四项准确率的等权平均。Easy、Normal、Hard 是同一批 600 条的难度标签，不应与四个域标签相加。

## BT multilingual

| 模型 | epochs | 总准确率 | HS2 | HS3 val | RM 四项均值 | Chat | Code | Math | Safety | Easy | Normal | Hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-14B-Instruct | 2 | 76.34 | 71.02 | 90.94 | **71.83** | **70.67** | **60.00** | 66.00 | 90.67 | **91.00** | **74.50** | **50.00** |
| Qwen3-8B | 3 | **76.66** | **74.15** | **92.50** | 69.83 | 64.67 | 53.33 | **70.00** | **91.33** | 87.00 | 74.00 | 48.50 |

Qwen3-8B 的总准确率较高，主要来自 HS2（+3.12）与 HS3 validation（+1.56）；Qwen2.5-14B 在 600 条 RM Bench 上较高（+2.00），且 Hard 高 +1.50。两者的训练轮数不同，以上是结果描述而非严格的模型尺寸结论。

| 模型 | Eval loss | Score diff | Train loss | 训练时长 |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-14B-Instruct | **0.6332** | 3.0955 | 0.2739 | 1:18:46 |
| Qwen3-8B | 0.7937 | **3.9538** | **0.2434** | 1:12:12 |

## SfL-binary multilingual

| 模型 | epochs | 总准确率 | HS2 | HS3 val | RM 四项均值 | Chat | Code | Math | Safety | Easy | Normal | Hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct | 3 | 75.24 | 66.19 | 90.31 | 72.50 | **74.00** | 55.33 | 66.67 | **94.00** | 80.50 | 69.50 | **67.50** |
| Qwen2.5-14B-Instruct | 2 | 72.88 | 66.19 | 77.19 | **74.50** | **79.33** | **64.67** | 66.00 | 88.00 | 84.00 | **76.00** | 63.50 |
| Qwen3-8B | 3 | **77.28** | **72.16** | **91.56** | 72.67 | 71.33 | 60.67 | **72.67** | 86.00 | **85.00** | 70.50 | 62.50 |

Qwen3-8B 相比同为 3 epochs 的 Qwen2.5-7B：总准确率 +2.04 个百分点，HS2 +5.97，HS3 validation +1.25；RM 四项均值仅 +0.17。其 Code（+5.34）与 Math（+6.00）提升明显，但 Chat（-2.67）、Safety（-8.00）与 Hard（-5.00）下降。Qwen2.5-14B 只训练 2 epochs，训练量不同，不能据此作严格尺寸结论。

| 模型 | Eval loss | Score diff | Train loss | 训练时长 |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-7B-Instruct | 0.9988 | 5.8999 | 0.2425 | 1:20:48 |
| Qwen2.5-14B-Instruct | 2.1780 | 12.0241 | — | 本地训练指标缺失 |
| Qwen3-8B | **0.7682** | 4.3368 | 0.2614 | 1:38:06 |

## SfL 1–5 multilingual

| 模型 | epochs | 总准确率 | HS2 | HS3 val | RM 四项均值 | Chat | Code | Math | Safety | Easy | Normal | Hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct | 3 | 72.25 | 67.33 | 82.50 | 69.67 | **72.67** | 49.33 | 68.67 | 88.00 | 85.50 | 74.50 | 49.00 |
| Qwen3-8B | 3 | **76.34** | **72.16** | 82.50 | **75.50** | 70.67 | **69.33** | **72.67** | **89.33** | **88.50** | **78.00** | **60.00** |

在相同的 3 epochs 设置下，Qwen3-8B 相比 Qwen2.5-7B：总准确率 +4.09 个百分点，RM 四项均值 +5.83，Code +20.00，Hard +11.00；Chat 是唯一下降项（-2.00）。

| 模型 | Eval loss | Score diff | Train loss | 训练时长 |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-7B-Instruct | 0.9050 | 0.7348 | 0.6408 | 1:21:56 |
| Qwen3-8B | **0.7903** | 0.8734 | 0.6112 | 1:36:47 |

## 配置与产物

| 目标 | 模型 | 训练配置 | LoRA 输出 | 合并 CKPT |
| --- | --- | --- | --- | --- |
| BT | Qwen2.5-14B | `examples/train_lora/helpsteer3_bt_rm_multilingual_14b.yaml` | `helpsteer3-bt-multilingual`（旧冲突目录） | `helpsteer3-bt-multilingual` |
| BT | Qwen3-8B | `examples/train_lora/helpsteer3_bt_rm_multilingual_qwen3_8b.yaml` | `helpsteer3-bt-multilingual-qwen3-8b` | `helpsteer3-bt-multilingual-qwen3-8b` |
| SfL-binary | Qwen2.5-7B | `examples/train_sfl/helpsteer3_sfl_binary_multilingual.yaml` | `saves/helpsteer3/sfl-binary-multilingual` | `helpsteer3-sfl-binary-multilingual-qwen2_5-7b` |
| SfL-binary | Qwen2.5-14B | `examples/train_sfl/helpsteer3_sfl_binary_multilingual_14b.yaml` | `saves/helpsteer3/sfl-binary-14b-multilingual` | — |
| SfL-binary | Qwen3-8B | `examples/train_sfl/helpsteer3_sfl_binary_multilingual_qwen3_8b.yaml` | `saves/helpsteer3/sfl-binary-multilingual-qwen3-8b` | `helpsteer3-sfl-binary-multilingual-qwen3-8b` |
| SfL 1–5 | Qwen2.5-7B | `examples/train_sfl/helpsteer3_sfl_1_5_multilingual.yaml` | `saves/helpsteer3/sfl-1-5-multilingual` | `helpsteer3-sfl-1-5-multilingual-qwen2_5-7b` |
| SfL 1–5 | Qwen3-8B | `examples/train_sfl/helpsteer3_sfl_1_5_multilingual_qwen3_8b.yaml` | `saves/helpsteer3/sfl-1-5-multilingual-qwen3-8b` | `helpsteer3-sfl-1-5-multilingual-qwen3-8b` |

## 未发现本地最终记录的 multilingual 配置

- BT：Qwen2.5-7B、Qwen3-14B。
- SfL-binary：Qwen3-14B（包括 ZeRO3 配置）。

这些配置保留在 `examples/train_lora/` 与 `examples/train_sfl/`，但没有对应的本地最终评估输出；后续完成后可追加到本页。Qwen2.5-7B 的原始 BT 结果已被旧版 Qwen2.5-14B 配置复用输出目录覆盖，无法从当前磁盘恢复。
