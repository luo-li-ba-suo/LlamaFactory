# HelpSteer3 English 训练结果汇总

更新时间：2026-08-10。本页汇总本地可找到 `all_results.json` 的 English 实验。训练产物分别位于 `saves/helpsteer3/`（SfL）及 `/root/model/ckpt/reward_model/`（BT）。

## 评估集与可比性

| 目标 | 评估集 | 样本数 | 说明 |
| --- | --- | ---: | --- |
| BT | `helpsteer3_bt_eval_english` | 1,592 | 仅在 BT 表内横比 |
| SfL-binary | `helpsteer3_sfl_binary_eval_english` | 1,592 | 仅在 SfL 表内横比 |

两份评估集的样本对不重合，不能把 BT 与 SfL-binary 的总准确率直接排序。两者均含 600 条 RM Bench：Chat、Code、Math、Safety 各 150 条；下文的「RM 四项均值」是这四项准确率的等权平均，也等价于这 600 条样本的准确率。Easy、Normal、Hard 是同一批 RM Bench 的正交难度标签，不应与领域标签相加。

## BT English

| 模型 | epochs | 总准确率 | HS2 | HS3 val | RM 四项均值 | Chat | Code | Math | Safety | Easy | Normal | Hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-14B-Instruct | 2 | 76.15 | **76.70** | 82.66 | 70.00 | **68.67** | 52.00 | 72.67 | 82.67 | **89.50** | **72.50** | 45.00 |
| Qwen3-8B | 3 | **78.48** | 75.57 | **87.19** | **70.83** | 68.00 | **57.33** | **74.67** | **83.33** | 85.50 | 72.50 | **54.50** |

Qwen3-8B 总准确率高 2.33 个百分点，RM Bench 高 0.83，且 Hard 高 9.50；Qwen2.5-14B 的 HS2 高 1.14、Easy 高 4.00。训练轮数不同，以上仅作实验结果描述。

| 模型 | Eval loss | Score diff | Train loss | 训练时长 |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-14B-Instruct | **0.5510** | 2.0239 | 0.4259 | 5:05:37 |
| Qwen3-8B | 1.0341 | **6.1369** | **0.2617** | 7:06:47 |

## SfL-binary English

| 模型 / 设置 | epochs | 总准确率 | HS2 | HS3 val | RM 四项均值 | Chat | Code | Math | Safety | Easy | Normal | Hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-7B, accum. 8 | 3 | **76.82** | 67.61 | **89.38** | 69.33 | **70.67** | 53.33 | **70.67** | 80.67 | **84.50** | 66.50 | 55.50 |
| Qwen2.5-7B, accum. 32 | 3 | 75.94 | 72.73 | 82.66 | **70.67** | **70.67** | **58.00** | 70.00 | 84.00 | 83.50 | **70.50** | **58.00** |
| Qwen2.5-14B, accum. 8 | 2 | 69.41 | 66.19 | 66.41 | **74.50** | **79.33** | **64.67** | 66.00 | 88.00 | 84.00 | **76.00** | **63.50** |
| Qwen3-8B, accum. 8 | 3 | **79.02** | **76.14** | **88.59** | 70.50 | 64.00 | 56.00 | **72.67** | **89.33** | **89.00** | 68.50 | 54.00 |

在可用的完整训练记录中，Qwen3-8B 的总准确率最高（79.02），并有最高的 HS2、Safety 和 Easy；Qwen2.5-14B 的 RM Bench 最高（74.50），但 HS2/HS3 较低。Qwen2.5-7B 的 accumulation 8 与 32 实验显示，总准确率分别为 76.82 与 75.94，而 RM Bench 分别为 69.33 与 70.67，存在总体与 RM Bench 的取舍。

| 模型 / 设置 | Eval loss | Score diff | Train loss | 训练时长 |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-7B, accum. 8 | 1.2568 | **7.5835** | **0.2726** | 7:42:59 |
| Qwen2.5-7B, accum. 32 | **0.6717** | 2.9497 | 0.4002 | 5:13:47 |
| Qwen2.5-14B, accum. 8 | 2.3419 | 9.7151 | — | 本地训练指标缺失 |
| Qwen3-8B, accum. 8 | 1.0209 | 5.9266 | 0.3093 | 9:21:31 |

## 仅有评估、不可作为完成训练结果的产物

下列目录只有评估文件（没有完整的训练状态、adapter 或训练指标），因此不并入上面的训练比较：

| 配置 / 产物 | 评估目标 | 总准确率 | RM 四项均值 | 结论 |
| --- | --- | ---: | ---: | --- |
| `sfl-binary-english-qwen3-14b` | SfL-binary | 72.48 | 79.33 | 未证实完成训练 |
| `helpsteer3-bt-english-qwen3-14b` | BT | 48.33 | 46.33 | 未证实完成训练 |
| `helpsteer3-rm-english-qwen3-14b-full-z3` | BT / full + ZeRO3 | 54.44 | 52.17 | 训练前评估后被停止 |

另外，Qwen2.5-14B 的 SfL accumulation 32 目录与 accumulation 8 记录完全相同，视为重复结果，不单列；Qwen3-8B 的 SfL accumulation 32 与 BT accumulation 32 尚无最终 `all_results.json`。

## 产物与已知目录冲突

| 目标 | 实际保留模型 | 配置 | 当前产物目录 |
| --- | --- | --- | --- |
| BT | Qwen2.5-14B | `examples/train_lora/helpsteer3_bt_rm_english_14b.yaml` | `helpsteer3-bt-english` |
| BT | Qwen3-8B | `examples/train_lora/helpsteer3_bt_rm_english_qwen3_8b.yaml` | `helpsteer3-bt-english-qwen3-8b` |
| SfL-binary | Qwen2.5-7B | `examples/train_sfl/helpsteer3_sfl_binary_english.yaml` | `saves/helpsteer3/sfl-binary-english` |
| SfL-binary | Qwen2.5-14B | `examples/train_sfl/helpsteer3_sfl_binary_english_14b.yaml` | `saves/helpsteer3/sfl-binary-14b-english` |
| SfL-binary | Qwen3-8B | `examples/train_sfl/helpsteer3_sfl_binary_english_qwen3_8b.yaml` | `saves/helpsteer3/sfl-binary-english-qwen3-8b` |

Qwen2.5-7B 与 Qwen2.5-14B 的旧 English BT 配置复用了 `helpsteer3-bt-english`，当前目录已保存为 14B 产物，7B 的 BT 结果不可恢复。后续运行 14B 前应先给该配置配置独立 `output_dir`，以免再次覆盖。
