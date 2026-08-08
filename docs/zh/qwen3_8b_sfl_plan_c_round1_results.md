# Qwen3-8B SfL Plan C Round 1 实验记录

训练日期：2026-08-07。三个实验在 8 张 GPU 上依次完成，均使用同一份 1,350 条带标签验证集。

## 共同训练设置

- 基座模型：`../../model/Qwen3-8B`
- 方法：SfL LoRA（`rank=16`、`alpha=32`、`target=all`）
- 模板：`qwen3`，`enable_thinking: false`
- 评分 token：`0,1`；评分前缀：`{"score": `；温度：`5.0`
- 训练：2 epochs，`per_device_train_batch_size=1`，8 卡 DDP，`gradient_accumulation_steps=16`，全局有效 batch size 为 128
- 学习率：`5e-5`，cosine scheduler，warmup ratio `0.1`，`cutoff_len=4096`
- 每个训练集约 20,000 个偏好样本；训练期间每 32 steps 验证一次。
- 验证集共 1,350 条：`B1`--`B5` 为 IID 子集（750 条），其余标签为 RM Bench 子集（600 条）。

## 最终结果

| Group | 采样/筛选策略 | IID 准确率 | RM Bench 准确率 | Eval loss | Score diff | Train loss | 训练耗时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| G1 | 原始数据，随机采样 | 76.27% | 70.92% | 0.5047 | 1.6617 | 0.4902 | 2:17:50 |
| G2 | Teacher-filtered，随机采样 | 79.87% | **73.07%** | 0.6503 | **3.8162** | 0.2892 | 2:20:13 |
| G3 | Teacher-filtered，主动采样 | **84.02%** | 58.90% | 0.6358 | 1.6814 | 0.6073 | 2:13:47 |

最终验证的分标签准确率：

| Group | B1 | B2 | B3 | B4 | B5 | Chat | Code | Easy | Hard | Math | Normal | Safety |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G1 | 100.00% | 95.33% | 75.33% | 61.33% | 49.33% | 70.00% | 66.67% | 93.00% | 50.50% | 62.00% | 69.00% | 84.67% |
| G2 | 98.68% | 95.33% | **82.00%** | 66.67% | 56.67% | **70.67%** | **68.67%** | 92.00% | **54.50%** | **67.33%** | **72.50%** | **85.33%** |
| G3 | 93.42% | 88.67% | **82.00%** | **78.67%** | **77.33%** | 52.00% | 62.67% | **95.00%** | 23.00% | 65.33% | 58.50% | 55.33% |

## 结论

- **G2 是最均衡的候选**：IID 与 RM Bench 都显著优于 G1，且 RM Bench 为三组最佳。
- **G3 明显偏向 IID 的 B3--B5**：IID 最高、B4/B5 最佳，但 RM Bench 降至 58.90%，说明主动采样引入了较强的分布偏置，不能作为通用 reward model 的首选。
- **G1 是稳健基线**：总体明显弱于 G2，尤其是 B3--B5，但其通用标签表现比 G3 均衡。

## 产物与可追溯性

| Group | 训练配置 | LoRA 输出目录 | W&B run name |
| --- | --- | --- | --- |
| G1 | `examples/train_sfl/curriculum_qwen3_8b_sfl_plan_c_g1_raw_random.yaml` | `saves/curriculum/qwen3-8b-sfl-plan-c-g1-raw-random` | `curriculum_qwen3_8b_sfl_plan_c_g1_raw_random` |
| G2 | `examples/train_sfl/curriculum_qwen3_8b_sfl_plan_c_g2_teacher_filtered_random.yaml` | `saves/curriculum/qwen3-8b-sfl-plan-c-g2-teacher-filtered-random` | `curriculum_qwen3_8b_sfl_plan_c_g2_teacher_filtered_random` |
| G3 | `examples/train_sfl/curriculum_qwen3_8b_sfl_plan_c_g3_teacher_filtered_active.yaml` | `saves/curriculum/qwen3-8b-sfl-plan-c-g3-teacher-filtered-active` | `curriculum_qwen3_8b_sfl_plan_c_g3_teacher_filtered_active` |

原始串行训练日志：`logs/rm_qwen3_8b_sfl_plan_c_8gpu_20260807.log`。各输出目录中的 `eval_results.json` 与 `train_results.json` 是本记录的数值来源。
