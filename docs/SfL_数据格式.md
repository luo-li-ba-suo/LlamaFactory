# LlamaFactory 偏好训练数据格式

## 1. BT/DPO 数据（标准 pairwise 格式）

用于 `stage: dpo`（及 `rm`、`kto` 等偏好训练阶段）。

### 格式：sharegpt JSON

```json
[
  {
    "conversations": [
      {"from": "human", "value": "第一个 user 消息"},
      {"from": "gpt", "value": "第一个 assistant 回复"},
      {"from": "human", "value": "第二个 user 消息"}
    ],
    "chosen": {
      "from": "gpt",
      "value": "好的 assistant 回复"
    },
    "rejected": {
      "from": "gpt",
      "value": "差的 assistant 回复"
    },
    "system": "可选的 system prompt（不填则用模型默认）"
  }
]
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `conversations` | `list` | 对话历史，user/assistant 交替。最后一条必须是 user（作为 prompt） |
| `chosen` | `dict` | `from: "gpt"` + `value`，更好的 assistant 回复 |
| `rejected` | `dict` | `from: "gpt"` + `value`，更差的 assistant 回复 |
| `system` | `str` | 可选。不填则用模型 chat template 的 `default_system`（如 "You are a helpful assistant."） |

### 拼接逻辑（repo 自动完成）

```
<  system  >  ← 可选，来自 system 字段或 default_system
<  user    >  ← conversations 中的 user 轮次
<assistant >  ← conversations 中的 assistant 轮次（如果有历史对话）
<  user    >  ← conversations 的最后一轮（prompt）
<assistant >  ← chosen.value 或 rejected.value
               ↑ format_assistant 自动包 <|im_start|>assistant\n{content}<|im_end|>
```

labels: prompt 部分为 IGNORE（-100），assistant 回复部分为真实 token ids（计算 CE loss）。

### dataset_info.json 配置

```json
{
  "your_dpo_data": {
    "file_name": "your_dpo_data.json",
    "formatting": "sharegpt",
    "ranking": true,
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt"
    }
  }
}
```

---

## 2. SfL 数据（Score-from-Logits 格式）

用于 `stage: sfl`。

### 格式：简单 JSON/JSONL

```json
[
  {
    "chosen": "你是一个文本质量评价员，负责评估助手回复的质量。\n\n# 对话历史\n### USER(idx: 0)\n你好\n### ASSISTANT(idx: 0)\n你好！有什么可以帮你的？\n\n# 当前回复（待评估）\n你好\n\n# 输出格式\n请严格按照以下 JSON 格式输出评分：",
    "rejected": "你是一个文本质量评价员，负责评估助手回复的质量。\n\n# 对话历史\n### USER(idx: 0)\n你好\n### ASSISTANT(idx: 0)\n你好！有什么可以帮你的？\n\n# 当前回复（待评估）\n滚开\n\n# 输出格式\n请严格按照以下 JSON 格式输出评分："
  }
]
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `chosen` | `str` | 完整的评估 prompt（含评价指令 + 对话历史 + 好的待评估回复）。对应最终得分更高的一方 |
| `rejected` | `str` | 完整的评估 prompt（含评价指令 + 对话历史 + 差的待评估回复）。对应最终得分更低的一方 |

### 拼接逻辑（repo 自动完成）

SfL 不走 conversations/chosen/rejected 的 sharegpt 路径，而是直接将 chosen/rejected 作为 user message 内容：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>          ← default_system（模板自动）
<|im_start|>user
{chosen 文本内容}<|im_end|>
<|im_start|>assistant\n                         ← add_generation_prompt=True 自动添加
{"score":                                        ← sfl_prefix_str 拼接（默认 '{"score": '）
                                                  ↑ 模型从此处预测下个 token（score digit）
```

rejected 同样结构，独立序列。

**关键特性**：
- labels 全部为 IGNORE（-100），不计算 CE loss
- 仅取 `{"score":` 拼接后的下一个位置 logit，只对 score token ids 做 pairwise ranking loss
- chosen 和 rejected 各自独立 tokenize，左 pad 到 batch 中最长长度
- 支持多轮 SfL 训练（无需 ref model）

### SfL loss 逻辑

- **二分类（score_tokens: "0,1"）**：`loss = -log_sigmoid(logit_1 - logit_0)_chosen - (logit_1 - logit_0)_rejected`
  即 chosen 的 "1" 相对 "0" 的优势应大于 rejected

- **多分类（score_tokens: "1,2,3,4,5"）**：
  ```
  p_chosen   = softmax(logits_chosen / temperature)     # [K] 概率分布
  p_rejected = softmax(logits_rejected / temperature)
  loss = Σ_i Σ_{j>i} p_chosen[j] * p_rejected[i]        # 上三角 pairwise
  ```
  即惩罚 rejected 得分高于 chosen 的配置

### 训练配置示例

```yaml
stage: sfl
finetuning_type: lora  # 或 full、freeze
sfl_score_tokens: "0,1"  # 二分类；多分类用 "1,2,3,4,5"
sfl_prefix_str: '{"score": '  # score 前缀（默认值）
sfl_temperature: 5.0  # 多分类 softmax 温度
```

### 数据集配置

SfL 不在 `dataset_info.json` 中注册，直接在训练配置的 `dataset` 字段指定文件路径：

```yaml
dataset: /path/to/sfl_data.json  # 或 data/sfl_train.json
```

---

## 3. 格式对比总结

| | BT/DPO | SfL |
|------|--------|-----|
| 文件格式 | sharegpt JSON | 简单 JSON/JSONL |
| prompt 来源 | `conversations` 列表 | chosen/rejected 文本自身 |
| 待评估内容 | `chosen` / `rejected`（assistant 回复） | 嵌入在 chosen/rejected 文本中 |
| chat template 处理 | `encode_oneturn`（prompt+response） | `apply_chat_template(user, add_generation_prompt=True)` |
| labels | prompt = IGNORE, response = token ids（CE loss） | 全部 IGNORE（无 CE loss） |
| loss | DPO/ORPO/SimPO 偏好 loss | 分数 token logit 的 pairwise ranking loss |
| ref model | 需要（DPO）或不需要（ORPO/SimPO） | 不需要 |
| 注册方式 | `dataset_info.json` | 配置文件 `dataset` 字段直接写路径 |
