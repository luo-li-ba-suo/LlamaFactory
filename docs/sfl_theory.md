# SfL (Score-from-Logits) 理论分析：为什么优于 BT Reward Model

## 1. 两种方法的数学形式

### 1.1 BT Reward Model

BT RM 在基础 LM 之上附加一个随机初始化的 value head，将隐藏状态映射为标量 reward：

$$r_c = \text{ValueHead}(\mathbf{h}_c^{\text{last}}), \quad r_r = \text{ValueHead}(\mathbf{h}_r^{\text{last}})$$

损失函数为 Bradley-Terry 模型的负对数似然：

$$\mathcal{L}_{\text{BT}} = -\log \sigma(r_c - r_r) = -\log \frac{1}{1 + e^{-(r_c - r_r)}} = \text{softplus}(-(r_c - r_r))$$

等价于最大化 $$\sigma(r_c - r_r) = P_{\text{BT}}(\text{chosen} \succ \text{rejected})$$。

### 1.2 SfL (1-5 多分类版本)

SfL 直接在预训练的 LM head 上读取 score token 的 logits，不引入额外参数。对每条 response，模型在 `{"score":` 后的最后一个位置预测分数：

$$\mathbf{s}_c = \text{LMHead}(\mathbf{h}_c^{\text{last}})[\text{score tokens}], \quad \mathbf{s}_c \in \mathbb{R}^K$$

其中 $K = 5$（tokens `"1", "2", "3", "4", "5"`），$\mathbf{s}_c$ 是从全词表 logits 中取出这 $K$ 个 score token 对应的 logits。

通过带温度 $T$ 的 softmax 得到 chosen 和 rejected 各自的分数分布：

$$\mathbf{p}_c = \text{softmax}(\mathbf{s}_c / T), \quad \mathbf{p}_r = \text{softmax}(\mathbf{s}_r / T)$$

核心：将 pairwise 排序建模为「chosen 分数高于 rejected 分数」的概率。

想象一个 $K \times K$ 的联合概率矩阵，第 $i$ 行、第 $j$ 列表示 rejected 得分为 $i$ 且 chosen 得分为 $j$ 的概率 $p_r[i] \cdot p_c[j]$：

```
           chosen score j →
           0        1        2        3        4
      ┌────────┬────────┬────────┬────────┬────────┐
   0  │ p[0,0] │ p[0,1] │ p[0,2] │ p[0,3] │ p[0,4] │  ← p_r[0]
      │   ✗    │   ✓    │   ✓    │   ✓    │   ✓    │
      ├────────┼────────┼────────┼────────┼────────┤
   1  │ p[1,0] │ p[1,1] │ p[1,2] │ p[1,3] │ p[1,4] │  ← p_r[1]
r     │   ✗    │   ✗    │   ✓    │   ✓    │   ✓    │
e  ├────────┼────────┼────────┼────────┼────────┤
j  2  │ p[2,0] │ p[2,1] │ p[2,2] │ p[2,3] │ p[2,4] │  ← p_r[2]
e     │   ✗    │   ✗    │   ✗    │   ✓    │   ✓    │
c  ├────────┼────────┼────────┼────────┼────────┤
t  3  │ p[3,0] │ p[3,1] │ p[3,2] │ p[3,3] │ p[3,4] │  ← p_r[3]
e     │   ✗    │   ✗    │   ✗    │   ✗    │   ✓    │
d  ├────────┼────────┼────────┼────────┼────────┤
   4  │ p[4,0] │ p[4,1] │ p[4,2] │ p[4,3] │ p[4,4] │  ← p_r[4]
s     │   ✗    │   ✗    │   ✗    │   ✗    │   ✗    │
c  └────────┴────────┴────────┴────────┴────────┘
o     p_c[0]   p_c[1]   p_c[2]   p_c[3]   p_c[4]
r
e  ✓ = chosen > rejected (j > i)  计入 P
i  ✗ = chosen ≤ rejected (j ≤ i)  不计入
↓
```

$P(\text{chosen} \succ \text{rejected})$ 就是矩阵**右上三角区域**（不含对角线）所有概率之和：

$$P_{\text{SfL}}(\text{chosen} \succ \text{rejected}) = \sum_{i=0}^{K-1} \sum_{j=i+1}^{K-1} p_c[j] \cdot p_r[i]$$

直观理解：行 $i$、列 $j$ 的单元格贡献了「rejected 得 $i$ 分 + chosen 得 $j$ 分」的联合概率。只有当 $j > i$（右上三角）时，chosen 才算赢了 rejected。loss 为 $-\log P$，训练推动概率质量向右上三角集中。

损失函数为标准最大似然：

$$\mathcal{L}_{\text{SfL}} = -\log\left(\sum_{i=0}^{K-1} \sum_{j=i+1}^{K-1} p_c[j] \cdot p_r[i] + \varepsilon\right)$$

### 1.3 为什么用分布求和而不是直接比期望分数？

一个自然的问题是：既然最终 accuracy 用 $\mathbb{E}[s_c] > \mathbb{E}[s_r]$ 判断，为什么 loss 不直接用 $\mathbb{E}[s_c] - \mathbb{E}[s_r]$（类似回归），而要构造 $\sum_{j>i} p_c[j] \cdot p_r[i]$ 这个求和形式？

**因为监督信号是 pairwise 的，不是 pointwise 的。** 我们不知道 chosen 的真实分数是多少，只知道 chosen > rejected。这个 1-bit label 无法驱动回归式的 pointwise loss（回归需要知道 target score）。

pairwise 求和形式 $\sum_{j>i} p_c[j] \cdot p_r[i]$ 的优点：

1. **正确的概率模型。** 将 $p_c$ 和 $p_r$ 视为两个独立的分类分布。事件「chosen 分数 > rejected 分数」的概率就是两个独立分类随机变量满足 $j > i$ 的联合概率之和。取 $-\log$ 后就是给定 pairwise label 下的负对数似然。

2. **自动处理分布重叠。** 如果 $p_c$ 和 $p_r$ 高度重叠（如都集中在中等分数），$\sum_{j>i} p_c[j] \cdot p_r[i]$ 接近 0.5，loss 较高；随着训练进行，两个分布逐渐分离（$p_c$ 右移、$p_r$ 左移），概率向 1 收敛。整个过程是平滑的、可微的。

3. **梯度分配到所有 token。** 即使 chosen 的 argmax 远大于 rejected 的 argmax，只要分布还有重叠区域，矩阵右上三角外的概率质量仍会产生 loss，驱动分布进一步分离。相比之下，仅比较期望分数 $\mathbb{E}[s_c] - \mathbb{E}[s_r]$ 在排序正确后就无法提供分布"锐度"的优化信号。

### 1.4 两种方法的统一视角

两者本质上都在做同一件事——最大化 $P(\text{chosen} \succ \text{rejected})$，区别在于如何建模这个概率：

| 方法 | 概率模型 | 参数来源 |
|------|---------|---------|
| BT RM | $\sigma(r_c - r_r)$ | value head（随机初始化） |
| SfL | $\sum_{i} \sum_{j>i} p_c[j] \cdot p_r[i]$ | LM head（预训练权重） |

## 2. SfL 为什么能实现热启动

### 2.1 关键机制：Judge Prompt × 预训练 LM Head

SfL 的输入构造流程（以 reward_bench 的 `build_sfl_inputs` 和 LlamaFactory 的 `SfLDatasetProcessor` 为参考）：

**Step 1**：将 judge prompt template（rubric + 评分标准 + 输出格式）填入对话和 response：

```python
judge_text = template.format(conversation=conv, response=response)
```

得到一段完整文本，包含评分标准、对话历史、待评估回复。

**Step 2**：将这段文本作为 user message，应用 chat template：

```python
messages = [{"role": "user", "content": judge_text}]
chat_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
```

**Step 3**：追加 score 前缀，引导模型在下一个 token 预测分数：

```python
full_text = chat_text + '{"score": '
```

最终输入为：

```
<|im_start|>user
{judge rubric — 评分标准/维度}
{conversation — 对话历史}
{response — 待评估回复}
{output format — 输出格式说明}<|im_end|>
<|im_start|>assistant
{"score":
```

模型在最后一个位置（`{"score":` 之后）的 logits 中，取出 score tokens（`"1"` 到 `"5"`）对应的值作为 $\mathbf{s}$。

末尾的 `{"score":` 不是随意选择的——它是 judge prompt 的收束点。

整个输入序列的角色分工：

- **judge prompt（user message 的全部内容）**：包含评分标准（rubric）、对话历史、待评估回复以及输出格式指示。这整段文本作为一个 user message 输入，设置「评判」的上下文。
- **`{"score":` 前缀**：跟在 `<|im_start|>assistant\n` 之后，是生成提示的延续——它模拟了 assistant 开始输出评分 JSON 的状态，迫使模型的下一个 token 必须是分数。

训练时，chosen 和 rejected 两条 response 各自构造一个完整输入，分别经过 LM head 读出 $\mathbf{s}_c$ 和 $\mathbf{s}_r$。

一个经过 instruction-tuning 的 chat model，其训练数据中包含了大量类似格式：一段用户指令（含评分标准和待评内容），紧接着 assistant 开始以 JSON 格式输出分数。模型在预训练和 SFT 阶段见过类似：

```
...请对以上回答从1-5打分：

{"score": 4}
```

`{"score":` 碰巧是 JSON 评分格式的自然开头。因此模型在看到 `{"score":` 时，LM head 已经知道接下来的 token 应该是一个数字——而且这个数字应该反映上文中的 response 质量。

1. **前缀触发「评分模式」**：`{"score":` 作为格式化前缀，提示模型「接下来要输出一个分数」。模型的 attention 机制已经在处理此类模式时学会了将 response 质量信号聚合到最后一个位置的 hidden state。

2. **LM Head 具备评分先验**：hidden state $\mathbf{h}^{\text{last}}$ 通过预训练的 LM head $\mathbf{W}_{\text{LM}} \in \mathbb{R}^{V \times d}$ 投影到词表空间。对于 score tokens（`"1"`, `"2"`, ..., `"5"`），$\mathbf{W}_{\text{LM}}$ 中对应的行向量已经在预训练中学会了：当上文暗示「需要一个评分」时，如何将 hidden state 映射为合理的分数分布。

3. **零样本评判能力**：即使在 SfL 训练之前，一个 instruction-tuned model 已经具备零样本评判 response 质量的能力——给定好的 response，它更倾向于输出大数字；给定差的 response，它更倾向于输出小数字。SfL 只是在这个已有能力上做 fine-tune，而非从零构建。

### 2.2 数据流对比：SfL vs BT

```
SfL:
  hidden state h  ──→  LM Head W_LM (预训练)  ──→  score logits s ∈ R^5
                                                      ↓ softmax
                                                   p ∈ Δ^4 (分数分布)

BT RM:
  hidden state h  ──→  Value Head W_v (随机初始化)  ──→  reward r ∈ R
```

两者的 hidden state $\mathbf{h}^{\text{last}}$ 都来自同一个 backbone，都编码了 response 的语义信息。区别全在**最后一公里的投影**：

| | 投影矩阵 | 初始状态 |
|---|---|---|
| SfL | $\mathbf{W}_{\text{LM}}$（词表嵌入，$V \times d$） | 已在数十亿 token 上训练，知道如何从语义中读出分数 |
| BT | $\mathbf{W}_v$（value head，$1 \times d$） | $\mathcal{N}(0, \sigma^2)$ 随机初始化，完全不知道「好」和「坏」的区别 |

**核心洞察**：SfL 的热启动不来自 backbone 本身的强大（backbone 对两者是共享的），而来自 **LM Head 这个投影矩阵已经预训练好了「从语义到评分」的映射**。BT 把同样的 hidden state 给了一个随机矩阵，当然起点更低。

### 2.3 定量分析：起始 accuracy 差异

在训练开始前（backbone 冻结或刚初始化 LoRA）：

- 给定一对 (chosen, rejected)，backbone 对两个 response 编码出的 hidden state 已有细微差异
- SfL 通过预训练 LM Head 读出：$p_c$ 倾向于高分，$p_r$ 倾向于低分
- 因此 $$P_{\text{SfL}}(\text{chosen} \succ \text{rejected}) = \sum_{j>i} p_c[j] \cdot p_r[i] > 0.5$$
- 典型起始 accuracy：60-70%

- BT 通过随机 Value Head 读出：$r_c - r_r \approx 0$（因为 $\mathbf{W}_v$ 初始均值为零）
- 因此 $$\sigma(r_c - r_r) \approx 0.5$$
- 典型起始 accuracy：~50%（≈ 随机猜测）

BT 需要先用大量训练数据教会 value head「hidden state 的什么方向对应高 reward」，而 SfL 的 LM Head 天生就会做这个投影。

### 2.4 扩展到专用 Critic Model

热启动的分析基于一个通用 chat model（如 Qwen2.5-7B-Instruct）。如果起点换成**专门训练过评判能力的 critic model**，SfL 和 BT 的差距会进一步拉大。

**SfL 可以直接继承 critic model 的全部评判能力。** 一个专门训练过的 critic model，其 LM head 已经针对评分任务充分优化——不仅知道「好回复应该得高分」这种粗略方向，还学会了精细区分相邻分数（如 3 vs 4 的边界）、理解不同评分维度之间的权衡。SfL 用这个模型的 LM head 作为初始化，pairwise fine-tuning 只需在现有分布上做微调（distribution shift 很小），训练极其高效。

**BT 无法完整利用 critic model。** BT 的训练流程是：

1. 取 critic model 的 backbone（去除 LM head）
2. 在 backbone 之上附加一个随机初始化的 value head
3. 用 pairwise 数据训练 value head + backbone

Critic model 的评判能力分布在 **backbone + LM head 的联合管线** 中。Backbone 负责理解对话内容和回复质量，输出富含评判信息的 hidden state；LM head 负责将 signal 映射到 score token。两者在 critic training 中协同优化，形成了端到端的评判通路。

BT 保留了 backbone（hidden state 质量仍然很高），但切断了这条通路——把 LM head 换成了随机 value head。即使 backbone 已经学会了「这个回复应该打 4 分」的表示，value head 需要从零开始学会如何从这个表示中读取出 reward。这等价于：

$$\text{BT on critic} = \text{已训练的 backbone} + \text{随机初始化的 value head}$$

$$\text{SfL on critic} = \text{已训练的 backbone} + \text{已训练的 LM head} + \text{微调}$$

两者都受益于 critic backbone 的强大表示，但 SfL 额外保留了 critic model 的端到端评判管线不做断裂。

## 3. 梯度分析：相同监督信号，不同参数化

两种方法接收的监督信号完全相同——1-bit pairwise label：「chosen 优于 rejected」。区别不在于信息量，而在于**如何用这个 1-bit 信号驱动不同参数化形式的模型**。

### 3.1 梯度方向

**BT RM**：

$$\frac{\partial \mathcal{L}_{\text{BT}}}{\partial r_c} = \sigma(r_c - r_r) - 1 < 0, \quad \frac{\partial \mathcal{L}_{\text{BT}}}{\partial r_r} = 1 - \sigma(r_c - r_r) > 0$$

直觉：拉高 $r_c$，压低 $r_r$。只有一个可调维度（标量差值）。

**SfL**（数值验证正确）：

$$\frac{\partial \mathcal{L}_{\text{SfL}}}{\partial s_c[j]} = -\frac{p_c[j]}{T \cdot P}\left(\sum_{i=0}^{j-1} p_r[i] - P\right)$$

$$\frac{\partial \mathcal{L}_{\text{SfL}}}{\partial s_r[i]} = -\frac{p_r[i]}{T \cdot P}\left(\sum_{j=i+1}^{K-1} p_c[j] - P\right)$$

梯度符号由括号内的项决定。以 $s_c[j]$ 为例：
- 当 $\sum_{i<j} p_r[i] < P$（即 rejected 在 $j$ 以下的质量不够多），梯度为正 → 压低 $s_c[j]$
- 当 $\sum_{i<j} p_r[i] > P$（即 rejected 集中在 $j$ 以下），梯度为负 → 拉高 $s_c[j]$

直观效果：梯度自动在 chosen 分布的高分区域施加正向推动、低分区域施加负向推动；对 rejected 分布则相反。但驱动这些梯度的根本信号，仍是那 1-bit 的 pairwise label。

### 3.2 为什么每次迭代的更新方式确实不同

虽然监督信号相同，**单次梯度更新的信息几何不同**：

- **BT**：在标量空间更新，梯度 $\propto \sigma(r_c - r_r) - 1$。当排序正确时（$r_c \gg r_r$），$\sigma \to 1$，梯度 $\to 0$，进入饱和区。
- **SfL**：在 $K$ 维单纯形 $\Delta^{K-1}$ 上更新。梯度通过 softmax Jacobian 传播到每个 score token。即使 $P(\text{chosen} \succ \text{rejected}) \approx 1$，只要分布形态有优化空间（例如 rejected 的概率质量尚未充分集中在低分区），各 token 仍可获得非零梯度。

换句话说，BT 只有一个自由度可以用这 1-bit 信号，而 SfL 有 $2(K-1)$ 个自由度（两个 softmax 分布的自由度之和）。更多的自由度意味着相同信息量下可以有更细粒度的参数更新。

### 3.3 这也解释了学习率迁移

SfL 可以接管 BT 的训练超参（learning rate 等）直接获得更好的效果，是因为：
- 预训练 LM head 提供了已在正确「语义方向」上的初始化
- 每个 score token 独立接收梯度，分布内部自动平衡，不需要像 BT value head 那样纯靠数据信号找到正确的投影方向

因此 SfL 的"优越性"本质上是**参数初始化和模型结构带来的数据效率优势**，而非获得了更多的监督信息。

## 4. 期望分数：从分布到标量

SfL 可以定义期望分数，将分布压缩为可解释的标量：

$$\mathbb{E}_c[s] = \sum_{k=0}^{K-1} k \cdot p_c[k], \quad \mathbb{E}_r[s] = \sum_{k=0}^{K-1} k \cdot p_r[k]$$

accuracy 用期望分数比较：

$$\text{accuracy} = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}\left[\mathbb{E}_c^{(b)}[s] > \mathbb{E}_r^{(b)}[s]\right]$$

这比 argmax 比较更合理——argmax 丢弃了分布中除最大值外的全部信息，而期望分数利用了完整分布。

## 5. 展望：替代 PPO 中的 Reward Model

RLHF 的 PPO 阶段有三个角色：

| 角色 | 功能 | 输入 | 输出 | 训练数据 |
|------|------|------|------|----------|
| Actor | 生成回复 | prompt | response tokens | PPO policy gradient |
| **Reward Model** | 对完整回复打分 | prompt + response | scalar reward | pairwise ranking |
| Critic | 估计状态价值 $V(s)$ | 当前已生成的 partial sequence | scalar value | MSE regression to return |

**SfL 可以替代的是 Reward Model，不是 Critic。**

Critic 评估的是**状态**——在生成过程中任意中间步，基于当前已输出的部分 token，预测最终会得到多少 reward。它的训练数据是 pointwise 的（目标值 = 实际 return），loss 是 MSE：

$$\mathcal{L}_{\text{critic}} = \frac{1}{T} \sum_t (V(s_t) - R_{\text{final}})^2$$

这与 SfL 的工作方式完全不兼容——SfL 评估的是**完整回复**（在 response 末尾的 `{"score":` 位置读取 logits），训练数据是 pairwise 的，loss 是 ranking loss。不能把 SfL 塞进 MSE regression 的框架里。

但 SfL 可以作为更高质量的 Reward Model，替代 BT RM 进入 PPO/GRPO 流程。

### 5.1 SfL 作为 Reward Model

PPO 中的 RM 对 actor 生成的完整回复输出一个标量 reward $R$，这正是 SfL 天然的能力：

```
Actor:  prompt → autoregressive decode → response
SfL RM: prompt + response → forward → last-position logits
                                       → softmax over score tokens
                                       → E[s] = Σ k · p[k]  → reward R
```

SfL 相比 BT RM 的优势在 PPO 场景中全部保留：
- **热启动**：在进入 PPO 之前已经是高质量的 scorer
- **分数分布**：$p$ 的熵可以衡量 RM 对这条回复的确定程度
- **共享 backbone**：如果 actor 也从同一 base model 初始化，SfL RM 和 actor 可以部分共享参数

### 5.2 配合 GRPO：无需独立 Critic

GRPO（Group Relative Policy Optimization）不训练独立 critic——advantage 通过组内标准化 reward 得到。SfL 可以直接提供 group rewards：

```
prompt → actor → {y_1, y_2, ..., y_N}  (N条采样)
         SfL → {R_1, R_2, ..., R_N}
         normalize → {A_1, A_2, ..., A_N}
```

$$A_k = \frac{R_k - \mu}{\sigma}, \quad \mu = \frac{1}{N}\sum_k R_k, \quad \sigma = \sqrt{\frac{1}{N}\sum_k (R_k - \mu)^2}$$

Actor 的 PPO 更新不变。SfL 的分布 $p_k$ 还能提供额外信息——当 $H(p_k)$ 很高时（RM 对此回复的评分不确定），可以降低该样本的 advantage 权重。

### 5.3 完整训练流程

```
# Phase 1: 训练 SfL Reward Model (pairwise data)
sfl = SfLModel(base_model)
for epoch in range(sfl_epochs):
    for chosen, rejected in pairwise_data:
        loss = -log(P(chosen > rejected))  # SfL loss
        sfl.backward(loss)

# Phase 2: GRPO with SfL reward
actor = PPOActor(base_model, sfl_shared_backbone=True)
for iteration in range(ppo_iterations):
    # 采样
    responses = actor.generate(prompts, num_samples=N)

    # SfL 打分 (no grad)
    rewards = sfl.score(prompts, responses).detach()

    # 组内标准化 → advantage
    advantages = (rewards - rewards.mean()) / rewards.std()

    # PPO 更新 Actor
    actor.backward(ppo_loss(responses, advantages))

    # 可选：SfL RM 继续用新 pairwise 数据更新
    if has_new_pairwise_data:
        sfl.backward(sfl_loss(chosen, rejected))
```

## 6. 总结

| 维度 | BT RM | SfL (1-5) |
|------|-------|-----------|
| 新增参数 | value head（随机初始化） | 无 |
| 热启动 | 否（~50% accuracy） | 是（>60% accuracy） |
| 梯度信号 | 标量、二元 | 全分布、$K$ 维 |
| 饱和行为 | 易饱和 | 不易饱和 |
| 表达能力 | 学到标量排序 | 学到完整分数分布 |
| 训练动态 | 快速收敛后 plateau | 多 epoch 持续上升 |
