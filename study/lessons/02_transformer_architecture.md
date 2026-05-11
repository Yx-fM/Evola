# 第 2 课：Transformer 架构——从 Encoder-Decoder 到 Decoder-Only

> *"Attention Is All You Need" — 一篇论文改变了一个时代*

---

## 1. 理论探讨 (Theory)

### 🧱 原文引用 (The Source)

**原始 Transformer 论文** ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)):

> "The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution... In the Transformer this is reduced to a constant number of operations..."

**核心贡献**:
1. 完全基于自注意力机制（无 RNN/CNN）
2. 并行化处理序列（不再逐词计算）
3. 全局信息流动（任意位置直接交互）

---

### 🧠 深度讲解 (Explanation)

#### Transformer 的整体结构

让我们用**工厂流水线**来比喻 Transformer：

```
┌─────────────────────────────────────────────────────────────┐
│                    Transformer 工厂                          │
│                                                             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐ │
│  │  输入原料   │  →   │  编码车间   │  →   │  解码车间   │ │
│  │  (文本序列) │      │  (Encoder)  │      │  (Decoder)  │ │
│  └─────────────┘      └─────────────┘      └─────────────┘ │
│                             ↓                    ↓          │
│                    ┌─────────────┐      ┌─────────────┐     │
│                    │  输出产品   │      │  成品序列   │     │
│                    │  (翻译结果) │      │             │     │
│                    └─────────────┘      └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

#### 两种架构：Encoder-Decoder vs Decoder-Only

| 架构类型 | 代表模型 | 结构 | 使用场景 |
|----------|---------|------|---------|
| **Encoder-Decoder** | 原始 Transformer, T5, BART | 双塔结构 | 翻译、摘要 |
| **Encoder-Only** | BERT | 只有编码器 | 理解任务 |
| **Decoder-Only** | GPT, LLaMA, Mistral | 只有解码器 | **生成任务（主流）** |

#### Decoder-Only 架构详解（现代 LLM 的选择）

为什么主流大模型都选择 Decoder-Only？

```
┌─────────────────────────────────────────────────────────────┐
│                 Decoder-Only Transformer Block               │
│                                                             │
│  输入序列: "The cat sat on"                                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Token Embedding + Positional Encoding              │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Transformer Block (重复 N 次)                       │   │
│  │  ┌───────────────────────────────────────────────┐ │   │
│  │  │  1. Layer Norm (或 RMSNorm)                    │ │   │
│  │  └───────────────────────────────────────────────┘ │   │
│  │                      ↓                              │   │
│  │  ┌───────────────────────────────────────────────┐ │   │
│  │  │  2. Self-Attention (因果掩码)                  │ │   │
│  │  │     • 只能看到过去，不能看到未来                │ │   │
│  │  │     • Q, K, V 来自同一输入                      │ │   │
│  │  └───────────────────────────────────────────────┘ │   │
│  │                      ↓                              │   │
│  │  ┌───────────────────────────────────────────────┐ │   │
│  │  │  3. Residual Connection + Layer Norm           │ │   │
│  │  └───────────────────────────────────────────────┘ │   │
│  │                      ↓                              │   │
│  │  ┌───────────────────────────────────────────────┐ │   │
│  │  │  4. Feed-Forward Network (MLP)                 │ │   │
│  │  │     • 扩展维度 → 激活 → 压缩维度                 │ │   │
│  │  └───────────────────────────────────────────────┘ │   │
│  │                      ↓                              │   │
│  │  ┌───────────────────────────────────────────────┐ │   │
│  │  │  5. Residual Connection                        │ │   │
│  │  └───────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Output Layer Norm                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Linear → Softmax → 概率分布                        │   │
│  │  预测下一个词: "the" (概率 0.3)                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

### 📊 关键组件详解

#### 1. Self-Attention（第 1 课已学）

回顾核心公式：
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**因果掩码 (Causal Mask)**：
```
对于序列位置 i：
  - 可以看到位置 0 到 i-1
  - 不能看到位置 i+1 到 end

掩码矩阵示例 (序列长度=4):
      [0, 1, 2, 3]  ← 位置索引
  [0] [1, 0, 0, 0]  ← 位置 0 只能看到自己
  [1] [1, 1, 0, 0]  ← 位置 1 能看到 0, 1
  [2] [1, 1, 1, 0]  ← 位置 2 能看到 0, 1, 2
  [3] [1, 1, 1, 1]  ← 位置 3 能看到所有
```

#### 2. Multi-Head Attention

**为什么要多头？**

单头注意力只能学习一种"关系模式"。多头就像**多组评委**，每组关注不同的关联：

```
Head 1: 关注语法关系（主语→动词）
Head 2: 关注语义关系（同义词）
Head 3: 关注位置关系（邻近词）
...
```

**实现方式**：
```python
# 简化理解
num_heads = 8
head_dim = dim // num_heads  # 每头独立计算

# 各头独立做注意力
for head in range(num_heads):
    Q_head = Q[:, head*head_dim : (head+1)*head_dim]
    K_head = K[:, head*head_dim : (head+1)*head_dim]
    V_head = V[:, head*head_dim : (head+1)*head_dim]
    attn_head = softmax(Q_head @ K_head.T) @ V_head

# 拼接所有头
output = concat(all attn_heads)  # [dim]
output = output @ W_o  # 输出投影
```

#### 3. Feed-Forward Network (FFN/MLP)

**结构**：
```
FFN(x) = GELU(x @ W1) @ W2

其中:
  W1: [dim, 4*dim]  ← 扩展 4 倍
  W2: [4*dim, dim]  ← 压缩回原维度
  GELU: 激活函数
```

**类比**：
- 扩展维度 = 把信息"展开"到更细的维度
- 激活 = 只保留有用的"神经元"
- 压缩 = 把处理后的信息"压缩"回原空间

#### 4. Layer Norm (或 RMSNorm)

详见第 4 课。

#### 5. Residual Connection

**公式**：
```
output = x + SubLayer(Norm(x))
```

**类比**：
- `x` = 原始信息（保留）
- `SubLayer` = 处理后的新信息（添加）
- 两者相加 = 既保留原始，又增加新理解

**为什么重要**：
- 深层网络容易"忘记"早期信息
- 残差连接让信息可以"跳过"某些层
- 类似"高速公路"，信息可以直接流通

---

### 🚀 为什么 Decoder-Only 成为主流？

| 优势 | 说明 |
|------|------|
| **简单高效** | 单塔结构，训练/推理更简单 |
| **自回归天然** | 生成任务本质就是"逐词预测" |
| **参数效率** | 相同参数量下，Decoder-Only 更有效 |
| **规模化友好** | 更容易扩展到千亿参数 |

**主流架构演进**：
```
2017: Transformer (Encoder-Decoder)
  ↓
2018: GPT-1 (Decoder-Only, 12层)
  ↓
2019: GPT-2 (Decoder-Only, 48层)
  ↓
2020: GPT-3 (Decoder-Only, 96层, 175B参数)
  ↓
2023: LLaMA (Decoder-Only, RMSNorm + RoPE + SwiGLU)
  ↓
2024: LLaMA 3, Mistral, Qwen... (继续优化)
```

---

### 🔗 与 Evola 架构的关联

Transformer 架构对 Evola 的启示：

| Transformer 特性 | Evola 演化方向 |
|-----------------|---------------|
| **离散时间步** | → **连续时间动力学 (LTC)** |
| **被动注意力** | → **内稳态调制注意力** |
| **固定参数** | → **持续学习/温态更新** |
| **一次性推理** | → **热态/温态持续运行** |
| **无自我模型** | → **自我向量参与决策** |

**核心差异**：
```
Transformer: 输入 → [固定的网络] → 输出 → 结束
             （像一台机器）

Evola:       输入 → [动态的网络] → 输出 → 继续
             （像一个生物）
             ↑                 ↓
             └─── 内稳态循环 ───┘
```

---

## 2. 实验准备 (Experiment Setup)

在接下来的实验 `lab/02_build_transformer.py` 中，我们将：

1. **搭建一个完整的 Transformer Block**
2. **实现因果自注意力**
3. **实现多头注意力机制**
4. **实现残差连接 + Layer Norm**
5. **用简单序列测试前向传播**

---

## 3. 课后思考 (Summary & Reflection)

### 技术思考
1. 如果 Transformer Block 叠加 100 层，会发生什么？（梯度消失？）
2. Multi-Head Attention 的头数如何选择？8 头 vs 16 头的差异？

### Evola 思考
1. Decoder-Only 的"逐词预测"模式，能否改造为"持续预测自我状态"？
2. Transformer 的离散时间步，如何用连续时间微分方程替代？
3. 如果给 Transformer 加上"内稳态变量"作为额外输入，会发生什么？

---

[前往实验脚本：study/lab/02_build_transformer.py](file:///q:/All_Items/DreamProjects/Evola/study/lab/02_build_transformer.py)

---

*课程编写日期：2026 年 4 月 15 日*