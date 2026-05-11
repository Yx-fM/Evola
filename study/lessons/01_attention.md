# 第 1 课：主流模型的灵魂——自注意力机制 (Self-Attention)

> “注意力是你唯一需要的。” —— *Vaswani et al., 2017*

## 1. 理论探讨 (Theory)

### 🧱 原文引用 (The Source)
在论文《Attention Is All You Need》中，作者定义自注意力为：
> "An attention function can be described as mapping a query and a set of key-value pairs to an output... The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key."

### 🧠 深度讲解 (Explanation)
为了理解这段话，我们需要抛弃“矩阵”和“张量”这种冷冰冰的词汇，把模型想象成一个**图书馆管理员**或一个**社交高手**。

自注意力的核心在于：**让序列中的每一个元素，都去环顾四周，看看谁对自己最重要。**

#### 核心概念：Q, K, V
我们可以用“相亲”或者“联谊”来打比方：
1.  **Query (Q - 查询)**：像是一个人手持的“征婚启事”——**“我正在找什么样的人？”**
2.  **Key (K - 键)**：像是一个人挂在胸前的“名牌”——**“我是什么样的人？”**
3.  **Value (V - 值)**：像是一个人的“真实内涵”——**“如果选中了我，我能提供什么信息？”**

#### 运作过程
1.  **打分 (Matching)**：模型算出每个人的 Q 和其他人的 K 的相似度（点积）。相似度越高，分数越高。
2.  **分配注意力 (Weighting)**：通过 Softmax 函数将分数转换成百分比（加起来等于 1）。比如：某个单词对自己贡献了 80% 的注意力，对隔壁单词贡献了 20%。
3.  **提取信息 (Aggregation)**：根据这些百分比，把大家的 V（真实内涵）加权求和，得到最终的输出。

### 🚀 为什么它是“主流”？
在它之前，循环神经网络（RNN）像一条细长的水管，信息必须按顺序一滴一滴流过去，流到后面，前面的就忘了。
而 **Self-Attention 是“全局视野”**：它像一张巨大的社交网络，所有单词一瞬间就能看到彼此，无论距离多远。

---

## 2. 实验准备 (Experiment Setup)

在接下来的实验 `lab/01_attention_minimal.py` 中，我们将手动模拟这个过程。
我们将用 3 个单词组成的序列，每个单词用一个简单的向量表示，看看它们是如何互相“吸引”的。

---

## 3. 课后思考 (Summary & Reflection)
- 如果 Q 和 K 完全一样，会发生什么？
- 在我们 `Evola` 的构想中，这种“环顾四周”的注意力能否被用来感知“内部饥饿感”？

[前往实验脚本：study/lab/01_attention_minimal.py](file:///q:/All_Items/DreamProjects/Evola/study/lab/01_attention_minimal.py)
