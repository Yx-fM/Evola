# 第 3 课：位置编码——从 Sinusoidal 到 RoPE

> *"位置是信息的骨架"*

---

## 1. 理论探讨 (Theory)

### 🧱 原文引用 (The Source)

**原始 Transformer 论文** ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)):

> "Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence..."

**RoPE 论文** ([Su et al., 2021](https://arxiv.org/abs/2104.09864)):

> "We propose Rotary Position Embedding (RoPE) to encode position information by rotating the query and key vectors in the attention mechanism..."

---

### 🧠 深度讲解 (Explanation)

#### 为什么需要位置编码？

**问题**：Self-Attention 是**位置无关的**

```
输入序列: "我 爱 AI"

如果不加位置编码：
  Attention("我", "爱") = Attention("爱", "我")  ← 顺序无关！
  
这就像：
  "我爱AI" 和 "AI爱我" 在模型看来是一样的 ← 错误！
```

**类比**：
- 如果把句子拆散，打乱顺序
- Self-Attention 的输出不变
- 这是因为它只看"内容"，不看"位置"

---

#### 方案一：Sinusoidal Position Encoding（原始方案）

**公式**：
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

其中：
  pos: 位置索引 (0, 1, 2, ...)
  i: 嵌入维度索引
  d: 嵌入总维度
```

**直观理解**：

```
位置 0: [sin(0), cos(0), sin(0/10000), cos(0/10000), ...]
位置 1: [sin(1), cos(1), sin(1/10000), cos(1/10000), ...]
位置 2: [sin(2), cos(2), sin(2/10000), cos(2/10000), ...]

特点：
  - 不同位置有不同的"波形"
  - 相邻位置的编码相似
  - 远距离位置的编码差异大
```

**为什么用 sin/cos？**

| 特性 | 说明 |
|------|------|
| **周期性** | 可以表示任意长度序列 |
| **可加性** | PE(pos+k) ≈ PE(pos) + PE(k)（相对位置） |
| **无参数** | 固定计算，不需要学习 |

---

#### 方案二：Learnable Position Embedding（BERT 方案）

**方式**：直接学习一个位置嵌入矩阵

```python
# 简化理解
max_seq_len = 512
dim = 768

position_embedding = nn.Embedding(max_seq_len, dim)

# 使用
pos_ids = torch.arange(seq_len)  # [0, 1, 2, ...]
pos_enc = position_embedding(pos_ids)  # 学习的位置向量
```

**优缺点**：
- ✅ 可以学习最优的位置表示
- ❌ 序列长度受限（max_seq_len 固定）
- ❌ 无法泛化到超长序列

---

#### 方案三：RoPE (Rotary Position Embedding)——现代主流

**核心思想**：通过**旋转**向量来编码位置

```
┌─────────────────────────────────────────────────────────────┐
│                    RoPE 直观理解                             │
│                                                             │
│  想象一个二维向量 [x, y]：                                   │
│                                                             │
│  ┌─────────┐                                                │
│  │    ↑ y  │                                                │
│  │    │    │    位置 0: 向量指向 (x, y)                      │
│  │    ●───→│    位置 1: 向量旋转 θ°                          │
│  │   / x   │    位置 2: 向量旋转 2θ°                         │
│  └─────────┘                                                │
│                                                             │
│  旋转角度 θ = pos × (基础频率)                               │
│                                                             │
│  旋转后：                                                    │
│    Q' = rotate(Q, θ)                                        │
│    K' = rotate(K, θ)                                        │
│                                                             │
│  注意力计算：                                                │
│    Q' · K' ≈ Q · K + 位置信息                               │
└─────────────────────────────────────────────────────────────┘
```

**数学实现**：

```python
def apply_rope(x, pos, dim):
    """应用 RoPE 旋转位置编码"""
    
    # 1. 计算旋转角度
    freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    theta = pos * freq  # 每个维度不同的旋转角度
    
    # 2. 分成 x, y 两部分
    x1 = x[..., 0::2]  # 偶数维度
    x2 = x[..., 1::2]  # 奇数维度
    
    # 3. 应用旋转
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    # 旋转公式：
    # x1' = x1 * cos - x2 * sin
    # x2' = x1 * sin + x2 * cos
    x1_rotated = x1 * cos_theta - x2 * sin_theta
    x2_rotated = x1 * sin_theta + x2 * cos_theta
    
    # 4. 拼接回来
    return torch.stack([x1_rotated, x2_rotated], dim=-1).flatten(-2)
```

**RoPE 的关键优势**：

| 特性 | Sinusoidal | Learnable | RoPE |
|------|-----------|-----------|------|
| **序列长度** | 无限 | 固定 | 无限 |
| **相对位置** | 间接 | 无 | **直接编码** |
| **长距离泛化** | 较好 | 差 | **很好** |
| **计算方式** | 加法 | 加法 | **乘法（旋转）** |

**为什么 RoPE 更好？**

```
相对位置保持：

对于位置 i 和 j：
  rotate(Q_i, θ_i) · rotate(K_j, θ_j)
  = Q · rotate(K, θ_j - θ_i)  ← 只取决于相对距离 (j-i)
  
这意味着：
  - 位置 1 和位置 5 的关系
  - 与位置 100 和位置 104 的关系
  - 在模型看来是相同的！（相对距离都是 4）
```

---

### 📊 主流模型的位置编码选择

| 模型 | 位置编码方案 | 特点 |
|------|-------------|------|
| **BERT** | Learnable PE | 最大长度 512 |
| **GPT-1/2/3** | Learnable PE | 最大长度 2048 |
| **LLaMA** | RoPE | 无限长度 + 相对位置 |
| **Mistral** | RoPE | 同 LLaMA |
| **PaLM** | RoPE | 同 LLaMA |

---

### 🔗 与 Evola 架构的关联

位置编码对 Evola 的启示：

| Transformer PE | Evola 演化方向 |
|----------------|---------------|
| **离散位置索引** (0, 1, 2, ...) | → **连续时间戳** |
| **固定频率** | → **可调节的时间常数** |
| **仅编码序列位置** | → **编码"存在时间"** |

**核心差异**：
```
Transformer PE:
  - 位置是离散的整数索引
  - 每个词有固定的位置 ID
  - 位置编码与内容相加

Evola 位置编码（构想）:
  - 时间是连续的实数
  - 每个状态有连续的时间戳
  - "位置"由内稳态循环定义
  - 可能用微分方程生成位置信号
```

**潜在设计**：
```python
# 概念性设计（非实现）
class ContinuousTimeEncoding(nn.Module):
    """连续时间位置编码"""
    
    def __init__(self, dim):
        self.time_constant = nn.Parameter(torch.ones(dim))
    
    def forward(self, x, continuous_time):
        """
        Args:
            x: 输入向量
            continuous_time: 连续时间戳（如 0.0, 0.1, 0.2, ...）
        """
        # 类似 RoPE，但时间是连续的
        theta = continuous_time * self.time_constant
        return rotate(x, theta)
```

---

## 2. 实验准备 (Experiment Setup)

在接下来的实验 `lab/03_rope_visualize.py` 中，我们将：

1. **可视化 Sinusoidal PE 的波形**
2. **实现 RoPE 旋转编码**
3. **对比不同位置编码的效果**
4. **观察相对位置保持特性**

---

## 3. 课后思考 (Summary & Reflection)

### 技术思考
1. RoPE 为什么能保持相对位置信息？（数学推导）
2. 为什么 LLaMA 选择 RoPE 而不是 Learnable PE？

### Evola 思考
1. 如果把"位置"改为"存在时间"，如何设计编码？
2. 内稳态循环的时间流速变化，如何影响位置编码？
3. 温态运行时（无外部输入），"位置"应该如何定义？

---

[前往实验脚本：study/lab/03_rope_visualize.py](file:///q:/All_Items/DreamProjects/Evola/study/lab/03_rope_visualize.py)

---

*课程编写日期：2026 年 4 月 15 日*