# 第 4 课：归一化——LayerNorm 到 RMSNorm

> *"归一化是网络的稳定器"*

---

## 1. 理论探讨 (Theory)

### 🧱 原文引用 (The Source)

**Layer Normalization 论文** ([Ba et al., 2016](https://arxiv.org/abs/1607.06450)):

> "We propose layer normalization, a normalization method that normalizes the inputs across the features instead of across the batch dimension..."

**RMSNorm 论文** ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)):

> "We propose Root Mean Square Layer Normalization (RMSNorm), a simplified variant of LayerNorm that computes the root mean square instead of mean and variance..."

---

### 🧠 深度讲解 (Explanation)

#### 为什么需要归一化？

**问题**：深层网络的数值会**爆炸或消失**

```
┌─────────────────────────────────────────────────────────────┐
│                    数值不稳定问题                             │
│                                                             │
│  层 1: 输入 [1.0, 2.0, 3.0]                                  │
│        ↓                                                    │
│  层 2: 输出 [10.0, 20.0, 30.0]  ← 放大了                     │
│        ↓                                                    │
│  层 3: 输出 [100.0, 200.0, 300.0]  ← 继续放大                │
│        ↓                                                    │
│  层 4: 输出 [1000.0, 2000.0, 3000.0]  ← 爆炸！               │
│                                                             │
│  结果：                                                      │
│    - 梯度计算不稳定                                          │
│    - 学习率需要极小                                          │
│    - 训练收敛困难                                            │
└─────────────────────────────────────────────────────────────┘
```

**类比**：
- 不归一化 = 没有刹车，速度越来越快
- 归一化 = 定期"减速"，保持稳定

---

#### 方案一：Batch Normalization (BN)

**公式**：
```
BN(x) = (x - μ_batch) / σ_batch * γ + β

其中：
  μ_batch: 批次均值
  σ_batch: 批次标准差
  γ, β: 可学习参数
```

**问题**：
- BN 需要**大批次**才能准确计算均值/方差
- 小批次时，均值/方差不准
- 对序列数据（Transformer）不友好

---

#### 方案二：Layer Normalization (LN)

**核心思想**：在**单个样本**内部归一化

```
┌─────────────────────────────────────────────────────────────┐
│              Batch Norm vs Layer Norm                        │
│                                                             │
│  Batch Norm:                                                │
│    ┌─────────────┬─────────────┬─────────────┐             │
│    │  样本 1     │  样本 2     │  样本 3     │             │
│    │  [a1,b1,c1] │  [a2,b2,c2] │  [a3,b3,c3] │             │
│    └─────────────┴─────────────┴─────────────┘             │
│         ↓            ↓            ↓                         │
│    沿着 **批次方向** 计算均值和方差                           │
│    μ = mean([a1,a2,a3])                                     │
│                                                             │
│  Layer Norm:                                                │
│    ┌─────────────┬─────────────┬─────────────┐             │
│    │  样本 1     │  样本 2     │  样本 3     │             │
│    │  [a1,b1,c1] │  [a2,b2,c2] │  [a3,b3,c3] │             │
│    └─────────────┴─────────────┴─────────────┘             │
│         ↓                                                   │
│    沿着 **特征方向** 计算均值和方差                           │
│    μ1 = mean([a1,b1,c1])  ← 每个样本独立计算                │
│    μ2 = mean([a2,b2,c2])                                    │
│    μ3 = mean([a3,b3,c3])                                    │
└─────────────────────────────────────────────────────────────┘
```

**公式**：
```
LN(x) = (x - μ_layer) / σ_layer * γ + β

其中：
  μ_layer = mean(x)  ← 单个样本所有特征的平均
  σ_layer = sqrt(mean((x - μ_layer)^2))
  γ, β: 可学习参数（缩放和偏移）
```

**代码实现**：
```python
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x):
        # x: [batch, seq_len, dim]
        
        # 1. 计算均值和方差（沿着 dim 方向）
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 2. 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 3. 缩放和偏移
        return self.gamma * x_norm + self.beta
```

---

#### 方案三：RMSNorm (Root Mean Square Normalization)

**LLaMA 的选择**——简化版的 LayerNorm

**核心思想**：只计算 RMS，不需要均值和方差

```
┌─────────────────────────────────────────────────────────────┐
│                    LN vs RMSNorm                             │
│                                                             │
│  LayerNorm:                                                 │
│    1. 计算均值 μ = mean(x)                                  │
│    2. 计算方差 σ² = mean((x - μ)^2)                         │
│    3. 归一化: (x - μ) / σ                                    │
│    4. 缩放偏移: γ * x_norm + β                               │
│                                                             │
│  RMSNorm:                                                   │
│    1. 计算 RMS = sqrt(mean(x^2))                            │
│    2. 归一化: x / RMS                                        │
│    3. 缩放: γ * x_norm                                        │
│                                                             │
│  简化：去掉均值计算和偏移参数                                  │
└─────────────────────────────────────────────────────────────┘
```

**公式**：
```
RMSNorm(x) = x / RMS(x) * γ

其中：
  RMS(x) = sqrt(mean(x^2) + ε)
  γ: 可学习的缩放参数
```

**代码实现**：
```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        # x: [batch, seq_len, dim]
        
        # 1. 计算 RMS
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # 2. 归一化（不需要减均值）
        x_norm = x / rms
        
        # 3. 只缩放（无偏移）
        return self.gamma * x_norm
```

**为什么 LLaMA 选择 RMSNorm？**

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| **计算量** | 需要均值+方差 | 只需 RMS |
| **参数量** | γ + β | 只有 γ |
| **训练速度** | 较慢 | **更快** |
| **效果** | 好 | **相近** |

---

### 📊 Transformer 中的归一化位置

有两种布局方式：

#### Pre-Norm（现代主流）
```
x → Norm → Attention → + → Norm → FFN → +
│                                    │
└───────────── 残差连接 ─────────────┘
```

#### Post-Norm（原始 Transformer）
```
x → Attention → + → Norm → FFN → + → Norm
│                                    │
└───────────── 残差连接 ─────────────┘
```

**对比**：
| 布局 | 稳定性 | 主流选择 |
|------|--------|---------|
| Pre-Norm | **更稳定** | LLaMA, GPT-3, Mistral |
| Post-Norm | 较不稳定 | 原始 Transformer |

---

### 🔗 与 Evola 架构的关联

归一化对 Evola 的启示：

| Transformer Norm | Evola 演化方向 |
|-----------------|---------------|
| **数值稳定** | → **内稳态的工程类比** |
| **固定归一化目标** | → **动态舒适区调节** |
| **对所有特征一视同仁** | → **选择性调制（重要特征保留）** |

**核心类比**：
```
Transformer 归一化:
  目标：保持数值在合理范围（避免爆炸）
  方式：统计归一化（均值/方差）
  
Evola 内稳态:
  目标：保持内在状态在舒适区
  方式：动态调节（偏离 → 驱动 → 回归）
```

**潜在设计**：
```python
# 概念性设计
class HomeostasisModulatedNorm(nn.Module):
    """内稳态调制的归一化"""
    
    def __init__(self, dim):
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x, homeostasis_state):
        # 标准归一化
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        x_norm = x / rms
        
        # 内稳态调制：根据当前状态调整 gamma
        # 例如：能量低时，整体减弱激活
        modulation = compute_modulation(homeostasis_state)
        
        return self.gamma * x_norm * modulation
```

---

## 2. 实验准备 (Experiment Setup)

在接下来的实验中，我们将：

1. **对比 LayerNorm 和 RMSNorm 的效果**
2. **可视化归一化前后数值分布**
3. **测试 Pre-Norm vs Post-Norm 的稳定性**
4. **模拟内稳态调制的归一化（Evola 概念）**

---

## 3. 课后思考 (Summary & Reflection)

### 技术思考
1. RMSNorm 真的比 LayerNorm 效果相近吗？（实验验证）
2. 为什么 Pre-Norm 更稳定？（梯度流分析）

### Evola 思考
1. 归一化的"稳定目标"，能否与内稳态的"舒适区"对应？
2. 如果归一化的 gamma 参数受内稳态调制，会发生什么？
3. 温态运行时，归一化应该如何更新？（持续学习）

---

*课程编写日期：2026 年 4 月 15 日*