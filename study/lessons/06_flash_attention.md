# 第 6 课：FlashAttention——IO 优化的艺术

> *"计算不是瓶颈，内存才是"*

---

## 1. 理论探讨 (Theory)

### 🧱 原文引用 (The Source)

**FlashAttention 论文** ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)):

> "We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM..."

**关键洞察**：注意力计算的计算量不是瓶颈，**内存访问**才是。

---

### 🧠 深度讲解 (Explanation)

#### 问题：标准注意力的内存瓶颈

**标准注意力流程**：

```
┌─────────────────────────────────────────────────────────────┐
│                标准 Attention 的内存访问                      │
│                                                             │
│  GPU 架构:                                                   │
│    HBM (高带宽内存): 40 GB, 1.5 TB/s                        │
│    SRAM (片上缓存): ~20 MB, ~10 TB/s                        │
│                                                             │
│  标准 Attention:                                             │
│    1. 从 HBM 加载 Q, K, V                                    │
│    2. 计算 QK^T → 写回 HBM (生成 S)                         │
│    3. 从 HBM 加载 S                                          │
│    4. 计算 Softmax(S) → 写回 HBM (生成 P)                   │
│    5. 从 HBM 加载 P, V                                       │
│    6. 计算 PV → 写回 HBM (生成 O)                           │
│                                                             │
│  内存访问次数:                                                │
│    O(N^2) 次 HBM 读/写                                       │
│                                                             │
│  问题:                                                       │
│    - N^2 大小的中间矩阵 (S, P)                               │
│    - 频繁的 HBM 读/写                                        │
│    - HBM 访问慢 → 成为瓶颈                                    │
└─────────────────────────────────────────────────────────────┘
```

**时间分析**：
```
计算时间: O(N^2 × d)  ← GPU 计算很快
内存时间: O(N^2)      ← HBM 访问慢，成为瓶颈

当 N 很大时（如 N=4096）：
  内存时间 >> 计算时间
```

---

#### 解决方案：FlashAttention

**核心思想**：**分块计算** + **不存储中间矩阵**

```
┌─────────────────────────────────────────────────────────────┐
│                 FlashAttention 的分块策略                    │
│                                                             │
│  将 Q, K, V 分成小块（适合 SRAM 容量）                       │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Q: [N, d] → 分成 B_r 块, 每块 [B_r, d]              │   │
│  │  K: [N, d] → 分成 B_c 块, 每块 [B_c, d]              │   │
│  │  V: [N, d] → 分成 B_c 块, 每块 [B_c, d]              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  分块计算流程:                                               │
│                                                             │
│  for i in range(B_r):  # Q 的块                             │
│      加载 Q_i 到 SRAM                                       │
│      for j in range(B_c):  # K, V 的块                      │
│          加载 K_j, V_j 到 SRAM                              │
│          在 SRAM 内计算:                                    │
│            S_ij = Q_i × K_j^T                               │
│            P_ij = softmax(S_ij)                             │
│            O_ij = P_ij × V_j                                │
│          更新输出 O_i (累积)                                 │
│      写回 O_i 到 HBM                                        │
│                                                             │
│  关键:                                                       │
│    - S, P 不写回 HBM（只存在于 SRAM）                        │
│    - 只写回最终输出 O                                        │
│    - 内存访问: O(N) 次而非 O(N^2)                            │
└─────────────────────────────────────────────────────────────┘
```

**效率对比**：
```
┌─────────────────────────────────────────────────────────────┐
│              标准 vs FlashAttention                          │
│                                                             │
│  标准 Attention:                                            │
│    HBM 读: Q, K, V, S, P (N^2 次访问)                       │
│    HBM 写: S, P, O (N^2 次访问)                             │
│    总访问: O(N^2)                                            │
│                                                             │
│  FlashAttention:                                            │
│    HBM 读: Q, K, V (分块加载, O(N) 次)                       │
│    HBM 写: O (只写输出, O(N) 次)                            │
│    总访问: O(N)                                              │
│                                                             │
│  加速: N^2 → N                                               │
│    N=1024: 加速 1024 倍                                      │
│    N=4096: 加速 4096 倍                                      │
└─────────────────────────────────────────────────────────────┘
```

---

#### Softmax 的分块计算（难点）

**问题**：Softmax 需要完整的输入才能计算

```
标准 Softmax:
  softmax(x) = exp(x) / sum(exp(x))
  
  需要知道所有 x 才能计算 sum(exp(x))
  
  分块后，只有部分 x → 如何计算？
```

**解决方案**：**Online Softmax**

```
┌─────────────────────────────────────────────────────────────┐
│                 Online Softmax 算法                          │
│                                                             │
│  维护两个全局统计量:                                          │
│    m: 最大值 (用于数值稳定)                                   │
│    l: exp(x - m) 的累积和                                    │
│                                                             │
│  分块更新:                                                   │
│    for each block j:                                        │
│        m_new = max(m_old, max(S_ij))                        │
│        l_new = l_old * exp(m_old - m_new) + sum(exp(S_ij))  │
│                                                             │
│  最终 Softmax:                                               │
│    P_ij = exp(S_ij - m_new) / l_new                         │
│                                                             │
│  这样不需要存储完整的 S                                       │
└─────────────────────────────────────────────────────────────┘
```

---

### 📊 FlashAttention-2 和 FlashAttention-3

**版本演进**：

| 版本 | 关键改进 | 速度提升 |
|------|---------|---------|
| **FlashAttention-1** | 分块 + 不存储中间矩阵 | 2-4x |
| **FlashAttention-2** | 并行化 + 减少 non-matmul | 2x vs FA1 |
| **FlashAttention-3** | H100 GPU 优化 + FP8 | 1.5-2x vs FA2 |

**FlashAttention-2 的改进**：
- 更好的并行化策略（work 分配）
- 减少 non-matmul 操作（GPU 上 matmul 更高效）
- 更好的内存对齐

---

### 🔗 与 Evola 架构的关联

FlashAttention 对 Evola 的启示：

| FlashAttention 特性 | Evola 演化方向 |
|--------------------|---------------|
| **分块计算** | → **温态时选择性处理记忆块** |
| **不存储中间矩阵** | → **只保留重要记忆** |
| **Online Softmax** | → **动态更新注意力分布** |
| **内存效率** | → **记忆容量动态调节** |

**潜在设计**：
```
Evola 的 FlashAttention-like 机制:

1. 温态运行时：
   - 只处理部分记忆块（而非全部）
   - 根据"记忆强度"选择处理哪些块
   
2. 记忆整理：
   - 类似 FlashAttention 的分块
   - 高强度记忆块 → 详细处理
   - 低强度记忆块 → 简化处理
   
3. 内稳态调制：
   - 高能量时 → 处理更多块
   - 低能量时 → 只处理关键块
```

---

## 2. 实验准备 (Experiment Setup)

在接下来的实验中，我们将：

1. **对比标准 Attention 和 FlashAttention 的速度**
2. **可视化分块计算流程**
3. **分析不同序列长度的性能差异**
4. **模拟 Evola 的选择性分块处理（概念）**

---

## 3. 课后思考 (Summary & Reflection)

### 技术思考
1. FlashAttention 的分块大小如何选择？
2. 为什么内存访问成为瓶颈而非计算？

### Evola 思考
1. 温态运行时，如何选择性处理记忆块？
2. 内稳态如何影响"处理哪些块"的决策？
3. 是否可以设计"IO-aware 的记忆整理"？

---

*课程编写日期：2026 年 4 月 15 日*