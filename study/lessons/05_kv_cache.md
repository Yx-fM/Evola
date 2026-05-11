# 第 5 课：KV Cache——增量推理的核心

> *"记忆是推理的加速器"*

---

## 1. 理论探讨 (Theory)

### 🧱 原文引用 (The Source)

**Transformers 库实现** ([HuggingFace, 2019](https://github.com/huggingface/transformers)):

> "Past key-values caches are used to speed up the generation process by avoiding recomputation of keys and values for previously generated tokens..."

**关键洞察**：在自回归生成中，已生成的 token 的 K/V 不需要重新计算。

---

### 🧠 深度讲解 (Explanation)

#### 问题：自回归生成的重复计算

**标准推理流程**（无 KV Cache）：

```
┌─────────────────────────────────────────────────────────────┐
│               无 KV Cache 的重复计算                          │
│                                                             │
│  生成第 1 个词：                                             │
│    输入: "The"                                               │
│    计算: Q1, K1, V1                                          │
│    输出: "cat"                                               │
│                                                             │
│  生成第 2 个词：                                             │
│    输入: "The cat"                                           │
│    计算: Q1, K1, V1  ← 重复计算！                            │
│          Q2, K2, V2                                          │
│    输出: "sat"                                               │
│                                                             │
│  生成第 3 个词：                                             │
│    输入: "The cat sat"                                       │
│    计算: Q1, K1, V1  ← 重复计算！                            │
│          Q2, K2, V2  ← 重复计算！                            │
│          Q3, K3, V3                                          │
│    输出: "on"                                                │
│                                                             │
│  生成第 n 个词：                                             │
│    计算量: O(n^2)  ← 每次都从头计算！                         │
└─────────────────────────────────────────────────────────────┘
```

**问题分析**：
- 已生成的 token 的 K/V 值是不变的
- 但每次推理都重新计算 → 浪费计算资源
- 长序列生成时，计算量爆炸

---

#### 解决方案：KV Cache

**核心思想**：缓存已计算的 K/V，避免重复计算

```
┌─────────────────────────────────────────────────────────────┐
│               有 KV Cache 的增量推理                          │
│                                                             │
│  生成第 1 个词：                                             │
│    输入: "The"                                               │
│    计算: K1, V1                                              │
│    Cache: [[K1], [V1]]                                       │
│    输出: "cat"                                               │
│                                                             │
│  生成第 2 个词：                                             │
│    输入: "cat"                                               │
│    计算: K2, V2                                              │
│    Cache: [[K1, K2], [V1, V2]]                               │
│    Attention: Q2 × [K1, K2]^T × [V1, V2]                     │
│    输出: "sat"                                               │
│                                                             │
│  生成第 3 个词：                                             │
│    输入: "sat"                                               │
│    计算: K3, V3                                              │
│    Cache: [[K1, K2, K3], [V1, V2, V3]]                       │
│    Attention: Q3 × [K1, K2, K3]^T × [V1, V2, V3]             │
│    输出: "on"                                                │
│                                                             │
│  生成第 n 个词：                                             │
│    计算量: O(n)  ← 只计算新的！                               │
└─────────────────────────────────────────────────────────────┘
```

**效率对比**：
```
无 KV Cache:  计算量 = O(n^2)  (n=1000 → 1,000,000)
有 KV Cache:  计算量 = O(n)    (n=1000 → 1,000)

加速比 = 1000 倍！
```

---

#### 实现细节

**数据结构**：
```python
class KVCache:
    """KV Cache 数据结构"""
    
    def __init__(self, num_layers, num_heads, head_dim, max_seq_len):
        # 每层都有独立的 KV Cache
        self.k_cache = torch.zeros(num_layers, max_seq_len, num_heads, head_dim)
        self.v_cache = torch.zeros(num_layers, max_seq_len, num_heads, head_dim)
        self.current_len = 0
    
    def update(self, layer_idx, k_new, v_new):
        """
        更新 Cache
        
        Args:
            layer_idx: 层索引
            k_new: 新的 Key [batch, num_heads, 1, head_dim]
            v_new: 新的 Value [batch, num_heads, 1, head_dim]
        """
        # 追加到 Cache 尾部
        self.k_cache[layer_idx, self.current_len] = k_new
        self.v_cache[layer_idx, self.current_len] = v_new
        self.current_len += 1
    
    def get(self, layer_idx):
        """获取到当前位置的 KV"""
        return (
            self.k_cache[layer_idx, :self.current_len],
            self.v_cache[layer_idx, :self.current_len],
        )
```

**增量推理流程**：
```python
def generate_with_cache(model, prompt, max_tokens):
    """带 KV Cache 的生成"""
    
    # 1. 初始化 Cache
    cache = KVCache(num_layers=6, num_heads=8, head_dim=64, max_seq_len=2048)
    
    # 2. 处理 prompt（一次性计算所有 KV）
    tokens = tokenize(prompt)
    hidden_states = model.embed(tokens)
    
    for layer_idx, layer in enumerate(model.layers):
        # 计算完整的 KV
        k, v = layer.compute_kv(hidden_states)
        cache.update(layer_idx, k, v)  # 存入 cache
        hidden_states = layer(hidden_states, past_kv=None)
    
    # 3. 增量生成
    for _ in range(max_tokens):
        # 只处理最新的 token
        new_token = hidden_states[-1:]  # 最后一个位置
        
        for layer_idx, layer in enumerate(model.layers):
            # 获取历史 KV
            past_k, past_v = cache.get(layer_idx)
            
            # 只计算新的 KV
            new_k, new_v = layer.compute_kv(new_token)
            
            # 更新 cache
            cache.update(layer_idx, new_k, new_v)
            
            # 合并历史和新的 KV
            full_k = torch.cat([past_k, new_k], dim=2)
            full_v = torch.cat([past_v, new_v], dim=2)
            
            # 注意力计算
            hidden_states = layer.attention(new_token, full_k, full_v)
        
        # 预测下一个 token
        next_token = model.predict(hidden_states)
        yield next_token
```

---

### 📊 KV Cache 的内存分析

**内存占用计算**：
```
假设:
  - num_layers = 32
  - num_heads = 32
  - head_dim = 128
  - max_seq_len = 2048
  - dtype = float16 (2 bytes)

KV Cache 内存:
  = 2 (K + V) × 32 layers × 2048 seq_len × 32 heads × 128 dim × 2 bytes
  = 2 × 32 × 2048 × 32 × 128 × 2
  = 536,870,912 bytes ≈ 512 MB
```

**LLaMA-7B 的 KV Cache**：
- 32 层，32 头，128 维
- 2048 序列长度 → 约 512 MB
- 4096 序列长度 → 约 1 GB

---

### 🚀 优化技术

#### 1. Multi-Query Attention (MQA)

**问题**：每头都有独立的 KV → 内存大

**解决方案**：所有头共享同一个 K/V

```
┌─────────────────────────────────────────────────────────────┐
│              Multi-Head vs Multi-Query                       │
│                                                             │
│  Multi-Head Attention (标准):                               │
│    Q: [num_heads, seq_len, head_dim]                        │
│    K: [num_heads, seq_len, head_dim]                        │
│    V: [num_heads, seq_len, head_dim]                        │
│    → KV Cache 大                                            │
│                                                             │
│  Multi-Query Attention:                                     │
│    Q: [num_heads, seq_len, head_dim]                        │
│    K: [1, seq_len, head_dim]        ← 共享                  │
│    V: [1, seq_len, head_dim]        ← 共享                  │
│    → KV Cache 小 (1/32)                                     │
└─────────────────────────────────────────────────────────────┘
```

**内存节省**：
```
标准: 512 MB
MQA:  512 MB / 32 = 16 MB  ← 节省 96%！
```

#### 2. Grouped-Query Attention (GQA)

**平衡方案**：分组共享，每组有独立 KV

```
┌─────────────────────────────────────────────────────────────┐
│              Grouped-Query Attention                         │
│                                                             │
│  LLaMA-2 70B 使用 GQA:                                       │
│                                                             │
│    32 个 Query 头                                            │
│    8 组 KV (每组 4 个 Query 共享)                             │
│                                                             │
│    Q: [32, seq_len, head_dim]                               │
│    K: [8, seq_len, head_dim]                                │
│    V: [8, seq_len, head_dim]                                │
│                                                             │
│  内存: 标准 / 4                                              │
│  效果: 与标准相近                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 🔗 与 Evola 架构的关联

KV Cache 对 Evola 的启示：

| Transformer KV Cache | Evola 记忆系统 |
|---------------------|---------------|
| **被动缓存** | → **主动记忆管理** |
| **固定窗口** | → **动态容量调节** |
| **无衰减** | → **可衰减记忆强度** |
| **无遗忘** | → **主动遗忘低价值信息** |

**核心差异**：
```
Transformer KV Cache:
  - 记住所有历史信息
  - 直到达到长度限制
  - 无记忆强度概念
  
Evola Memory System:
  - 主动决定记住什么
  - 记忆可衰减（遗忘曲线）
  - 有记忆强度和元记忆
  - 温态运行时整理记忆
```

**Evola 的 KV Cache 演化**：
```python
# 概念性设计
class ActiveKVCache(nn.Module):
    """主动管理的 KV Cache"""
    
    def __init__(self, capacity, decay_rate=0.01):
        self.capacity = capacity
        self.decay_rate = decay_rate
        
        # KV Cache + 记忆强度
        self.k_cache = torch.zeros(capacity, ...)
        self.v_cache = torch.zeros(capacity, ...)
        self.strengths = torch.ones(capacity)  # 记忆强度
    
    def update(self, k_new, v_new, importance):
        """更新时考虑重要性"""
        # 高重要性 → 低衰减率
        # 低重要性 → 高衰减率
        decay = self.decay_rate * (1 - importance)
        
        # 存入 cache
        ...
    
    def decay_all(self, dt):
        """时间衰减所有记忆"""
        self.strengths *= torch.exp(-self.decay_rate * dt)
        
        # 低强度记忆可被覆盖
        ...
    
    def warm_mode_consolidation(self):
        """温态记忆整理"""
        # 强化重要记忆
        # 遗忘低强度记忆
        ...
```

---

## 2. 实验准备 (Experiment Setup)

在接下来的实验中，我们将：

1. **实现基础 KV Cache**
2. **对比有无 Cache 的推理速度**
3. **分析不同序列长度的内存占用**
4. **实现简化版 ActiveKVCache（Evola 概念）**

---

## 3. 课后思考 (Summary & Reflection)

### 技术思考
1. KV Cache 对推理速度的具体加速比例？
2. MQA 和 GQA 对模型效果的影响？

### Evola 思考
1. KV Cache 如何演化为"主动记忆系统"？
2. 记忆强度衰减如何影响推理？
3. 温态运行时，应该如何整理 KV Cache？

---

*课程编写日期：2026 年 4 月 15 日*