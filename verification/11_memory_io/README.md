# 11_memory_io — 记忆读写适配层

---

## 步骤 1：边界定义

> 输入：感知事件 或 决策上下文  
> 输出：编码后的记忆条目 或 检索到的相关记忆列表  
> 不考虑：记忆内部存储结构（属于 03）、元记忆策略（属于 04）

---

## 步骤 2：假设识别

| # | 假设 | 风险 |
|---|------|------|
| H1 | 事件可以用 (时间, 位置, 类型, 内稳态变化) 四元组充分描述 | 低 |
| H2 | 重要性评分（记忆是否值得存）可以启发式计算，无需 NN | **高** |
| H3 | 检索只需简单的向量距离（位置邻近 + 时间衰减），不需要语义理解 | 低 |
| H4 | 内存上限 + 过期淘汰能满足初期需求，不需要主动遗忘（那是 03 的活） | 中 |

---

## 步骤 3：快速否决 —— H2

- **风险**：启发式评分可能存太多"无聊"记忆，或漏掉"微妙但重要"的模式
- **验证**：初期用简单规则——
  - 生存事件（吃到食物 / 受伤）→ importance = 1.0（必存）
  - 探索事件（访问新区）→ importance = 0.5
  - 无事件（空步）→ 不存
  - 留接口：`importance` 可被 03 的注意力机制覆盖
- **结论**：**H2 成立但有限制**。初版规则够用，后期由 03 接管重要性评估。

---

## 步骤 4：技术选型

| 类别 | 选型 | 说明 |
|------|------|------|
| 拿来用 | **Python `dataclass`** | MemoryEvent 结构 |
| 拿来用 | **NumPy** | 向量距离计算（`np.linalg.norm`）做检索排序 |
| 拿来用 | **`heapq`** | Top-K 检索结果排序 |
| 改装 | **Prioritized Experience Replay (PER)** | PER 用 TD-error 做重要性权重。改装：用"内稳态变化量"替代 TD-error 做重要性评分 |
| 自研 | **事件编码格式** | (timestamp, pos, type, homeo_delta, importance) 元组设计 |
| 自研 | **检索相似度函数** | 基于位置距离 + 时间衰减的复合距离度量 |

**事件编码格式：**
```python
@dataclass
class MemoryEvent:
    timestamp: int
    position: tuple[int, int]
    event_type: str        # "food_found" | "danger" | "new_area"
    homeo_delta: dict      # {"energy": +0.3, "novelty": +0.1}
    importance: float      # 0..1
```

**检索接口：**
```python
def encode(event: MemoryEvent) -> bytes   # → 03 存储
def retrieve(context: DecisionContext, k: int) -> list[MemoryEvent]  # ← 03 检索
```

**检索相似度（复合距离）：**
```python
def similarity(item: MemoryEvent, context: DecisionContext) -> float:
    spatial_dist = np.linalg.norm(pos - context.pos)     # 空间近
    temporal_decay = exp(-(now - item.timestamp) / TAU)  # 时间新
    type_match = 1.0 if item.event_type in context.relevant_types else 0.1
    return type_match * temporal_decay / (1 + spatial_dist)
```

---

## 步骤 5：定向查文献

1. **Prioritized Experience Replay (Schaul et al., 2016)**：用 TD-error 做重要性权重。我们改用生存相关性，逻辑等效。
2. **FadeMem 双层级**：短时效记忆自动衰减，长时效记忆需高重要性阈值才能进入——提示 `importance` 字段需要两个阈值。
3. **事件边界**：用"内稳态变量显著变化"作为事件边界（如能量变化 > 0.1）。

---

## 输出物
```
11_memory_io/
├── impl/
│   ├── encoder.py         # 事件编码器
│   └── retriever.py       # 检索器
├── test/
│   └── test_memory_io.py
└── notes/
```
