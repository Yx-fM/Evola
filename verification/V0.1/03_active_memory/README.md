# 03_active_memory - 主动记忆管理

---

## 步骤 1：边界定义

> 提供记忆的**存储结构 + 衰减逻辑 + 遗忘策略**（03 的职责）  
> 记忆的**编码和检索**由 11_memory_io 负责  
> 不考虑：语义理解、大语言模型集成、注意力机制的完整实现（那是 P2 后期的事）

**最小成功标准**：一个带有双层级（短时/长时）衰减 + 重要性阈值自动遗忘的记忆存储系统。

---

## 步骤 2：假设识别

| # | 假设 | 风险 |
|---|------|------|
| H1 | 双层级记忆（STM→衰减→LTM 或丢弃）是合适的设计粒度 | 低 |
| H2 | 基于内稳态状态调节遗忘速率（"感觉记忆满了→加速遗忘"）可行 | **高** |
| H3 | 内存上限 + 重要性阈值的自动淘汰比手动调参更稳定 | 中 |
| H4 | STM 用 Python list，LTM 用 dict/sqlite 满足初期需求 | 低 |

---

## 步骤 3：快速否决 —— H2

- **风险**："内稳态驱动遗忘"是核心创新但不是无风险的——如果真的"感觉记忆满了就忘"，可能忘掉重要但低频的信息
- **验证**：需对比三种策略——
  - 策略 A：纯基于时间的 FIFO 淘汰（无内稳态输入）
  - 策略 B：基于 time × importance 排序淘汰（静态 rule）
  - 策略 C：基于内稳态（memory_load_drive）动态调节 forgetting_rate（自研核心）
  - 观察：策略 C 是否比 A/B 保留了更多"生存关键"记忆
- **结论**：**H2 不能当场否决，需要实验验证**。先实现策略 A+B 做基线，在此基础上加 C。

---

## 步骤 4：技术选型

| 类别 | 选型 | 说明 |
|------|------|------|
| 拿来用 | **Python `list` + `dict`** | STM 用 list（顺序访问），LTM 用 dict（key 检索） |
| 拿来用 | **`heapq`** | 按 （importance × time_decay）维护淘汰优先级堆 |
| 拿来用 | **`dataclass`** | MemoryItem 结构 |
| 改装 | **FadeMem 双层级衰减** | FadeMem (2026) 的短时/长时记忆衰减模型。改装：加入内稳态变量作为 forgetting_rate 的调制输入 |
| 自研 | **内稳态耦合遗忘** | 核心创新：`forgetting_rate = base_rate * memory_load_drive`——记忆越满，忘得越快 |
| 自研 | **记忆状态感知** | 能被 02_homeostasis 查询的"记忆负载度"（占总容量比例） |

**双层级结构：**

```python
@dataclass
class MemoryItem:
    id: str
    event: MemoryEvent       # 由 11 编码
    layer: str               # "stm" | "ltm"
    importance: float        # 0..1
    created_at: int          # step 时间戳
    accessed_at: int         # 最后一次被检索的时间
    access_count: int        # 检索次数
    decay_factor: float = 1.0  # 时间衰减因子

class ActiveMemory:
    def __init__(self, stm_capacity=100, ltm_capacity=1000):
        self.stm: list[MemoryItem] = []
        self.ltm: dict[str, MemoryItem] = {}
    
    def write(self, item: MemoryItem):
        self.stm.append(item)
        if item.importance > LTM_THRESHOLD:
            self.ltm[item.id] = item
        self._evict_if_full()
    
    def decay(self, memory_load_drive: float):
        """每 step 调用，受内稳态调制"""
        rate = BASE_DECAY_RATE * (1.0 + memory_load_drive)
        for item in self.stm:
            item.decay_factor *= (1.0 - rate)
        # 淘汰 decay_factor 低于阈值的记忆
        self.stm = [i for i in self.stm if i.decay_factor > 0.1]
    
    def get_memory_load(self) -> float:
        """供 02 查询记忆负载度"""
        return len(self.stm) / STM_CAPACITY
```

---

## 步骤 5：定向查文献

1. **FadeMem (2026)**：当前 AI 主动遗遗忘的 SOTA。双层级衰减 + LLM 引导的冲突解决。我们借其双层级结构，但不借 LLM 部分（P0-P1 不涉及 LLM）。关键坑：FadeMem 的 LTM 准入机制依赖 LLM 判断冲突，我们用 `importance > threshold` 替代。
2. **EverMemOS**：四层架构（感记/短时/长时/工作记忆），比我们的双层级复杂很多。初期不需要四层，但它的"记忆处理器"概念（独立的内存管理单元）值得参考——03 就是这个"记忆处理器"。
3. **Catastrophic Forgetting in Continual Learning**：持续学习中的灾难性遗忘——学新东西就忘旧东西。FadeMem 和我们的主动遗忘正好相反：我们**主动**忘，CL 研究**防止**忘。对比学习有助于理解"什么该忘、什么不该忘"。
4. **Ebbinghaus Forgetting Curve**：经典心理学——遗忘是指数衰减。我们的 `decay_factor *= (1-rate)` 直接模拟这个。加内稳态调制后变成"可变参数的指数衰减"。

---

## 输出物

```
03_active_memory/
├── impl/
│   ├── memory_store.py       # STM/LTM 存储结构
│   ├── decay.py              # 衰减 + 淘汰逻辑（内稳态耦合）
│   └── item.py                # MemoryItem 数据类
├── test/
│   ├── test_memory_store.py
│   ├── test_decay.py
│   └── test_forgetting_strategy.py  # 对比 A/B/C 三种策略
└── notes/
    └── forgetting_experiments.md
```
