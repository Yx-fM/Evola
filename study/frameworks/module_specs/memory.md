# 记忆系统规格 (Memory System Specification)

**版本**: 0.1.0  
**状态**: 设计稿  
**所属架构层级**: Layer 2 - 认知层  
**最后更新**: 2026 年 4 月 15 日

---

## 1. 模块概述

### 1.1 核心职责

记忆系统是 Evola **信息存储与管理的核心**，负责：

- **主动存储**: 决定哪些信息值得记住
- **主动检索**: 根据需求回忆相关信息
- **主动遗忘**: 清除低价值记忆，释放资源
- **元记忆**: 对自身记忆状态的监控与反思

**核心创新**: 记忆不是被动的仓库，而是可被主动操控的动态系统。

### 1.2 设计目标

| 目标 | 说明 | 优先级 |
|------|------|--------|
| **可衰减记忆** | 记忆强度随时间衰减 | 🔴 必须 |
| **主动遗忘** | 基于价值判断的清除 | 🔴 必须 |
| **元记忆** | 对记忆状态的监控 | 🔴 必须 |
| **温态巩固** | 无输入时的记忆整理 | 🟡 可选 |
| **关联网络** | 记忆间的语义关联 | 🟡 可选 |

---

## 2. 与标准 KV Cache 的对比

| 特性 | 标准 Transformer | Evola Memory |
|------|-----------------|---------------|
| **存储模式** | 被动缓存 | **主动管理** |
| **遗忘机制** | 无（固定窗口） | **可衰减 + 主动清除** |
| **检索方式** | 精确匹配位置 | **语义检索 + 关联** |
| **容量** | 固定（上下文窗口） | **可扩展 + 可压缩** |
| **元认知** | 无 | **有（记忆质量评估）** |

---

## 3. 核心类设计

### 3.1 记忆槽：`MemorySlot`

```python
@dataclass
class MemorySlot:
    """
    单个记忆槽
    
    每个记忆包含：
    - 内容（向量表示）
    - 强度（可衰减）
    - 时间戳
    - 关联索引
    """
    
    content: Tensor
    """记忆内容 [dim]"""
    
    strength: float = 1.0
    """
    记忆强度 (0-1)
    - 1.0: 新形成/刚强化
    - 0.0: 完全遗忘（可被覆盖）
    """
    
    decay_rate: float = 0.01
    """
    衰减速率 (每时间步)
    - 高 decay: 快速遗忘（临时信息）
    - 低 decay: 长期保留（重要信息）
    """
    
    created_at: float = 0.0
    """创建时间戳"""
    
    last_accessed_at: float = 0.0
    """最后访问时间戳"""
    
    access_count: int = 0
    """被访问次数"""
    
    tags: List[str] = None
    """语义标签（用于检索）"""
    
    associations: List[int] = None
    """关联记忆槽的索引"""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.associations is None:
            self.associations = []
    
    def decay(self, dt: float) -> float:
        """
        衰减记忆强度
        
        Args:
            dt: 经过的时间
            
        Returns:
            remaining_strength: 剩余强度
        """
        self.strength = self.strength * exp(-self.decay_rate * dt)
        return self.strength
    
    def reinforce(self, amount: float = 0.1):
        """
        强化记忆
        
        Args:
            amount: 强化量 (0-1)
        """
        self.strength = min(1.0, self.strength + amount)
        self.decay_rate = max(0.0, self.decay_rate - 0.001)  # 略微降低衰减
    
    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'content': self.content.tolist(),
            'strength': self.strength,
            'decay_rate': self.decay_rate,
            'created_at': self.created_at,
            'last_accessed_at': self.last_accessed_at,
            'access_count': self.access_count,
            'tags': self.tags,
            'associations': self.associations,
        }
```

---

### 3.2 主动记忆系统：`ActiveMemorySystem`

```python
class ActiveMemorySystem(nn.Module):
    """
    主动记忆系统
    
    核心功能:
    1. 存储决策（记什么）
    2. 检索（回忆什么）
    3. 遗忘（清除什么）
    4. 温态巩固（整理记忆）
    """
    
    def __init__(
        self,
        capacity: int = 1000,
        dim: int = 512,
        default_decay_rate: float = 0.01,
    ):
        super().__init__()
        
        self.capacity = capacity
        self.dim = dim
        self.default_decay_rate = default_decay_rate
        
        # 记忆槽参数化（可学习）
        self.memory_slots = nn.Parameter(
            torch.randn(capacity, dim) * 0.02
        )
        
        # 记忆状态（非参数）
        self.strengths = nn.Parameter(torch.zeros(capacity))  # 初始为 0（空）
        self.decay_rates = nn.Parameter(
            torch.ones(capacity) * default_decay_rate
        )
        
        # 使用掩码：1=使用中，0=空闲
        self.usage_mask = torch.zeros(capacity, dtype=torch.bool)
        
        # 元记忆模块
        self.meta_memory = MetaMemory(capacity)
        
        # 关联网络（可选）
        self.association_network = AssociationNetwork(capacity)
    
    def store(
        self,
        content: Tensor,
        importance: float,
        tags: Optional[List[str]] = None,
    ) -> int:
        """
        主动存储记忆
        
        Args:
            content: 记忆内容 [dim]
            importance: 重要性评分 (0-1) → 决定 decay_rate
            tags: 语义标签
            
        Returns:
            slot_index: 存储位置索引
        """
        # 1. 寻找空闲槽位
        available = (~self.usage_mask).nonzero(as_tuple=True)[0]
        
        if len(available) == 0:
            # 无空闲槽位，触发主动遗忘
            self.active_forget(forget_count=10)
            available = (~self.usage_mask).nonzero(as_tuple=True)[0]
        
        # 2. 选择槽位（优先使用空闲的）
        slot_idx = available[0].item()
        
        # 3. 写入记忆
        with torch.no_grad():
            self.memory_slots[slot_idx] = content
            self.strengths[slot_idx] = 1.0  # 新记忆强度为 1
            # 重要性越高，衰减越慢
            self.decay_rates[slot_idx] = self.default_decay_rate * (1.0 - importance)
        
        self.usage_mask[slot_idx] = True
        
        # 4. 添加标签
        if tags:
            self.meta_memory.add_tags(slot_idx, tags)
        
        return slot_idx
    
    def retrieve(
        self,
        query: Tensor,
        top_k: int = 5,
        min_strength: float = 0.1,
    ) -> RetrievedMemories:
        """
        主动检索记忆
        
        Args:
            query: 查询向量 [dim]
            top_k: 返回数量
            min_strength: 最低强度阈值
            
        Returns:
            RetrievedMemories: 检索结果
        """
        # 1. 获取使用中的槽位
        used_indices = self.usage_mask.nonzero(as_tuple=True)[0]
        
        if len(used_indices) == 0:
            return RetrievedMemories(memories=[], indices=[])
        
        # 2. 计算相似度
        used_slots = self.memory_slots[used_indices]
        query_normalized = F.normalize(query.unsqueeze(0), dim=-1)
        slots_normalized = F.normalize(used_slots, dim=-1)
        
        similarities = (query_normalized @ slots_normalized.T).squeeze(0)
        
        # 3. 考虑记忆强度（强度越高，越容易被检索）
        strengths = self.strengths[used_indices]
        weighted_scores = similarities * strengths
        
        # 4. 选择 top-k
        top_values, top_positions = torch.topk(weighted_scores, min(top_k, len(used_indices)))
        
        # 5. 构建结果
        retrieved_indices = used_indices[top_positions].tolist()
        retrieved_memories = []
        
        for idx in retrieved_indices:
            memory = MemorySlot(
                content=self.memory_slots[idx].clone(),
                strength=self.strengths[idx].item(),
                decay_rate=self.decay_rates[idx].item(),
            )
            retrieved_memories.append(memory)
            
            # 更新访问统计
            with torch.no_grad():
                self.strengths[idx] = min(1.0, self.strengths[idx] + 0.05)  # 访问即强化
        
        return RetrievedMemories(
            memories=retrieved_memories,
            indices=retrieved_indices,
        )
    
    def active_forget(
        self,
        forget_threshold: float = 0.2,
        forget_count: Optional[int] = None,
    ) -> List[int]:
        """
        主动遗忘
        
        Args:
            forget_threshold: 遗忘阈值（强度低于此值的可被遗忘）
            forget_count: 强制遗忘数量（可选）
            
        Returns:
            forgotten_indices: 被遗忘的槽位索引
        """
        used_indices = self.usage_mask.nonzero(as_tuple=True)[0]
        
        if len(used_indices) == 0:
            return []
        
        # 1. 找出低强度记忆
        strengths = self.strengths[used_indices]
        low_strength_mask = strengths < forget_threshold
        
        candidates = used_indices[low_strength_mask]
        
        # 2. 如果指定了数量，选择最弱的
        if forget_count is not None and len(candidates) < forget_count:
            # 需要更多，按强度排序
            sorted_values, sorted_positions = torch.sort(strengths)
            candidates = used_indices[sorted_positions[:forget_count]]
        
        # 3. 清除记忆
        forgotten_indices = []
        with torch.no_grad():
            for idx in candidates:
                self.usage_mask[idx] = False
                self.strengths[idx] = 0.0
                self.decay_rates[idx] = self.default_decay_rate
                forgotten_indices.append(idx.item())
        
        return forgotten_indices
    
    def warm_mode_consolidation(self) -> ConsolidationResult:
        """
        温态记忆巩固
        
        无外部输入时的记忆整理活动：
        1. 强化重要记忆
        2. 衰减陈旧记忆
        3. 建立关联
        """
        result = ConsolidationResult()
        
        used_indices = self.usage_mask.nonzero(as_tuple=True)[0]
        
        with torch.no_grad():
            for idx in used_indices:
                # 1. 时间衰减
                strength = self.strengths[idx].item()
                decay = self.decay_rates[idx].item()
                new_strength = strength * exp(-decay)
                
                self.strengths[idx] = new_strength
                
                # 2. 如果强度过低，标记为待遗忘
                if new_strength < 0.1:
                    result.decayed_memories.append(idx.item())
                
                # 3. 高强度记忆，进一步巩固
                if new_strength > 0.8:
                    self.decay_rates[idx] = max(0.001, decay * 0.95)  # 降低衰减
                    result.consolidated_memories.append(idx.item())
        
        # 4. （可选）建立关联
        if self.association_network is not None:
            associations = self.association_network.discover_associations(
                self.memory_slots[self.usage_mask]
            )
            result.new_associations = associations
        
        return result
```
