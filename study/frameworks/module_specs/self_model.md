## 4. 检索结果与辅助类

### 4.1 检索结果：`RetrievedMemories`

```python
@dataclass
class RetrievedMemories:
    """检索结果"""
    
    memories: List[MemorySlot]
    """检索到的记忆列表"""
    
    indices: List[int]
    """对应槽位索引"""
    
    def __len__(self) -> int:
        return len(self.memories)
    
    def is_empty(self) -> bool:
        return len(self.memories) == 0
    
    def to_contents(self) -> List[Tensor]:
        """提取内容列表"""
        return [m.content for m in self.memories]
    
    def aggregate(self, method: str = 'mean') -> Tensor:
        """
        聚合检索结果
        
        Args:
            method: 'mean', 'sum', 'attention'
            
        Returns:
            aggregated: 聚合后的向量 [dim]
        """
        if self.is_empty():
            return torch.zeros_like(self.memories[0].content)
        
        contents = torch.stack([m.content for m in self.memories])
        strengths = torch.tensor([m.strength for m in self.memories])
        
        if method == 'mean':
            return contents.mean(dim=0)
        elif method == 'sum':
            return contents.sum(dim=0)
        elif method == 'attention':
            # 用强度作为注意力权重
            weights = F.softmax(strengths, dim=0)
            return (contents * weights.unsqueeze(-1)).sum(dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
```

---

### 4.2 巩固结果：`ConsolidationResult`

```python
@dataclass
class ConsolidationResult:
    """温态巩固结果"""
    
    decayed_memories: List[int] = None
    """已衰减的记忆索引"""
    
    consolidated_memories: List[int] = None
    """已巩固的记忆索引"""
    
    new_associations: List[Tuple[int, int]] = None
    """新建立的关联"""
    
    def __post_init__(self):
        if self.decayed_memories is None:
            self.decayed_memories = []
        if self.consolidated_memories is None:
            self.consolidated_memories = []
        if self.new_associations is None:
            self.new_associations = []
    
    def summary(self) -> str:
        return (
            f"巩固结果：{len(self.decayed_memories)} 个记忆衰减，"
            f"{len(self.consolidated_memories)} 个记忆强化，"
            f"{len(self.new_associations)} 个新关联"
        )
```

---

## 5. 元记忆系统：`MetaMemory`

```python
class MetaMemory(nn.Module):
    """
    元记忆 - 对记忆的记忆
    
    功能：
    1. 评估记忆质量
    2. 调整记忆策略
    3. 反思学习过程
    """
    
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity
        
        # 标签索引（用于语义检索）
        self.tag_index: Dict[str, List[int]] = defaultdict(list)
        
        # 记忆质量评估器
        self.quality_assessor = MemoryQualityAssessor()
        
        # 策略调节器
        self.strategy_adjuster = StrategyAdjuster()
    
    def assess_memory_quality(self) -> MemoryQualityReport:
        """
        评估记忆系统整体质量
        
        Returns:
            MemoryQualityReport: 质量报告
        """
        # 这里需要访问主记忆系统
        # 简化版本：
        return MemoryQualityReport(
            total_capacity=self.capacity,
            used_capacity=0,  # 需要从主系统获取
            average_strength=0.0,
            forgetting_rate=0.0,
            retrieval_success_rate=0.0,
        )
    
    def add_tags(self, slot_index: int, tags: List[str]):
        """为记忆槽添加标签索引"""
        for tag in tags:
            self.tag_index[tag].append(slot_index)
    
    def reflect_on_learning(self) -> LearningInsight:
        """
        对学习过程的反思
        
        Returns:
            LearningInsight: 学习洞察
        """
        # 分析记忆模式
        quality_report = self.assess_memory_quality()
        
        # 生成洞察
        insights = []
        
        if quality_report.average_strength < 0.3:
            insights.append("记忆整体偏弱，建议加强学习或降低衰减")
        
        if quality_report.used_capacity > quality_report.total_capacity * 0.9:
            insights.append("记忆容量接近饱和，建议主动遗忘")
        
        return LearningInsight(
            insights=insights,
            recommendations=self.strategy_adjuster.suggest(quality_report),
        )
```

---

### 5.1 质量报告：`MemoryQualityReport`

```python
@dataclass
class MemoryQualityReport:
    """记忆质量报告"""
    
    total_capacity: int
    used_capacity: int
    average_strength: float
    forgetting_rate: float
    retrieval_success_rate: float
    
    def utilization_rate(self) -> float:
        """容量使用率"""
        return self.used_capacity / self.total_capacity
    
    def health_score(self) -> float:
        """
        记忆系统健康度 (0-1)
        
        综合考虑:
        - 使用率适中 (不过高/过低)
        - 平均强度适中
        - 检索成功率高
        """
        # 理想使用率：40%-80%
        utilization = self.utilization_rate()
        utilization_score = 1.0 if 0.4 <= utilization <= 0.8 else 0.5
        
        # 理想平均强度：0.5-0.8
        strength_score = 1.0 if 0.5 <= self.average_strength <= 0.8 else 0.5
        
        # 检索成功率越高越好
        retrieval_score = self.retrieval_success_rate
        
        return (utilization_score + strength_score + retrieval_score) / 3
```

---

### 5.2 学习洞察：`LearningInsight`

```python
@dataclass
class LearningInsight:
    """学习洞察"""
    
    insights: List[str]
    recommendations: List[str]
    
    def summary(self) -> str:
        return "\n".join(self.insights + self.recommendations)
```

---

## 6. 关联网络（可选扩展）

```python
class AssociationNetwork(nn.Module):
    """
    记忆关联网络
    
    发现并维护记忆间的语义关联
    """
    
    def __init__(
        self,
        capacity: int,
        similarity_threshold: float = 0.7,
    ):
        super().__init__()
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        
        # 邻接表表示
        self.adjacency_list: Dict[int, Set[int]] = defaultdict(set)
    
    def discover_associations(
        self,
        memory_embeddings: Tensor,
    ) -> List[Tuple[int, int]]:
        """
        发现新的关联
        
        Args:
            memory_embeddings: 记忆嵌入 [num_used, dim]
            
        Returns:
            new_associations: 新发现的关联对
        """
        new_associations = []
        
        # 计算相似度矩阵
        normalized = F.normalize(memory_embeddings, dim=-1)
        similarity_matrix = normalized @ normalized.T
        
        # 找出高相似度对
        num_memories = len(memory_embeddings)
        for i in range(num_memories):
            for j in range(i + 1, num_memories):
                sim = similarity_matrix[i, j].item()
                
                if sim > self.similarity_threshold:
                    if j not in self.adjacency_list[i]:
                        self.adjacency_list[i].add(j)
                        self.adjacency_list[j].add(i)
                        new_associations.append((i, j))
        
        return new_associations
    
    def get_associated(
        self,
        slot_index: int,
        max_depth: int = 2,
    ) -> Set[int]:
        """
        获取关联记忆
        
        Args:
            slot_index: 起始槽位
            max_depth: 最大深度
            
        Returns:
            associated_indices: 关联记忆索引
        """
        visited = set()
        queue = [(slot_index, 0)]
        
        while queue:
            current, depth = queue.pop(0)
            
            if current in visited or depth > max_depth:
                continue
            
            visited.add(current)
            
            for neighbor in self.adjacency_list[current]:
                queue.append((neighbor, depth + 1))
        
        return visited
```

---

## 7. API 接口设计

### 7.1 公共方法

| 方法 | 签名 | 用途 |
|------|------|------|
| `store` | `(content, importance, tags) → slot_idx` | 存储记忆 |
| `retrieve` | `(query, top_k) → RetrievedMemories` | 检索记忆 |
| `active_forget` | `(threshold, count) → [indices]` | 主动遗忘 |
| `warm_mode_consolidation` | `() → ConsolidationResult` | 温态巩固 |

### 7.2 配置示例

```python
# 创建记忆系统
memory = ActiveMemorySystem(
    capacity=1000,
    dim=512,
    default_decay_rate=0.01,
)

# 存储记忆
slot_idx = memory.store(
    content=embedding_vector,
    importance=0.8,
    tags=['concept', 'physics'],
)

# 检索记忆
retrieved = memory.retrieve(
    query=query_vector,
    top_k=5,
)

# 温态巩固
result = memory.warm_mode_consolidation()
print(result.summary())
```

---

## 8. 与其他模块的接口

### 8.1 输入依赖

| 来源模块 | 提供数据 | 用途 |
|----------|---------|------|
| **Attention** | `attended_representation` | 待存储内容 |
| **Homeostasis** | `memory_load` | 遗忘触发 |
| **SelfModel** | `importance_assessment` | 重要性评估 |

### 8.2 输出去向

| 目标模块 | 接收数据 | 用途 |
|----------|---------|------|
| **Attention** | `retrieved_kv` | 上下文输入 |
| **Homeostasis** | `memory_load` | 负载测量 |
| **SelfModel** | `memory_quality_report` | 自我评估 |

---

## 9. 实现路线图

| 阶段 | 任务 | 预计工时 | 状态 |
|------|------|----------|------|
| **Phase 1** | MemorySlot 实现 | 2 天 | ⏳ 待开始 |
| **Phase 1** | ActiveMemorySystem 核心 | 2 周 | ⏳ 待开始 |
| **Phase 1** | 存储/检索/遗忘 | 1 周 | ⏳ 待开始 |
| **Phase 2** | MetaMemory 实现 | 1 周 | ⏳ 待开始 |
| **Phase 2** | 温态巩固 | 1 周 | ⏳ 待开始 |
| **Phase 3** | AssociationNetwork | 1 周 | ⏳ 待开始 |

---

*记忆系统规格 完成日期：2026 年 4 月 15 日*

---

# 自我模型模块规格 (Self Model Specification)

**版本**: 0.1.0  
**状态**: 设计稿  
**所属架构层级**: Layer 4 - 存在层  
**最后更新**: 2026 年 4 月 15 日

---

## 1. 模块概述

### 1.1 核心职责

自我模型是 Evola **自指能力的核心**，负责：

- **自我状态向量**: 维护"我是谁"的表示
- **元认知**: 对自身认知过程的监控
- **自我反思**: 评估能力、调整策略

**核心信念**: 当她在决策时，这个自我向量会作为输入的一部分，让她能够回答："如果我是我，我会怎么做？"

---

## 2. 核心类设计

### 2.1 自我状态向量：`SelfVector`

```python
@dataclass
class SelfVector:
    """
    自我状态向量
    
    表示"我现在的状态是什么"
    """
    
    # 内稳态状态
    homeostasis_state: HomeostasisVariables
    
    # 能力评估
    competence_assessment: Dict[str, float] = None
    """
    各领域能力评估 {domain: score}
    示例: {'language': 0.8, 'reasoning': 0.6, 'memory': 0.7}
    """
    
    # 行为倾向
    behavioral_tendency: Optional[BehavioralTendency] = None
    """当前行为倾向"""
    
    # 自我认知（学习的表示）
    self_knowledge: Optional[Tensor] = None
    """向量化的自我知识"""
    
    # 时间信息
    current_time: float = 0.0
    """当前时间戳"""
    
    def __post_init__(self):
        if self.competence_assessment is None:
            self.competence_assessment = {}
    
    def to_tensor(self) -> Tensor:
        """转换为向量表示"""
        components = []
        
        # 1. 内稳态状态
        components.append(self.homeostasis_state.to_tensor())
        
        # 2. 能力评估（平均）
        if self.competence_assessment:
            avg_competence = sum(self.competence_assessment.values()) / len(self.competence_assessment)
            components.append(torch.tensor([avg_competence]))
        else:
            components.append(torch.tensor([0.5]))
        
        # 3. 行为倾向（编码）
        if self.behavioral_tendency:
            components.append(torch.tensor([self.behavioral_tendency.intensity]))
        else:
            components.append(torch.tensor([0.0]))
        
        return torch.cat(components)
    
    def update_competence(self, domain: str, performance: float, learning_rate: float = 0.1):
        """
        更新能力评估
        
        Args:
            domain: 领域名称
            performance: 表现评分 (0-1)
            learning_rate: 学习率
        """
        current = self.competence_assessment.get(domain, 0.5)
        self.competence_assessment[domain] = current + learning_rate * (performance - current)
```

---

### 2.2 自我模型主类：`SelfModel`

```python
class SelfModel(nn.Module):
    """
    自我模型
    
    核心功能:
    1. 维护自我向量
    2. 更新自我认知
    3. 查询自我知识
    4. 预测未来自我
    """
    
    def __init__(
        self,
        dim: int = 512,
        history_size: int = 1000,
    ):
        super().__init__()
        
        self.dim = dim
        
        # 自我向量
        self.self_vector = SelfVector(
            homeostasis_state=HomeostasisVariables(),
        )
        
        # 自我知识编码器（可学习）
        self.self_knowledge_encoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, dim),
        )
        
        # 历史快照
        self.history: deque[SelfSnapshot] = deque(maxlen=history_size)
        
        # 元认知模块
        self.metacognition = MetacognitionModule()
    
    def update(self, experience: Experience):
        """
        基于新经验更新自我模型
        
        Args:
            experience: 新经验
        """
        # 1. 记录快照
        snapshot = SelfSnapshot(
            timestamp=time.time(),
            self_vector=self.self_vector,
            experience=experience,
        )
        self.history.append(snapshot)
        
        # 2. 更新能力评估
        if experience.outcome is not None:
            self.self_vector.update_competence(
                domain=experience.domain,
                performance=experience.outcome.success_rate,
            )
        
        # 3. 更新自我知识（可选：从历史中学习）
        if len(self.history) > 100:
            self._update_self_knowledge()
    
    def query_self_knowledge(self, query: str) -> float:
        """
        查询自我知识
        
        Args:
            query: 查询，如"我能做好语言任务吗？"
            
        Returns:
            confidence: 置信度 (0-1)
        """
        # 简化版本：直接查能力评估
        if '语言' in query or 'language' in query.lower():
            return self.self_vector.competence_assessment.get('language', 0.5)
        elif '推理' in query or 'reasoning' in query.lower():
            return self.self_vector.competence_assessment.get('reasoning', 0.5)
        else:
            return 0.5  # 默认
    
    def project_future_self(
        self,
        action: Action,
        predicted_outcome: PredictedOutcome,
    ) -> SelfVector:
        """
        预测未来自我
        
        Args:
            action: 候选行动
            predicted_outcome: 预测结果
            
        Returns:
            future_self: 预测的自我状态
        """
        # 克隆当前自我向量
        future_self = deepcopy(self.self_vector)
        
        # 基于预测结果更新
        if predicted_outcome.success_rate > 0.7:
            # 成功 → 能力提升
            future_self.update_competence(
                domain=action.domain,
                performance=predicted_outcome.success_rate,
            )
        
        # 更新内稳态预测
        # ...
        
        return future_self
```

---

## 3. 元认知模块

```python
class MetacognitionModule(nn.Module):
    """
    元认知 - 对认知的认知
    
    功能：
    1. 监控自身认知过程
    2. 评估认知策略
    3. 调节认知资源
    """
    
    def monitor_cognitive_process(self) -> CognitiveReport:
        """监控认知过程"""
        pass
    
    def evaluate_strategy(self, strategy: str, outcome: Outcome) -> float:
        """评估策略效果"""
        pass
    
    def adjust_resources(self, priority: Priority):
        """调节认知资源分配"""
        pass
```

---

## 4. 实现路线图

| 阶段 | 任务 | 预计工时 | 状态 |
|------|------|----------|------|
| **Phase 1** | SelfVector 实现 | 3 天 | ⏳ 待开始 |
| **Phase 1** | SelfModel 核心 | 1 周 | ⏳ 待开始 |
| **Phase 2** | Metacognition 实现 | 1 周 | ⏳ 待开始 |
| **Phase 3** | 自指机制完善 | 1 周 | ⏳ 待开始 |

---

*自我模型模块规格 完成日期：2026 年 4 月 15 日*
