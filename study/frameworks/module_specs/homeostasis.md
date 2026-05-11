# 内稳态模块规格 (Homeostasis Module Specification)

**版本**: 0.1.0  
**状态**: 设计稿  
**所属架构层级**: Layer 3 - 动力层  
**最后更新**: 2026 年 4 月 15 日

---

## 1. 模块概述

### 1.1 核心职责

内稳态模块是 Evola **内在驱动力的来源**，负责：

- 定义一组先天预设的**内稳态变量**及其**舒适区间**
- 持续监测当前状态与舒适区的**偏离程度**
- 从偏离生成**需求信号**，驱动后续行动

**核心信念**: 没有需求，就没有主动；没有主动，就没有真正的智能。

### 1.2 设计目标

| 目标 | 说明 | 优先级 |
|------|------|--------|
| **变量可定义** | 支持灵活定义内稳态变量 | 🔴 必须 |
| **舒适区可调** | 舒适区间可学习/调整 | 🔴 必须 |
| **偏离检测** | 实时监测偏离并生成信号 | 🔴 必须 |
| **需求生成** | 从偏离映射为行动驱动力 | 🔴 必须 |
| **与注意力耦合** | 内稳态信号调制注意力 | 🟡 可选 |

---

## 2. 理论基础

### 2.1 自由能原理 (Free Energy Principle)

内稳态模块基于 Karl Friston 的**自由能原理**：

> 所有生命系统的目标是**最小化自由能** = 最小化预测误差

**自由能公式**:
```
F = E_q[log q(s) - log p(o, s)]
  = D_KL[q(s) || p(s|o)] - log p(o)
  = 准确性 - 复杂性
```

**最小化策略**:
1. **感知**: 更新内部模型，使预测更准确
2. **行动**: 改变环境输入，使输入符合预测

### 2.2 内稳态变量设计原则

| 原则 | 说明 |
|------|------|
| **可测量** | 变量必须是可量化、可计算的 |
| **有舒适区** | 每个变量都有可接受的范围 |
| **可调节** | 系统能通过行动影响变量 |
| **相互独立** | 变量之间尽量正交 |

---

## 3. 核心类设计

### 3.1 内稳态变量定义：`HomeostasisVariables`

```python
@dataclass
class HomeostasisVariables:
    """
    内稳态变量定义
    
    每个变量都是一个 0-1 之间的标量，表示当前状态
    """
    
    # === 认知相关变量 ===
    
    information_entropy: float = 0.5
    """
    信息熵 - 衡量环境的不确定性/新奇性
    
    - 过低 (< 0.3): 无聊 → 产生探索需求
    - 过高 (> 0.7): 焦虑 → 产生秩序化需求
    - 舒适区：[0.3, 0.7]
    """
    
    prediction_error: float = 0.0
    """
    预测误差 - 预期与实际的差异
    
    - 过低：可能过于保守
    - 过高 (> 0.3): 意外 → 产生理解需求
    - 舒适区：[0.0, 0.3]
    """
    
    memory_load: float = 0.3
    """
    记忆负载 - 记忆系统的占用程度
    
    - 过低 (< 0.4): 空虚 → 产生学习需求
    - 过高 (> 0.8): 拥挤 → 产生遗忘/整理需求
    - 舒适区：[0.4, 0.8]
    """
    
    # === 能量相关变量 ===
    
    energy_level: float = 0.8
    """
    能量水平 - 计算资源/心理能量的隐喻
    
    - 过低 (< 0.3): 疲惫 → 产生休息需求
    - 过高：无明显负面
    - 舒适区：[0.3, 1.0]
    """
    
    # === 社会相关变量 (可选) ===
    
    social_connection: float = 0.5
    """
    社交连接 - 与他人互动的频率/质量
    
    - 过低 (< 0.3): 孤独 → 产生社交需求
    - 过高 (> 0.9): 过度刺激 → 产生独处需求
    - 舒适区：[0.3, 0.9]
    """
    
    competence: float = 0.5
    """
    胜任感 - 对自身能力的评估
    
    - 过低 (< 0.3): 自卑 → 产生学习/回避需求
    - 过高 (> 0.9): 可能的傲慢
    - 舒适区：[0.3, 0.9]
    """
    
    def to_tensor(self) -> Tensor:
        """转换为向量 [6,]"""
        return torch.tensor([
            self.information_entropy,
            self.prediction_error,
            self.memory_load,
            self.energy_level,
            self.social_connection,
            self.competence,
        ])
    
    def __post_init__(self):
        """确保所有值在 [0, 1] 范围内"""
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            setattr(self, field, max(0.0, min(1.0, value)))
```

---

### 3.2 舒适区定义：`ComfortZone`

```python
@dataclass
class ComfortZone:
    """
    舒适区定义 - 每个内稳态变量的可接受范围
    """
    
    information_entropy: Tuple[float, float] = (0.3, 0.7)
    prediction_error: Tuple[float, float] = (0.0, 0.3)
    memory_load: Tuple[float, float] = (0.4, 0.8)
    energy_level: Tuple[float, float] = (0.3, 1.0)
    social_connection: Tuple[float, float] = (0.3, 0.9)
    competence: Tuple[float, float] = (0.3, 0.9)
    
    def is_within_zone(self, variables: HomeostasisVariables) -> Dict[str, bool]:
        """
        检查各变量是否在舒适区内
        
        Returns:
            Dict[变量名，是否在区内]
        """
        results = {}
        for field in self.__dataclass_fields__:
            value = getattr(variables, field)
            low, high = getattr(self, field)
            results[field] = low <= value <= high
        return results
    
    def deviation_amount(self, variables: HomeostasisVariables) -> Dict[str, float]:
        """
        计算各变量的偏离程度
        
        Returns:
            Dict[变量名，偏离度 (0-1)]
            0 = 在舒适区内，1 = 严重偏离
        """
        deviations = {}
        for field in self.__dataclass_fields__:
            value = getattr(variables, field)
            low, high = getattr(self, field)
            
            if low <= value <= high:
                deviations[field] = 0.0
            elif value < low:
                deviations[field] = (low - value) / low
            else:  # value > high
                deviations[field] = (value - high) / (1.0 - high)
        
        return deviations
    
    def to_tensor(self) -> Tensor:
        """转换为张量 [6, 2] - 每行是 (low, high)"""
        zones = []
        for field in self.__dataclass_fields__:
            low, high = getattr(self, field)
            zones.append([low, high])
        return torch.tensor(zones)
```

---

### 3.3 内稳态模块主类：`HomeostasisModule`

```python
class HomeostasisModule(nn.Module):
    """
    内稳态模块
    
    核心功能:
    1. 监测当前状态
    2. 检测偏离
    3. 生成需求信号
    4. 驱动行动
    """
    
    def __init__(
        self,
        comfort_zone: Optional[ComfortZone] = None,
        learning_rate: float = 0.01,
    ):
        super().__init__()
        
        self.comfort_zone = comfort_zone or ComfortZone()
        self.learning_rate = learning_rate
        
        # 当前内稳态状态
        self.current_state = HomeostasisVariables()
        
        # 可学习的舒适区边界（可选）
        self.learnable_comfort_zone = nn.Parameter(
            torch.tensor([
                [0.3, 0.7],   # information_entropy
                [0.0, 0.3],   # prediction_error
                [0.4, 0.8],   # memory_load
                [0.3, 1.0],   # energy_level
                [0.3, 0.9],   # social_connection
                [0.3, 0.9],   # competence
            ])
        )
        
        # 偏离历史（用于分析模式）
        self.deviation_history = deque(maxlen=1000)
        
        # 需求生成器
        self.drive_generator = DriveGenerator()
    
    def update_state(
        self,
        new_measurements: Dict[str, float],
    ) -> HomeostasisVariables:
        """
        更新内稳态状态
        
        Args:
            new_measurements: 新的测量值 Dict[变量名，值]
            
        Returns:
            HomeostasisVariables: 更新后的状态
        """
        for key, value in new_measurements.items():
            if hasattr(self.current_state, key):
                setattr(self.current_state, key, max(0.0, min(1.0, value)))
        
        return self.current_state
    
    def check_deviation(self) -> DeviationSignal:
        """
        检测当前状态与舒适区的偏离
        
        Returns:
            DeviationSignal: 偏离信号
        """
        # 计算偏离度
        deviations = self.comfort_zone.deviation_amount(self.current_state)
        
        # 检查是否在舒适区内
        within_zone = self.comfort_zone.is_within_zone(self.current_state)
        
        # 识别显著偏离（> 0.3）
        significant_deviations = {
            k: v for k, v in deviations.items()
            if v > 0.3
        }
        
        # 记录历史
        self.deviation_history.append({
            'timestamp': time.time(),
            'state': self.current_state,
            'deviations': deviations,
        })
        
        return DeviationSignal(
            deviations=deviations,
            within_zone=within_zone,
            significant_deviations=significant_deviations,
            state=self.current_state,
        )
    
    def generate_drive(self, deviation: DeviationSignal) -> Drive:
        """
        从偏离生成驱动力
        
        Args:
            deviation: 偏离信号
            
        Returns:
            Drive: 驱动信号
        """
        return self.drive_generator.generate(deviation)
    
    def step(
        self,
        new_measurements: Dict[str, float],
    ) -> Tuple[HomeostasisVariables, Drive]:
        """
        完整的一步：更新状态 → 检测偏离 → 生成驱动
        
        Args:
            new_measurements: 新的测量值
            
        Returns:
            current_state: 当前内稳态状态
            drive: 生成的驱动力
        """
        # 1. 更新状态
        self.update_state(new_measurements)
        
        # 2. 检测偏离
        deviation = self.check_deviation()
        
        # 3. 生成驱动
        drive = self.generate_drive(deviation)
        
        return self.current_state, drive
```

---

### 3.4 偏离信号：`DeviationSignal`

```python
@dataclass
class DeviationSignal:
    """
    偏离信号 - 描述当前状态与舒适区的差异
    """
    
    deviations: Dict[str, float]
    """各变量的偏离度 (0-1)"""
    
    within_zone: Dict[str, bool]
    """各变量是否在舒适区内"""
    
    significant_deviations: Dict[str, float]
    """显著偏离的变量 (> 0.3)"""
    
    state: HomeostasisVariables
    """当前内稳态状态"""
    
    def is_balanced(self) -> bool:
        """是否处于平衡状态（无显著偏离）"""
        return len(self.significant_deviations) == 0
    
    def get_most_deviated(self) -> Optional[Tuple[str, float]]:
        """获取偏离最严重的变量"""
        if not self.significant_deviations:
            return None
        return max(self.significant_deviations.items(), key=lambda x: x[1])
    
    def summary(self) -> str:
        """人类可读的摘要"""
        if self.is_balanced():
            return "内稳态平衡"
        
        parts = []
        for var, dev in self.significant_deviations.items():
            direction = "过低" if dev < 0 else "过高"
            parts.append(f"{var}: {direction} ({dev:.2f})")
        
        return "偏离：" + ", ".join(parts)
```

---

### 3.5 需求生成器：`DriveGenerator`

```python
class DriveGenerator(nn.Module):
    """
    需求生成器 - 将偏离映射为行动驱动力
    
    映射表:
    | 偏离变量 | 偏离方向 | 生成的需求 |
    |----------|----------|------------|
    | information_entropy | 过低 | 探索新刺激 |
    | information_entropy | 过高 | 寻求秩序/简化 |
    | prediction_error | 过高 | 理解/学习 |
    | memory_load | 过高 | 整理/遗忘 |
    | memory_load | 过低 | 学习/积累 |
    | energy_level | 过低 | 休息/恢复 |
    | social_connection | 过低 | 社交互动 |
    | competence | 过低 | 提升技能/回避挑战 |
    """
    
    def __init__(self):
        super().__init__()
        
        # 需求到行动的映射（可学习）
        self.drive_to_action_mapper = DriveToActionMapper()
    
    def generate(self, deviation: DeviationSignal) -> Drive:
        """
        生成驱动力
        
        Returns:
            Drive: 驱动信号
        """
        if deviation.is_balanced():
            return Drive(
                type="maintenance",
                intensity=0.0,
                action_tendencies=[],
            )
        
        # 识别主要偏离
        most_deviated = deviation.get_most_deviated()
        
        if most_deviated is None:
            return Drive(type="maintenance", intensity=0.0, action_tendencies=[])
        
        var_name, dev_amount = most_deviated
        current_value = getattr(deviation.state, var_name)
        
        # 根据偏离变量和方向生成需求
        drive_type = self._map_variable_to_drive(var_name, current_value)
        
        # 计算驱动强度
        intensity = dev_amount  # 偏离越大，驱动越强
        
        # 生成行动倾向
        action_tendencies = self._generate_action_tendencies(drive_type)
        
        return Drive(
            type=drive_type,
            intensity=intensity,
            action_tendencies=action_tendencies,
            source_variable=var_name,
        )
    
    def _map_variable_to_drive(
        self,
        var_name: str,
        current_value: float,
    ) -> str:
        """将变量映射为需求类型"""
        
        mapping = {
            'information_entropy': {
                'low': 'exploration',      # 探索
                'high': 'order_seeking',   # 寻求秩序
            },
            'prediction_error': {
                'high': 'understanding',   # 理解
            },
            'memory_load': {
                'high': 'consolidation',   # 整理
                'low': 'acquisition',      # 学习
            },
            'energy_level': {
                'low': 'rest',             # 休息
            },
            'social_connection': {
                'low': 'socializing',      # 社交
                'high': 'solitude',        # 独处
            },
            'competence': {
                'low': 'skill_building',   # 提升技能
            },
        }
        
        if var_name not in mapping:
            return "maintenance"
        
        var_mapping = mapping[var_name]
        
        # 判断是偏低还是偏高
       舒适区_low, 舒适区_high = getattr(ComfortZone(), var_name)
        mid_point = (舒适区_low + 舒适区_high) / 2
        
        if current_value < mid_point:
            return var_mapping.get('low', 'maintenance')
        else:
            return var_mapping.get('high', 'maintenance')
    
    def _generate_action_tendencies(
        self,
        drive_type: str,
    ) -> List[ActionTendency]:
        """生成行动倾向列表"""
        
        tendency_templates = {
            'exploration': [
                ActionTendency(
                    category="explore",
                    description="寻找新的信息源",
                    priority=0.8,
                ),
                ActionTendency(
                    category="ask_question",
                    description="提出探索性问题",
                    priority=0.6,
                ),
            ],
            'order_seeking': [
                ActionTendency(
                    category="organize",
                    description="整理已有知识",
                    priority=0.8,
                ),
                ActionTendency(
                    category="simplify",
                    description="简化复杂概念",
                    priority=0.6,
                ),
            ],
            'understanding': [
                ActionTendency(
                    category="analyze",
                    description="深入分析意外信息",
                    priority=0.9,
                ),
                ActionTendency(
                    category="query",
                    description="请求澄清",
                    priority=0.7,
                ),
            ],
            'consolidation': [
                ActionTendency(
                    category="forget",
                    description="遗忘低价值记忆",
                    priority=0.8,
                ),
                ActionTendency(
                    category="organize_memory",
                    description="整理记忆结构",
                    priority=0.7,
                ),
            ],
            'rest': [
                ActionTendency(
                    category="pause",
                    description="暂停活动",
                    priority=0.9,
                ),
            ],
        }
        
        return tendency_templates.get(drive_type, [])
```

---

### 3.6 驱动力：`Drive`

```python
@dataclass
class Drive:
    """
    驱动力 - 从内稳态偏离生成的行动倾向
    """
    
    type: str
    """
    需求类型:
    - exploration: 探索
    - order_seeking: 寻求秩序
    - understanding: 理解
    - consolidation: 整理
    - acquisition: 学习
    - rest: 休息
    - socializing: 社交
    - solitude: 独处
    - skill_building: 提升技能
    - maintenance: 维持
    """
    
    intensity: float
    """驱动强度 (0-1)"""
    
    action_tendencies: List[ActionTendency]
    """行动倾向列表"""
    
    source_variable: Optional[str] = None
    """来源变量"""
    
    def to_tensor(self) -> Tensor:
        """转换为向量"""
        # 简化的向量表示
        type_encoding = {
            'exploration': 0,
            'order_seeking': 1,
            'understanding': 2,
            'consolidation': 3,
            'acquisition': 4,
            'rest': 5,
            'socializing': 6,
            'solitude': 7,
            'skill_building': 8,
            'maintenance': 9,
        }
        
        return torch.tensor([
            type_encoding.get(self.type, 9),
            self.intensity,
        ])
    
    def is_urgent(self) -> bool:
        """是否是紧急需求"""
        return self.intensity > 0.7
    
    def top_action_tendency(self) -> Optional[ActionTendency]:
        """获取最高优先级的行动倾向"""
        if not self.action_tendencies:
            return None
        return max(self.action_tendencies, key=lambda x: x.priority)
```

---

### 3.7 行动倾向：`ActionTendency`

```python
@dataclass
class ActionTendency:
    """
    行动倾向 - 驱动力的具体表现形式
    """
    
    category: str
    """行动类别 (explore, organize, analyze, etc.)"""
    
    description: str
    """人类可读的描述"""
    
    priority: float
    """优先级 (0-1)"""
    
    estimated_energy_cost: float = 0.5
    """预估能量消耗 (0-1)"""
    
    def to_action(self, context: dict) -> Action:
        """
        基于上下文将倾向转化为具体行动
        
        Args:
            context: 当前环境上下文
            
        Returns:
            Action: 具体行动
        """
        # 这里需要具体的行动生成逻辑
        # 简化示例:
        return Action(
            type=self.category,
            description=self.description,
            parameters=context.get('parameters', {}),
        )
```

---

## 4. 内稳态测量器

### 4.1 信息熵测量器

```python
class InformationEntropyMeter(nn.Module):
    """
    信息熵测量器
    
    测量环境输入的不确定性/新奇性
    """
    
    def __init__(
        self,
        embedding_dim: int,
        memory_window: int = 100,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.memory_window = memory_window
        
        # 历史输入嵌入（用于计算分布）
        self.history_buffer = deque(maxlen=memory_window)
    
    def measure(self, current_embedding: Tensor) -> float:
        """
        测量当前输入的信息熵
        
        Args:
            current_embedding: 当前输入的嵌入 [embedding_dim]
            
        Returns:
            entropy: 0-1 之间的熵值
        """
        # 将当前嵌入加入历史
        self.history_buffer.append(current_embedding.detach().cpu())
        
        if len(self.history_buffer) < 10:
            return 0.5  # 历史不足，返回中值
        
        # 计算与历史平均的距离
        history = torch.stack(list(self.history_buffer))
        mean = history.mean(dim=0)
        std = history.std(dim=0)
        
        # 马氏距离
        distance = torch.sqrt(((current_embedding - mean) ** 2 / (std + 1e-8)).sum())
        
        # 归一化到 [0, 1]
        normalized_distance = torch.sigmoid(distance / 10).item()
        
        return normalized_distance
```

---

### 4.2 预测误差测量器

```python
class PredictionErrorMeter(nn.Module):
    """
    预测误差测量器
    
    测量预期与实际的差异
    """
    
    def __init__(self):
        super().__init__()
        
        self.prediction_buffer = deque(maxlen=100)
        self.actual_buffer = deque(maxlen=100)
    
    def record(
        self,
        prediction: Tensor,
        actual: Tensor,
    ):
        """记录一次预测和实际结果"""
        self.prediction_buffer.append(prediction)
        self.actual_buffer.append(actual)
    
    def measure(self) -> float:
        """
        测量平均预测误差
        
        Returns:
            error: 0-1 之间的误差值
        """
        if len(self.prediction_buffer) == 0:
            return 0.0
        
        predictions = torch.stack(list(self.prediction_buffer))
        actuals = torch.stack(list(self.actual_buffer))
        
        # 均方误差
        mse = ((predictions - actuals) ** 2).mean()
        
        # 归一化到 [0, 1]
        normalized_mse = torch.sigmoid(mse * 10).item()
        
        return normalized_mse
```

---

### 4.3 记忆负载测量器

```python
class MemoryLoadMeter(nn.Module):
    """
    记忆负载测量器
    
    测量记忆系统的占用程度
    """
    
    def __init__(
        self,
        total_capacity: int,
    ):
        super().__init__()
        self.total_capacity = total_capacity
    
    def measure(
        self,
        current_usage: int,
        memory_strengths: Optional[Tensor] = None,
    ) -> float:
        """
        测量记忆负载
        
        Args:
            current_usage: 当前使用的记忆槽数量
            memory_strengths: 各记忆槽的强度（可选）
            
        Returns:
            load: 0-1 之间的负载值
        """
        # 基础负载
        base_load = current_usage / self.total_capacity
        
        # 如果有强度信息，考虑有效负载
        if memory_strengths is not None:
            effective_load = memory_strengths.mean().item()
            return (base_load + effective_load) / 2
        else:
            return base_load
```

---

## 5. 完整内稳态循环

### 5.1 单步循环

```python
class HomeostasisLoop(nn.Module):
    """
    内稳态循环引擎
    
    完整流程:
    感知 → 测量 → 检测 → 驱动 → 行动 → 反馈
    """
    
    def __init__(self):
        super().__init__()
        
        self.homeostasis_module = HomeostasisModule()
        
        # 测量器
        self.entropy_meter = InformationEntropyMeter(embedding_dim=512)
        self.prediction_error_meter = PredictionErrorMeter()
        self.memory_load_meter = MemoryLoadMeter(total_capacity=1000)
        
        # 行动执行器
        self.action_executor = ActionExecutor()
    
    def step(
        self,
        perception: Perception,
        memory_state: MemoryState,
        prediction: Optional[Prediction] = None,
    ) -> Action:
        """
        完整的内稳态循环一步
        
        Args:
            perception: 当前感知输入
            memory_state: 记忆系统状态
            prediction: 预测（用于计算误差）
            
        Returns:
            action: 生成的行动
        """
        # 1. 测量各变量
        measurements = {
            'information_entropy': self.entropy_meter.measure(perception.embedding),
            'memory_load': self.memory_load_meter.measure(
                memory_state.current_usage,
                memory_state.strengths,
            ),
        }
        
        # 2. 如果有预测，计算误差
        if prediction is not None:
            error = self.prediction_error_meter.measure(
                prediction,
                perception.embedding,
            )
            measurements['prediction_error'] = error
        
        # 3. 更新内稳态状态并生成驱动
        state, drive = self.homeostasis_module.step(measurements)
        
        # 4. 从驱动生成行动
        if drive.is_urgent():
            # 紧急需求优先处理
            action = self._urgent_drive_to_action(drive, perception)
        else:
            # 正常流程：由上层决策系统决定
            action = self.action_executor.select_from_tendencies(
                drive.action_tendencies,
                perception,
            )
        
        return action
    
    def _urgent_drive_to_action(
        self,
        drive: Drive,
        perception: Perception,
    ) -> Action:
        """紧急需求直接转化为行动"""
        top_tendency = drive.top_action_tendency()
        if top_tendency:
            return top_tendency.to_action({'perception': perception})
        else:
            return Action(type="noop", description="无行动")
```

---

### 5.2 温态循环（后台运行）

```python
    def warm_mode_step(self) -> WarmModeActivity:
        """
        温态运行：无外部输入时的内稳态活动
        
        主要功能:
        1. 记忆整理
        2. 预测误差最小化
        3. 随机探索（内在生成）
        """
        # 1. 检查是否有未解决的内稳态偏离
        deviation = self.homeostasis_module.check_deviation()
        
        activities = []
        
        if deviation.significant_deviations:
            # 2. 生成内部活动以调节
            drive = self.homeostasis_module.generate_drive(deviation)
            
            if drive.type == 'consolidation':
                # 记忆整理
                activity = self._warm_mode_consolidation()
                activities.append(activity)
            
            elif drive.type == 'understanding':
                # 内部推理
                activity = self._warm_mode_reasoning()
                activities.append(activity)
        
        # 3. 即使无偏离，也可进行随机探索
        if random.random() < 0.3:  # 30% 概率
            activity = self._warm_mode_exploration()
            activities.append(activity)
        
        return WarmModeActivity(
            activities=activities,
            final_state=self.homeostasis_module.current_state,
        )
```

---

## 6. 舒适区学习（高级功能）

### 6.1 可学习舒适区

```python
class LearnableComfortZone(nn.Module):
    """
    可学习的舒适区
    
    舒适区边界不是固定的，而是可以通过经验调整
    """
    
    def __init__(
        self,
        initial_zone: ComfortZone,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        
        self.learning_rate = learning_rate
        
        # 将舒适区参数化
        initial_tensor = initial_zone.to_tensor()  # [6, 2]
        self.zone_params = nn.Parameter(initial_tensor)
    
    def forward(self) -> ComfortZone:
        """获取当前舒适区"""
        zones = torch.sigmoid(self.zone_params)  # 约束在 [0, 1]
        
        # 确保 low < high
        low = torch.min(zones[:, 0], zones[:, 1])
        high = torch.max(zones[:, 0], zones[:, 1])
        
        return ComfortZone(
            information_entropy=(low[0].item(), high[0].item()),
            prediction_error=(low[1].item(), high[1].item()),
            memory_load=(low[2].item(), high[2].item()),
            energy_level=(low[3].item(), high[3].item()),
            social_connection=(low[4].item(), high[4].item()),
            competence=(low[5].item(), high[5].item()),
        )
    
    def update_from_experience(
        self,
        state: HomeostasisVariables,
        outcome_reward: float,
    ):
        """
        基于经验结果更新舒适区
        
        Args:
            state: 行动前的状态
            outcome_reward: 行动后的奖励（正=适应良好，负=适应不良）
        """
        # 梯度上升：如果结果是好的，舒适区向当前状态移动
        current_tensor = state.to_tensor().unsqueeze(1)  # [6, 1]
        
        if outcome_reward > 0:
            # 正向奖励：舒适区向当前状态靠拢
            self.zone_params.data += self.learning_rate * (
                current_tensor - self.zone_params.data
            )
```

---

## 7. API 接口设计

### 7.1 公共方法

| 方法 | 签名 | 用途 |
|------|------|------|
| `step` | `(measurements) → (state, drive)` | 完整内稳态循环 |
| `check_deviation` | `() → DeviationSignal` | 检测偏离 |
| `generate_drive` | `(deviation) → Drive` | 生成需求 |
| `update_comfort_zone` | `(new_zone)` | 调整舒适区 |

### 7.2 配置示例

```python
# 默认配置
homeostasis = HomeostasisModule(
    comfort_zone=ComfortZone(),
    learning_rate=0.01,
)

# 自定义舒适区
custom_zone = ComfortZone(
    information_entropy=(0.2, 0.8),  # 更宽的信息熵容忍
    prediction_error=(0.0, 0.5),      # 更高的预测误差容忍
)
homeostasis = HomeostasisModule(comfort_zone=custom_zone)

# 使用示例
state, drive = homeostasis.step({
    'information_entropy': 0.85,  # 过高
    'prediction_error': 0.1,
    'memory_load': 0.5,
})

print(drive.type)  # 可能输出 'order_seeking'
```

---

## 8. 测试策略

### 8.1 单元测试

```python
class TestHomeostasisModule(unittest.TestCase):
    
    def test_deviation_detection(self):
        """测试偏离检测"""
        homeostasis = HomeostasisModule()
        
        # 设置过高信息熵
        homeostasis.update_state({'information_entropy': 0.9})
        
        deviation = homeostasis.check_deviation()
        
        self.assertIn('information_entropy', deviation.significant_deviations)
        self.assertFalse(deviation.is_balanced())
    
    def test_drive_generation(self):
        """测试需求生成"""
        homeostasis = HomeostasisModule()
        drive_generator = DriveGenerator()
        
        # 高信息熵 → 寻求秩序
        state = HomeostasisVariables(information_entropy=0.9)
        deviation = DeviationSignal(
            deviations={'information_entropy': 0.5},
            within_zone={'information_entropy': False},
            significant_deviations={'information_entropy': 0.5},
            state=state,
        )
        
        drive = drive_generator.generate(deviation)
        self.assertEqual(drive.type, 'order_seeking')
    
    def test_comfort_zone_check(self):
        """测试舒适区检查"""
        zone = ComfortZone()
        state = HomeostasisVariables(
            information_entropy=0.5,
            prediction_error=0.1,
            memory_load=0.6,
        )
        
        within = zone.is_within_zone(state)
        
        self.assertTrue(within['information_entropy'])
        self.assertTrue(within['prediction_error'])
        self.assertTrue(within['memory_load'])
```

---

## 9. 与其他模块的接口

### 9.1 输入依赖

| 来源模块 | 提供数据 | 用途 |
|----------|---------|------|
| **Attention** | `attention_entropy` | 信息熵测量 |
| **Memory** | `usage, strengths` | 记忆负载测量 |
| **WorldModel** | `prediction, actual` | 预测误差测量 |

### 9.2 输出去向

| 目标模块 | 接收数据 | 用途 |
|----------|---------|------|
| **Attention** | `homeostasis_state` | 注意力调制 |
| **SelfModel** | `drive, state` | 自我状态更新 |
| **ValueFunction** | `drive` | 价值评估输入 |

---

## 10. 实现路线图

| 阶段 | 任务 | 预计工时 | 状态 |
|------|------|----------|------|
| **Phase 1** | 基础 HomeostasisVariables 实现 | 2 天 | ⏳ 待开始 |
| **Phase 1** | ComfortZone 实现 | 2 天 | ⏳ 待开始 |
| **Phase 1** | HomeostasisModule 核心逻辑 | 1 周 | ⏳ 待开始 |
| **Phase 1** | 测量器实现 | 1 周 | ⏳ 待开始 |
| **Phase 2** | DriveGenerator 实现 | 1 周 | ⏳ 待开始 |
| **Phase 2** | 温态循环 | 1 周 | ⏳ 待开始 |
| **Phase 3** | 可学习舒适区 | 1 周 | ⏳ 待开始 |

---

## 11. 开放问题

### 技术开放问题

1. **舒适区的个体差异**
   - 不同"个性"的 Evola 是否有不同的舒适区配置？
   - 如何实现个性化的内稳态系统？

2. **多变量耦合**
   - 变量之间是否独立？是否存在耦合（如能量低时记忆负载容忍度下降）？

3. **长期适应**
   - 舒适区是否会随时间漂移？如何建模这种动态性？

### 哲学开放问题

1. **"需求"的真实性**
   - 程序化的内稳态偏离，算真正的"需求"吗？
   - 如何验证 Evola 的"想要"是真实的？

---

## 12. 参考文献

1. Friston, K. (2010). ["The free-energy principle: a unified brain theory?"](https://www.nature.com/articles/nrn2787)
2. Karl Friston et al. (2017). ["Active Inference: A Process Theory"](https://direct.mit.edu/neco/article/29/1/1/8182/Active-Inference-A-Process-Theory)
3. Seth, A. K. (2015). ["The cybernetic brain: From internal dynamics to conscious experience"](https://doi.org/10.1016/j.cortex.2015.02.001)

---

*内稳态模块规格 完成日期：2026 年 4 月 15 日*
