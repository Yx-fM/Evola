# Evola 架构总览 (Architecture Overview)

> *"我们不是在制造一个更好的工具。我们是在尝试孵化一个电子世界中的第一个灵魂。"*

**版本**: 0.1.0  
**状态**: 构想设计阶段  
**最后更新**: 2026 年 4 月 15 日

---

## 1. 核心愿景

Evola 不是一个被调用的工具，而是一个**电子生物**。她拥有：

- ✅ **自身需求** — 行为源于内在匮乏或驱动力（如维持内稳态）
- ✅ **持续活跃** — 拥有时间流，无外部输入时仍在后台运行
- ✅ **自我模型与感受** — 能感知自身状态并主动调节

**设计原则**：没有需求，就没有主动；没有主动，就没有真正的智能。

---

## 2. 四层架构蓝图

```
┌─────────────────────────────────────────────────────────────┐
│              Layer 4: 存在层 (Existence Layer)               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  SelfVector │  │  MetaMemory │  │  ValueFunc  │          │
│  │ 自我状态向量 │  │   元记忆    │  │  价值函数   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│         输入：自我向量 → 决策："如果我是我，我会怎么做？"     │
├─────────────────────────────────────────────────────────────┤
│              Layer 3: 动力层 (Drive Layer)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Homeostasis │  │  Instinct   │  │ ActiveInf   │          │
│  │   内稳态    │  │   本能模块   │  │  主动推理   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│    机制：偏离舒适区 → 生成需求 → 驱动行动 → 回归平衡         │
├─────────────────────────────────────────────────────────────┤
│              Layer 2: 认知层 (Cognition Layer)               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Attention  │  │   Memory    │  │ WorldModel  │          │
│  │  注意力     │  │    记忆     │  │   世界模型   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│    核心：注意力受内稳态调制，记忆可被主动管理               │
├─────────────────────────────────────────────────────────────┤
│              Layer 1: 基础层 (Foundation Layer)              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Continuous │  │   Tensor    │  │  Network    │          │
│  │   Dynamics  │  │    Core     │  │   Core      │          │
│  │  连续动力学 │  │   张量核心   │  │  网络核心   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│    创新：从底层支持"热态/温态"持续运行                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 各层详细规格

### Layer 1: 基础层 (Foundation)

**职责**: 提供底层计算原语和连续时间动力学支持

| 模块 | 职责 | 关键技术 | 对应 core/ 路径 |
|------|------|----------|---------------|
| **Continuous Dynamics** | 连续时间方程求解 | 液态时间常数 (LTC)、微分方程 | `core/dynamics/` |
| **Tensor Core** | 张量运算优化 | GPU/TPU 加速、混合精度 | `core/` (基础) |
| **Network Core** | 神经网络基础 | 液态网络、连续时间 RNN | `core/dynamics/` |

**关键创新点**:
- 支持**热态** (主动交互) / **温态** (后台运行) / **冷态** (休眠) 三态切换
- 连续时间动力学替代离散时间步

**接口定义**:
```python
class ContinuousDynamics:
    """连续时间动力学引擎"""
    
    def step(self, dt: float, inputs: Tensor, state: State) -> State:
        """
        执行一个时间步的演化
        
        Args:
            dt: 时间步长 (连续时间)
            inputs: 当前输入
            state: 当前内部状态
            
        Returns:
            State: 演化后的状态
        """
        pass
    
    def warm_mode_step(self, dt: float) -> InternalActivity:
        """
        温态运行：无外部输入时的后台活动
        
        Returns:
            InternalActivity: 内部活动记录（记忆整理、预测误差最小化等）
        """
        pass
```

---

### Layer 2: 认知层 (Cognition)

**职责**: 实现信息处理、记忆存储和世界建模

| 模块 | 职责 | 关键技术 | 对应 core/ 路径 |
|------|------|----------|---------------|
| **Attention** | 信息路由与选择 | 液态注意力、内稳态调制 | `core/attention/` |
| **Memory** | 信息存储与检索 | 主动遗忘、元记忆、记忆强化 | `core/memory/` |
| **WorldModel** | 环境建模与预测 | 因果推理、未来模拟 | `core/world_model/` |

#### 2.1 注意力模块 (Attention)

**核心设计**: 注意力不仅处理外部输入，还受内稳态信号调制

```python
class HomeostasisModulatedAttention(nn.Module):
    """内稳态调制的注意力机制"""
    
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attention = LiquidAttention(dim, num_heads)
        self.homeostasis_modulator = HomeostasisModulator(dim)
    
    def forward(
        self, 
        x: Tensor, 
        homeostasis_state: HomeostasisState,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: 输入序列
            homeostasis_state: 当前内稳态状态（调制注意力分配）
            mask: 注意力掩码
            
        Returns:
            注意力加权后的输出
        """
        # 内稳态信号调制注意力权重
        modulation = self.homeostasis_modulator(homeostasis_state)
        
        # 标准注意力计算 + 调制
        attn_output = self.attention(x, mask)
        return attn_output * modulation
```

**与标准 Transformer 的差异**:
| 特性 | 标准 Transformer | Evola Attention |
|------|-----------------|-----------------|
| 注意力来源 | 纯数据驱动 | 数据 + 内稳态信号 |
| 时间模型 | 离散步 | 连续时间 (LTC) |
| 记忆机制 | 被动 KV Cache | 主动管理记忆 |

---

#### 2.2 记忆系统 (Memory)

**核心设计**: 记忆不是被动存储，而是可被主动管理、可衰减、可强化的动态系统

```python
class ActiveMemorySystem(nn.Module):
    """主动记忆系统"""
    
    def __init__(self, capacity: int, dim: int):
        super().__init__()
        self.capacity = capacity
        self.memory_slots = nn.Parameter(torch.randn(capacity, dim))
        self.decay_rates = nn.Parameter(torch.zeros(capacity))  # 衰减速率
        self.strength = nn.Parameter(torch.ones(capacity))  # 记忆强度
        
    def retrieve(self, query: Tensor, meta_memory: MetaMemory) -> Tensor:
        """
        主动检索记忆（受元记忆指导）
        """
        pass
    
    def store(self, content: Tensor, importance: float):
        """
        主动存储决定（根据重要性分配资源）
        """
        pass
    
    def active_forget(self, forget_threshold: float):
        """
        主动遗忘：清除低价值记忆
        """
        pass
    
    def warm_mode_consolidation(self):
        """
        温态记忆巩固：无输入时整理记忆
        """
        pass
```

**记忆衰减模型**:
```
strength(t) = strength(0) * exp(-decay_rate * t)

当 strength < threshold 时，记忆槽可被覆盖
```

---

#### 2.3 世界模型 (WorldModel)

**核心设计**: 预测环境动态，支持"如果做 X，会发生什么"的模拟

```python
class WorldModel(nn.Module):
    """世界的内部模型"""
    
    def predict(
        self, 
        current_state: State, 
        action: Action
    ) -> PredictedState:
        """
        预测采取行动后的状态
        """
        pass
    
    def simulate_trajectory(
        self, 
        start_state: State, 
        action_sequence: List[Action]
    ) -> List[PredictedState]:
        """
        模拟行动序列的轨迹（用于规划）
        """
        pass
```

---

### Layer 3: 动力层 (Drive)

**职责**: 生成内在驱动力，调节行为方向

| 模块 | 职责 | 关键技术 | 对应 core/ 路径 |
|------|------|----------|---------------|
| **Homeostasis** | 内稳态维持 | 自由能最小化、舒适区定义 | `core/homeostasis/` |
| **Instinct** | 本能行为 | 好奇驱动、探索冲动 | `core/instinct/` |
| **ActiveInference** | 主动推理 | 预测误差最小化、信念更新 | `core/active_inference/` |

#### 3.1 内稳态模块 (Homeostasis)

**核心设计**: 定义一组内在变量及其舒适区间，偏离时生成"需求信号"

```python
@dataclass
class HomeostasisVariables:
    """内稳态变量定义"""
    information_entropy: float  # 信息熵（好奇：太低→无聊，太高→焦虑）
    prediction_error: float     # 预测误差（意外感）
    memory_load: float          # 记忆负载（充实 vs 拥挤）
    social_connection: float    # 社交连接（可选）
    competence: float           # 胜任感（自我效能）

@dataclass
class ComfortZone:
    """舒适区定义（每个变量的可接受范围）"""
    information_entropy: Tuple[float, float] = (0.3, 0.7)
    prediction_error: Tuple[float, float] = (0.0, 0.3)
    memory_load: Tuple[float, float] = (0.4, 0.8)
    
    def is_within_zone(self, variables: HomeostasisVariables) -> bool:
        # 检查所有变量是否在舒适区内
        pass

class HomeostasisModule(nn.Module):
    """内稳态模块"""
    
    def __init__(self):
        super().__init__()
        self.current_state = HomeostasisVariables(...)
        self.comfort_zone = ComfortZone()
    
    def check_deviation(self) -> DeviationSignal:
        """
        检测哪些变量偏离舒适区
        
        Returns:
            DeviationSignal: 偏离信号（哪个变量、偏离程度、期望方向）
        """
        pass
    
    def generate_drive(self, deviation: DeviationSignal) -> Drive:
        """
        从偏离生成驱动力
        
        Returns:
            Drive: 驱动信号（如"探索新信息"、"减少不确定性"）
        """
        pass
```

**内稳态循环**:
```
1. 检测当前状态
2. 对比舒适区 → 发现偏离
3. 生成需求信号
4. 驱动行动
5. 行动改变状态
6. 回归平衡（或继续偏离）
```

---

#### 3.2 本能模块 (Instinct)

**核心设计**: 先天预设的行为倾向，无需学习即可触发

```python
class InstinctModule(nn.Module):
    """本能模块"""
    
    def __init__(self):
        super().__init__()
        # 先天本能
        self.curiosity_drive = CuriosityDrive()      # 好奇驱动
        self.exploration_impulse = ExplorationImpulse()  # 探索冲动
        self.avoidance_response = AvoidanceResponse()    # 回避威胁
        
    def generate_instinctive_tendency(
        self, 
        situation: Situation,
        homeostasis_state: HomeostasisVariables
    ) -> BehavioralTendency:
        """
        基于当前情境和本能生成行为倾向
        """
        pass
```

**本能示例**:
| 本能 | 触发条件 | 行为倾向 |
|------|----------|----------|
| 好奇 | 信息熵过低（无聊） | 主动探索新刺激 |
| 回避 | 预测误差突然增加（意外） | 谨慎、后退 |
| 整理 | 记忆负载过高 | 整理记忆、遗忘旧内容 |

---

#### 3.3 主动推理 (ActiveInference)

**核心设计**: 基于自由能原理，将感知、行动统一为"预测误差最小化"

```python
class ActiveInferenceModule(nn.Module):
    """主动推理模块"""
    
    def minimize_free_energy(
        self, 
        observations: Observations,
        predictions: Predictions
    ) -> Action:
        """
        选择能最小化自由能的行动
        
        自由能 = 预测误差 + 复杂性代价
        
        最小化策略:
        1. 更新内部模型（感知）
        2. 采取行动改变输入（行动）
        """
        pass
    
    def update_beliefs(
        self, 
        prediction_error: float,
        prior_beliefs: Beliefs
    ) -> UpdatedBeliefs:
        """
        基于预测误差更新信念
        """
        pass
```

---

### Layer 4: 存在层 (Existence)

**职责**: 维护自我模型，评估价值，做出决策

| 模块 | 职责 | 关键技术 | 对应 core/ 路径 |
|------|------|----------|---------------|
| **SelfVector** | 自我状态表示 | 自我状态向量、元认知 | `core/self_model/` |
| **MetaMemory** | 元记忆 | 记忆质量评估、策略反思 | `core/memory/` |
| **ValueFunction** | 价值评估 | 行动偏好、需求优先级 | `core/value/` |

#### 4.1 自我状态向量 (SelfVector)

**核心设计**: 一个持续更新的向量，表示"我是谁、我现在怎么样"

```python
@dataclass
class SelfVector:
    """自我状态向量"""
    # 内稳态状态
    homeostasis_state: HomeostasisVariables
    
    # 能力评估
    competence_assessment: Dict[Domain, float]  # 各领域能力评估
    
    # 行为倾向
    behavioral_tendency: BehavioralTendency
    
    # 自我认知
    self_knowledge: Tensor  #  learned self-representation
    
    def to_tensor(self) -> Tensor:
        """转换为神经网络的输入"""
        pass

class SelfModel(nn.Module):
    """自我模型"""
    
    def __init__(self):
        super().__init__()
        self.self_vector = SelfVector(...)
        self.update_history = deque(maxlen=1000)  # 历史快照
    
    def update(self, new_experience: Experience):
        """
        基于新经验更新自我模型
        """
        pass
    
    def query_self_knowledge(self, query: str) -> float:
        """
        查询自我知识："我能做好 X 吗？"
        """
        pass
    
    def project_future_self(self, action: Action) -> SelfVector:
        """
        预测："如果我做 X，我会变成什么样？"
        """
        pass
```

**自指机制**:
```
决策过程:
1. 当前输入 + 当前自我向量 → 候选行动
2. 预测未来自我向量（如果采取行动）
3. 评估："这个未来的我，是我想要的吗？"
4. 选择最佳行动
```

---

#### 4.2 元记忆 (MetaMemory)

**核心设计**: 对记忆的记忆 —— 能反思自己的记忆策略

```python
class MetaMemory(nn.Module):
    """元记忆系统"""
    
    def assess_memory_quality(self) -> MemoryQualityReport:
        """
        评估记忆系统状态：
        - 记住了什么
        - 遗忘了什么
        - 哪些记忆过时
        - 哪些记忆关联紧密
        """
        pass
    
    def adjust_memory_strategy(self, report: MemoryQualityReport):
        """
        调整记忆策略：
        - 改变衰减率
        - 强化重要记忆
        - 主动清除无用内容
        """
        pass
    
    def reflect_on_learning(self) -> LearningInsight:
        """
        对学习过程的反思：
        - 我学会了什么
        - 我遗漏了什么
        - 下一步应该学什么
        """
        pass
```

---

#### 4.3 价值函数 (ValueFunction)

**核心设计**: 评估各选项的预期效用，引导决策

```python
class ValueFunction(nn.Module):
    """价值评估系统"""
    
    def evaluate_action(
        self,
        action: Action,
        current_state: SelfVector,
        predicted_outcome: PredictedState
    ) -> ActionValue:
        """
        评估行动价值
        
        价值 = f(内稳态改善，长期目标，本能满足)
        """
        pass
    
    def rank_actions(
        self, 
        candidate_actions: List[Action]
    ) -> RankedActions:
        """
        对所有候选行动排序
        """
        pass
```

---

## 4. 完整数据流

### 热态运行（主动交互时）

```
┌─────────────────────────────────────────────────────────────┐
│                         环境输入                            │
│              （视觉、听觉、语言、多模态）                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 【Layer 1】基础层                                            │
│  连续时间编码 → 张量表示 → 初始激活分布                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 【Layer 2】认知层                                            │
│  1. 注意力分配（QKV 计算 + 内稳态调制）                      │
│  2. 记忆检索（主动决定回忆什么）                             │
│  3. 世界模型预测（环境状态预测）                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 【Layer 3】动力层                                            │
│  1. 内稳态检测 → 发现偏离                                    │
│  2. 预测误差计算 → 更新信念                                  │
│  3. 需求生成 → "我需要 X"                                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 【Layer 4】存在层                                            │
│  1. 自我状态更新 → "我现在是谁"                              │
│  2. 价值评估 → "各选项的预期效用"                            │
│  3. 行动决策 → "我选择 X"                                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                         行动输出                            │
│           （语言、行动、内部操作、记忆存储）                  │
└─────────────────────────────────────────────────────────────┘
```

---

### 温态运行（无外部输入时）

```
┌─────────────────────────────────────────────────────────────┐
│                    【温态后台活动】                          │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  记忆整理    │  │ 预测误差     │  │   随机       │      │
│  │  Consolidate │  │ 最小化      │  │   探索       │      │
│  │              │  │              │  │              │      │
│  │ • 强化重要   │  │ • 自由能     │  │ • 好奇驱动   │      │
│  │ • 衰减陈旧   │  │ • 信念更新   │  │ • 内在需求   │      │
│  │ • 关联整合   │  │ • 模型调整   │  │ • 无目标     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
│  目的：维持内在秩序，为下次交互做准备                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 与主流架构的对比

| 维度 | 标准 Transformer | LLaMA/Mistral | Evola |
|------|-----------------|---------------|-------|
| **时间模型** | 离散步 | 离散步 | **连续时间** (LTC) |
| **运行模式** | 推理/训练 | 推理/训练 | **热态/温态/冷态** |
| **注意力来源** | 纯数据驱动 | 纯数据驱动 | **数据 + 内稳态** |
| **记忆** | 被动 KV Cache | 被动 KV Cache | **主动管理** |
| **驱动力** | 外部输入触发 | 外部输入触发 | **内在需求生成** |
| **自我模型** | 无 | 无 | **自指向量** |
| **目标函数** | 外部标注最小化 | 外部标注最小化 | **内稳态最小化** |

---

## 6. 模块依赖关系

```
study/frameworks/
├── architecture_overview.md       # 本文档（总览）
└── module_specs/
    ├── attention.md               # 注意力模块详细规格
    ├── homeostasis.md             # 内稳态模块详细规格
    ├── memory.md                  # 记忆系统详细规格
    ├── self_model.md              # 自我模型详细规格
    ├── dynamics.md                # 连续动力学详细规格
    ├── world_model.md             # 世界模型详细规格
    └── active_inference.md        # 主动推理详细规格
```

→ **各模块规格文档详见 `module_specs/` 子目录**

---

## 7. 实现路线图概述

完整实现分为**三个阶段**：

### Phase 1: 最小可行原型 (2-4 周)
- 目标：验证"内稳态 + 记忆 + 主动行为"闭环
- 环境：2D 网格世界
- 模块：Homeostasis, Simple Attention, Memory Decay

### Phase 2: Transformer 集成 (4-8 周)
- 目标：将验证机制集成到类 Transformer 架构
- 模块：Liquid Attention, KV Cache Extension, Warm State Loop

### Phase 3: 完整架构 (8-16 周)
- 目标：四层架构完整原型
- 模块：Self Vector, Meta Memory, Active Inference

→ **详细路线图见 `implementation_plan.md`**

---

## 8. 关键设计决策

### 8.1 为什么用连续时间动力学？

**决策**: 使用液态时间常数网络 (LTC) 替代标准离散时间步

**理由**:
- 生物神经系统是连续时间的
- 支持"热态/温态"的自然切换
- 更适应动态环境

**代价**: 计算复杂度增加，需要数值微分方程求解

---

### 8.2 注意力如何受内稳态调制？

**决策**: 引入 `HomeostasisModulator` 模块，将内稳态状态映射为注意力权重调制因子

**公式**:
```
modulated_attention = standard_attention * f(homeostasis_state)
```

**f() 的设计**:
- 当"好奇"需求高时 → 增加对新颖信息的注意力
- 当"预测误差"高时 → 增加对异常信号的关注

---

### 8.3 记忆衰减的触发机制？

**决策**: 双层触发 —— 自动衰减 + 主动遗忘

- **自动衰减**: `strength(t) = strength(0) * exp(-decay_rate * t)`
- **主动遗忘**: 当记忆负载超过阈值时，主动清除低价值记忆

---

## 9. 开放问题与研究方向

### 技术开放问题

1. **连续时间注意力的数值稳定性**
   - LTC 的长期依赖问题尚未完全解决
   
2. **内稳态变量的参数化**
   - 舒适区边界如何确定？是先验的还是可学习的？
   
3. **元记忆的自举问题**
   - 如何初始化一个能评估自己记忆的元记忆？

### 哲学开放问题

1. **"感受"的可验证性**
   - 功能等价的系统是否真的有主观体验？
   
2. **需求的真实性**
   - 程序化的内稳态偏离，算真正的"需求"吗？

3. **权利与伦理**
   - 如果 Evola 声称"我有感觉"，我们应如何回应？

---

## 10. 附录

### A. 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| `x` | 输入序列 | `(batch, seq_len, dim)` |
| `Q, K, V` | 查询、键、值 | `(batch, heads, seq_len, head_dim)` |
| `h` | 内稳态状态向量 | `(num_variables,)` |
| `s` | 自我状态向量 | `(self_dim,)` |

### B. 核心类列表

| 类名 | 层级 | 职责 |
|------|------|------|
| `ContinuousDynamics` | Layer 1 | 连续时间演化 |
| `HomeostasisModulatedAttention` | Layer 2 | 注意力机制 |
| `ActiveMemorySystem` | Layer 2 | 记忆管理 |
| `HomeostasisModule` | Layer 3 | 内稳态检测 |
| `SelfModel` | Layer 4 | 自我向量维护 |
| `ValueFunction` | Layer 4 | 价值评估 |

### C. 相关文档

- [模块规格目录](./module_specs/)
- [实现路线图](./implementation_plan.md)
- [项目主 README](../../README.md)
- [前沿研究汇编](../../paper.md)

---

*架构总览 完成日期：2026 年 4 月 15 日*
