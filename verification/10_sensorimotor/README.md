# 10_sensorimotor — 感知-行动桥接层

---

## 步骤 1：边界定义

> 两个永久层：
> - **行为规范层**（不变）：定义智能体能做什么、驱动力如何映射到动作——这是"基因"的一部分
> - **感知层**（不变框架，可替换填充）：世界→智能体内部表征
>
> P0 用规则填充，P1 用 LTC 替换行为规范层的实现。框架本身永不变。  
> 不考虑：高层认知（规划、工具使用、多智能体协作）

---

## 步骤 2：假设识别

| # | 假设 | 风险 |
|---|------|------|
| H1 | 行为可以抽象为一组"行为原语"，驱动力选择原语而非直接选动作 | 低 |
| H2 | 静态优先级掩码（安全>生存>探索）作为 P0 仲裁足够 | **高** |
| H3 | 行为规范层的接口可以在 P0（规则）和 P1（LTC）之间无缝替换 | 高 |
| H4 | 显著性映射（哪些事实"重要"）可以纯规则驱动 | 中 |
| H5 | 不同智慧层级的智能体可以共享同一套行为规范框架，只增不减原语 | 低 |

---

## 步骤 3：快速否决 —— H2 + H3

### H2：静态优先级掩码

- **风险**：安全 > 生存 > 探索 的固定顺序过于机械化。真实生物不会因为"能量 0.29"就完全放弃探索
- **验证**：静态掩码的问题是"二值化"——mask 是 0 或 1，没有中间态。应升级为**动态权重衰减**：
  - 安全驱动力 > 0.5 时，探索驱动力 × 0.05（几乎禁止，但不是 0——极端情况下还要探索）
  - 能量 < 0.3 时，探索驱动力 × (energy / 0.3)——饥饿越重，探索越被抑制
  - 而非一刀切的 if-else
- **结论**：**H2 部分否决**。改为连续衰减函数，而非二值掩码。

### H3：P0→P1 的无缝替换

- **风险**：规则写的 `select_action()` 和一个 LTC 网络的 `forward()` 签名完全不同
- **验证**：定义**仲裁器抽象接口**，P0 和 P1 都实现它：
  ```python
  class IArbiter(ABC):
      """驱动力 + 事实 + 显著性 → 动作。P0 用规则实现，P1 用 LTC 实现"""
      @abstractmethod
      def select(self, facts: Facts, drive: DriveVector,
                 salience: SalienceMap, context: DecisionContext) -> Action: ...
  ```
- **结论**：**H3 成立，但需要抽象接口**。不是 P0 的代码"替换"为 P1，而是通过同一接口换实现。

---

## 步骤 4：技术选型

| 类别 | 选型 | 说明 |
|------|------|------|
| 拿来用 | **NumPy** | `np.argmin` 找最近食物/危险，`np.where` 筛选格子 |
| 拿来用 | **Python `set`** | 已访问集合 |
| 拿来用 | **Python `ABC`** | 行为规范接口定义 |
| 改装 | **Brooks Subsumption Architecture** | 分层抑制逻辑→升级为动态权重衰减 |
| 改装 | **Ethology Action Selection (McFarland)** | 因果因子排序→改为连续驱动力评分 |
| 自研 | **行为原语体系** | 定义不可约的行为原子，组成所有复杂行为 |
| 自研 | **显著性映射** | 哪些事实对当前驱动力是"重要的" |
| 自研 | **智慧层级动作空间** | 不同智慧层级可用的动作集合 |

---

## 两层的具体设计

### 第一层：行为规范层（Behavior Standard）—— 永久的"基因"

**行为原语 (Behavior Primitives)：**

```python
class BehaviorPrimitive(Enum):
    APPROACH = "approach"   # 向目标移动（食物、新区域）
    AVOID    = "avoid"      # 远离目标（危险）
    CONSUME  = "consume"    # 原地消耗（吃、收集）
    WAIT     = "wait"       # 静止不动
    EXPLORE  = "explore"    # 往没去过的地方走
    WANDER   = "wander"     # 无目标随机移动
```

**仲裁器接口：**

```python
class IArbiter(ABC):
    @abstractmethod
    def select_primitive(self, facts: Facts, drive: DriveVector,
                         salience: SalienceMap, context: DecisionContext) -> BehaviorPrimitive: ...

class IActuator(ABC):
    @abstractmethod
    def execute(self, primitive: BehaviorPrimitive, facts: Facts) -> Action: ...
```

**动作空间按智慧层级：**

| 层级 | 可用动作 | 原语映射 |
|------|----------|----------|
| 小鱼 (P0) | UP/DOWN/LEFT/RIGHT/EAT/WAIT | APPROACH→移动方向, CONSUME→EAT, AVOID→远离, WANDER→随机移动 |
| 小猫 (P2) | + PUSH/PULL/DROP | 增加对象操作原语 |
| 小狗 (P2) | + BARK/CALL/FOLLOW | 增加社会原语 |
| 猴子 (远期) | + USE_TOOL/BUILD | 增加工具原语 |

**关键设计：动作空间只增不减，低层级智能体只使用原语的一个子集。框架永远不变。**

### P0 仲裁器实现：Dynamic Weight Arbiter

```
驱动力向量 {energy, novelty, safety}
        ↓
   动态权重衰减（连续函数）
     safety ≥ 0.5 → novelty *= 0.05
     energy → novelty *= clamp(energy/COMFORT, 0, 1)
     safety ≥ 0.5 → energy 保持（逃命也要有力气）
        ↓
   有效驱动力 × 显著性
     显著性 = 每个原语在当前上下文的"吸引力"
        ↓
   评分最高的原语 → 执行器 → Action
```

### P1 替换方案：LTC Arbiter

```
驱动力 + 事实 + 上下文 → concat → LTC(hidden_state) → 原语 logits → softmax → 原语
                                                                          ↓
                                                                     执行器 → Action
```
接口不变，只换仲裁器实现。

---

### 第二层：感知层 (Perception)

```python
class Facts:
    entities_near: dict[str, str]   # {"food": "UP", "danger": "LEFT"}
    is_new_tile: bool
    terrain_near: dict[str, int]    # {"walls": 3, "open": 22}
    self_position: tuple[int, int]

class SalienceMap:
    """对当前驱动力而言，每个事实的'重要性'"""
    food_relevance:  float  # 0..1，饥饿时高
    danger_relevance: float  # 0..1，安全驱动力高时高
    novelty_relevance: float # 0..1，探索驱动力高时高
```

**显著性动态计算：**
```
food_relevance  = drive.energy    （越饿，食物越重要）
danger_relevance = drive.safety   （越危险，危险信息越重要）
novelty_relevance = drive.novelty （越想探索，新区域越重要）
```

---

## 步骤 5：定向查文献

1. **Subsumption Architecture (Brooks, 1986)**：分层行为抑制。我们取其"抑制"概念，但改为连续衰减。
2. **Action Selection in Ethology (McFarland & Sibly, 1975)**：驱动力竞争有限的运动输出。证实：加权竞争比二值掩码更接近生物。
3. **Affordance Theory (Gibson, 1979)**：环境为生物"提供"行为可能性。我们的"显著性"就是 affordance 的可计算版本——不同驱动力下，环境"看起来"不同。
4. **Motor Primitives in Neuroscience**：脊椎动物的运动皮层的"运动原语"（如 locomotion, grasping）是预编译的动作单元。我们的行为原语（APPROACH/AVOID/CONSUME...）是其 AI 等价物。
5. **动态系统理论 (Thelen & Smith, 1994)**：行为不是离散"触发"的，而是在吸引子场中连续演化的。支撑了"动态权重衰减"而非"if-else"的设计。

---

## 输出物（更新后）

```
10_sensorimotor/
├── impl/
│   ├── behavior_standard/         # ★ 行为规范层（永久框架）
│   │   ├── primitives.py          #   行为原语枚举
│   │   ├── action_space.py        #   动作空间 + 原语→动作映射（按智慧层级）
│   │   ├── arbitration.py         #   仲裁器抽象接口 + P0 RuleArbiter
│   │   └── actuator.py            #   执行器（原语→具体Action）
│   ├── perception/                # 感知层
│   │   ├── perception.py          #   5×5网格→Facts
│   │   └── salience.py            #   显著性映射
│   └── __init__.py
├── test/
│   ├── test_arbitration.py
│   ├── test_perception.py
│   └── test_salience.py
└── notes/
    └── arbitration_tuning.md
```
