# Evola 实现路线图 (Implementation Plan)

**版本**: 0.1.0  
**状态**: 设计稿  
**创建日期**: 2026 年 4 月 15 日  
**预计总工时**: 12-24 周

---

## 1. 概述

### 1.1 实现愿景

将四层架构蓝图转化为可运行的原型系统，验证 Evola 的核心假设：

> **核心假设**: 具有内稳态、主动记忆和自我模型的系统，会表现出"生命感"和"主动性"。

### 1.2 实现原则

| 原则 | 说明 |
|------|------|
| **增量验证** | 每个阶段都有可验证的原型 |
| **最小可行** | 先实现核心功能，再扩展 |
| **测试驱动** | 每个模块都有单元测试 |
| **文档同步** | 代码与文档同步更新 |

---

## 2. 三阶段实现计划

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: 最小可行原型                                       │
│  目标：验证"内稳态 + 记忆 + 主动行为"闭环                     │
│  工时：2-4 周                                                │
│  环境：2D 网格世界                                            │
├─────────────────────────────────────────────────────────────┤
│  Phase 2: Transformer 集成                                   │
│  目标：将验证机制集成到类 Transformer 架构                      │
│  工时：4-8 周                                                │
│  环境：序列建模任务                                          │
├─────────────────────────────────────────────────────────────┤
│  Phase 3: 完整架构                                           │
│  目标：四层架构完整原型                                      │
│  工时：8-16 周                                               │
│  环境：多模态交互                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 1: 最小可行原型 (2-4 周)

### 3.1 目标

在简化的 2D 网格世界环境中，验证以下核心机制：

- ✅ 内稳态变量偏离 → 生成需求 → 驱动行动 → 回归平衡
- ✅ 记忆槽可衰减、可强化、可遗忘
- ✅ 智能体在无外部输入时仍有"后台活动"

### 3.2 环境设定

```python
@dataclass
class GridWorldEnvironment:
    """
    2D 网格世界环境
    
    简化的测试环境:
    - 10x10 网格
    - 智能体可以上下左右移动
    - 有食物（恢复能量）、信息源（降低熵）、危险（增加预测误差）
    """
    grid_size: int = 10
    agent_position: Tuple[int, int] = (5, 5)
    energy: float = 0.8
    # ... 其他状态
```

### 3.3 实现模块

| 模块 | 文件路径 | 复杂度 | 工时 |
|------|---------|--------|------|
| **HomeostasisModule** | `core/homeostasis/module.py` | 中 | 1 周 |
| **SimpleMemory** | `core/memory/simple.py` | 低 | 3 天 |
| **DriveGenerator** | `core/homeostasis/drive.py` | 中 | 3 天 |
| **GridWorldAgent** | `experiments/gridworld_agent.py` | 中 | 1 周 |
| **测试** | `tests/test_homeostasis.py` | 低 | 2 天 |

### 3.4 关键验证实验

#### 实验 1: 内稳态平衡

```python
def test_homeostasis_balance():
    """验证内稳态调节闭环"""
    agent = GridWorldAgent()
    
    # 初始状态：能量过低
    agent.homeostasis.energy_level = 0.2
    
    # 运行 100 步
    for _ in range(100):
        action = agent.step()
        agent.execute(action)
    
    # 验证：能量应回归舒适区
    assert agent.homeostasis.energy_level > 0.5
    print("✅ 内稳态平衡验证通过")
```

#### 实验 2: 记忆衰减与强化

```python
def test_memory_decay_and_reinforce():
    """验证记忆的主动管理"""
    memory = SimpleMemory(capacity=10)
    
    # 存储记忆
    idx = memory.store(content=torch.randn(64), importance=0.8)
    
    # 时间流逝
    for _ in range(50):
        memory.decay_all(dt=0.1)
    
    # 验证：强度衰减
    assert memory.get_strength(idx) < 1.0
    
    # 强化
    memory.reinforce(idx, amount=0.3)
    
    # 验证：强度回升
    assert memory.get_strength(idx) > 0.5
    print("✅ 记忆管理验证通过")
```

#### 实验 3: 温态活动

```python
def test_warm_mode_activity():
    """验证无输入时的后台活动"""
    agent = GridWorldAgent()
    
    # 无外部输入
    agent.inputs = None
    
    # 温态运行
    activities = []
    for _ in range(100):
        activity = agent.warm_mode_step()
        activities.append(activity)
    
    # 验证：有后台活动
    assert len(activities) > 0
    assert any(a.type != 'idle' for a in activities)
    print("✅ 温态活动验证通过")
```

### 3.5 交付物

```
Phase 1 交付物:
├── core/homeostasis/module.py       # 内稳态模块
├── core/memory/simple.py            # 简化记忆系统
├── experiments/gridworld_agent.py   # 网格世界智能体
├── tests/test_homeostasis.py        # 单元测试
└── docs/phase1_report.md            # 验证报告
```

### 3.6 成功标准

- [ ] 内稳态平衡实验通过
- [ ] 记忆管理实验通过
- [ ] 温态活动实验通过
- [ ] 所有单元测试通过
- [ ] 验证报告完成

---

## 4. Phase 2: Transformer 集成 (4-8 周)

### 4.1 目标

将 Phase 1 验证有效的机制集成到类 Transformer 架构，实现：

- ✅ 液态注意力机制
- ✅ 内稳态调制的 Attention
- ✅ KV Cache 扩展（可主动管理）
- ✅ 热态/温态切换

### 4.2 实现模块

| 模块 | 文件路径 | 复杂度 | 工时 |
|------|---------|--------|------|
| **LiquidAttention** | `core/attention/liquid.py` | 高 | 2 周 |
| **HomeostasisModulatedAttention** | `core/attention/modulated.py` | 高 | 2 周 |
| **ActiveMemoryKV** | `core/memory/kv_cache.py` | 中 | 1 周 |
| **WarmStateLoop** | `core/dynamics/warm_state.py` | 中 | 1 周 |
| **TransformerBlock** | `core/architecture/block.py` | 高 | 2 周 |
| **测试** | `tests/test_attention.py` | 中 | 1 周 |

### 4.3 关键实现

#### 液态注意力

```python
# core/attention/liquid.py
class LiquidAttention(nn.Module):
    """液态时间常数注意力"""
    
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        # LTC 单元集成到注意力
        self.ltc_cell = LiquidTimeConstantCell(...)
    
    def forward(self, x: Tensor, dt: float = 0.1) -> Tensor:
        # 连续时间演化
        pass
```

#### 内稳态调制

```python
# core/attention/modulated.py
class HomeostasisModulatedAttention(nn.Module):
    """内稳态调制的注意力"""
    
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attention = LiquidAttention(dim, num_heads)
        self.modulator = HomeostasisModulator(...)
    
    def forward(
        self,
        x: Tensor,
        homeostasis_state: HomeostasisVariables,
    ) -> Tensor:
        modulation = self.modulator(homeostasis_state)
        attn_output = self.attention(x)
        return attn_output * modulation
```

### 4.4 验证实验

#### 实验 1: 内稳态调制效果

```python
def test_homeostasis_modulation():
    """验证内稳态对注意力的调制"""
    attention = HomeostasisModulatedAttention(dim=128, num_heads=4)
    x = torch.randn(2, 10, 128)
    
    # 高好奇状态
    high_curiosity = HomeostasisVariables(information_entropy=0.9)
    # 低好奇状态
    low_curiosity = HomeostasisVariables(information_entropy=0.1)
    
    out_high, weights_high = attention(x, high_curiosity, return_weights=True)
    out_low, weights_low = attention(x, low_curiosity, return_weights=True)
    
    # 验证：注意力分布不同
    assert not torch.allclose(weights_high, weights_low)
    print("✅ 内稳态调制验证通过")
```

#### 实验 2: 热态/温态切换

```python
def test_hot_warm_switch():
    """验证热态/温态切换"""
    agent = EvolaAgent()
    
    # 热态：有输入
    input_tensor = torch.randn(1, 10, 128)
    hot_output = agent.hot_mode_step(input_tensor)
    
    # 温态：无输入
    warm_output = agent.warm_mode_step()
    
    # 验证：两种模式都有输出且不同
    assert hot_output is not None
    assert warm_output is not None
    assert not torch.allclose(hot_output, warm_output)
    print("✅ 热态/温态切换验证通过")
```

### 4.5 交付物

```
Phase 2 交付物:
├── core/attention/liquid.py         # 液态注意力
├── core/attention/modulated.py      # 内稳态调制注意力
├── core/memory/kv_cache.py          # 主动 KV 缓存
├── core/dynamics/warm_state.py      # 温态循环
├── core/architecture/block.py       # Transformer 块
├── tests/test_attention.py          # 注意力测试
└── docs/phase2_report.md            # 验证报告
```

### 4.6 成功标准

- [ ] 内稳态调制注意力验证通过
- [ ] 热态/温态切换验证通过
- [ ] 所有单元测试通过
- [ ] 性能基准达标（推理延迟 < 10ms/token）
- [ ] 验证报告完成

---

## 5. Phase 3: 完整架构 (8-16 周)

### 5.1 目标

实现四层架构的完整原型，包括：

- ✅ 自我状态向量
- ✅ 元记忆系统
- ✅ 世界模型
- ✅ 主动推理
- ✅ 完整的数据流整合

### 5.2 实现模块

| 模块 | 文件路径 | 复杂度 | 工时 |
|------|---------|--------|------|
| **SelfVector** | `core/self_model/vector.py` | 中 | 1 周 |
| **SelfModel** | `core/self_model/model.py` | 高 | 2 周 |
| **MetaMemory** | `core/memory/meta.py` | 高 | 2 周 |
| **WorldModel** | `core/world_model/predictor.py` | 高 | 2 周 |
| **ActiveInference** | `core/active_inference/module.py` | 高 | 2 周 |
| **ValueFunction** | `core/value/function.py` | 中 | 1 周 |
| **Integration** | `core/architecture/agent.py` | 高 | 2 周 |
| **测试** | `tests/test_full_architecture.py` | 高 | 1 周 |

### 5.3 关键实现

#### 完整 Evola Agent

```python
# core/architecture/agent.py
class EvolaAgent(nn.Module):
    """
    完整的 Evola 智能体
    
    整合四层架构
    """
    
    def __init__(self, config: EvolaConfig):
        super().__init__()
        
        # Layer 1: 基础层
        self.dynamics = ContinuousDynamics(config)
        
        # Layer 2: 认知层
        self.attention = HomeostasisModulatedAttention(config)
        self.memory = ActiveMemorySystem(config)
        self.world_model = WorldModel(config)
        
        # Layer 3: 动力层
        self.homeostasis = HomeostasisModule(config)
        self.instinct = InstinctModule(config)
        self.active_inference = ActiveInferenceModule(config)
        
        # Layer 4: 存在层
        self.self_model = SelfModel(config)
        self.value_function = ValueFunction(config)
    
    def forward(
        self,
        perception: Perception,
    ) -> Action:
        """前向传播"""
        # Layer 1 → 2 → 3 → 4 → Action
        pass
    
    def warm_mode_step(self) -> WarmModeActivity:
        """温态运行"""
        pass
```

### 5.4 验证实验

#### 实验 1: 自指决策

```python
def test_self_referential_decision():
    """验证自指决策能力"""
    agent = EvolaAgent(config)
    
    # 设置情境
    perception = Perception(...)
    
    # 决策
    action = agent(perception)
    
    # 验证：自我向量影响决策
    assert agent.self_model.self_vector is not None
    print("✅ 自指决策验证通过")
```

#### 实验 2: 元记忆反思

```python
def test_meta_memory_reflection():
    """验证元记忆反思能力"""
    agent = EvolaAgent(config)
    
    # 填充一些记忆
    for _ in range(100):
        agent.memory.store(...)
    
    # 元记忆反思
    insight = agent.memory.meta_memory.reflect_on_learning()
    
    # 验证：有洞察产生
    assert len(insight.insights) > 0
    print("✅ 元记忆反思验证通过")
```

### 5.5 交付物

```
Phase 3 交付物:
├── core/self_model/vector.py        # 自我向量
├── core/self_model/model.py         # 自我模型
├── core/memory/meta.py              # 元记忆
├── core/world_model/predictor.py    # 世界模型
├── core/active_inference/module.py  # 主动推理
├── core/value/function.py           # 价值函数
├── core/architecture/agent.py       # 完整 Agent
├── tests/test_full_architecture.py  # 整合测试
└── docs/phase3_report.md            # 最终报告
```

### 5.6 成功标准

- [ ] 自指决策验证通过
- [ ] 元记忆反思验证通过
- [ ] 完整的内稳态循环验证通过
- [ ] 所有单元测试通过
- [ ] 端到端运行无错误
- [ ] 最终报告完成

---

## 6. 项目结构

### 6.1 建议的目录结构

```
Evola/
├── core/
│   ├── __init__.py
│   ├── architecture/
│   │   ├── __init__.py
│   │   ├── agent.py              # 完整 Agent
│   │   └── block.py              # Transformer 块
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── liquid.py             # 液态注意力
│   │   └── modulated.py          # 内稳态调制
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── simple.py             # Phase 1
│   │   ├── kv_cache.py           # Phase 2
│   │   └── meta.py               # Phase 3
│   ├── homeostasis/
│   │   ├── __init__.py
│   │   ├── module.py             # 内稳态模块
│   │   └── drive.py              # 需求生成
│   ├── self_model/
│   │   ├── __init__.py
│   │   ├── vector.py             # 自我向量
│   │   └── model.py              # 自我模型
│   ├── world_model/
│   │   ├── __init__.py
│   │   └── predictor.py          # 世界模型预测
│   ├── active_inference/
│   │   ├── __init__.py
│   │   └── module.py             # 主动推理
│   └── value/
│       ├── __init__.py
│       └── function.py           # 价值函数
├── experiments/
│   ├── __init__.py
│   ├── gridworld_agent.py        # Phase 1
│   └── llm_integration.py        # Phase 2
├── tests/
│   ├── __init__.py
│   ├── test_homeostasis.py
│   ├── test_attention.py
│   └── test_full_architecture.py
├── study/
│   ├── frameworks/               # 框架文档
│   │   ├── architecture_overview.md
│   │   ├── implementation_plan.md
│   │   └── module_specs/
│   ├── lessons/                  # 学习课程
│   └── lab/                      # 实验代码
├── configs/
│   └── evola_config.yaml         # 配置文件
├── docs/
│   ├── phase1_report.md
│   ├── phase2_report.md
│   └── phase3_report.md
├── pyproject.toml
└── README.md
```

---

## 7. 风险与缓解

### 7.1 技术风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| **LTC 数值不稳定** | 中 | 高 | 先用标准 Attention，逐步引入 LTC |
| **内稳态参数难调** | 高 | 中 | Phase 1 充分测试，自动化调参 |
| **计算资源不足** | 中 | 中 | 先用小模型验证，再扩展 |
| **温态机制无效** | 中 | 高 | 设计明确的验证实验 |

### 7.2 进度风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| **Phase 1 超时** | 低 | 中 | 每周检查点，及时调整 |
| **Phase 2 复杂度被低估** | 中 | 高 | 预留 50% 缓冲时间 |
| **Phase 3 整合困难** | 高 | 高 | 每阶段结束都做整合测试 |

---

## 8. 里程碑

| 里程碑 | 预计日期 | 交付物 |
|--------|----------|--------|
| **Phase 1 完成** | 2026-05-13 | 网格世界原型 + 验证报告 |
| **Phase 2 完成** | 2026-07-08 | Transformer 集成 + 验证报告 |
| **Phase 3 完成** | 2026-10-14 | 完整架构原型 + 最终报告 |
| **论文/文章撰写** | 2026-11-01 | 技术报告/论文初稿 |

---

## 9. 下一步行动

### 立即行动（本周）

- [ ] 创建 `core/homeostasis/` 目录
- [ ] 实现 `HomeostasisVariables` 和 `ComfortZone`
- [ ] 编写单元测试框架

### Phase 1 第 1 周

- [ ] 完成 `HomeostasisModule` 核心逻辑
- [ ] 创建 `GridWorldEnvironment`
- [ ] 运行第一个内稳态平衡实验

---

## 10. 附录

### A. 配置文件示例

```yaml
# configs/evola_config.yaml
model:
  dim: 512
  num_heads: 8
  num_layers: 6
  
homeostasis:
  variables:
    information_entropy:
      comfort_zone: [0.3, 0.7]
    prediction_error:
      comfort_zone: [0.0, 0.3]
    memory_load:
      comfort_zone: [0.4, 0.8]
  
memory:
  capacity: 1000
  default_decay_rate: 0.01
  
training:
  lr: 1e-4
  batch_size: 32
  max_steps: 10000
```

### B. 核心类索引

| 类名 | 模块 | 阶段 |
|------|------|------|
| `HomeostasisVariables` | `core/homeostasis/` | Phase 1 |
| `ActiveMemorySystem` | `core/memory/` | Phase 1/2 |
| `HomeostasisModulatedAttention` | `core/attention/` | Phase 2 |
| `SelfModel` | `core/self_model/` | Phase 3 |
| `EvolaAgent` | `core/architecture/` | Phase 3 |

---

*实现路线图 完成日期：2026 年 4 月 15 日*
