# 00_prototype - 2D 网格世界原型（集成验证）

---

## 步骤 1：边界定义

> 集成 P0-P2 所有模块，在 2D 网格中验证自主行为  
> 输入：完整的模块链路  
> 输出：可观察的自主行为记录（通过 09_junjian）  
> 不考虑：3D、多智能体、人类交互

**最小成功标准**：
- P0 集成：智能体在网格中存活 500+ steps，行为由内稳态驱动
- P1 集成：LTC 网络替换硬编码决策，在线学习可见行为改善
- P2 集成：记忆影响决策、世界模型支持想象、态切换可观察

---

## 步骤 2：假设识别

| # | 假设 | 风险 |
|---|------|------|
| H1 | 各模块的接口在集成时不需要大规模修改 | 中 |
| H2 | 完整链路的计算延迟在 Python 单线程中可接受 | 低 |
| H3 | 多个内稳态变量的协同不会产生非预期的边缘行为 | 中 |
| H4 | 钧鉴记录的轨迹足以支持行为分析 | 低 |

---

## 步骤 3：快速否决 —— H1

- **风险**：独立开发时假设的接口，集成时发现不兼容（如 10 输出 Action 枚举但 08 需要 int；02 期望 Facts 但 10 输出 dict）
- **验证**：预防而非否决——在每个子模块完成时就做**接口契约测试**（contract test）：
  - 08 `step()` 的签名是否能被 10 调用？
  - 02 `get_drive()` 的返回值是否能被 10 消费？
  - 09 `log_action()` 的输入格式是否与其他模块一致？
- **结论**：**H1 不能被否决，但能通过提前定义接口契约来降低风险**。00_prototype 应首先写一个 `contracts.py` 定义所有模块的输入/输出类型。

---

## 步骤 4：技术选型

| 类别 | 选型 | 说明 |
|------|------|------|
| 拿来用 | **Python 的 `abc.ABC`** | 定义接口契约（抽象基类） |
| 拿来用 | **pytest** | 集成测试 |
| 拿来用 | **matplotlib** | 全程可视化（09_junjian 的重放功能） |
| 拿来用 | **PyTorch** | P1+ 的 LTC/MLP 权重加载 |
| 改装 | — | 仅做集成，不引入新算法 |
| 自研 | **接口契约** | 模块间输入/输出类型定义，防止集成断裂 |
| 自研 | **集成测试场景** | 设计特定场景验证每个模块的集成效果 |

**接口契约示例：**
```python
from abc import ABC, abstractmethod

class IWorld(ABC):
    @abstractmethod
    def step(self, action: Action, dt: float) -> tuple[Obs, float]: ...
    @abstractmethod
    def get_obs(self) -> Obs: ...

class IHomeostasis(ABC):
    @abstractmethod
    def update(self, facts: Facts) -> None: ...
    @abstractmethod
    def get_drive(self) -> DriveVector: ...

class IPerception(ABC):
    @abstractmethod
    def perceive(self, raw_obs: Obs) -> Facts: ...
```

**集成里程碑：**

| 阶段 | 包含模块 | 验证目标 |
|------|----------|----------|
| P0 集成 | 08 + 10 + 02 + 09 | 内稳态驱动的存活 > 500 steps |
| + 基线 | + 硬编码规则基线 | 对比：规则 vs 本能哪个更"像活的" |
| P1 集成 | + 01 + 04 | LTC 在线学习 → 行为改善 > 基线 |
| P1 消融 | - 04 | self_vector 消融实验 |
| P2 集成 | + 03 + 05 + 11 + 12 | 记忆影响决策、态切换可观察 |
| 完整 | + 00 全部 | 自主行为 > 2000 steps |

---

## 步骤 5：定向查文献

1. **ML 系统设计的契约测试**：微服务架构中的 Consumer-Driven Contracts——下游定义期望，上游实现。应用到 ML 模块：下游模块先声明"我需要的接口长这样"，上游模块实现它。
2. **RL 集成模式**：标准的 RL 训练循环是 `env.step() → agent.act() → env.step()` 的简单闭环。我们多了 homeo / memory / self_model 等侧支，但主循环不变。主循环设计应最简。
3. **Continuous Integration for ML**：每个模块提交代码时触发自己的单元测试；00_prototype 的集成测试只在"所有子模块都通过"后才运行。用 pytest 标记区分 `unit` 和 `integration`。

---

## 输出物

```
00_prototype/
├── impl/
│   ├── contracts.py        # 接口契约（ABC）
│   ├── main_loop.py         # 主循环（env→perception→homeo→action→env）
│   ├── p0_agent.py          # P0 阶段：硬编码规则智能体
│   ├── p1_agent.py          # P1 阶段：LTC 智能体
│   └── p2_agent.py          # P2 阶段：完整智能体
├── test/
│   ├── test_contracts.py    # 接口契约验证
│   ├── test_p0_loop.py      # P0 集成测试
│   └── test_ablation.py     # 消融实验
└── notes/
    └── integration_log.md   # 集成问题记录
```
