# 坤舆 (Kunyu) - AI居住的世界

> *名称取自中国古代宇宙观，大地被视为承载万物的宏大马车。"坤"（承载万物的地）+ "舆"（马车/承载之物）*

---

## 步骤 1：边界定义

> 输入：`step(action, dt=1.0)` 带动作指令  
> 输出：5×5 视图矩阵 + 奖励信号  
> 不考虑：3D 渲染、物理引擎（重力/碰撞体）、多智能体通信、网络同步

**最小成功标准**：键盘操控一个"代理"在 2D 网格中移动、吃食物、躲危险、撞墙被拦截。

---

## 步骤 2：假设识别

| # | 假设 | 风险 |
|---|------|------|
| H1 | 2D 网格 + 食物/墙壁/危险三种实体足以产生非平凡的行为选择 | 低 |
| H2 | 5×5 局部视野（非全知）满足低等智能的感知需求 | 低 |
| H3 | 离散 `step` 循环与后续 01_liquid_dynamics 的连续 ODE 不冲突 | **高** |
| H4 | Python + NumPy 纯数组运算，渲染用 matplotlib，性能足够 | 低 |
| H5 | `gymnasium` 接口规范（非依赖）适合对接后续 RL/LLM 生态 | 低 |

---

## 步骤 3：快速否决 —— H3

- **风险**：LTC 基于连续 ODE (`dx/dt`)，环境是一次性 `step()`，两者时间粒度不一致
- **验证**：这是 RL 领域的标准模式——Gym/MuJoCo 都是离散 step，LTC 研究者（Liquid AI、MIT）也是在这个框架下工作的。解决方案是在 LTC 的 ODE solver 内部用 `dt` 多次子步积分
- **结论**：**H3 成立**。环境侧只需 `step()` 接受可变 `dt` 参数并返回 `(obs, reward, done, info)`；LTC 侧在两次 `step` 之间自行做 ODE 积分

---

## 步骤 4：技术选型

| 类别 | 选型 | 说明 |
|------|------|------|
| 拿来用 | **NumPy** | 2D 数组运算，网格状态存储 |
| 拿来用 | **matplotlib** | `imshow()` 渲染网格，`FuncAnimation` 做动画回放 |
| 拿来用 | **`gymnasium` 接口规范** | 不安装 gymnasium 依赖，只模仿其 `step/obs/reward/done` 签名，确保未来兼容 |
| 拿来用 | **Python `enum`** | 实体类型（EMPTY/WALL/FOOD/DANGER/AGENT）和动作类型（UP/DOWN/LEFT/RIGHT/EAT/WAIT） |
| 改装 | — | — |
| 自研 | **实体系统 + 规则引擎** | 碰撞检测、食物再生、视野范围（Ray 扇形→简化为 5×5 窗口）、实体生命周期 |

**为什么不用游戏引擎（Pygame/Godot）？**
- 网格世界不需要碰撞体、物理引擎、实时渲染
- 纯数组运算推理速度快，适合批量实验
- 无额外依赖，降低环境搭建成本

**为什么不用 gymnasium 直接安装？**
- 初期自己写更灵活，需要非标准接口（如可变 dt）
- 后期若需对接 RL 社区，加一个 `gym.Wrapper` 即可

---

## 步骤 5：定向查文献

1. **GridWorld 经典实现（Sutton & Barto, 2018 §3）**：网格世界是 RL 教科书第一课的标准环境。核心规则（状态-动作-奖励）已在数十年实践中验证，无坑。
2. **MiniGrid (Chevalier-Boisvert et al., 2018)**：最接近我们需求的成熟开源项目。可以借鉴其任务定义 DSL 和多 room 地图生成，但不直接依赖——它的 gymn asium 绑定太重。
3. **实体-组件系统 (ECS) 设计模式**：游戏开发中用于管理大量实体的标准模式。我们的世界实体少（<100），不需要 ECS 的查询优化，用简单 list 或 dict 即可。
4. **离散-连续混合系统**：ODE solver + event handler 是数学标准做法。LTC 论文的原作者就是用 `torchdiffeq` 的 `odeint` 在标准 Gym 环境中跑的。环境侧只要提供 `(s, a) → (s', r)` transition，连续时间由模型侧处理。

---

## 输出物

```
08_kunyu/
├── impl/
│   ├── world.py             # 网格世界引擎（实体管理、规则）
│   ├── entities.py          # Entity 类型定义
│   ├── event_bus.py          # 事件总线（08↔09 桥梁）
│   ├── render_mpl.py         # matplotlib 渲染（回放分析+论文用图）
│   ├── render_pygame.py      # Pygame 实时交互窗口 ★
│   ├── render_terminal.py    # ANSI 字符渲染（headless/smoke test）
│   └── demo.py               # 键盘操控验证
├── test/
│   └── test_world.py         # 世界规则单元测试
└── notes/
    └── parameter_tuning.md   # 食物再生率、能量衰减等参数实验记录
```

---

## 步骤 6：UI 设计

### 渲染器架构

三种渲染器并存，按场景选择：

| 渲染器 | 技术 | 帧率 | 适用场景 |
|--------|------|------|----------|
| `render_terminal.py` | ANSI 字符串 + `print()` | ~100 step/s | CI / 单元测试 / 批量实验 |
| `render_pygame.py` | Pygame 窗口 | 60 fps | 人工观察 / 实验演示 / 日常开发 |
| `render_mpl.py` | matplotlib | 静态 | 论文用图 / 回放分析 / 报告 |

**统一接口：**
```python
class IRenderer(ABC):
    @abstractmethod
    def render(self, world: KunyuWorld) -> None: ...
    @abstractmethod
    def close(self) -> None: ...
```

### 地图大小规范

| 场景 | 尺寸 | 用途 |
|------|------|------|
| 微缩 (Tiny) | 8×8 或 10×10 | 单元测试 / 快速原型 |
| 标准 (Standard) | 20×15 或 20×20 | 日常实验（默认） |
| 大场景 (Large) | 50×50 | 长期存活 / 复杂行为 |

当前默认值：`height=15, width=20`

### Pygame 窗口布局（实时模式）

```
┌───────────────────────────────────────────────┐
│  坤舆 (Kunyu) - Step 142                      │
│                                               │
│  ┌─────────────────────┐  ┌────────────────┐  │
│  │                     │  │  Agent View    │  │
│  │   全地图             │  │  ┌───────┐     │  │
│  │   (20×15 grid)      │  │  │  5×5    │     │  │
│  │                     │  │  │ obs    │     │  │
│  │    A        F       │  │  └───────┘     │  │
│  │       F    X        │  │               │  │
│  │   F           X     │  │  Stats:       │  │
│  │                     │  │  Energy: 0.7  │  │
│  └─────────────────────┘  │  Visited: 23  │  │
│                           └────────────────┘  │
│  [Space] Pause | [→] Step | [R] Reset       │
└───────────────────────────────────────────────┘
```

### 与 09_junjian（钧鉴）的集成

**问题：08（世界）和 09（观察）需要共享数据但不应硬耦合。**

**方案：事件总线（EventBus）**

```python
class EventBus:
    """08 发布事件，09（及其他订阅者）接收事件。松耦合。"""
    def publish(self, event_type: str, data: dict): ...
    def subscribe(self, event_type: str, callback: callable): ...
```

**08_kunyu 在每个 step 发布以下事件：**

```python
# world.step() 内：
bus.publish("step.begin",   {"step": n, "agent_pos": (r, c)})
# ... 执行动作 ...
bus.publish("step.action",  {"action": action, "success": ...})
bus.publish("step.outcome", {"energy_gained": 0.3, "damage_taken": 0.0, "collision": False})
bus.publish("step.end",     {"step": n})
```

**10_sensorimotor 发布：**

```python
bus.publish("agent.decision", {"drive": {...}, "chosen_primitive": "APPROACH", "salience": {...}})
```

**09_junjian 订阅并持久化：**

```python
class JunJian:
    def __init__(self, bus, privacy=PUBLIC):
        bus.subscribe("step.action",   self._log_action)
        bus.subscribe("step.outcome",  self._log_outcome)
        bus.subscribe("agent.decision", self._log_decision)
```

**数据流全景：**

```
08_kunyu ───── publish ────┐
10_sensorimotor ── publish ─┤
02_homeostasis ── publish ──┤──→ EventBus ──→ 09_junjian.subscribe()
                              │                         ↓
                              │                    JSON Lines 日志
                              │                         ↓
                              │                    replay.py 回放
```

这样 09 完全不需要 `import` 08——它只依赖 EventBus。
