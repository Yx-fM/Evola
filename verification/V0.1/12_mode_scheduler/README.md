# 12_mode_scheduler — 态调度器

---

## 步骤 1：边界定义

> 输入：内稳态状态 + 当前态  
> 输出：目标态 + 各模块计算频率系数  
> 不考虑：硬件级功耗管理、分布式调度、容错恢复

---

## 步骤 2：假设识别

| # | 假设 | 风险 |
|---|------|------|
| H1 | 内稳态变量能有效判断"是否需要高频率计算" | 低 |
| H2 | 阈值切换（>某值=HOT, <某值=WARM）不产生抖动 | **高** |
| H3 | 温态降频不影响核心闭环（感知-驱动-行动） | 中 |
| H4 | 冷态 = 完全暂停，无需复杂逻辑 | 低 |

---

## 步骤 3：快速否决 —— H2

- **风险**：阈值边界附近状态快速切换（如能量在 0.49↔0.51 振荡），模块被高频启停，开销比一直跑还大
- **验证**：标准解决方案——**迟滞（hysteresis）**：
  - 进入 HOT：能量 < 0.3
  - 退出 HOT：能量 > 0.5（不是 0.4）
  - 即上升阈值和下降阈值不同，消除抖动
- **结论**：**H2 被否决，但技术方案成熟**。引入迟滞区间。

---

## 步骤 4：技术选型

| 类别 | 选型 | 说明 |
|------|------|------|
| 拿来用 | **Python `enum.Enum`** | 态枚举 |
| 拿来用 | **Python `dataclass`** | FreqConfig 结构 |
| 改装 | **迟滞逻辑（EE 标准技术）** | 施密特触发器（Schmitt Trigger）的软件实现——上升阈值 ≠ 下降阈值 |
| 改装 | **计算神经科学"唤醒水平"模型** | 大脑从睡眠到警觉的连续谱。改装：离散三态 + 阈值 |
| 自研 | **降频系数矩阵** | 每态/每模块的频率系数设计 |

**态枚举与降频表：**
```python
class AgentMode(enum.Enum):
    HOT  = "hot"
    WARM = "warm"
    COLD = "cold"

@dataclass
class FreqConfig:
    perception:    float = 1.0  # 不可降频
    ltc_forward:   float = 1.0
    ltc_learning:  float = 1.0
    memory:        float = 0.5
    world_model:   float = 0.5
    self_model:    float = 1.0

FREQ_TABLE = {
    AgentMode.HOT:  FreqConfig(1.0, 1.0, 1.0, 0.5, 0.5, 1.0),
    AgentMode.WARM: FreqConfig(1.0, 0.3, 0.0, 1.0, 1.0, 0.3),
    AgentMode.COLD: FreqConfig(0,   0,   0,   0,   0,   0),
}
```

**切换逻辑（含迟滞）：**
```python
def determine_mode(homeo, current_mode):
    if homeo.safety_drive > SAFETY_HOT:
        return AgentMode.HOT          # 危险立即热态
    if homeo.energy < ENERGY_HOT_ENTER:  # 0.3
        return AgentMode.HOT
    if current_mode == AgentMode.HOT and homeo.energy > ENERGY_HOT_EXIT:  # 0.5
        return AgentMode.WARM
    if homeo.energy > ENERGY_COLD:
        return AgentMode.COLD
    return current_mode  # 迟滞区内维持当前态
```

---

## 步骤 5：定向查文献

1. **Cognitive Load & DMN**：人脑静息时默认模式网络活跃，任务时切换→对应 WARM→HOT。温态可维持低成本认知。
2. **A3C 异步更新**：多个 worker 异步运行，证明非同步设计可行。
3. **延迟警告**：温态下感知模块必须全频，否则无法感知突然出现的危险。

---

## 输出物
```
12_mode_scheduler/
├── impl/
│   ├── mode_state.py      # AgentMode + FreqConfig
│   └── hysteresis.py      # 迟滞切换逻辑
├── test/
│   └── test_scheduler.py  # 迟滞边界测试
└── notes/
```
