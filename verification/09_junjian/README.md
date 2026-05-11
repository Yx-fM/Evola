# 钧鉴 (JunJian) - 坤舆的孪生观察系统

> *"钧鉴" 意为"明镜可鉴"——坤舆的镜面孪生。负责一切渲染与行为记录。*
>
> **架构原则**：坤舆(08)只管世界运作，钧鉴(09)负责"这个世界长什么样"和"发生了什么"。

---

## 步骤 1：边界定义

> 孪生系统的职责：
> - **渲染层**：Terminal / Pygame / matplotlib 三种渲染器
> - **日志层**：订阅 EventBus，写 JSON Lines 行为轨迹
> - **回放层**：逐帧重现实验过程
>
> 输入：EventBus 事件流  
> 输出：可视画面 + JSON Lines + 帧图序列  
> 不考虑：实时 Web 面板、多用户权限、数据库

---

## 步骤 2：假设识别

| # | 假设 | 风险 |
|---|------|------|
| H1 | 只看行为不看内部向量，仍能有效分析智能体行为模式 | 低 |
| H2 | 状态标签（"饿"/"好奇"）足以替代具体数值做行为分析 | **高** |
| H3 | JSON Lines + CSV 满足初期数据需求，不需要专用数据库 | 低 |
| H4 | matplotlib 静态帧图足以做回放，不需要实时动画 | 低 |

---

## 步骤 3：快速否决 —— H2

- **风险**：只用标签（"饿"）丢失精细度——"能量 0.49"和"能量 0.01"行为差异巨大但标签相同
- **验证**：标签与数值并非互斥——对内保留精确值（开发者调试），对外只暴露标签（隐私原则）。用配置开关控制暴露级别
- **结论**：H2 调整为**双层模式**——
  - `privacy_level = "public"` → 只输出标签，不暴露原始向量
  - `privacy_level = "debug"` → 输出完整数值（开发者使用）

---

## 步骤 4：技术选型

| 类别 | 选型 | 说明 |
|------|------|------|
| 拿来用 | **Python `json` 标准库** | JSON Lines（每行一个 JSON），可直接 `cat` 或导入 pandas |
| 拿来用 | **Python `csv` 标准库** | 可选导出 CSV 给 Excel 分析 |
| 拿来用 | **`pathlib.Path`** | 日志文件管理，跨平台 |
| 拿来用 | **matplotlib** | `imshow` 渲染帧图，`savefig` 保存为 PNG 序列 |
| 改装 | — | — |
| 自研 | **日志格式规范** | 事件结构设计（见下方格式） |
| 自研 | **标签映射规则** | 数值→离散标签的阈值定义 |
| 自研 | **因果链追踪** | 行为→位置变化→内稳态变化的完整链路记录 |

**日志格式：**

```jsonl
{"step": 0,  "type": "action",  "action": "MOVE_UP",  "pos": [5,3], "cause": "hunger"}
{"step": 0,  "type": "homeo",   "labels": {"energy": "low", "novelty": "neutral", "safety": "safe"}, "_values": {...}}
{"step": 0,  "type": "world",   "food_near": true, "danger_near": false}
{"step": 1,  "type": "action",  "action": "EAT",     "pos": [5,2], "cause": "hunger"}
```

**隐私开关：**
```python
class PrivacyLevel(enum.Enum):
    PUBLIC = 0     # 只输出 labels（默认）
    DEBUG  = 1     # 输出 labels + _values

class JunJian:
    def __init__(self, privacy: PrivacyLevel = PrivacyLevel.PUBLIC):
        ...
```

**标签映射规则：**
```python
def energy_label(energy: float) -> str:
    if energy < 0.2:  return "starving"
    if energy < 0.5:  return "low"
    if energy < 0.8:  return "ok"
    return "full"
```

---

## 步骤 5：定向查文献

1. **OpenAI Gym Monitor (legacy)**：Gym 自带的 `Monitor` wrapper 录视频+日志，设计简洁。参考其"每个 episode 一个目录"的文件组织方式，但不直接依赖（它录视频太重，我们只需文本+帧图）。
2. **MLflow / Weights & Biases**：成熟的 ML 实验跟踪工具。远超当前需求（需要服务端），但日志结构（step/tag/value）值得参考。我们保持简单，手动维护实验记录即可。
3. **Event Sourcing 模式**：软件架构中的"事件溯源"——存储所有状态变化而非仅当前状态。我们的 `log_action` 就是事件溯源，每次行为都是"不可变的日志事件"。天然适合因果追溯。

---

## 输出物

```
09_junjian/
├── impl/
│   ├── logger.py              # JunJian 核心类（订阅 EventBus + 日志写入）
│   ├── render_terminal.py     # ANSI 终端渲染（CI / headless / smoke test）
│   ├── render_pygame.py       # Pygame 60fps 实时窗口（日常开发）
│   ├── render_mpl.py          # matplotlib 渲染（论文用图 / 回放分析）
│   ├── labels.py              # 数值→标签映射
│   ├── replay.py              # 帧图回放生成
│   └── demo.py                # 集成演示（坤舆 + 钧鉴联动）
├── logs/                      # 实验日志输出目录
├── test/
│   └── test_junjian.py
└── notes/
    └── label_threshold_tuning.md
```

### 08 ↔ 09 数据流

```
08_kunyu       publish ──→  EventBus  ──→  09_junjian.subscribe()
  (世界引擎)                     │              ├─ logger.py → JSON Lines
                              │              ├─ render_terminal → 终端
                              │              ├─ render_pygame → 窗口
                              │              └─ render_mpl → 图片
                              │
10_sensorimotor publish ──────┘
02_homeostasis  publish ──────┘
```
