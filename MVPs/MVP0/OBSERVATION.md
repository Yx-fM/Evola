# MVP 0 观察手册

> 如何启动、操作、观察和分析 Evola 第一个完整原型的运行结果。

---

## 一、文件结构

```
MVPs/MVP0/
├── evola_pygame.py         ★ Pygame 图形完整系统（推荐）
├── evola_cli.py             ★ CLI 仪表盘
├── mvp_0_demo.py            一次性批量实验
├── mvp_0_config.py          参数配置
├── mvp_0_stats.py           统计模块
├── theme.py                 色彩体系
├── README.md
├── OBSERVATION.md           本文件
├── minds/                   心智快照 (.mind.evola)
├── stats/                   统计 CSV + PNG
└── logs/                    JunJian JSONL
```

依赖模块：

```
verification/
├── 08_kunyu/impl/      (world, entities, event_bus)
├── 02_homeostasis/impl/ (drives, variables, comfort_zones)
├── 09_junjian/impl/    (logger)
├── 10_sensorimotor/impl/ (agent, behavior_standard/, perception/)
```

---

## 二、Pygame 图形窗口（推荐）

```bash
cd Q:\All_Items\DreamProjects\Evola
python MVPs\MVP0\evola_pygame.py
```

### 启动后自动创建世界和智能体，跳出窗口：

```
┌─ Evola MVP 0 ────────────────────────────────────────────┐
│ [Start] [Spawn] [Run 100] [Pause] [Extract] [Debug] [Quit]│
├──────────────────┬───────────────────────────────────────┤
│                   │  evo_001 @(5,3)                      │
│   世界地图         │  正在觅食...                         │
│   (彩色网格)       │  能量 ████████░░  0.72  舒适         │
│   金色圆=智能体     │  好奇 ██████░░░░  0.45  好奇         │
│   绿色格=食物       │  安全 ░░░░░░░░░░  0.00  安全         │
│   红色格=危险       │  足迹: 31/300 (10.3%)              │
│   灰点=已访问       │  - - JunJian Mirror - -            │
│                   │  Events:                            │
│                   │  142 ate food                       │
│                   │  127 saw food                       │
├──────────────────┴───────────────────────────────────────┤
│ Step: 142 | Agents: 1 | FPS: 30 | [Space]Pause [Q]Quit  │
└─────────────────────────────────────────────────────────┘
```

### 操作

| 操作 | 方式 |
|------|------|
| 启动平台 | 点击 `[Start]` |
| 放入智能体 | 点击 `[Spawn]` |
| 自动跑 100 步 | 点击 `[Run 100]` |
| 暂停/继续 | 点击 `[Pause]` 或空格 |
| 取出智能体 | 点击 `[Extract]` |
| 切换 debug | 点击 `[Debug]` 或按 D |
| 关闭平台 | 点击 `[Stop]` |
| 退出程序 | 点击 `[Quit]` 或按 Q |

---

## 三、CLI 观察站

```bash
cd Q:\All_Items\DreamProjects\Evola
python MVPs\MVP0\evola_cli.py
```

### 界面

```
┌─ 钧鉴·观察站 ─── 坤舆: 15×20 ─── Step 142 ────┐
│ ┌─ 坤舆 15x20 ─┐ ┌─ 钧鉴·日志 ─┐ ┌─ 未晞 ────┐│
│ │ ···F·X···A·  │ │ 142 觅食...  │ │ evo_001   ││
│ │ ···#··F····  │ │ 127 发现食物 │ │ 状态: 觅食 ││
│ │ ··A···F····  │ │ 124 探索新区 │ │ 能量 ████░ ││
│ └──────────────┘ └─────────────┘ │ 好奇 ██░░░ ││
│                                  │ 安全 ░░░░░ ││
│                                  │ ─ ─ 明镜 ─ ││
│                                  └────────────┘│
│ [s]tart [sp]awn [r]un [o]bserve [d]ebug [q]uit│
├───────────────────────────────────────────────┤
│ evola> _                                       │
└───────────────────────────────────────────────┘
```

三栏从左到右：**坤舆**（世界 ASCII 地图）、**钧鉴**（事件日志）、**未晞**（智能体面板）。

### 命令

| 命令 | 别名 | 说明 |
|------|------|------|
| `start` | `s` | 启动坤舆+钧鉴 |
| `spawn <id>` | `sp` | 创建智能体并放入世界（不填 id 自动命名） |
| `run <n>` | `r` | 推进世界 N 步 |
| `observe <n>` | `o` | 打开观察。**不填 n = 持续观察直到按 Q** |
| `extract <id>` | `e` | 取出智能体，保存 .mind.evola |
| `load <id>` | `lo` | 从 .mind.evola 恢复到世界 |
| `stop` | — | 关闭平台（需先 extract 所有智能体） |
| `debug` | `d` | 切换隐私膜（见下文） |
| `quit` | `q` | 退出 |

### 典型会话

```
evola> s                    # 启动平台
[Platform started] 15x20

evola> sp evo               # 她进入世界
[Spawned] evo at (4, 19)

evola> o                    # 观察她（持续直到按 Q）
(pygame 窗口弹出，看到她移动、觅食、探索)
(pygame 窗口按 Q 关闭，或 CLI 按 Ctrl+C)

evola> e evo                # 拉她出来
[Extracted] evo -> evo.mind.evola

evola> stop                 # 关闭平台
evola> q                    # 退出
```

---

## 四、隐私膜

未晞面板默认只显示状态**标签**，不暴露精确数值：

```
默认:   能量 ████████░░  饥饿        ← 只有标签
按 d:   能量 ████████░░  0.42  drive:0.58  ← 精确值
        面板边框变红                     ← 提示已跨越隐私边界
```

底部隐私线：
```
默认:   ─ ─ 钧鉴·明镜 ─ ─   (虚线，表示未跨越)
debug:  ━━━ 内部状态可见 ━━━  (实线红线，表示已跨越)
```

**这是伦理表达而非安全机制**——提醒你选择不去看她的内部变量。

---

## 五、观察什么

### CLI 面板

| 观察项 | 正常 | 异常 |
|--------|------|------|
| 世界窗口 `A` 位置变化 | 不会超过 10 步不动 | 一直停在一格 |
| 事件日志滚动 | 有觅食/探索/危险事件混合 | 只有"撞墙" |
| 能量进度条 | 在 0.3~1.0 间波动 | 持续降到空 |
| 新奇度进度条 | 逐步缩短（探索越多越低） | 一直满（可能卡住） |
| 安全进度条 | 多数时候接近空 | 长期满（被危险包围） |
| 足迹计数 | 持续增长 | 一直不变 |

### Pygame 窗口

| 观察项 | 视觉线索 |
|--------|----------|
| 觅食行为 | `A` 向绿色 `F` 移动 |
| 回避行为 | `A` 远离红色 `X` |
| 探索行为 | `A` 在未访问区域（深色）移动 |
| 足迹 | 灰色小圆点标记已访问过的格子 |

---

## 六、一次性模式（旧方式）

```bash
# 终端模式
python MVPs\MVP0\mvp_0_demo.py

# 批量实验
python MVPs\MVP0\mvp_0_demo.py --steps 2000 --render none --seed 42
```

参数：`--steps` `--render` `--delay` `--seed` `--no-save` `--no-log`

输出：`.mind.evola` + `stats/*.csv` + `stats/*.png`

---

## 七、参数调整

编辑 `mvp_0_config.py`：

```python
WORLD = {
    "n_food": 8,             # 食物数量（越多越容易活）
    "n_dangers": 2,          # 危险数量
    "wall_density": 0.03,    # 墙壁密度
}

HOMEOSTASIS = {
    "energy_decay": 0.003,   # 每步能量衰减（越小活得越久）
}

ARBITER = {
    "energy_urgent": 0.5,    # 多饿开始优先觅食
    "safety_threshold": 0.3, # 多危险开始逃跑
}
```

改完后重新启动生效。

---

## 八、实验建议

### 实验 A：Pygame 基准观察

```bash
python MVPs\MVP0\evola_pygame.py
```
启动后自动创建世界+智能体。点击 `[Run 100]` 看她跑 100 步，观察右侧面板的能量波动和事件日志。

### 实验 B：CLI 实验

```bash
python MVPs\MVP0\evola_cli.py
evola> s
evola> sp evo
evola> r 200           # 快跑
evola> status          # 看结果
evola> e evo           # 保存
```

### 实验 C：存档与恢复

```bash
evola> sp evo
evola> r 200
evola> e evo           # 保存 .mind.evola
evola> stop
evola> s               # 重启
evola> lo evo          # 从存档恢复——她的记忆还在
evola> o
```

### 实验 D：批量实验（大量步数）

```bash
python MVPs\MVP0\mvp_0_demo.py --steps 5000 --render none --seed 42
```
结束后查看 `stats/` 下的 PNG 统计图。

---

## 九、常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| Pygame 窗口不弹 | 没装 pygame | `pip install pygame` |
| 屏闪 | rich 逐帧重绘 | 正常现象，4Hz 已缓解 |
| 智能体原地打转 | 能量充足无探索驱动 | 提高 energy_decay |
| 能量一直掉 | 食物太少 | 增加 n_food |
| 探索率不涨 | 食物太足不愿离开 | 降低 n_food 或提高 novelty_threshold |
| debug 后忘记恢复 | — | 再按一次 `d` |

---

## 十、版本记录

| 日期 | 变更 |
|------|------|
| 2026-05-12 | 新增 evola_pygame.py 完整图形系统，双入口（pygame+CLI） |
| 2026-05-11 | MVP 0 发布：P0 四模块集成、CLI 观察站、.mind.evola |
