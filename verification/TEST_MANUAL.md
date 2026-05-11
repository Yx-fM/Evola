# 测试手册 Test Manual

> 每个模块的测试命令、观察步骤、开发状态。每次开发后同步更新。

---

## 模块状态一览

| 模块 | 状态 | 测试通过率 | 上次更新 |
|------|------|-----------|----------|
| 08_kunyu | ✅ 基础完成 | 15/15 | 2026-05-11 |
| 09_junjian | ✅ 基础完成 | smoke verified | 2026-05-11 |
| 10_sensorimotor | ✅ 基础完成 | 22/22 | 2026-05-11 |
| 02_homeostasis | ✅ 基础完成 | 19/19 | 2026-05-11 |
| 01_liquid_dynamics | 🔴 未开始 | - | - |
| 04_self_model | 🔴 未开始 | - | - |
| 06_hot_warm_cold | 🔴 未开始 | - | - |
| 03_active_memory | 🔴 未开始 | - | - |
| 05_world_model | 🔴 未开始 | - | - |
| 07_evolution | 🔴 未开始 | - | - |
| 11_memory_io | 🔴 未开始 | - | - |
| 12_mode_scheduler | 🔴 未开始 | - | - |
| 00_prototype | 🔴 未开始 | - | - |

---

## 08_kunyu — 坤舆世界引擎

### 状态：✅ 基础完成（支持多智能体）

文件：
```
08_kunyu/impl/entities.py         — EntityType(5种) + Action(6种)
08_kunyu/impl/event_bus.py        — 事件总线（08↔09 桥梁）
08_kunyu/impl/agent_interface.py  — IAgent 接口 + RandomAgent
08_kunyu/impl/world.py            — KunyuWorld 引擎（多智能体支持）
```

### 架构原则
- **世界是舞台，智能体是演员** —— 智能体的记忆/内稳态/自我模型独立存储，不在世界中
- `register_agent(id)` → 智能体"进入世界"；`remove_agent(id)` → "离开世界"
- `step(action, agent_id=...)` → 操作指定智能体

### 测试命令

```bash
cd Q:\All_Items\DreamProjects\Evola
python -m pytest verification/08_kunyu/test/test_world.py -v
```

### 预期输出
```
15 passed in 0.xx s
```

### 测试覆盖内容
（同前 12 项 + 新增 3 项）
- test_multi_agent — 多个智能体同时存在
- test_remove_agent — 智能体离开世界
- test_agent_enter_event — EventBus 事件发布

### 观察方式

```bash
# 方式 1：manual 模式（WASD 键盘操控）
python verification/09_junjian/impl/demo.py --mode manual

# 方式 2：auto 模式（RandomAgent 自主运行）
python verification/09_junjian/impl/demo.py --mode auto --steps 200 --delay 0.05

# 方式 3：查看事件日志
cat verification/09_junjian/logs/evola_*.jsonl | head -20
```

---

## 09_junjian — 钧鉴观察系统

### 状态：✅ 基础完成

文件：
```
09_junjian/impl/logger.py          — JunJian 核心（订阅 EventBus，写 JSON Lines）
09_junjian/impl/render_terminal.py — ANSI 终端渲染
09_junjian/impl/render_pygame.py   — Pygame 实时窗口
09_junjian/impl/render_mpl.py      — matplotlib 渲染
09_junjian/impl/demo.py            — 集成演示入口
```

### 测试命令

```bash
# Smoke test（EventBus + JunJian 集成验证）
python verification/09_junjian/smoke.py
```

### 预期输出
```
OK: 4 events logged to evola_20260511_xxxxxx.jsonl
```

### 观察方式

```bash
# 方式 1：matplotlib 窗口（demo.py）
python verification/09_junjian/impl/demo.py

# 方式 2：查看日志
cat verification/09_junjian/logs/evola_*.jsonl
```

### 日志格式
```jsonl
{"type": "step.begin", "step": 1, "agent_pos": [5, 3], "action": "UP"}
{"type": "step.outcome", "step": 1, "energy_gained": 0.0, "damage_taken": 0.0, ...}
```

---

## 10_sensorimotor — 感知-行动桥接

### 状态：✅ 基础完成

文件：
```
10_sensorimotor/impl/
├── behavior_standard/           # ★ 行为规范层（永久框架）
│   ├── primitives.py            #   行为原语(6种)
│   ├── action_space.py          #   动作空间（按智慧层级）
│   ├── arbitration.py           #   IArbiter + RuleArbiter(动态权重衰减)
│   └── actuator.py              #   执行器（原语→Action）
├── perception/                  # 感知层
│   ├── perception.py            #   5×5网格→Facts
│   └── salience.py              #   驱动力→显著性
└── agent.py                     # SensorimotorAgent 组装
```

### 测试命令
```bash
cd Q:\All_Items\DreamProjects\Evola
python -m pytest verification/10_sensorimotor/test/test_sensorimotor.py -v
```

### 预期输出
```
22 passed
```

### 测试覆盖
- 感知：空视野 / 食物方向 / 危险方向 / 脚下食物 / 墙壁计数 (5 tests)
- 显著性：饥饿→食物重要 / 危险→危险信息重要 (2 tests)
- 仲裁：饥饿→趋近 / 脚下食物→消耗 / 危险→回避 / 安全→探索 / 默认→漫游 (5 tests)
- 执行：消耗→EAT / 等待→WAIT / 趋近→方向 / 回避→反向 / 漫游→合法动作 (5 tests)
- 智能体：创建 / 空视野行动 / 吃附近食物 / 观察更新能量 / 观察更新访问 (5 tests)

### 可直接 demo
```bash
# SensorimotorAgent 在坤舆中自主运行
# (需要写一个集成脚本 —— 接下来做)
```

---

## 02_homeostasis — 内稳态本能

### 状态：✅ 基础完成（已与 10 集成）

文件：
```
02_homeostasis/impl/
├── comfort_zones.py    — ComfortZone + 默认区间
├── variables.py        — EnergyVariable / NoveltyVariable / SafetyVariable
└── drives.py           — Homeostasis + DriveVector + 标签映射
```

### 测试命令

```bash
cd Q:\All_Items\DreamProjects\Evola
python -m pytest verification/02_homeostasis/test/test_homeostasis.py -v
```

### 预期输出
```
19 passed
```

### 测试覆盖
- 舒适区间：区内/区下/区上/边界 (4 tests)
- 变量：能量衰减/补充/钳制 (3 tests)
- 新奇度：初始高/访问降低/饱和 (3 tests)
- 安全：危险上升/无危险衰减 (2 tests)
- Homeostasis：初始驱动/饥饿驱动/eating恢复/访问跟踪/危险驱动/标签/状态 (7 tests)

### 已集成
- SensorimotorAgent (10) 现在使用 Homeostasis 而非合成驱动
- `agent.act(obs)` → 内部调用 `homeo.get_drive()` → 喂给 arbiter
- `agent.observe(info)` → 内部调用 `homeo.update(info)`

---

## 其余模块

### 状态：🔴 均未开始（README 已就绪）

| 模块 | 启动前置条件 |
|------|-------------|
| 01_liquid_dynamics | P0 全部就绪 |
| 04_self_model | 02_homeostasis |
| 06_hot_warm_cold | 01 + 02 |
| 03_active_memory | 01 + 02 |
| 05_world_model | 08 + 02 |
| 07_evolution | P0-P2 |
| 11_memory_io | 10 + 03 |
| 12_mode_scheduler | 02 + 06 |
| 00_prototype | P0-P2 全部 |

---

## 快速全量测试

```bash
# 运行所有已有测试
cd Q:\All_Items\DreamProjects\Evola
python -m pytest verification/ -v --tb=short
```

---

## 更新日志

| 日期 | 模块 | 变更 |
|------|------|------|
| 2026-05-11 | 02_homeostasis | 舒适区间+变量+Homeostasis 完成，19 tests pass；已与 10 集成 |
| 2026-05-11 | 10_sensorimotor | 重构 agent.py 使用 Homeostasis 替换合成驱动 |
| 2026-05-11 | 10_sensorimotor | 行为规范层+感知层完成，22 tests pass |
| 2026-05-11 | 08_kunyu | 重构多智能体：支持 register/remove，IAgent 接口，15 tests pass |
| 2026-05-11 | 09_junjian | demo 支持 --mode manual/auto |
| 2026-05-11 | 08_kunyu | 基础引擎完成，12 tests pass |
| 2026-05-11 | 09_junjian | 孪生系统完成，EventBus 集成，smoke verified |
| 2026-05-11 | - | 创建测试手册 |
