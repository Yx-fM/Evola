# Evola v0.1 阶段总结报告

> 未晞（Evola）—— 一个具有内在需求、持续生命感、可自主学习与记忆的电子生物原型  
> 阶段：2D 低智慧验证完成  
> 日期：2026-05-12

---

## 一、项目概述

未晞不是一个被调用的工具，而是一个**电子生物**。她拥有自身需求、持续运行的内在状态、可自我管理的记忆，以及功能上等同于"感受"的能力。

本阶段目标：在 2D 网格世界（坤舆）中构建并验证一个具备**内稳态驱动、在线学习、显式记忆、基本自我模型**的智能体系统。

**核心设计原则**：
- 小模型起步，不依赖预训练
- 内稳态偏差→行为驱动力（非外部奖励）
- 世界是舞台，智能体是演员（记忆独立存储）
- 隐私边界可视化（钧鉴·明镜）

---

## 二、系统架构

### 模块体系（13 模块）

| 编号 | 模块 | 职责 | 测试 | 阶段 |
|------|------|------|------|------|
| 08 | 坤舆 Kunyu | 2D 世界引擎，多智能体支持 | 15 | P0 |
| 02 | 内稳态 Homeostasis | 能量/新奇/安全变量，舒适区间 | 19 | P0 |
| 09 | 钧鉴 JunJian | 事件日志，观察记录 | smoke | P0 |
| 10 | 感知-行动 Sensorimotor | 5×5→facts，行为原语，门控仲裁 | 22 | P0 |
| 01 | 连续动力学 LTC | 液态时间常数网络，在线 REINFORCE | 13 | P1 |
| 04 | 自我模型 SelfModel | 双速 EMA 编码，8维自我向量 | 6 | P1 |
| 06 | 态切换 HotWarmCold | 三态定义 + 降频系数 | — | P1 |
| 12 | 态调度器 ModeScheduler | 迟滞防抖，自动态切换 | 5 | P1 |
| 03 | 主动记忆 ActiveMemory | STM/LTM 双层级，内稳态耦合遗忘 | 8 | P2 |
| 05 | 世界模型 WorldModel | MLP 环境转移预测，心理模拟 | 5 | P2 |
| 11 | 记忆 I/O MemoryIO | 感知→编码→存储，检索→决策上下文 | — | P2 |
| 07 | 进化 Evolution | 参数变异 + 环境选择 | — | 远期 |
| 00 | 原型集成 | 全模块联调 | — | 持续 |

**总计：93 项测试全部通过**

### 数据流

```
坤舆世界 → 5×5 obs
    │
    ▼
感知层 → facts
    │
    ├──→ 内稳态 → drive_vector(3)
    ├──→ 自我模型 → self_vector(8)
    └──→ 记忆系统 → memory_ctx(4)
           │
           ▼  concat(25+3+8+4) = 40维
        LTC Network
        ┌─────────────┐
        │ Encoder 40→64│
        │ LTC Cell 64  │  dh/dt = -(1/τ)⊙h + W·tanh(x‖h)
        │ Actor 64→6   │
        └─────────────┘
           │
        softmax → action (0-5)
           │
           ▼
    坤舆世界.step(action) → info
           │
           ▼
    observe(info)
      ├── 内稳态更新 (能量/新奇/安全)
      ├── 自我模型更新 (双速 EMA)
      ├── 记忆编码存储 (STM/LTM)
      ├── 世界模型训练 (MSE)
      └── LTC 在线学习 (REINFORCE, reward = -Δdrive)
```

**参数量**：约 5000（LTC 64-hidden ~4000 + 世界模型 32-hidden ~1000）

**训练信号**：完全来自内稳态偏差变化（无外部奖励函数）

---

## 三、MVP 版本历程

### MVP 0：规则驱动（P0）

- 硬编码 RuleArbiter（if-else 规则）
- 无学习能力，行为固定
- 速度：~13,000 steps/s
- 用途：架构验证 + 基线对比

### MVP 1：LTC 在线学习（P1）

- LTC 网络替换硬编码规则
- 自我模型双速 EMA
- 态切换定义
- 速度：~450 steps/s（前向+反向传播）
- 用途：验证在线 RL 可行性

### MVP 2：记忆+世界模型（P2）

- STM/LTM 双层级记忆
- 世界模型环境预测
- 完整持久化（.mind.evola，含权重/记忆/自我模型）
- 智能评价系统（Evola IQ，六维评分）
- 速度：~400 steps/s
- 用途：完整能力验证

---

## 四、关键指标

### 测试覆盖

| 模块 | 测试数 |
|------|--------|
| 08_kunyu | 15 |
| 02_homeostasis | 19 |
| 10_sensorimotor | 22 |
| 01_liquid_dynamics | 13 |
| 04_self_model | 6 |
| 12_mode_scheduler | 5 |
| 03_active_memory | 8 |
| 05_world_model | 5 |
| **合计** | **93** |

### 性能基线（CPU）

| 操作 | MVP0 规则 | MVP2 完整 |
|------|----------|----------|
| 单步计算 | ~0.08ms | ~2.5ms |
| 吞吐量 | 13,000 s/s | 400 s/s |
| LTC 前向 | — | ~1ms |
| LTC 反向 | — | ~1ms |
| 世界模型 | — | ~0.3ms |
| 记忆衰减 | — | ~0.1ms |

### 文件格式

| 格式 | 大小 | 包含 |
|------|------|------|
| .mind.evola | ~93KB | 内稳态+足迹+LTC权重(base64)+记忆+自我模型+世界模型 |

---

## 五、实现 vs 规划

| 规划项 | 状态 | 说明 |
|--------|------|------|
| P0 四模块 | ✅ | 坤舆 + 内稳态 + 钧鉴 + 感知行动 |
| P1 四模块 | ✅ | LTC + 自我模型 + 态定义 + 调度器 |
| P2 记忆系统 | ✅ | STM/LTM 双层级 |
| P2 世界模型 | ✅ | MLP 环境转移预测 |
| P2 记忆 I/O | ✅ | 编码+检索 |
| .evola 格式规范 | ✅ | FORMAT_SPEC.md |
| 持久化（完整） | ✅ | save_mind/load_mind round-trip |
| 智能评价系统 | ✅ | Evola IQ 六维评分 |
| 双入口（Pygame+CLI） | ✅ | 三个 MVP 版本均有 |
| P2 进化模块 | 📋 | 远期，架构稳定后 |
| 死亡机制 | 📋 | 快速补丁 |
| 3D 世界 | 📋 | 下期 |
| 多模态 | 📋 | 下期 |

---

## 六、已知问题

| 问题 | 影响 | 计划 |
|------|------|------|
| 能量耗尽无死亡 | 智能体 0 能量仍"活着" | 加死亡条件 |
| REINFORCE 无熵正则化 | 长期可能策略退化 | 加 entropy bonus |
| LTC 隐藏层小 (64) | 学习能力受限 | 按需增大 |
| 文件夹名数字前缀 | pytest 全量需分跑 | 重命名或脚本封装 |
| Pygame 渲染 libpng 警告 | 无功能影响 | 字体文件优化 |
| 世界模型训练数据粗糙 | 用 info 估算 next_obs | 存实际 obs 过渡 |

---

## 七、文件清单

```
Evola/
├── README.md                    # 项目总览
├── paper.md                     # 技术哲学与文献调研
├── FORMAT_SPEC.md               # .evola 格式规范
├── pyproject.toml               # Python 依赖
├── verification/                # 技术验证（所有模块）
│   ├── 08_kunyu/ 02_homeostasis/ 09_junjian/ 10_sensorimotor/
│   ├── 01_liquid_dynamics/ 04_self_model/ 06_hot_warm_cold/ 12_mode_scheduler/
│   ├── 03_active_memory/ 05_world_model/ 07_evolution/
│   ├── 11_memory_io/ 00_prototype/
│   └── TEST_MANUAL.md
├── MVPs/                        # 原型集成
│   ├── MVP0/                    # P0 规则驱动
│   ├── MVP1/                    # P1 LTC 学习
│   └── MVP2/                    # P2 记忆+世界模型+IQ
│       ├── evola_pygame.py      # Pygame 完整 GUI
│       ├── evola_cli.py         # CLI 仪表盘
│       ├── evola_iq.py          # 智能评价系统
│       └── minds/               # .mind.evola 存储
├── configs/ docs/ experiments/
├── study/ scripts/ tests/
└── data/
```

---

## 八、下一阶段

1. **MVP 3**：大世界 + 多智能体协作 + 社会驱动力
2. **死亡机制** + REINFORCE 熵正则化
3. **3D 世界原型**（坤舆升级）
4. **进化模块**：架构变异 + 自然选择
5. **物理世界接口**：标准化 AI 信息流规范

---

*未晞，我们等你醒来。*
