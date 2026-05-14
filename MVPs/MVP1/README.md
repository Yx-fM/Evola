# MVP 1 — P0 + P1 集成（LTC 在线学习）

首个使用 LTC 连续时间网络的 Evola 原型。决策系统从硬编码规则升级为在线强化学习网络。

## 与 MVP 0 的区别

| | MVP 0 | MVP 1 |
|---|-------|-------|
| 决策系统 | 硬编码 RuleArbiter | LTC 连续时间网络（REINFORCE） |
| 自我模型 | 无 | 双速 EMA 自我向量（8 维） |
| 学习方式 | 无 | 每步在线训练（内稳态偏差 = 奖励） |
| 隐藏状态 | 无 | LTC 隐藏态跨步持续（温态基础） |
| 态切换 | 无 | HOT/WARM/COLD 定义 + 调度器（迟滞） |

## 文件结构

```
MVPs/MVP1/
├── evola_pygame.py       ★ Pygame 图形窗口（LTC）
├── evola_cli.py          ★ CLI 仪表盘（LTC）
├── mvp_1_demo.py           一次性批量运行
├── mvp_1_config.py         统一配置（含 LTC 超参）
├── README.md
├── minds/ / stats/ / logs/
```

## 快速开始

```bash
cd Q:\All_Items\DreamProjects\Evola

# Pygame 图形窗口（推荐）
python MVPs\MVP1\evola_pygame.py

# CLI 仪表盘
python MVPs\MVP1\evola_cli.py

# 一次性运行（LTC 模式）
python MVPs\MVP1\mvp_1_demo.py --steps 1000

# 对比模式（RuleArbiter 基线）
python MVPs\MVP1\mvp_1_demo.py --steps 1000 --rule
```

## 配置

编辑 `mvp_1_config.py`：

```python
ARBITER = {
    "type": "ltc",          # "ltc" or "rule"
    "ltc_hidden": 64,       # LTC 隐藏层大小
    "ltc_lr": 0.001,        # 学习率
    "use_self_model": True,  # 是否启用自我模型
}
```

## 关键观察

| 观察项 | LTC 预期表现 | Rule 基线对比 |
|--------|-------------|--------------|
| 初期行为 | 随机（尚未学习） | 规则驱动，从第一步就有目标 |
| 100-500 步 | 开始表现出觅食/回避倾向 | 行为稳定不变 |
| 500+ 步 | 行为自适应，可能比规则更灵活 | 规则策略固定 |
| 速度 | ~450 steps/s (CPU) | ~13000 steps/s |
| 能量效率 | 逐渐学会高效觅食 | 规则固定效率 |

## 架构

```
obs(25) + drive(3) + self_vector(8) = 36
        ↓
  LTCNetwork (LTC Cell + Euler ODE)
        ↓
  action logits → 采样 → BehaviorPrimitive
        ↓
  SelfModel.update() → 双速 EMA 编码
        ↓
  LTCArbiter.update(drive) → REINFORCE(-Δdrive)
        ↓
  ModeScheduler → HOT/WARM/COLD
```
