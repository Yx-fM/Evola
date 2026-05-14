# 01_liquid_dynamics - 连续时间动力学

---

## 步骤 1：边界定义

> 输入：当前观察 + 驱动力向量 + 上一个隐藏状态  
> 输出：下一个动作 + 新的隐藏状态  
> 不考虑：Transformer 架构、注意力机制、大规模并行训练、多模态输入

**最小成功标准**：一个 LTC 小网络替换 10_sensorimotor 的行动子模块，在 2D 网格世界中比硬编码规则表现得"更灵活"（能学会吃食物不一定是最近的那个）。

---

## 步骤 2：假设识别

| # | 假设 | 风险 |
|---|------|------|
| H1 | LTC 的小网络（<1000 参数）足以驱动 P0 级别的 2D 网格行为 | **高** |
| H2 | `torchdiffeq` 的 `odeint` 能稳定运行在我们的训练循环中 | 中 |
| H3 | 连续时间动态真的比离散 MLP 带来行为优势（如行动更平滑） | 中 |
| H4 | 内稳态驱动力可以作为 LTC 的额外条件输入 | 低 |
| H5 | 训练可以在线完成（边跑边学），不需要离线数据集 | 中 |

---

## 步骤 3：快速否决 —— H1

- **风险**：LTC 虽高效，但参数太少可能学不到"食物方向→移动"这种基本映射
- **验证**：LTC 的已知最低能力边界——
  - 线虫（C. elegans）302 个神经元即可实现趋利避害
  - LTC 论文演示了在 Gym 环境（CartPole, Mountain Car）中用约 100 参数的网络成功
  - 我们的 2D 网格比 CartPole 简单（离散动作 + 低维观察）
- **结论**：**H1 成立**。128-256 hidden 的 LTC 足够。如果不行，可以先用小型 MLP 替代做基线对比。

---

## 步骤 4：技术选型

| 类别 | 选型 | 说明 |
|------|------|------|
| 拿来用 | **PyTorch** | 张量运算，自动微分 |
| 拿来用 | **`torchdiffeq`** (`odeint`) | 清华大学开源的 ODE solver。标准选择，LTC 原始论文就用它 |
| 拿来用 | **Adam 优化器** | 小参数量的标准选择 |
| 改装 | **LTC 参考实现** | 参考 Liquid AI 开源代码中的 LTC cell 结构。改装：将"预测下一个 observation"的损失替换为"内稳态偏差最小化 + 行为多样性 bonus" |
| 自研 | **内稳态奖励函数** | 核心创新：不预测 next token，不看外部 reward——而是用 `sum(drive_i)` 的减小量作为训练信号 |
| 自研 | **在线持续学习循环** | 每次 step 都做一次小 gradient update（类似 online SGD），而非等 episode 结束后批量训练 |

**LTC Cell 核心结构：**

```
输入 x(t) + 隐藏状态 h(t)
    ↓
┌───────────────────────┐
│  LTC Cell              │
│  dh/dt = f(x, h, θ)   │  ← ODE 定义
│  τ = sigmoid(W_τ·[x,h])│  ← 时间常数（LTC 核心创新：每个神经元有自己的时间常数）
│  h_new = odeint(       │
│    dh/dt, h, [t, t+dt] │  ← ODE solver 积分
│  )                     │
└───────────────────────┘
    ↓
输出层 → action logits
```

**训练信号（自研核心）：**

```python
# 不是：loss = cross_entropy(prediction, ground_truth) 
# 而是：loss = - (drive_before - drive_after)
# 即：行动减轻了多少内稳态偏差
def compute_reward(drive_before, drive_after, action):
    delta = sum(max(d, 0) for d in (drive_before - drive_after).values())
    entropy_bonus = -0.01 * entropy(action_logits)  # 鼓励探索
    return delta + entropy_bonus
```

**在线学习循环：**

```python
for step in range(max_steps):
    obs = env.get_obs()
    action, h = ltc_net(obs, drive, h_prev)        # 前向
    new_obs = env.step(action)
    new_drive = homeo.update(perceive(new_obs))     # 得到新驱动
    reward = compute_reward(drive, new_drive, action)
    loss = -reward * log_prob(action)               # Policy Gradient
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    h_prev = h.detach()                             # 截断梯度（BPTT 只做 1 步）
```

---

## 步骤 5：定向查文献

1. **LTC 原始论文 (Hasani et al., AAAI 2021)**："Liquid Time-Constant Networks"——定义了 LTC 的数学形式：`dx/dt = -(1/τ)(x + S) + A * sigmoid(I + x + S)`，其中 τ 是神经元的可学习时间常数。我们不需要从头推导，直接用标准实现。
2. **`torchdiffeq` 使用注意事项**：`odeint` 的 `atol`/`rtol` 参数影响精度和速度。CartPole 级别任务用默认值 (`1e-3`) 即可。关键坑：LTC 的 τ 必须 > 0（用 `sigmoid + epsilon` 保证），否则 ODE 发散。
3. **Online RL with Policy Gradient**：A3C/A2C 是标准的在线策略梯度方法。我们的简化版（1-step REINFORCE）在 CartPole 上可行。关键坑：熵正则化系数不能太大（>0.1），否则行为变随机而非优化。
4. **BPTT truncation**：RNN 通常需要截断反向传播。我们的 1-step truncation 是最保守的，如果行为策略不够长，需改为 N-step（如 8 步）。暂用 1-step 快速原型，观察效果。

---

## 输出物

```
01_liquid_dynamics/
├── impl/
│   ├── ltc_cell.py        # LTC 单元定义（参考 LTC 论文）
│   ├── ltc_network.py     # 完整网络（encoder + LTC + output）
│   ├── reward_signal.py   # 内稳态奖励函数（自研核心）
│   └── online_trainer.py  # 在线学习循环
├── test/
│   ├── test_ltc_cell.py   # LTC 单元数学正确性
│   └── test_training.py   # 训练循环集成测试
└── notes/
    └── ltc_experiments.md
```
