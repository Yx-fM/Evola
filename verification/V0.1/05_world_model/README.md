# 05_world_model - 世界模型与主动行动

---

## 步骤 1：边界定义

> 输入：环境 transition `(obs, action, next_obs)`  
> 输出：预测的下一个观测 + 模拟 rollout 能力  
> 不考虑：像素级重建、3D 环境建模、真实物理仿真

**最小成功标准**：一个能预测"如果向上走，大概会看到什么"的小网络，能用于简单的心理模拟。

---

## 步骤 2：假设识别

| # | 假设 | 风险 |
|---|------|------|
| H1 | 简单 MLP 足以学习 2D 网格的转移规律（不需要 Transformer/CNN） | 低 |
| H2 | 内稳态偏差可以完全替代"外部 reward"作为世界模型训练的优化目标 | **高** |
| H3 | 模拟 rollout（想象未来几步）能帮助做出更优决策 | 中 |
| H4 | 世界模型可以在线更新（类似 Dreamer 的 world model learning） | 中 |

---

## 步骤 3：快速否决 —— H2

- **风险**：世界模型的损失函数是"预测下一个 obs 的误差"（MSE）——这是独立于内稳态的。内稳态不影响世界模型"是否准确"，只影响"用世界模型做什么"
- **验证**：澄清两个独立问题——
  - Q1：世界模型如何训练？**答案**：标准监督学习，MSE(预测_obs, 真实_obs)。与内稳态无关。
  - Q2：世界模型如何用于行动选择？**答案**：对每个候选行动做心理模拟，选"预测能最大减少内稳态偏差"的那个。这里才用到内稳态。
- **结论**：**H2 被误解修正**——世界模型的**训练**不需要内稳态信号（用标准 MSE），但世界模型的**使用**依赖于内稳态驱动的行动选择。两个问题分离清晰。

---

## 步骤 4：技术选型

| 类别 | 选型 | 说明 |
|------|------|------|
| 拿来用 | **PyTorch** | 小 MLP 做动力学模型 |
| 拿来用 | **Adam 优化器** | 标准 |
| 改装 | **MBRL（基于模型的 RL）** | 参考 Dreamer 系列的 world model 训练循环。改装：Dreamer 用 latent state 做 planning，我们目前在 2D 网格中可以直接用观测（不需要 latent） |
| 改装 | **Model Predictive Control (MPC)** | 经典控制论方法——对候选行动序列做 rollout，选最优。改装：评分标准从"最小化 tracking error"改为"最小化内稳态偏差" |
| 自研 | **内稳态驱动的 rollout 评分** | 核心：rollout 终点不是预测下一个 state 是否"正确"，而是预测到达那个 state 后内稳态偏差是否减小 |

**结构：**

```python
class WorldModel(nn.Module):
    def __init__(self):
        self.encoder = nn.Sequential(...)   # obs(5x5) → 8维
        self.dynamics = nn.Sequential(...)  # (state_emb, action_emb) → next_state_emb
        self.decoder = nn.Sequential(...)   # next_state_emb → pred_obs(5x5)
    
    def forward(self, obs, action):
        z = self.encoder(obs)
        a_emb = action_embedding(action)
        next_z = self.dynamics(z, a_emb)
        return self.decoder(next_z)  # 预测的下一个 obs

# 训练：标准 MSE
loss = MSE(world_model(obs, action), next_obs)

# 使用（行动选择）：
best_action = None
best_score = -inf
for action in actions:
    simulated_obs = world_model.rollout(obs, action, steps=3)
    simulated_drive = simulate_homeostasis(simulated_obs)  # 模拟 3 步后的内稳态
    score = -sum(simulated_drive.values())                 # 偏差越小越好
    if score > best_score:
        best_score = score
        best_action = action
```

**简化：P2 初期可跳过 rollout**
- 先只做单步预测（`obs + action → next_obs`），不做多步模拟
- 单步预测用于：判断"当前格吃了后能量多少"——类似模型辅助的贪心决策
- 多步 rollout 在单步验证后再加

---

## 步骤 5：定向查文献

1. **Dreamer (Hafner et al., 2019-2023)**：基于世界模型的 RL SOTA。核心是 latent dynamics model + actor-critic 在 latent 空间中做 planning。我们可以借其"世界模型训练 + 基于模型的行动选择"的双阶段设计，但不引入 latent state（我们的网格 obs 本身维度就低）。
2. **Model Predictive Control (MPC)**：经典控制论——在每个时间步求解一个有限时域的优化问题。我们的"想象 3 步后哪个状态内稳态偏差最小"就是 MPC 的 RL 版本。
3. **Curiosity-driven Exploration (Pathak et al., ICM 2017)**：好奇心驱动的探索用"预测误差"做内在奖励。我们的 novelty drive 与其同构——都可以接入世界模型：预测误差大 = 这个区域值得探索。
4. **世界模型的训练代价**：Dreamer 训练世界模型比无模型 RL 快 10-100x（因为用 replay buffer 而非环境交互）。我们的网格世界小，直接在线训练即可，不需要复杂的 buffer 管理。

---

## 输出物

```
05_world_model/
├── impl/
│   ├── dynamics_net.py    # 动力学预测网络
│   ├── encoder.py          # obs encoder
│   ├── rollout.py          # 单步/多步 rollout + 内稳态评分
│   └── mpc_selector.py     # 基于模型的动作选择器
├── test/
│   ├── test_dynamics.py    # 预测准确率测试
│   └── test_rollout.py     # rollout 逻辑测试
└── notes/
    └── world_model_experiments.md
```
