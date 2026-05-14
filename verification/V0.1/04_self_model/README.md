# 04_self_model - 自我状态向量与自指

---

## 步骤 1：边界定义

> 输入：内稳态状态 `{energy, novelty, safety}` + 近期行为历史  
> 输出：自我状态向量（低维 embedding）  
> 不考虑：自然语言自我报告（"我感觉..."）、情感模型、意识哲学

**最小成功标准**：一个持续更新的低维向量，能被 01_liquid_dynamics 作为额外输入，使智能体在决策时"参考自身状态"。

---

## 步骤 2：假设识别

| # | 假设 | 风险 |
|---|------|------|
| H1 | 低维自我向量（16-32 维）足以捕捉"我是谁"所需的信息 | 中 |
| H2 | 自我向量在决策中的实际影响是可测量的（消融实验） | **高** |
| H3 | 近期行为历史对自我状态的贡献可通过简单统计量（成功率、探索率）表达 | 低 |
| H4 | 自我向量的更新可以用指数移动平均（EMA），不需要复杂的贝叶斯更新 | 低 |

---

## 步骤 3：快速否决 —— H2

- **风险**：加了 self_vector 和没加，行为没区别——那自我模型就是个"摆设"
- **验证**：设计消融实验——
  - 条件 A：LTC 输入不含 self_vector（只含 obs + drive）
  - 条件 B：LTC 输入含 self_vector
  - 观察行为差异：条件 B 是否更少做"自己能力达不到的事"？（如能量很低时不探索远处）
- **结论**：**H2 暂不否决，但需要消融实验验证**。如果 P1 完成后消融无差异，说明 self_vector 设计无效，需重新设计或降低优先级

---

## 步骤 4：技术选型

| 类别 | 选型 | 说明 |
|------|------|------|
| 拿来用 | **PyTorch** | 一个小 MLP 做 encoder |
| 拿来用 | **NumPy** | 统计量计算（成功率、移动距离等） |
| 改装 | **EMA（指数移动平均）** | 经典时序平滑方法。改装：用不同衰减率区分"长期自我"（α=0.001）和"短期状态"（α=0.1） |
| 自研 | **自我向量维度设计** | 核心创新：设计"自我状态"应包含哪些维度 |
| 自研 | **行为自我聚合** | 如何把行为历史压缩为"我擅长什么"的信号 |

**自我向量维度设计提案：**

```python
SelfVector = [
    # 内稳态（来自 02）
    energy_level,          # 当前能量
    novelty_saturation,    # 探索了多少
    
    # 能力自评（来自行为历史统计）
    efficiency,            # 最近 N 步内吃到食物的比例
    survival_time,         # 存活步数 / 总步数
    exploration_rate,      # 最近 N 步访问新格子的比例
    
    # 元状态
    state_freshness,       # 距离上次"有意义事件"的步数（时间感知）
    stability,             # 内稳态变量的方差（是否稳定）
]
```

**编码器结构：**

```
raw_self_state (6-8 维) ──→ MLP(8→16→8) ──→ self_vector (8 维)
                                              ↓
                                    作为 LTC 的额外条件输入
                                    与 [obs, drive] concat
```

**更新机制（双速度 EMA）：**

```python
# 长期自我（缓慢变化）——"我是谁"
self_long = 0.999 * self_long + 0.001 * current_raw_state

# 短期状态（快速变化）——"我现在怎么样"
self_short = 0.9 * self_short + 0.1 * current_raw_state

# 最终自我向量 = concat
self_vector = concat(self_long, self_short)
```

---

## 步骤 5：定向查文献

1. **MAGELLAN (ICML 2025)**：赋予 AI 体"元认知"——预测自己的学习进度。我们的"能力自评"部分（efficiency）就是元认知的简化版。他们的方法用的是 learning progress 预测器，我们初期用简单统计即可。
2. **Recursive Self-Modeling (Coda 双智能体系统)**：两个智能体通过互相预测对方的信念维持自身结构。我们不做双智能体，但核心思想——"自我模型是稳定的动态吸引子"——启示：self_vector 应该在 P2 能够被 06_warm_cold 利用，温态时自我向量继续保持更新。
3. **EMA 作为认知模型**：心理学中"自我概念"的更新速度——我们对自己的认知不会因一次失败就改变（类似 α 极小的 EMA）。我们的双速度 EMA 设计受此启发。
4. **消融实验设计 (Ablation Study)**：标准 DL 方法学——验证某个组件是否有用的黄金标准。我们必须在 P1 结束时对 self_vector 做消融实验，否则就是在自欺欺人。

---

## 输出物

```
04_self_model/
├── impl/
│   ├── self_state.py      # 自我状态原始变量定义
│   ├── encoder.py          # MLP encoder
│   └── aggregator.py       # EMA 双速度更新
├── test/
│   ├── test_self_state.py
│   └── test_ablation.py    # 消融实验脚本（带/不带 self_vector）
└── notes/
    └── ablation_results.md
```
