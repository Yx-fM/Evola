# 07_evolution - 进化与繁衍

---

## 步骤 1：边界定义

> 探索 AI 自我进化的可能性：基因（架构参数）变异 + 环境选择  
> 输出：理论方案 + 可行性验证（远期）  
> 不考虑：实际部署的进化闭环、多智能体协同进化、生物安全

**最小成功标准**（远期）：在 P0-P2 全部就绪后，验证"基因变异 + 环境选择 → 后代表现优于亲代"是否成立。

---

## 步骤 2：假设识别

| # | 假设 | 风险 |
|---|------|------|
| H1 | 架构参数可以类比为"基因"，进行可遗传的变异 | **高** |
| H2 | 内稳态达标程度是一个有效的"适应性"（fitness）度量 | 中 |
| H3 | 从亲代复制 + 变异产生子代的方案在工程上可行 | **高** |
| H4 | 自我进化（同一智能体修改自己的架构）比亲代-子代模型更合理 | **高** |

---

## 步骤 3：快速否决 —— H3

- **风险**：直接复制整个网络参数 + 随机扰动（高斯噪声）在 Torch 中很简单。但问题是——什么样的变异是有意义的？
  - Layer 数量变化？→ 需要复杂的结构变异（NAS）
  - 权重噪声？→ 可能和随机初始化一样差
  - 超参数变异？→ 调参而非进化
- **验证**：最简单的"进化"方案——
  1. 取一个表现良好的亲代模型
  2. 对其权重施加小高斯噪声（μ=0, σ=0.01）
  3. 新模型作为"子代"，在同一环境从头开始跑
  4. 比较亲代和子代的 survival time
- **结论**：**H3 可行但简单权重噪声可能不够**。真实进化需要对**架构**做变异（增加/删除神经元、调整 τ 参数），这引向 NAS。复杂度过高，**07 保持远期规划，不现在启动**。

---

## 步骤 4：技术选型

| 类别 | 选型 | 说明 |
|------|------|------|
| 拿来用 | **PyTorch 的 `state_dict()`** | 模型参数序列化/复制/修改 |
| 拿来用 | **NumPy** | 变异噪声生成 |
| 改装 | **NEAT（NeuroEvolution of Augmenting Topologies）** | 经典神经进化算法——同时进化权重和拓扑。改装：不做复杂的 speciation 机制，先做最简单的"复制+噪声" |
| 改装 | **Open-ended Learning** | 如 POET（Paired Open-Ended Trailblazer）——同时进化环境和智能体。改装：我们的"环境"可以从 08 的不同参数配置中抽样，产生"不同难度"的选择压力 |
| 自研 | **内稳态适应性函数** | `fitness = 平均 survival_time × 能量效率`——完全基于内稳态，无外部 reward |

**远期实验设计：**

```python
# 概念代码——不在 P0-P2 实现
def evolve(population_size=10, generations=50):
    population = [create_random_agent() for _ in range(population_size)]
    for gen in range(generations):
        fitnesses = [evaluate(agent) for agent in population]
        # 选择 Top-K
        elite = select_top(population, fitnesses, k=3)
        # 变异：权重噪声 + 小概率架构变异
        offspring = [mutate(copy(agent)) for agent in elite for _ in range(3)]
        population = elite + offspring
    return best(population)

def mutate(agent):
    for param in agent.parameters():
        param += torch.randn_like(param) * MUTATION_RATE
    # 小概率架构变异：增减 LTC 神经元
    if random() < 0.05:
        agent.ltc.add_neuron()
    return agent
```

---

## 步骤 5：定向查文献

1. **NEAT (Stanley & Miikkulainen, 2002)**：开创性的神经进化算法——同时进化权重和拓扑。Speciation（物种分化）机制防止创新被过早淘汰。我们不需要 speciation（初期群体规模太小），但核心思想（保护新颖性）值得考虑。
2. **POET (Wang et al., 2019)**：开放-ended learning 的里程碑——同时进化环境和智能体，互相创造选择压力。映射到我们的框架：08 的世界参数可以作为另一个"进化维度"。
3. **Weight Perturbation**：简单的参数噪声在低维问题（如 CartPole）中有时有效，但在高维网络中几乎等同于随机初始化。除非配合"从已有的好解出发"（warm start），这在我们的框架中是 natural 的（因为初始化就是"本能"）。
4. **AI Safety & Open-endedness**：开放-ended 进化可能产生意想不到的、潜在危险的行为。在 2D 网格环境中风险为零，但应在进入物理世界前建立 safety sandbox。

---

## 输出物

要求：此模块为远期规划，暂不实际产出代码。仅保留：
- 本 README（技术选型 + 实验设计文档）
- 后续补充实验记录到 notes/
