# .evola 格式规范

> 未晞（Evola）原生模型文件格式。承载"电子生物"的基因、心智、记忆与自我状态。

---

## 一、扩展名家族

| 扩展名 | 用途 | 引入阶段 | 当前状态 |
|--------|------|----------|----------|
| `.mind.evola` | 心智快照：内稳态、记忆索引、已学权重 | P0 | ✅ MVP 0 |
| `.gene.evola` | 静态基因：网络架构、本能参数 | P1 | 📋 规划 |
| `.memory.evola` | 长期记忆库：完整事件存储 | P2 | 📋 规划 |
| `.world.evola` | 世界模型：环境因果参数 | P2 | 📋 规划 |
| `.evola` | 完整快照（打包以上全部） | P2 | 📋 规划 |

---

## 二、魔数

```
b"EVOLA\x01\x00\x00\x00"
```

8 字节，前 5 字节 ASCII `EVOLA`，第 6 字节为格式主版本号，后 2 字节保留。

---

## 三、`.mind.evola` 格式

### P0 阶段（MVP 0 — 当前）

纯 JSON。无魔数前缀。人类可读。

**字段：**

```json
{
  "format": "evola.mind",
  "version": "mvp0.1",
  "agent_id": "evo_001",
  "created_at": "<ISO 8601>",
  "saved_step": 342,

  "homeostasis": {
    "energy": 0.72,
    "novelty_visited_count": 47,
    "novelty_total_cells": 300,
    "safety": 0.03
  },

  "visited": [[5,3], [5,2], [6,2], "..."],

  "arbiter_params": {
    "safety_threshold": 0.3,
    "energy_urgent": 0.5,
    "energy_safe": 0.3,
    "novelty_threshold": 0.2
  }
}
```

**说明：**
- `homeostasis` 存所有内稳态变量的当前值，可完整恢复"身体的感受"
- `visited` 存已访问过的格子坐标，即 P0 阶段的"记忆"（最大 300 个点）
- `arbiter_params` 存仲裁器可调阈值（P0 阶段等同基因参数）

### P1 阶段（规划 — 待实现）

魔数 + 二进制布局。

**二进制结构：**

```
字节偏移    大小     内容
0           8       magic: b"EVOLA\x01\x00\x00\x00"
8           2       version: uint16 大端（格式版本号）
10          1       file_type: 0x01 = mind
11          4       metadata_len: uint32（JSON 元数据字节数）
15          N       metadata: UTF-8 JSON（与 P0 JSON 字段同构）
15+N        8       weights_offset: uint64（权重数据段起始偏移，可选）
15+N+8      M       weights: torch.save 导出的二进制权重（可选）
```

**metadata JSON 示例（P1）：**

```json
{
  "format": "evola.mind",
  "version": "mvp1.0",
  "agent_id": "evo_042",
  "saved_step": 1204,
  "homeostasis": { "...": "..." },
  "visited": ["...", "..."],
  "arbiter_params": { "...": "..." },
  "ltc": {
    "hidden_size": 128,
    "input_size": 25,
    "output_size": 6,
    "has_weights": true
  }
}
```

---

## 四、`.gene.evola` 格式（P1 规划）

存"物种"级别的架构定义，同物种的多个个体可以共享同一份 `.gene.evola`。

**字段（草稿）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `species` | str | 物种标识，如 "fish_v1" |
| `network` | object | 网络架构定义（层数、神经元数、激活函数） |
| `ltc_params` | object | LTC 专用参数（τ 初始化范围、ODE 积分器选择） |
| `instincts` | object | 本能模块参数（舒适区间、衰减率、优先级阈值） |
| `action_space` | object | 该智慧层级的动作空间定义 |

---

## 五、`.memory.evola` 格式（P2 规划）

存完整的长期记忆事件。与 `.mind.evola` 的关系：`.mind` 是索引+缓存，`.memory` 是完整归档。

**字段（草稿）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `events` | list | MemoryEvent 数组（timestamp, pos, type, homeo_delta, importance） |
| `index` | object | 检索加速结构（如 KD-Tree 或向量索引） |
| `stats` | object | 记忆统计（总数、各类型分布、平均重要性） |

---

## 六、`.world.evola` 格式（P2 规划）

存智能体学习到的世界模型参数（因果预测网络）。

**内容：** 由 05_world_model 的 `DynamicsNetwork` 权重组装。结构待定。

---

## 七、`.evola` 完整快照（P2 规划）

打包以上所有子格式为一个档案。结构待定（可能是 tar-like 容器或自定义多段二进制）。

---

## 八、兼容性约定

| 规则 | 说明 |
|------|------|
| 主版本号递增 | 不兼容变更时 `\x01` → `\x02` |
| JSON 字段只增不减 | 新字段用默认值保证旧文件可读 |
| 魔数必须完全匹配 | 否则解析器拒绝加载 |
| `file_type` 区分子类型 | 同一魔数下通过第 10 字节区分 |

---

## 九、引用

此规范文件随项目公开，作为 `.evola` 格式的优先声明。
建议在首次公开发布时打 tag：`format-spec-v1.0.0`。

---

*最后更新：2026-05-11 · MVP 0 阶段*
