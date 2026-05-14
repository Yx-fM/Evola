# .evola 格式规范 v1.0

> 未晞（Evola）原生模型文件格式。承载"电子生物"的基因、心智、记忆与自我状态。

---

## 一、扩展名家族

| 扩展名 | 用途 | 阶段 | 实现 |
|--------|------|------|:---:|
| `.mind.evola` | 心智快照：内稳态+足迹+LTC权重+记忆+自我模型+世界模型 | P2 | ✅ |
| `.gene.evola` | 静态基因：网络架构定义+本能参数（同物种共享） | P2 | 📋 |
| `.memory.evola` | 长期记忆归档：完整事件存储（.mind 是索引+缓存） | P2 | ⚡ 已在 .mind 内 |
| `.world.evola` | 世界模型：环境预测网络权重 | P2 | ⚡ 已在 .mind 内 |
| `.evola` | 完整快照（打包以上全部） | P3 | 📋 |

---

## 二、魔数

```
b"EVOLA\x01\x00\x00\x00"
```

8 字节。前 5 字节 ASCII `EVOLA`。第 6 字节为格式主版本号。后 2 字节保留。

---

## 三、`.mind.evola` 格式（当前实现 v0.1）

**容器**：纯 JSON，无魔数前缀，人类可读。**网络权重以 base64 内嵌**。

**字段（v0.1 实际）：**

```json
{
  "format": "evola.mind",
  "version": "mvp2.0",
  "agent_id": "evo_001",
  "step_count": 1204,

  "homeostasis": {
    "energy": 0.72,
    "safety": 0.03,
    "novelty_visited_count": 47,
    "novelty_total_cells": 300
  },

  "visited": [[5,3], [5,2], "..."],

  "arbiter_type": "ltc",
  "ltc_weights": "<base64 编码的 torch state_dict，约 64KB>",
  "wm_weights": "<base64 编码的世界模型权重，约 2KB>",

  "memory": {
    "stm": [{"step": 294, "pos": [10,18], "event_type": "food_visible",
              "importance": 0.3, "decay": 0.94, "...": "..."}],
    "ltm": [{"step": 300, "pos": [12,18], "event_type": "food_found",
              "importance": 1.0, "decay": 0.99, "...": "..."}]
  },

  "self_model": {
    "slow": [0.7, 0.5, 0.1, 0.0, 0.0, 0.0],
    "fast": [0.3, 0.8, 0.0, 0.1, 0.0, 0.3]
  }
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `homeostasis` | object | 内稳态当前值（可恢复"身体感受"） |
| `visited` | list | 已访问格子坐标（P0 记忆） |
| `ltc_weights` | base64 string | LTC 网络完整参数（约 64KB 编码后） |
| `wm_weights` | base64 string | 世界模型网络参数 |
| `memory` | object | STM/LTM 记忆条目（P2 记忆） |
| `self_model` | object | 双速 EMA 向量（slow/fast） |

**文件大小**：约 93KB（含 1 个 agent 的完整状态）。

---

## 四、`.gene.evola` 格式（规划）

存"物种"级别的架构定义。同物种多个个体共享同一份 `.gene.evola`。

| 字段 | 类型 | 说明 |
|------|------|------|
| `species` | str | "fish_v1" 等 |
| `network` | object | 层数、神经元数、激活函数 |
| `ltc_params` | object | τ 初始化、ODE solver 选择 |
| `instincts` | object | 舒适区间、衰减率、优先级阈值 |
| `action_space` | object | 智慧层级动作空间 |

> 当前这些参数散落在 `mvp_2_config.py` 中。`.gene.evola` 将其统一为一个可加载的"DNA 文件"。

---

## 五、`.memory.evola` 格式（规划）

独立于 `.mind.evola` 的长期记忆归档。当 STM/LTM 超过容量时，溢出内容写入此文件。

| 字段 | 类型 | 说明 |
|------|------|------|
| `events` | list | MemoryEvent 数组 |
| `stats` | object | 记忆统计（总数、类型分布） |

---

## 六、`.world.evola` 格式（规划）

独立存储世界模型参数，支持跨世界迁移。

**内容**：`05_world_model` 的完整网络参数 + 训练统计。

---

## 七、`.evola` 完整快照（v0.2 规划）

二进制容器：

```
字节偏移    大小     内容
0           8       magic: b"EVOLA\x01\x00\x00\x00"
8           2       version: uint16
10          1       file_type: 0xFF = full snapshot
11          4       section_count: uint32
15          N×16    section_table: [(type:uint8, offset:uint64, size:uint64)]
15+N×16     ...     sections: [mind, gene, memory, world]
```

四个 section 按上面定义的子格式独立存储，共享一个魔数前缀。

---

## 八、兼容性约定

| 规则 | 说明 |
|------|------|
| 主版本号递增 | 不兼容变更时 `\x01` → `\x02` |
| JSON 字段只增不减 | 新字段用默认值保证旧文件可读 |
| 魔数必须完全匹配 | 否则解析器拒绝加载 |
| base64 权重可随时升级为二进制段 | 二进制段追加在 JSON 之后，保留向后兼容 |
| `file_type` 区分子格式 | 同一魔数下通过第 10 字节区分 |

---

## 九、当前实现状态（v0.1）

| 能力 | 状态 |
|------|:---:|
| .mind.evola JSON 序列化 | ✅ |
| .mind.evola JSON 反序列化 | ✅ |
| LTC 权重持久化 (base64) | ✅ |
| 世界模型权重持久化 (base64) | ✅ |
| 记忆 STM/LTM 持久化 | ✅ |
| 自我模型持久化 | ✅ |
| Round-trip 验证 | ✅ |
| 二进制容器 (.evola) | 📋 v0.2 |
| .gene.evola 独立文件 | 📋 v0.2 |
| 跨版本迁移 | 📋 |

---

*最后更新：2026-05-12 · v0.1 完成*
