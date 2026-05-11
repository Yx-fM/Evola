# 注意力模块规格 (Attention Module Specification)

**版本**: 0.1.0  
**状态**: 设计稿  
**所属架构层级**: Layer 2 - 认知层  
**最后更新**: 2026 年 4 月 15 日

---

## 1. 模块概述

### 1.1 核心职责

注意力模块负责**信息路由与选择性聚焦**，是 Evola 认知系统的核心组件。

**关键创新**: 与标准 Transformer 的注意力机制不同，Evola 的注意力受**内稳态信号调制**，使信息处理具有内在驱动力导向。

### 1.2 设计目标

| 目标 | 说明 | 优先级 |
|------|------|--------|
| **内稳态调制** | 注意力分配受内在需求影响 | 🔴 必须 |
| **连续时间动力学** | 使用 LTC 替代离散时间步 | 🔴 必须 |
| **多尺度注意力** | 同时处理局部和全局信息 | 🟡 可选 |
| **温态 Attention** | 无输入时仍能后台运行 | 🟡 可选 |

---

## 2. 与标准 Transformer 的对比

| 特性 | 标准 Transformer | LLaMA/Mistral | Evola Attention |
|------|-----------------|---------------|-----------------|
| **时间模型** | 离散步 | 离散步 | **连续时间 (LTC)** |
| **注意力来源** | QKV 点积 | QKV 点积 + RoPE | **QKV + 内稳态调制** |
| **运行模式** | 前向传播 | 前向传播 | **热态 + 温态** |
| **记忆机制** | KV Cache (被动) | KV Cache (被动) | **主动记忆管理** |
| **可解释性** | 注意力权重可视化 | 注意力权重可视化 | **+ 内稳态影响分析** |

---

## 3. 核心类设计

### 3.1 主类：`HomeostasisModulatedAttention`

```python
class HomeostasisModulatedAttention(nn.Module):
    """
    内稳态调制的注意力机制
    
    核心思想：注意力不仅由数据驱动，还受内在需求调节
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        causal: bool = False,
        liquid_time_constant: bool = True,
    ):
        """
        Args:
            dim: 嵌入维度
            num_heads: 注意力头数
            head_dim: 每头维度 (默认 dim // num_heads)
            dropout: 注意力 dropout
            causal: 是否因果掩码
            liquid_time_constant: 是否使用 LTC
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV 投影
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        # 内稳态调制器
        self.homeostasis_modulator = HomeostasisModulator(
            homeostasis_dim=5,  # 内稳态变量数量
            num_heads=num_heads,
        )
        
        # 可选：液态时间常数
        if liquid_time_constant:
            self.ltc_cell = LiquidTimeConstantCell(
                input_dim=self.head_dim,
                hidden_dim=self.head_dim,
            )
        else:
            self.ltc_cell = None
        
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
```

---

### 3.2 前向传播

```python
    def forward(
        self,
        x: Tensor,                                    # [batch, seq_len, dim]
        homeostasis_state: HomeostasisVariables,      # 内稳态状态
        mask: Optional[Tensor] = None,                # 注意力掩码
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,  # 历史 KV 缓存
        return_weights: bool = False,                 # 是否返回注意力权重
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            x: 输入序列
            homeostasis_state: 当前内稳态状态
            mask: 注意力掩码（如因果掩码、padding 掩码）
            past_kv: 历史 KV 缓存（用于增量解码）
            return_weights: 是否返回注意力权重（用于分析）
            
        Returns:
            output: 注意力加权输出 [batch, seq_len, dim]
            attn_weights (可选): 注意力权重 [batch, heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 计算 Q, K, V
        query = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. 可选：应用 RoPE 位置编码
        # query, key = apply_rope(query, key, position_ids)
        
        # 3. 可选：使用 LTC 处理 QKV（连续时间动力学）
        if self.ltc_cell is not None:
            query, key, value = self._apply_ltc(query, key, value)
        
        # 4. 合并历史 KV 缓存
        if past_kv is not None:
            past_key, past_value = past_kv
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
        
        # 5. 计算注意力分数
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # 6. 应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # 7. Softmax 获取注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 8. 【关键创新】内稳态调制
        modulation_factors = self.homeostasis_modulator(homeostasis_state)
        # modulation_factors: [batch, num_heads, 1, 1]
        # 对内稳态敏感的注意力头进行增强/抑制
        attn_weights = attn_weights * modulation_factors
        
        # 9. 加权求和
        output = torch.matmul(attn_weights, value)
        
        # 10. 拼接 + 输出投影
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.o_proj(output)
        
        if return_weights:
            return output, attn_weights
        else:
            return output
```

---

### 3.3 内稳态调制器：`HomeostasisModulator`

```python
class HomeostasisModulator(nn.Module):
    """
    将内稳态状态映射为注意力调制因子
    
    原理：不同内稳态状态会影响注意力的分配策略
    - 高好奇 → 增强对新异刺激的注意力
    - 高预测误差 → 增强对异常信号的关注
    - 低能量 → 降低注意力广度（聚焦关键信息）
    """
    
    def __init__(
        self,
        homeostasis_dim: int = 5,
        num_heads: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        self.homeostasis_dim = homeostasis_dim
        self.num_heads = num_heads
        
        # 调制网络：内稳态 → 头级调制因子
        self.network = nn.Sequential(
            nn.Linear(homeostasis_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
            nn.Sigmoid(),  # 输出范围 [0, 1]，作为注意力缩放因子
        )
        
        # 初始化：默认全 1（无调制）
        nn.init.ones_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, homeostasis_state: HomeostasisVariables) -> Tensor:
        """
        Args:
            homeostasis_state: 内稳态变量
            
        Returns:
            modulation_factors: [batch, num_heads, 1, 1]
        """
        # 将内稳态变量转换为向量
        h_vector = homeostasis_state.to_tensor()  # [batch, homeostasis_dim]
        
        # 通过网络生成调制因子
        modulation = self.network(h_vector)  # [batch, num_heads]
        
        # 重塑为可广播的形状
        modulation = modulation.view(-1, self.num_heads, 1, 1)
        
        return modulation


@dataclass
class HomeostasisVariables:
    """内稳态变量定义"""
    information_entropy: float   # 信息熵 (0-1)
    prediction_error: float      # 预测误差 (0-1)
    memory_load: float           # 记忆负载 (0-1)
    energy_level: float          # 能量水平 (0-1)
    social_connection: float     # 社交连接 (0-1)
    
    def to_tensor(self) -> Tensor:
        """转换为向量"""
        return torch.tensor([
            self.information_entropy,
            self.prediction_error,
            self.memory_load,
            self.energy_level,
            self.social_connection,
        ]).unsqueeze(0)
```

---

### 3.4 液态时间常数单元：`LiquidTimeConstantCell`

```python
class LiquidTimeConstantCell(nn.Module):
    """
    液态时间常数 (LTC) 单元
    
    核心思想：使用微分方程描述隐藏状态的连续时间演化
    dh/dt = f(h, x, t) - h / tau
    
    其中 tau 是可学习的时间常数
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_units: int = 16,
        time_constant_init: float = 1.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_units = num_units
        
        # 输入到隐藏的权重
        self.input_weights = nn.Linear(input_dim, num_units * hidden_dim, bias=False)
        
        # 隐藏到隐藏的递归权重
        self.recurrent_weights = nn.Linear(hidden_dim, num_units * hidden_dim, bias=False)
        
        # 时间常数 tau（可学习）
        self.time_constant = nn.Parameter(
            torch.ones(num_units, hidden_dim) * time_constant_init
        )
        
        # 激活函数
        self.activation = nn.Tanh()
    
    def forward(
        self,
        x: Tensor,      # [batch, seq_len, input_dim]
        h0: Tensor,     # [batch, num_units, hidden_dim]
        dt: float = 0.1, # 时间步长
    ) -> Tensor:
        """
        求解微分方程：dh/dt = f(h, x) - h / tau
        
        使用欧拉法数值求解：
        h(t+dt) = h(t) + dt * (f(h, x) - h / tau)
        """
        batch_size, seq_len, _ = x.shape
        h = h0
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, input_dim]
            
            # 计算输入驱动和递归驱动
            input_drive = self.input_weights(x_t).view(batch_size, self.num_units, self.hidden_dim)
            recurrent_drive = self.recurrent_weights(h).view(batch_size, self.num_units, self.hidden_dim)
            
            # 激活
            f_h = self.activation(input_drive + recurrent_drive)
            
            # LTC 微分方程：dh/dt = f(h, x) - h / tau
            dh_dt = f_h - h / self.time_constant
            
            # 欧拉积分
            h = h + dt * dh_dt
            
            outputs.append(h)
        
        # 输出：[batch, seq_len, num_units, hidden_dim]
        output = torch.stack(outputs, dim=1)
        
        # 求和 num_units 维度 → [batch, seq_len, hidden_dim]
        output = output.sum(dim=2)
        
        return output
```

---

## 4. 温态注意力机制

### 4.1 概念

**温态**是 Evola 在无外部输入时的后台运行状态。温态注意力机制允许注意力模块在此状态下继续活动。

### 4.2 实现

```python
    def warm_mode_step(
        self,
        memory_replay_buffer: Tensor,
        homeostasis_state: HomeostasisVariables,
    ) -> WarmModeActivity:
        """
        温态运行：无外部输入时的注意力活动
        
        主要功能：
        1. 记忆回放（replay）
        2. 预测误差最小化
        3. 随机探索
        
        Args:
            memory_replay_buffer: 待回放记忆 [batch, replay_len, dim]
            homeostasis_state: 当前内稳态状态
            
        Returns:
            activity: 温态活动记录
        """
        # 1. 自发记忆回放
        replay_output, replay_weights = self.forward(
            x=memory_replay_buffer,
            homeostasis_state=homeostasis_state,
            return_weights=True,
        )
        
        # 2. 分析注意力模式（元认知）
        attention_entropy = compute_attention_entropy(replay_weights)
        
        # 3. 更新突触强度（基于回放）
        self._synaptic_consolidation(replay_output)
        
        return WarmModeActivity(
            replay_output=replay_output,
            attention_entropy=attention_entropy,
        )
```

---

## 5. 多尺度注意力（可选扩展）

### 5.1 设计思想

标准注意力对所有 token 一视同仁。多尺度注意力引入**时间尺度的层次性**：

- **快速尺度**: 处理局部、即时信息
- **慢速尺度**: 整合长期、全局上下文

### 5.2 实现草图

```python
class MultiScaleAttention(nn.Module):
    """多尺度注意力"""
    
    def __init__(self, dim: int, num_scales: int = 3):
        super().__init__()
        
        self.num_scales = num_scales
        self.attention_heads = nn.ModuleList([
            HomeostasisModulatedAttention(
                dim=dim,
                num_heads=8,
                liquid_time_constant=True,
            )
            for _ in range(num_scales)
        ])
        
        # 时间常数（不同尺度有不同的 tau）
        self.time_constants = nn.Parameter(torch.tensor([0.1, 0.5, 1.0]))
        
        # 跨尺度整合
        self.cross_scale_integration = nn.Linear(dim * num_scales, dim)
    
    def forward(
        self,
        x: Tensor,
        homeostasis_state: HomeostasisVariables,
    ) -> Tensor:
        """
        多尺度注意力前向传播
        """
        scale_outputs = []
        
        for scale_idx, attention in enumerate(self.attention_heads):
            # 不同尺度使用不同的时间步长
            dt = self.time_constants[scale_idx]
            
            # 应用注意力
            output = attention(x, homeostasis_state)
            scale_outputs.append(output)
        
        # 拼接所有尺度并整合
        combined = torch.cat(scale_outputs, dim=-1)
        integrated = self.cross_scale_integration(combined)
        
        return integrated
```

---

## 6. API 接口设计

### 6.1 公共方法

| 方法 | 签名 | 用途 |
|------|------|------|
| `forward` | `(x, homeostasis_state, mask, past_kv) → output` | 标准前向传播 |
| `warm_mode_step` | `(memory_buffer, homeostasis_state) → activity` | 温态运行 |
| `get_attention_weights` | `(x, homeostasis_state) → weights` | 获取注意力权重（分析） |
| `update_time_constant` | `(tau: float)` | 动态调整时间常数 |

### 6.2 配置示例

```python
# 标准配置（用于热态交互）
hot_attention = HomeostasisModulatedAttention(
    dim=512,
    num_heads=8,
    causal=True,
    liquid_time_constant=True,
)

# 温态配置（用于后台记忆整理）
warm_attention = HomeostasisModulatedAttention(
    dim=512,
    num_heads=8,
    causal=False,  # 温态可访问未来信息
    liquid_time_constant=True,
)

# 使用示例
output = hot_attention(
    x=input_tensor,
    homeostasis_state=current_homeostasis,
    mask=causal_mask,
)
```

---

## 7. 训练策略

### 7.1 损失函数

注意力模块的训练损失包括：

```python
class AttentionLoss(nn.Module):
    """注意力模块损失函数"""
    
    def __init__(self):
        super().__init__()
        self.task_loss = nn.CrossEntropyLoss()  # 主任务损失
        self.sparsity_loss = SparsityLoss()     # 注意力稀疏性正则
        self.stability_loss = StabilityLoss()   # 稳定性正则
    
    def forward(
        self,
        output: Tensor,
        target: Tensor,
        attn_weights: Tensor,
    ) -> LossDict:
        # 1. 主任务损失
        task_loss = self.task_loss(output, target)
        
        # 2. 注意力稀疏性正则（鼓励聚焦）
        sparse_reg = self.sparsity_loss(attn_weights)
        
        # 3. 稳定性正则（防止剧烈波动）
        stability_reg = self.stability_loss(attn_weights)
        
        return {
            'total': task_loss + 0.01 * sparse_reg + 0.01 * stability_reg,
            'task': task_loss,
            'sparsity': sparse_reg,
            'stability': stability_reg,
        }
```

### 7.2 优化器配置

```python
def configure_optimizer(model: nn.Module, lr: float = 1e-4) -> torch.optim.Optimizer:
    """配置优化器"""
    
    # 分离权重衰减参数
    weight_decay_params = []
    no_weight_decay_params = []
    
    for name, param in model.named_parameters():
        if 'bias' in name or 'time_constant' in name:
            no_weight_decay_params.append(param)
        else:
            weight_decay_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': weight_decay_params, 'weight_decay': 0.1},
        {'params': no_weight_decay_params, 'weight_decay': 0.0},
    ], lr=lr)
    
    return optimizer
```

---

## 8. 测试策略

### 8.1 单元测试

```python
class TestHomeostasisModulatedAttention(unittest.TestCase):
    
    def test_basic_forward(self):
        """测试基本前向传播"""
        attention = HomeostasisModulatedAttention(dim=128, num_heads=4)
        x = torch.randn(2, 10, 128)
        homeostasis = HomeostasisVariables(
            information_entropy=0.5,
            prediction_error=0.2,
            memory_load=0.3,
            energy_level=0.8,
            social_connection=0.5,
        )
        
        output = attention(x, homeostasis)
        self.assertEqual(output.shape, (2, 10, 128))
    
    def test_causal_masking(self):
        """测试因果掩码"""
        attention = HomeostasisModulatedAttention(
            dim=128, num_heads=4, causal=True
        )
        x = torch.randn(1, 5, 128)
        homeostasis = HomeostasisVariables(...)
        
        _, weights = attention(x, homeostasis, return_weights=True)
        
        # 检查未来位置是否被屏蔽
        self.assertTrue(torch.all(weights[:, :, 0, 1:] == 0))
    
    def test_homeostasis_modulation(self):
        """测试内稳态调制效果"""
        attention = HomeostasisModulatedAttention(dim=128, num_heads=4)
        x = torch.randn(1, 10, 128)
        
        # 高好奇状态
        high_curiosity = HomeostasisVariables(
            information_entropy=0.9,
            prediction_error=0.1,
            memory_load=0.3,
            energy_level=0.8,
            social_connection=0.5,
        )
        
        # 低好奇状态
        low_curiosity = HomeostasisVariables(
            information_entropy=0.1,
            prediction_error=0.1,
            memory_load=0.3,
            energy_level=0.8,
            social_connection=0.5,
        )
        
        out_high, weights_high = attention(x, high_curiosity, return_weights=True)
        out_low, weights_low = attention(x, low_curiosity, return_weights=True)
        
        # 注意力分布应有所不同
        self.assertFalse(torch.allclose(weights_high, weights_low))
```

---

## 9. 性能基准

### 9.1 预期性能指标

| 指标 | 目标值 | 测量方法 |
|------|--------|----------|
| **推理延迟** | < 10ms / token | 单 GPU, batch=1 |
| **训练吞吐量** | > 1000 tokens/s | 多 GPU, batch=32 |
| **内存效率** | < 2GB / 1k seq_len | 峰值内存使用 |
| **注意力稀疏度** | > 70% 零权重 | 平均稀疏度 |

### 9.2 与标准 Attention 的对比

```python
# 性能对比实验设计
import time

def benchmark_attention():
    x = torch.randn(32, 512, 512).cuda()  # batch=32, seq=512, dim=512
    
    # 标准自注意力
    standard_attn = StandardSelfAttention(dim=512, num_heads=8).cuda()
    
    # Evola 注意力
    evola_attn = HomeostasisModulatedAttention(
        dim=512, num_heads=8
    ).cuda()
    
    # 预热
    for _ in range(10):
        standard_attn(x)
        evola_attn(x, homeostasis_state)
    
    # 测量
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        standard_attn(x)
    torch.cuda.synchronize()
    standard_time = time.time() - t0
    
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        evola_attn(x, homeostasis_state)
    torch.cuda.synchronize()
    evola_time = time.time() - t0
    
    print(f"标准注意力：{standard_time/100*1000:.2f} ms")
    print(f"Evola 注意力：{evola_time/100*1000:.2f} ms")
    # 预期：Evola 慢 10-20%（由于内稳态调制）
```

---

## 10. 与其他模块的接口

### 10.1 输入依赖

| 来源模块 | 提供数据 | 用途 |
|----------|---------|------|
| **Homeostasis** | `HomeostasisVariables` | 注意力调制 |
| **Memory** | `past_kv` | 历史上下文 |
| **SelfModel** | `attention_focus_preference` | 聚焦偏好 |

### 10.2 输出去向

| 目标模块 | 接收数据 | 用途 |
|----------|---------|------|
| **Memory** | `new_kv` | 更新 KV 缓存 |
| **Homeostasis** | `attention_entropy` | 信息摄入监测 |
| **WorldModel** | `attended_representation` | 世界状态预测 |
| **SelfModel** | `attention_patterns` | 自我反思 |

---

## 11. 实现路线图

| 阶段 | 任务 | 预计工时 | 状态 |
|------|------|----------|------|
| **Phase 1** | 基础 Attention 实现 | 1 周 | ⏳ 待开始 |
| **Phase 1** | HomeostasisModulator 实现 | 1 周 | ⏳ 待开始 |
| **Phase 1** | 单元测试 | 3 天 | ⏳ 待开始 |
| **Phase 2** | LTC 单元实现 | 2 周 | ⏳ 待开始 |
| **Phase 2** | 温态注意力机制 | 1 周 | ⏳ 待开始 |
| **Phase 3** | 多尺度注意力（可选） | 2 周 | ⏳ 待开始 |

---

## 12. 开放问题与研究笔记

### 12.1 技术开放问题

1. **LTC 的长期依赖问题**
   - 连续时间 RNN 的梯度消失问题如何解决？
   - 可能的方案：门控机制 + 残差连接

2. **内稳态调制的可解释性**
   - 如何将"好奇"等抽象概念映射为具体的注意力权重？
   - 需要实验验证调制策略的有效性

3. **温态注意力的训练**
   - 温态活动没有外部监督信号，如何训练？
   - 可能的方案：自监督 + 预测误差最小化

### 12.2 实验建议

1. **消融实验**
   - 移除内稳态调制 → 性能下降多少？
   - 移除 LTC → 时间建模能力是否受损？

2. **可视化分析**
   - 不同内稳态状态下的注意力模式差异
   - 温态活动的内容特征

---

## 13. 参考文献

1. Vaswani et al. (2017). ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
2. Hasani et al. (2021). ["Liquid Time-constant Networks"](https://aaai.org/papers/07657-liquid-time-constant-networks/)
3. Touvron et al. (2023). ["LLaMA: Open and Efficient Foundation Language Models"](https://arxiv.org/abs/2302.13971)
4. Dao et al. (2022). ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"](https://arxiv.org/abs/2205.14135)

---

*注意力模块规格 完成日期：2026 年 4 月 15 日*
