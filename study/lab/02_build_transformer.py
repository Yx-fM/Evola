"""
第 2 课实验：从零搭建 Transformer Block

目标：
1. 实现 Self-Attention 模块
2. 实现 Multi-Head Attention
3. 实现 Transformer Block（含残差连接和归一化）
4. 验证因果掩码的正确性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.set_printoptions(precision=3, sci_mode=False)


# ============================================
# 1. Self-Attention 模块
# ============================================

class SelfAttention(nn.Module):
    """
    自注意力模块
    
    核心公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(self, dim, causal=False):
        super().__init__()
        
        self.dim = dim
        self.causal = causal  # 是否使用因果掩码
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        # 输出投影
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        # 缩放因子
        self.scale = dim ** -0.5
    
    def forward(self, x, mask=None):
        """
        Args:
            x: 输入 [batch, seq_len, dim]
            mask: 注意力掩码 [seq_len, seq_len]
            
        Returns:
            output: 注意力输出 [batch, seq_len, dim]
            attn_weights: 注意力权重 [batch, seq_len, seq_len]
        """
        batch_size, seq_len, dim = x.shape
        
        # 1. 计算 Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, dim]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. 注意力打分
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # [batch, seq_len, seq_len]
        
        # 3. 应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        if self.causal:
            # 生成因果掩码：只能看到过去，不能看到未来
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # 4. Softmax 获取注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 5. 加权求和
        output = torch.matmul(attn_weights, v)
        
        # 6. 输出投影
        output = self.o_proj(output)
        
        return output, attn_weights


# ============================================
# 2. Multi-Head Attention
# ============================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力
    
    将 dim 分成 num_heads 个独立的头，每个头独立计算注意力
    """
    
    def __init__(self, dim, num_heads, causal=False):
        super().__init__()
        
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        
        # Q, K, V 投影（一次性投影到所有头）
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        # 输出投影
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        """
        Args:
            x: 输入 [batch, seq_len, dim]
            mask: 注意力掩码
            
        Returns:
            output: [batch, seq_len, dim]
            attn_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, dim = x.shape
        
        # 1. Q, K, V 投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. 重塑为多头形式
        # [batch, seq_len, dim] → [batch, seq_len, num_heads, head_dim] → [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. 注意力打分
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # [batch, num_heads, seq_len, seq_len]
        
        # 4. 应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # 5. Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 6. 加权求和
        attn_output = torch.matmul(attn_weights, v)
        # [batch, num_heads, seq_len, head_dim]
        
        # 7. 拼接所有头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, dim)
        
        # 8. 输出投影
        output = self.o_proj(attn_output)
        
        return output, attn_weights


# ============================================
# 3. Feed-Forward Network (MLP)
# ============================================

class FeedForward(nn.Module):
    """
    前馈网络
    
    结构: dim → 4*dim → 激活 → dim
    """
    
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        
        hidden_dim = hidden_dim or 4 * dim
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


# ============================================
# 4. Layer Normalization
# ============================================

class LayerNorm(nn.Module):
    """
    Layer Normalization
    """
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        # 计算均值和方差（沿着最后一维）
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和偏移
        return self.gamma * x_norm + self.beta


# ============================================
# 5. Transformer Block
# ============================================

class TransformerBlock(nn.Module):
    """
    Transformer Block
    
    Pre-Norm 结构:
      x → Norm → Attention → + → Norm → FFN → +
      │                                   │
      └─────── 残差连接 ───────────────────┘
    """
    
    def __init__(self, dim, num_heads, causal=False):
        super().__init__()
        
        # Pre-Norm Attention
        self.norm1 = LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, num_heads, causal=causal)
        
        # Pre-Norm FFN
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, dim]
            
        Returns:
            output: [batch, seq_len, dim]
        """
        # 1. Attention 子层 + 残差
        attn_out, attn_weights = self.attention(self.norm1(x), mask)
        x = x + attn_out
        
        # 2. FFN 子层 + 残差
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x, attn_weights


# ============================================
# 6. 简化 Transformer (多层堆叠)
# ============================================

class SimpleTransformer(nn.Module):
    """
    简化的 Decoder-Only Transformer
    
    结构:
      Embedding → [Block × N] → Norm → Linear → Softmax
    """
    
    def __init__(self, vocab_size, dim, num_heads, num_layers, max_seq_len=512):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        
        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Position Embedding (Learnable)
        self.pos_embedding = nn.Embedding(max_seq_len, dim)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, causal=True)
            for _ in range(num_layers)
        ])
        
        # 最终 Layer Norm
        self.final_norm = LayerNorm(dim)
        
        # 输出层
        self.output_proj = nn.Linear(dim, vocab_size)
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch, seq_len] token 索引
            
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Token + Position Embedding
        token_emb = self.embedding(input_ids)
        pos_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(pos_ids)
        
        x = token_emb + pos_emb
        
        # 2. 通过所有 Transformer Blocks
        all_attn_weights = []
        for block in self.blocks:
            x, attn_weights = block(x)
            all_attn_weights.append(attn_weights)
        
        # 3. 最终 Layer Norm
        x = self.final_norm(x)
        
        # 4. 输出投影
        logits = self.output_proj(x)
        
        return logits, all_attn_weights


# ============================================
# 7. 实验：验证因果掩码
# ============================================

def test_causal_mask():
    """验证因果掩码的正确性"""
    print("\n=== 实验：验证因果掩码 ===\n")
    
    attn = SelfAttention(dim=64, causal=True)
    
    # 输入序列
    x = torch.randn(1, 5, 64)
    
    # 前向传播
    output, weights = attn(x)
    
    print("输入序列长度: 5")
    print("\n注意力权重矩阵 (因果掩码):")
    print("        位置0  位置1  位置2  位置3  位置4")
    
    for i, row in enumerate(weights[0]):
        row_str = " ".join([f"{v:.3f}" for v in row])
        print(f"位置{i}:  {row_str}")
    
    print("\n【导师解读】")
    print("因果掩码的效果:")
    print("- 位置0 只能看到自己 (权重只在位置0)")
    print("- 位置1 能看到位置0和1")
    print("- 位置2 能看到位置0, 1, 2")
    print("- 位置4 能看到所有位置")
    
    # 验证：未来位置应该为 0
    for i in range(5):
        for j in range(i + 1, 5):
            assert weights[0, i, j].item() == 0.0, f"位置{i}不应该看到位置{j}"
    
    print("\n✅ 因果掩码验证通过！")


# ============================================
# 8. 实验：多头注意力可视化
# ============================================

def visualize_multihead_attention():
    """可视化多头注意力"""
    print("\n=== 实验：多头注意力可视化 ===\n")
    
    mha = MultiHeadAttention(dim=64, num_heads=4, causal=True)
    
    # 输入序列（模拟）
    x = torch.randn(1, 8, 64)
    
    output, attn_weights = mha(x)
    
    print(f"配置: dim=64, num_heads=4, head_dim=16")
    print(f"序列长度: 8")
    
    print("\n各头的注意力权重:")
    
    for head_idx in range(4):
        print(f"\n--- Head {head_idx} ---")
        weights = attn_weights[0, head_idx]
        
        # 只显示前 4 个位置
        print("        位置0  位置1  位置2  位置3")
        for i in range(4):
            row_str = " ".join([f"{v:.2f}" for v in weights[i, :4]])
            print(f"位置{i}:  {row_str}")
    
    print("\n【导师解读】")
    print("多头注意力的差异:")
    print("- 不同头关注不同的关系模式")
    print("- Head 0 可能关注语法关系")
    print("- Head 1 可能关注语义关系")
    print("- 这就是多头'分工合作'的效果")


# ============================================
# 9. 实验：完整 Transformer Block
# ============================================

def test_transformer_block():
    """测试完整 Transformer Block"""
    print("\n=== 实验：完整 Transformer Block ===\n")
    
    block = TransformerBlock(dim=128, num_heads=8, causal=True)
    
    # 输入
    x = torch.randn(2, 10, 128)
    
    # 前向传播
    output, attn_weights = block(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 验证形状
    assert output.shape == x.shape, "输出形状应该与输入相同"
    
    print("\nTransformer Block 结构:")
    print("1. Layer Norm → Multi-Head Attention → 残差连接")
    print("2. Layer Norm → Feed-Forward → 残差连接")
    
    print("\n✅ Transformer Block 验证通过！")


# ============================================
# 10. 主实验运行
# ============================================

def run_experiment():
    """运行所有实验"""
    print("=" * 60)
    print("第 2 课实验：从零搭建 Transformer Block")
    print("=" * 60)
    
    # 实验 1: 因果掩码验证
    test_causal_mask()
    
    # 实验 2: 多头注意力可视化
    visualize_multihead_attention()
    
    # 实验 3: Transformer Block
    test_transformer_block()
    
    print("\n" + "=" * 60)
    print("所有实验完成！")
    print("=" * 60)
    
    print("\n【课后思考】")
    print("1. 尝试修改 num_heads，观察不同头数的效果")
    print("2. 尝试增加 num_layers，观察多层堆叠的输出")
    print("3. 思考：如果把这个 Block 用于 Evola，应该如何改造？")


if __name__ == "__main__":
    run_experiment()