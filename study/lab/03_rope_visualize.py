"""
第 3 课实验：位置编码可视化 (RoPE)

目标：
1. 可视化 Sinusoidal PE 的波形
2. 实现 RoPE 旋转位置编码
3. 对比不同位置编码的效果
4. 观察相对位置保持特性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(precision=3, sci_mode=False)


# ============================================
# 1. Sinusoidal Position Encoding
# ============================================

class SinusoidalPositionEncoding(nn.Module):
    """
    Sinusoidal 位置编码 (Transformer 原始方案)
    
    公式:
      PE(pos, 2i)   = sin(pos / 10000^(2i/d))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """
    
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # 预计算位置编码
        pe = torch.zeros(max_seq_len, dim)
        
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        
        # 注册为 buffer（不参与训练）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: 输入 [batch, seq_len, dim]
            
        Returns:
            x + PE: 加入位置编码后的输入
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len].unsqueeze(0)


# ============================================
# 2. RoPE (Rotary Position Embedding)
# ============================================

class RotaryPositionEmbedding(nn.Module):
    """
    RoPE - 旋转位置编码
    
    通过旋转向量来编码位置，保持相对位置信息
    
    公式:
      rotate(x, θ) = [x1*cos - x2*sin, x1*sin + x2*cos]
      θ = pos * freq
    """
    
    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # 计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算 cos 和 sin
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        
        # 重复以匹配维度
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def rotate_half(self, x):
        """
        旋转一半维度
        
        [x1, x2, x3, x4] → [-x2, x1, -x4, x3]
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rotary_pos_emb(self, x, seq_len):
        """
        应用旋转位置编码
        
        Args:
            x: 输入向量 [batch, seq_len, dim] 或 [batch, heads, seq_len, head_dim]
            seq_len: 序列长度
        """
        cos = self.cos_cached[:seq_len].unsqueeze(0)  # [1, seq_len, dim]
        sin = self.sin_cached[:seq_len].unsqueeze(0)
        
        # 广播到 batch
        if x.dim() == 4:
            cos = cos.unsqueeze(1)  # [1, 1, seq_len, dim]
            sin = sin.unsqueeze(1)
        
        # 旋转公式
        x_rotated = (x * cos) + (self.rotate_half(x) * sin)
        
        return x_rotated
    
    def forward(self, q, k):
        """
        对 Q 和 K 应用 RoPE
        
        Args:
            q: Query [batch, seq_len, dim] 或 [batch, heads, seq_len, head_dim]
            k: Key
            
        Returns:
            q_rotated, k_rotated
        """
        seq_len = q.shape[-2]
        
        q_rotated = self.apply_rotary_pos_emb(q, seq_len)
        k_rotated = self.apply_rotary_pos_emb(k, seq_len)
        
        return q_rotated, k_rotated


# ============================================
# 3. 实验可视化函数
# ============================================

def visualize_sinusoidal_pe():
    """可视化 Sinusoidal PE 的波形"""
    print("\n=== 实验：可视化 Sinusoidal PE ===\n")
    
    dim = 64
    max_seq_len = 100
    
    pe = SinusoidalPositionEncoding(dim, max_seq_len)
    
    try:
        plt.figure(figsize=(12, 8))
        
        # 子图 1: 选择几个维度展示波形
        plt.subplot(2, 1, 1)
        
        dims_to_plot = [0, 1, 2, 3, 20, 21]
        for d in dims_to_plot:
            plt.plot(pe.pe[:50, d].numpy(), label=f'维度 {d}')
        
        plt.xlabel('位置')
        plt.ylabel('PE 值')
        plt.title('Sinusoidal PE: 不同维度的波形')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # 子图 2: 所有维度的热图
        plt.subplot(2, 1, 2)
        plt.imshow(pe.pe[:50, :32].numpy().T, cmap='RdBu', aspect='auto')
        plt.xlabel('位置')
        plt.ylabel('维度')
        plt.title('Sinusoidal PE 热图 (前 32 维)')
        plt.colorbar(label='PE 值')
        
        plt.savefig('sinusoidal_pe.png', dpi=100)
        print("✅ 图像已保存: sinusoidal_pe.png")
        
        plt.show()
    except Exception as e:
        print(f"⚠️ 无法显示图像: {e}")
        
        # 文本输出
        print("\n位置编码示例 (位置 0-9, 维度 0-3):")
        print("        dim0    dim1    dim2    dim3")
        for pos in range(10):
            vals = " ".join([f"{pe.pe[pos, d].item():.3f}" for d in range(4)])
            print(f"pos{pos}:  {vals}")
    
    print("\n【导师解读】")
    print("Sinusoidal PE 的特点:")
    print("- 不同维度有不同的频率（波长）")
    print("- 低维度: 高频 → 快速变化的波形")
    print("- 高维度: 低频 → 缓慢变化的波形")
    print("- 这样不同位置有独特的'签名'")


def visualize_rope_rotation():
    """可视化 RoPE 的旋转效果"""
    print("\n=== 实验：可视化 RoPE 旋转 ===\n")
    
    dim = 32
    
    rope = RotaryPositionEmbedding(dim, max_seq_len=50)
    
    # 创建一个向量
    x = torch.ones(1, 10, dim)
    
    # 应用 RoPE
    x_rotated = rope.apply_rotary_pos_emb(x, 10)
    
    try:
        plt.figure(figsize=(12, 6))
        
        # 选择两个维度展示旋转
        dim1, dim2 = 0, dim // 2  # 对应的一对
        
        original_x = x[0, :, dim1].numpy()
        original_y = x[0, :, dim2].numpy()
        rotated_x = x_rotated[0, :, dim1].numpy()
        rotated_y = x_rotated[0, :, dim2].numpy()
        
        # 绘制旋转轨迹
        plt.scatter(original_x, original_y, c='blue', s=50, label='原始向量')
        plt.scatter(rotated_x, rotated_y, c='red', s=50, label='旋转后')
        
        # 连接每个位置的旋转
        for i in range(10):
            plt.plot([original_x[i], rotated_x[i]], 
                     [original_y[i], rotated_y[i]], 
                     'gray', alpha=0.5, linestyle='--')
            plt.annotate(f'pos{i}', (rotated_x[i], rotated_y[i]), fontsize=8)
        
        plt.xlabel(f'维度 {dim1}')
        plt.ylabel(f'维度 {dim2}')
        plt.title('RoPE: 向量随位置的旋转')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.savefig('rope_rotation.png', dpi=100)
        print("✅ 图像已保存: rope_rotation.png")
        
        plt.show()
    except Exception as e:
        print(f"⚠️ 无法显示图像: {e}")
    
    print("\n【导师解读】")
    print("RoPE 的旋转原理:")
    print("- 每个位置的向量被旋转一定角度")
    print("- 位置 0: θ=0 (不旋转)")
    print("- 位置 1: θ=1×freq")
    print("- 位置 2: θ=2×freq")
    print("- 相邻位置的旋转角度差 = freq")
    print("- 这使得相对位置信息被编码")


def compare_pe_methods():
    """对比不同位置编码方法"""
    print("\n=== 实验：对比 PE 方法 ===\n")
    
    dim = 64
    
    # Sinusoidal PE
    sin_pe = SinusoidalPositionEncoding(dim)
    
    # Learnable PE
    learnable_pe = nn.Embedding(50, dim)
    
    # RoPE
    rope = RotaryPositionEmbedding(dim)
    
    print("三种位置编码方法:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("| 方法          | 特点                    |")
    print("| Sinusoidal    | 固定，可无限长          |")
    print("| Learnable     | 可学习，长度受限        |")
    print("| RoPE          | 旋转，相对位置保持      |")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    try:
        plt.figure(figsize=(12, 6))
        
        # 计算不同位置的相似度
        positions = [0, 10, 20, 30]
        
        for method_name, pe_values in [
            ('Sinusoidal', sin_pe.pe[:40, :8]),
            ('Learnable', learnable_pe.weight[:40, :8]),
        ]:
            # 计算位置间的余弦相似度
            similarity = F.cosine_similarity(
                pe_values.unsqueeze(1), 
                pe_values.unsqueeze(0), 
                dim=-1
            )
            
            plt.subplot(1, 2, 1 if method_name == 'Sinusoidal' else 2)
            plt.imshow(similarity.numpy(), cmap='RdBu', aspect='auto')
            plt.title(f'{method_name} PE: 位置相似度')
            plt.xlabel('位置')
            plt.ylabel('位置')
            plt.colorbar()
        
        plt.savefig('pe_comparison.png', dpi=100)
        print("✅ 图像已保存: pe_comparison.png")
        
        plt.show()
    except Exception as e:
        print(f"⚠️ 无法显示图像: {e}")
    
    print("\n【导师解读】")
    print("LLaMA 为什么选择 RoPE?")
    print("- 相对位置保持: Q_i · K_j 只取决于 (i-j)")
    print("- 无限长度: 不需要固定 max_seq_len")
    print("- 更好的泛化: 相同相对距离关系一致")


def test_relative_position():
    """测试 RoPE 的相对位置保持特性"""
    print("\n=== 实验：相对位置保持测试 ===\n")
    
    dim = 64
    
    rope = RotaryPositionEmbedding(dim, max_seq_len=100)
    
    # 创建 Q, K
    q = torch.randn(1, 1, dim)  # 单个 Query
    k = torch.randn(1, 100, dim)  # 100 个 Key
    
    # 应用 RoPE
    q_rotated, k_rotated = rope(q.expand(1, 100, dim), k)
    
    # 计算注意力分数
    scores = torch.matmul(q_rotated.squeeze(0), k_rotated.squeeze(0).T)
    
    print("测试相对位置保持:")
    print("- 对于固定 Query 位置和不同 Key 位置")
    print("- 相对距离相同的对应该有相似的注意力分数")
    
    # 提取相对距离为 5 的所有对的分数
    print("\n相对距离 = 5 的所有对的分数:")
    for q_pos in range(10, 90):
        # 获取 q 在位置 q_pos，k 在位置 q_pos+5 的分数
        score = scores[q_pos, q_pos + 5].item()
        print(f"  Q在{q_pos}, K在{q_pos+5}: {score:.4f}")
    
    print("\n【导师解读】")
    print("RoPE 的相对位置保持:")
    print("- 相对距离相同的对，注意力分数相近")
    print("- 这使得模型对'相对关系'的理解一致")
    print("- 例如：'位置1和5'的关系 = '位置100和104'的关系")


# ============================================
# 4. 主实验运行
# ============================================

def run_experiment():
    """运行所有实验"""
    print("=" * 60)
    print("第 3 课实验：位置编码可视化")
    print("=" * 60)
    
    # 实验 1: Sinusoidal PE
    visualize_sinusoidal_pe()
    
    # 实验 2: RoPE 旋转
    visualize_rope_rotation()
    
    # 实验 3: PE 方法对比
    compare_pe_methods()
    
    # 实验 4: 相对位置保持
    test_relative_position()
    
    print("\n" + "=" * 60)
    print("所有实验完成！")
    print("=" * 60)
    
    print("\n【课后思考】")
    print("1. RoPE 为什么能保持相对位置？（数学推导）")
    print("2. 如果把'位置'改为'连续时间'，如何设计 Evola 的 PE?")
    print("3. 内稳态循环如何定义'时间'的概念？")


if __name__ == "__main__":
    run_experiment()