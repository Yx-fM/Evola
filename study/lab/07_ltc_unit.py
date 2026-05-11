"""
第 7 课实验：液态神经网络 (LTC) 单元

目标：
1. 实现基础 LTC 单元
2. 可视化连续时间状态演化
3. 对比不同 τ 值的记忆曲线
4. 实现内稳态调制的 LTC（Evola 核心）
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

torch.set_printoptions(precision=3, sci_mode=False)


# ============================================
# 1. 基础 LTC 单元
# ============================================

class LTCCell(nn.Module):
    """
    液态时间常数 (LTC) 单元
    
    微分方程:
      dh/dt = -h / τ + σ(x·W + h·U) * (1 - h)
    
    其中:
      h: 隐藏状态
      τ: 时间常数（决定衰减速度）
      σ: 激活函数
      (1-h): 饱和因子
    """
    
    def __init__(self, input_dim, hidden_dim, tau_init=1.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入权重
        self.W = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # 递归权重
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 时间常数 τ（可学习）
        self.tau = nn.Parameter(torch.ones(hidden_dim) * tau_init)
        
        # 激活函数
        self.sigma = nn.Tanh()
    
    def step(self, x, h, dt=0.1):
        """
        单步 LTC 更新
        
        Args:
            x: 输入 [input_dim]
            h: 当前状态 [hidden_dim]
            dt: 时间步长
            
        Returns:
            h_new: 新状态 [hidden_dim]
        """
        # LTC 微分方程
        # dh/dt = -h/τ + σ(x·W + h·U) * (1-h)
        
        input_drive = self.sigma(self.W(x) + self.U(h))
        decay = h / self.tau
        
        dh_dt = -decay + input_drive * (1 - h)
        
        # 欧拉积分
        h_new = h + dt * dh_dt
        
        return h_new
    
    def forward(self, x_sequence, h0=None, dt=0.1, return_history=True):
        """
        处理整个输入序列
        
        Args:
            x_sequence: 输入序列 [seq_len, input_dim]
            h0: 初始状态 [hidden_dim]
            dt: 时间步长
            return_history: 是否返回历史
            
        Returns:
            h_final: 最终状态 [hidden_dim]
            h_history: 状态历史 [seq_len, hidden_dim] (可选)
        """
        seq_len = x_sequence.shape[0]
        
        if h0 is None:
            h0 = torch.zeros(self.hidden_dim)
        
        h = h0
        h_history = []
        
        for t in range(seq_len):
            x_t = x_sequence[t]
            h = self.step(x_t, h, dt)
            
            if return_history:
                h_history.append(h.clone())
        
        if return_history:
            return h, torch.stack(h_history)
        else:
            return h


# ============================================
# 2. 动态时间常数 LTC
# ============================================

class DynamicTauLTCCell(nn.Module):
    """
    动态时间常数 LTC
    
    τ 不再是固定值，而是随输入动态变化
    
    τ(t) = τ_base * f(x(t))
    
    这实现了:
      - 输入变化剧烈 → τ 小 → 快速响应
      - 输入平稳 → τ 大 → 长期记忆
    """
    
    def __init__(self, input_dim, hidden_dim, tau_base=1.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LTC 核心参数
        self.W = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 基础时间常数
        self.tau_base = nn.Parameter(torch.ones(hidden_dim) * tau_base)
        
        # 动态 τ 调制器
        self.tau_modulator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),  # 输出 0-1，用于调制 τ
        )
        
        self.sigma = nn.Tanh()
    
    def step(self, x, h, dt=0.1):
        """动态 τ 的单步更新"""
        
        # 计算动态 τ
        tau_modulation = self.tau_modulator(x)
        tau = self.tau_base * (0.5 + tau_modulation)  # τ 在 [0.5*τ_base, 1.5*τ_base] 范围
        
        # LTC 微分方程
        input_drive = self.sigma(self.W(x) + self.U(h))
        decay = h / tau
        
        dh_dt = -decay + input_drive * (1 - h)
        
        h_new = h + dt * dh_dt
        
        return h_new, tau
    
    def forward(self, x_sequence, h0=None, dt=0.1, return_history=True):
        """处理序列"""
        
        seq_len = x_sequence.shape[0]
        
        if h0 is None:
            h0 = torch.zeros(self.hidden_dim)
        
        h = h0
        h_history = []
        tau_history = []
        
        for t in range(seq_len):
            x_t = x_sequence[t]
            h, tau = self.step(x_t, h, dt)
            
            if return_history:
                h_history.append(h.clone())
                tau_history.append(tau.clone())
        
        if return_history:
            return h, torch.stack(h_history), torch.stack(tau_history)
        else:
            return h


# ============================================
# 3. 内稳态调制 LTC（Evola 核心）
# ============================================

class HomeostasisModulatedLTC(nn.Module):
    """
    内稳态调制的 LTC
    
    这是 Evola 基础层的核心组件
    
    τ 受内稳态变量调制:
      - 高能量 → τ 大 → 记忆持久
      - 低能量 → τ 小 → 快速遗忘
      - 高预测误差 → τ 小 → 快速响应
      - 低预测误差 → τ 大 → 保持稳定
    """
    
    def __init__(self, input_dim, hidden_dim, homeostasis_dim=6):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LTC 核心参数
        self.W = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 基础时间常数
        self.tau_base = nn.Parameter(torch.ones(hidden_dim) * 1.0)
        
        # 内稳态调制器
        # 输入: 6 个内稳态变量
        # 输出: τ 的调制因子
        self.homeostasis_modulator = nn.Sequential(
            nn.Linear(homeostasis_dim, hidden_dim),
            nn.Tanh(),  # 输出 -1 到 1
        )
        
        self.sigma = nn.Tanh()
    
    def step(self, x, h, homeostasis_state, dt=0.1):
        """
        Args:
            x: 输入 [input_dim]
            h: 当前状态 [hidden_dim]
            homeostasis_state: 内稳态向量 [homeostasis_dim]
            dt: 时间步长
        """
        
        # 内稳态调制 τ
        # 正向调制 → τ 增大 → 记忆持久
        # 负向调制 → τ 减小 → 快速遗忘
        tau_modulation = self.homeostasis_modulator(homeostasis_state)
        tau = self.tau_base * (1.0 + tau_modulation)
        tau = torch.clamp(tau, min=0.1)  # τ 至少为 0.1
        
        # LTC 微分方程
        input_drive = self.sigma(self.W(x) + self.U(h))
        decay = h / tau
        
        dh_dt = -decay + input_drive * (1 - h)
        
        h_new = h + dt * dh_dt
        
        return h_new, tau
    
    def forward(self, x_sequence, homeostasis_sequence=None, h0=None, dt=0.1):
        """
        Args:
            x_sequence: 输入序列 [seq_len, input_dim]
            homeostasis_sequence: 内稳态序列 [seq_len, homeostasis_dim]
            h0: 初始状态
            dt: 时间步长
        """
        
        seq_len = x_sequence.shape[0]
        
        if h0 is None:
            h0 = torch.zeros(self.hidden_dim)
        
        if homeostasis_sequence is None:
            # 默认内稳态（平衡状态）
            homeostasis_sequence = torch.zeros(seq_len, 6)
        
        h = h0
        h_history = []
        tau_history = []
        
        for t in range(seq_len):
            x_t = x_sequence[t]
            h_state = homeostasis_sequence[t]
            
            h, tau = self.step(x_t, h, h_state, dt)
            
            h_history.append(h.clone())
            tau_history.append(tau.clone())
        
        return h, torch.stack(h_history), torch.stack(tau_history)


# ============================================
# 4. 实验：可视化 LTC 状态演化
# ============================================

def visualize_ltc_evolution():
    """可视化 LTC 的状态演化曲线"""
    print("\n=== 实验：可视化 LTC 状态演化 ===\n")
    
    ltc = LTCCell(input_dim=1, hidden_dim=10, tau_init=1.0)
    
    # 创建输入序列
    # 模拟脉冲输入：前半部分有输入，后半部分无输入
    seq_len = 100
    x_sequence = torch.zeros(seq_len, 1)
    x_sequence[20:30] = 1.0  # 脉冲输入
    x_sequence[60:70] = 0.5  # 另一个脉冲
    
    # 运行 LTC
    h_final, h_history = ltc(x_sequence, dt=0.1)
    
    # 可视化
    try:
        plt.figure(figsize=(12, 6))
        
        # 选择 3 个神经元可视化
        neurons_to_plot = [0, 4, 9]
        
        for i in neurons_to_plot:
            plt.plot(h_history[:, i].numpy(), label=f'神经元 {i}')
        
        # 标记输入脉冲
        plt.axvspan(20, 30, alpha=0.2, color='red', label='输入脉冲 1')
        plt.axvspan(60, 70, alpha=0.2, color='blue', label='输入脉冲 2')
        
        plt.xlabel('时间步 (dt=0.1)')
        plt.ylabel('神经元状态')
        plt.title('LTC 状态演化：脉冲输入 → 状态上升 → 自然衰减')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('ltc_evolution.png', dpi=100)
        print("✅ 图像已保存: ltc_evolution.png")
        
        plt.show()
    except Exception as e:
        print(f"⚠️ 无法显示图像: {e}")
        print("使用文本可视化代替...")
        
        print("\n状态演化 (选择神经元 0):")
        for t in [0, 20, 30, 50, 60, 70, 99]:
            h_val = h_history[t, 0].item()
            print(f"  t={t}: h={h_val:.4f}")
    
    print("\n【导师解读】")
    print("LTC 的状态演化特点:")
    print("1. 输入脉冲 → 状态快速上升")
    print("2. 无输入时 → 自然衰减 (-h/τ)")
    print("3. τ 决定衰减速度: τ 大衰减慢，τ 小衰减快")
    print("4. 这就是'液态'特性：状态持续流动")


# ============================================
# 5. 实验：对比不同 τ 值
# ============================================

def compare_tau_values():
    """对比不同 τ 值的记忆曲线"""
    print("\n=== 实验：对比不同 τ 值 ===\n")
    
    tau_values = [0.5, 1.0, 2.0, 5.0]
    
    seq_len = 100
    x_sequence = torch.zeros(seq_len, 1)
    x_sequence[10:20] = 1.0  # 单次脉冲
    
    print("τ 值对比:")
    
    try:
        plt.figure(figsize=(10, 6))
        
        for tau in tau_values:
            ltc = LTCCell(input_dim=1, hidden_dim=1, tau_init=tau)
            h_final, h_history = ltc(x_sequence, dt=0.1)
            
            plt.plot(h_history[:, 0].numpy(), label=f'τ={tau}')
            
            # 计算衰减时间（状态降到 0.1 以下）
            decay_time = None
            for t in range(20, seq_len):
                if h_history[t, 0].item() < 0.1:
                    decay_time = t - 20
                    break
            
            print(f"  τ={tau}: 衰减时间 ≈ {decay_time} 步")
        
        plt.axvspan(10, 20, alpha=0.2, color='red', label='输入脉冲')
        plt.xlabel('时间步')
        plt.ylabel('状态值')
        plt.title('不同 τ 值的衰减曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('tau_comparison.png', dpi=100)
        print("✅ 图像已保存: tau_comparison.png")
        
        plt.show()
    except Exception as e:
        print(f"⚠️ 无法显示图像: {e}")
    
    print("\n【导师解读】")
    print("τ 的作用:")
    print("- τ=0.5: 快速衰减（短期记忆）")
    print("- τ=1.0: 中等衰减")
    print("- τ=2.0: 较慢衰减（中期记忆）")
    print("- τ=5.0: 慢衰减（长期记忆）")
    print("\n这可以类比:")
    print("- 工作记忆（τ 小）")
    print("- 长期记忆（τ 大）")


# ============================================
# 6. 实验：动态 τ 的效果
# ============================================

def demonstrate_dynamic_tau():
    """演示动态 τ 的效果"""
    print("\n=== 实验：动态 τ 的效果 ===\n")
    
    ltc = DynamicTauLTCCell(input_dim=1, hidden_dim=5)
    
    seq_len = 50
    
    # 创建输入序列：强度变化
    x_sequence = torch.zeros(seq_len, 1)
    x_sequence[5:15] = 1.0   # 强脉冲
    x_sequence[25:35] = 0.3  # 弱脉冲
    
    h_final, h_history, tau_history = ltc(x_sequence, dt=0.1)
    
    print("动态 τ 的变化:")
    print("输入强度 → τ 变化 → 记忆长度自动调整")
    
    try:
        plt.figure(figsize=(12, 8))
        
        # 子图 1: 状态演化
        plt.subplot(2, 1, 1)
        plt.plot(h_history[:, 0].numpy(), label='神经元状态')
        plt.axvspan(5, 15, alpha=0.2, color='red', label='强脉冲')
        plt.axvspan(25, 35, alpha=0.2, color='blue', label='弱脉冲')
        plt.ylabel('状态')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图 2: τ 变化
        plt.subplot(2, 1, 2)
        plt.plot(tau_history[:, 0].numpy(), label='动态 τ', color='green')
        plt.ylabel('τ 值')
        plt.xlabel('时间步')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle('动态 τ: 输入强度 → τ 自动调整')
        
        plt.savefig('dynamic_tau.png', dpi=100)
        print("✅ 图像已保存: dynamic_tau.png")
        
        plt.show()
    except Exception as e:
        print(f"⚠️ 无法显示图像: {e}")
    
    print("\n【导师解读】")
    print("动态 τ 的自适应:")
    print("- 输入强 → τ 可能变大 → 记忆更持久")
    print("- 输入弱 → τ 可能变小 → 快速遗忘")
    print("- 这是 LTC 的核心创新: '液态'适应性")


# ============================================
# 7. 实验：内稳态调制 LTC（Evola 核心）
# ============================================

def demonstrate_homeostasis_modulated_ltc():
    """演示内稳态调制的 LTC"""
    print("\n=== 实验：内稳态调制 LTC ===\n")
    
    ltc = HomeostasisModulatedLTC(input_dim=1, hidden_dim=5)
    
    seq_len = 100
    x_sequence = torch.zeros(seq_len, 1)
    x_sequence[20:30] = 1.0  # 输入脉冲
    
    # 创建不同的内稳态序列
    # 高能量 vs 低能量
    homeostasis_high_energy = torch.zeros(seq_len, 6)
    homeostasis_high_energy[:, 3] = 0.9  # energy_level = 0.9
    
    homeostasis_low_energy = torch.zeros(seq_len, 6)
    homeostasis_low_energy[:, 3] = 0.2  # energy_level = 0.2
    
    # 运行两种情况
    h_high, h_history_high, tau_high = ltc(x_sequence, homeostasis_high_energy, dt=0.1)
    h_low, h_history_low, tau_low = ltc(x_sequence, homeostasis_low_energy, dt=0.1)
    
    print("内稳态调制效果:")
    
    try:
        plt.figure(figsize=(12, 8))
        
        # 状态对比
        plt.subplot(2, 1, 1)
        plt.plot(h_history_high[:, 0].numpy(), label='高能量 (τ 大)', color='blue')
        plt.plot(h_history_low[:, 0].numpy(), label='低能量 (τ 小)', color='red')
        plt.axvspan(20, 30, alpha=0.2, color='green', label='输入脉冲')
        plt.ylabel('状态')
        plt.title('内稳态调制: 能量 → τ → 记忆长度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # τ 对比
        plt.subplot(2, 1, 2)
        plt.plot(tau_high[:, 0].numpy(), label='高能量时的 τ', color='blue')
        plt.plot(tau_low[:, 0].numpy(), label='低能量时的 τ', color='red')
        plt.ylabel('τ')
        plt.xlabel('时间步')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('homeostasis_ltc.png', dpi=100)
        print("✅ 图像已保存: homeostasis_ltc.png")
        
        plt.show()
    except Exception as e:
        print(f"⚠️ 无法显示图像: {e}")
        
        # 文本输出
        print("\n高能量状态:")
        for t in [20, 30, 50, 70, 99]:
            print(f"  t={t}: h={h_history_high[t, 0].item():.4f}, τ={tau_high[t, 0].item():.4f}")
        
        print("\n低能量状态:")
        for t in [20, 30, 50, 70, 99]:
            print(f"  t={t}: h={h_history_low[t, 0].item():.4f}, τ={tau_low[t, 0].item():.4f}")
    
    print("\n【导师解读 - Evola 核心概念】")
    print("内稳态对 LTC 的调制:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("高能量:")
    print("  → τ 调大")
    print("  → 记忆衰减慢")
    print("  → 长期记忆能力增强")
    print("  → 资源充足时，保留更多信息")
    print("")
    print("低能量:")
    print("  → τ 调小")
    print("  → 记忆衰减快")
    print("  → 快速遗忘低价值信息")
    print("  → 资源不足时，节省内存")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("\n这就是 Evola 的核心机制:")
    print("'根据内在需求，自动调节记忆长度'")


# ============================================
# 8. 主实验运行
# ============================================

def run_experiment():
    """运行所有实验"""
    print("=" * 60)
    print("第 7 课实验：液态神经网络 (LTC) 单元")
    print("=" * 60)
    
    # 实验 1: LTC 状态演化
    visualize_ltc_evolution()
    
    # 实验 2: τ 值对比
    compare_tau_values()
    
    # 实验 3: 动态 τ
    demonstrate_dynamic_tau()
    
    # 实验 4: 内稳态调制 LTC（核心）
    demonstrate_homeostasis_modulated_ltc()
    
    print("\n" + "=" * 60)
    print("所有实验完成！")
    print("=" * 60)
    
    print("\n【课后思考】")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("1. LTC 的 τ 如何与内稳态的其他变量关联？")
    print("   - 预测误差 → τ 应该如何变化？")
    print("   - 记忆负载 → τ 应该如何变化？")
    print("")
    print("2. LTC 如何与 Transformer 结合？")
    print("   - 替代标准 Attention？")
    print("   - 作为温态运行的基础？")
    print("")
    print("3. Evola 的温态运行:")
    print("   - 无输入时，LTC 的状态如何自然衰减？")
    print("   - 温态时，τ 应该如何设置？")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    run_experiment()