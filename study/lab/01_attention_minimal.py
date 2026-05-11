import torch
import torch.nn.functional as F

# === 视觉优化：禁用科学计数法，让数字看起来更清爽 ===
torch.set_printoptions(precision=3, sci_mode=False)

"""
第 1 课实验 1.1：带“剧情”的自注意力模拟
目标：通过简化数值和增加可视化，直观理解注意力是如何在词与词之间流动的。
"""

def run_experiment():
    print("=== 实验 1.1：直观理解 Self-Attention ===\n")
    print("【背景故事】")
    print("我们有三个词：'I', 'love', 'AI'。")
    print("- 'I' 作为一个主语，它的 Query (查询) 是：'动作在哪里？'")
    print("- 'love' 作为一个动词，它的 Key (键) 是：'我是个动作。'")
    print("- 'AI' 作为一个名词，它的 Key (键) 是：'我是个东西。'\n")

    # 1. 简化的输入数据 (3个词 x 3个原始特征)
    # 特征维度：[是主语吗, 是动作吗, 是名词吗]
    x = torch.tensor([
        [1.0, 0.0, 0.0],   # "I"
        [0.0, 1.0, 0.0],   # "love"
        [0.0, 0.0, 1.0]    # "AI"
    ])

    # 2. 模拟权重矩阵
    # 这里的 w_query 我们设定为只看“动作”维度的需求
    w_query = torch.tensor([
        [0.0, 10.0, 0.0],  # 如果输入有[主语]特征，我强烈寻找[动作]
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    
    # 这里的 w_key 我们设定为只展示“动作”维度的特征
    w_key = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],  # 如果输入有[动作]特征，我展示[动作]属性
        [0.0, 0.0, 0.0]
    ])

    # Value 矩阵我们设为单位矩阵，即保留原始含义
    w_value = torch.eye(3)

    # 3. 计算 Q, K, V
    queries = x @ w_query
    keys = x @ w_key
    values = x @ w_value

    # 4. 计算注意力打分 (Q点积K的转置)
    attn_scores = queries @ keys.T
    
    # 5. 计算权重 (Softmax)
    # 这里我们省去 Scale，直接看效果
    attn_weights = F.softmax(attn_scores, dim=-1)

    print("--- 注意力分配图 (越长代表关注度越高) ---")
    words = ["I   ", "love", "AI  "]
    print("        " + "    ".join(words))
    for i, row in enumerate(attn_weights):
        # 简单的 ASCII 进度条可视化
        bar = ["■" * int(val * 10) for val in row]
        # 补齐空格对齐
        bar_str = "  ".join([f"{b:<10}" for b in bar])
        print(f"{words[i]}: {bar_str}  {row}")

    print("\n【导师解读】")
    print(f"注意力矩阵显示，当处理 'I' 时，它有很高的权重分配给了 'love'。")
    print("这意味着 'I' 成功地在序列中找到了与它匹配的‘动作’。")

    # 6. 最终输出
    output = attn_weights @ values
    print(f"\n最终融合后的特征 (Output):\n{output}")
    print("\n对比 'I' 融合前后的特征：")
    print(f"融合前: {x[0]}")
    print(f"融合后: {output[0]} <- 你看，'I' 现在带上了 'love' 的(动作)属性！")

    print("\n=== 实验结束 ===")

if __name__ == "__main__":
    run_experiment()

if __name__ == "__main__":
    run_experiment()
