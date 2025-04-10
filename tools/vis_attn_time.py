import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = np.load('attn_time.npy')[1:]  # 替换成你的npy文件路径
print(data.shape)

# 沿N轴平均（假设第一维是N）
channels = data.mean(axis=0)  # shape (1, 13, 32)

# 创建折线图
plt.figure(figsize=(15, 8))  # 调整画布大小以适应13条线

# 绘制所有通道的折线
for i in range(13):
    x = np.arange(32)  # 假设x轴是0~31的位置索引
    y = channels[i]
    plt.plot(x, y, label=f'Stage {i}', linewidth=1.5)

# 添加图表元素
plt.title('Infinity Stage Time Consumption')
plt.xlabel('Layer Id')
plt.ylabel('Average Time(ms)')
plt.legend()  # 显示图例
plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线
plt.tight_layout()  # 自动调整布局防止重叠

plt.savefig('infinity_blocks.png', dpi=300)