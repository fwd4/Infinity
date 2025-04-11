import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 数据处理
data = {
    '高宽比': [0.333, 0.4, 0.5, 0.571, 0.666, 0.75, 0.8, 1.0, 1.25, 1.333, 1.5, 1.75, 2.0, 2.5, 3.0],
    '样本数': [2089, 2479, 7865, 206016, 158327, 88799, 19486, 177421, 16062, 33887, 47749, 5487, 1604, 663, 319],
    '分片数': [3, 3, 8, 207, 159, 89, 20, 178, 17, 34, 48, 6, 2, 1, 1]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 生成markdown表格
markdown_table = df.to_markdown(index=False, floatfmt='.3f')
print("### 模板统计数据")
print(markdown_table)

# 创建饼图
plt.figure(figsize=(12, 8))

# 计算占比阈值
total_samples = df['样本数'].sum()
threshold_ratio = 0.02  # 2%
threshold = total_samples * threshold_ratio

# 计算占比并排序
df['占比'] = df['样本数'] / total_samples
df_sorted = df.sort_values('占比', ascending=False)

# 分离主要数据和其他数据
major_samples = df_sorted[df_sorted['样本数'] >= threshold]
other_samples = df_sorted[df_sorted['样本数'] < threshold]['样本数'].sum()

sizes = major_samples['样本数'].tolist()
labels = [f'Aspect Ratio {ratio:.3f}' for ratio in major_samples['高宽比']]

# 添加其他类别
if other_samples > 0:
    sizes.append(other_samples)
    labels.append('Others')

colors = plt.cm.Pastel1(np.linspace(0, 1, len(sizes)))

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Template Aspect Ratios', fontsize=12)
plt.axis('equal')

# 添加图例，调整位置和字体大小
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

# 保存图片
plt.savefig('template_distribution.png', bbox_inches='tight', dpi=300)
print("\n饼图已保存为 template_distribution.png")

# 输出一些基本统计信息
print("\n### 基本统计信息")
print(f"总样本数: {df['样本数'].sum():,}")
print(f"总分片数: {df['分片数'].sum()}")
print(f"重复数据: 305,911")