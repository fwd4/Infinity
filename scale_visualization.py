'''
## 功能
该脚本用于可视化 scale_schedule 中的 ph 和 pw 的值，并将结果保存为 PDF 文件。

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import torch

# 定义 scale_schedule
scale_schedule = [
    (1, 1, 1), (2, 2, 2), (3, 4, 4), (4, 6, 6), (5, 8, 8), 
    (6, 12, 12), (7, 16, 16), (9, 20, 20), (11, 24, 24), 
    (13, 32, 32), (15, 40, 40), (17, 48, 48), (21, 64, 64)
]

# 定义 uph 和 upw
_, uph, upw = scale_schedule[-1]

# 创建 PDF 文件
pdf_filename = "scale_schedule_visualization.pdf"
with PdfPages(pdf_filename) as pdf:
    # 每页的图像数量
    rows, cols = 2, 4
    num_per_page = rows * cols

    # 遍历 scale_schedule 中的 (ph, pw)
    for page_start in range(0, len(scale_schedule), num_per_page):
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            idx = page_start + i
            if idx >= len(scale_schedule):
                ax.axis('off')  # 如果没有更多的图像，隐藏多余的子图
                continue

            _, ph, pw = scale_schedule[idx]

            # 计算 indices
            indices = torch.stack([
                (torch.arange(ph) * (uph / ph)).reshape(ph, 1).expand(ph, pw),
                (torch.arange(pw) * (upw / pw)).reshape(1, pw).expand(ph, pw),
            ], dim=-1).round().int()  # (ph, pw, 2)
            indices = indices.reshape(-1, 2)  # (ph*pw, 2)

            # 绘制网格
            ax.set_xlim(0, upw)
            ax.set_ylim(0, uph)
            ax.set_aspect('equal')
            ax.set_title(f"ph={ph}, pw={pw}")

            # 绘制空白网格
            for y in range(uph):
                for x in range(upw):
                    rect = patches.Rectangle((x, uph - y - 1), 1, 1, linewidth=0.5, edgecolor='gray', facecolor='white')
                    ax.add_patch(rect)

            # 根据 indices 填色
            for idx in indices:
                y, x = idx.tolist()
                rect = patches.Rectangle((x, uph - y - 1), 1, 1, linewidth=0.5, edgecolor='black', facecolor='blue', alpha=0.5)
                ax.add_patch(rect)

        # 调整布局并保存当前页
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"PDF 文件已保存为 {pdf_filename}")
'''


'''
## 功能: 绘制叠加图
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import torch
import numpy as np

# 定义 scale_schedule
scale_schedule = [
    (1, 1, 1), (2, 2, 2), (3, 4, 4), (4, 6, 6), (5, 8, 8), 
    (6, 12, 12), (7, 16, 16), (9, 20, 20), (11, 24, 24), 
    (13, 32, 32), (15, 40, 40), (17, 48, 48), (21, 64, 64)
]

# 定义 uph 和 upw
_, uph, upw = scale_schedule[-1]

# 初始化网格计数器，用于记录每个网格被填色的次数
grid_counter = np.zeros((uph, upw), dtype=int)

# 创建 PDF 文件
pdf_filename = "scale_schedule_visualization_accumulated.pdf"
with PdfPages(pdf_filename) as pdf:
    # 每页的图像数量
    rows, cols = 2, 4
    num_per_page = rows * cols

    # 遍历 scale_schedule 中的 (ph, pw)，逐步叠加
    for page_start in range(0, len(scale_schedule), num_per_page):
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            idx = page_start + i
            if idx >= len(scale_schedule):
                ax.axis('off')  # 如果没有更多的图像，隐藏多余的子图
                continue

            # 叠加当前 (ph, pw) 的 indices
            for j in range(idx + 1):  # 累计到当前 idx 的所有 (ph, pw)
                _, ph, pw = scale_schedule[j]

                # 计算 indices
                indices = torch.stack([
                    (torch.arange(ph) * (uph / ph)).reshape(ph, 1).expand(ph, pw),
                    (torch.arange(pw) * (upw / pw)).reshape(1, pw).expand(ph, pw),
                ], dim=-1).round().int()  # (ph, pw, 2)
                indices = indices.reshape(-1, 2)  # (ph*pw, 2)

                # 更新网格计数器
                for idx in indices:
                    y, x = idx.tolist()
                    grid_counter[y, x] += 1

            # 绘制网格
            ax.set_xlim(0, upw)
            ax.set_ylim(0, uph)
            ax.set_aspect('equal')
            ax.set_title(f"Up to ph={ph}, pw={pw}")

            # 绘制空白网格并填色
            for y in range(uph):
                for x in range(upw):
                    count = grid_counter[y, x]
                    if count > 0:
                        # 根据计数器调整颜色深度
                        # color_intensity = min(1.0, count / len(scale_schedule))  # 最大深度为 1.0
                        color_intensity = min(1.0, 0.2 + 0.8 * (count / grid_counter.max()))  # 最小深度为 0.2，最大为 1.0                        
                        rect = patches.Rectangle((x, uph - y - 1), 1, 1, linewidth=0.5, edgecolor='black', facecolor=(0, 0, 1, color_intensity))
                    else:
                        rect = patches.Rectangle((x, uph - y - 1), 1, 1, linewidth=0.5, edgecolor='gray', facecolor='white')
                    ax.add_patch(rect)

        # 调整布局并保存当前页
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"PDF 文件已保存为 {pdf_filename}")
'''

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import torch
import numpy as np

# 定义 scale_schedule
scale_schedule = [
    (1, 1, 1), (2, 2, 2), (3, 4, 4), (4, 6, 6), (5, 8, 8), 
    (6, 12, 12), (7, 16, 16), (9, 20, 20), (11, 24, 24), 
    (13, 32, 32), (15, 40, 40), (17, 48, 48), (21, 64, 64)
]

# 定义 uph 和 upw
_, uph, upw = scale_schedule[-1]

# 初始化网格计数器，用于记录每个网格被填色的次数
grid_counter = np.zeros((uph, upw), dtype=int)

# 创建 PDF 文件
pdf_filename = "scale_schedule_visualization_distinguish.pdf"
with PdfPages(pdf_filename) as pdf:
    # 每页的图像数量
    rows, cols = 2, 4
    num_per_page = rows * cols

    # 遍历 scale_schedule 中的 (ph, pw)，逐步叠加
    for page_start in range(0, len(scale_schedule), num_per_page):
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            idx = page_start + i
            if idx >= len(scale_schedule):
                ax.axis('off')  # 如果没有更多的图像，隐藏多余的子图
                continue

            # 当前 (ph, pw)
            _, ph, pw = scale_schedule[idx]

            # 计算 indices
            indices = torch.stack([
                (torch.arange(ph) * (uph / ph)).reshape(ph, 1).expand(ph, pw),
                (torch.arange(pw) * (upw / pw)).reshape(1, pw).expand(ph, pw),
            ], dim=-1).round().int()  # (ph, pw, 2)
            indices = indices.reshape(-1, 2)  # (ph*pw, 2)

            # 初始化当前图的网格状态
            new_blocks = []  # 当前新增的块
            old_blocks = []  # 当前重复的块

            # 更新网格计数器并分类新旧块
            for idx in indices:
                y, x = idx.tolist()
                if grid_counter[y, x] == 0:
                    new_blocks.append((y, x))  # 新块
                else:
                    old_blocks.append((y, x))  # 旧块
                grid_counter[y, x] += 1

            # 绘制网格
            ax.set_xlim(0, upw)
            ax.set_ylim(0, uph)
            ax.set_aspect('equal')
            ax.set_title(f"Up to ph={ph}, pw={pw}")

            # 移除坐标轴
            ax.axis('off')
            
            # 绘制空白网格
            for y in range(uph):
                for x in range(upw):
                    rect = patches.Rectangle((x, uph - y - 1), 1, 1, linewidth=0.5, edgecolor='gray', facecolor='white')
                    ax.add_patch(rect)

            # 绘制旧块（用橘色表示）
            for y, x in old_blocks:
                rect = patches.Rectangle((x, uph - y - 1), 1, 1, linewidth=0.5, edgecolor='black', facecolor='lightcoral', alpha=0.8)
                ax.add_patch(rect)

            # 绘制新块（用绿色表示）
            for y, x in new_blocks:
                rect = patches.Rectangle((x, uph - y - 1), 1, 1, linewidth=0.5, edgecolor='black', facecolor='green', alpha=0.8)
                ax.add_patch(rect)

            # 计算新旧块比例
            total_blocks = len(new_blocks) + len(old_blocks)
            new_ratio = len(new_blocks) / total_blocks if total_blocks > 0 else 0
            old_ratio = len(old_blocks) / total_blocks if total_blocks > 0 else 0

            # 在图中显示比例
            ax.text(0.5, -0.1, f"New: {len(new_blocks)}/{total_blocks}, {new_ratio:.2%}\n",
                               f"Old: {len(old_blocks)}/{total_blocks}, {old_ratio:.2%}" ,
                    transform=ax.transAxes, ha='center', fontsize=10, color='black')
            
        # 调整布局并保存当前页
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"PDF 文件已保存为 {pdf_filename}")