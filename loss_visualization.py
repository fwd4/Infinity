import glob
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# 查找所有前缀为 loss_data_ 的 pkl 文件
pkl_files = glob.glob('outputs/loss/loss_data_*.pkl')

# 读取所有 pkl 文件并存储到一个列表中
all_loss_data = []

for pkl_file in pkl_files:
    with open(pkl_file, 'rb') as f:
        loss_data = pickle.load(f)
        all_loss_data.append(loss_data)

all_keys = all_loss_data[0].keys()

# 创建一个 PDF 文件来保存所有图表
with PdfPages('loss_plots.pdf') as pdf:
    # 为每个 key 绘制折线图
    for key in all_keys:
        plt.figure()
        for i, loss_data in enumerate(all_loss_data):
            if key in loss_data:
                plt.plot(loss_data[key], label=f'Data from {os.path.basename(pkl_files[i])}')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'Loss for {key}')
        plt.legend()
        plt.grid(True)
        pdf.savefig()  # 保存当前图表到PDF
        plt.close()
