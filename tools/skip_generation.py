import glob
import pickle
import os


# 查找所有前缀为 loss_data_ 的 pkl 文件
pkl_files = glob.glob('outputs/loss/loss_data_*.pkl')

# 读取所有 pkl 文件并存储到一个列表中
all_loss_data = []

for pkl_file in pkl_files:
    with open(pkl_file, 'rb') as f:
        loss_data = pickle.load(f)
        all_loss_data.append(loss_data)

loss_data = all_loss_data[0]

# 获取 loss_data 中所有值大于 15 的索引
skip_list = {si:[] for si in range(len(loss_data))}
for si, values in loss_data.items():
    for bi, value in enumerate(values):
        if value > 15:
            skip_list[si].append(bi)

with open("skip_list.pkl", 'wb') as f:
    pickle.dump(skip_list, f)
print(skip_list)
print("Skip list has been saved to skip_list.pkl")