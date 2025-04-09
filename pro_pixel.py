import pickle
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

# 读取 pkl 文件
def read_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# 绘制折线图并保存为 PDF
def plot_tensors(data, pdf_filename):
    with PdfPages(pdf_filename) as pdf:  # 这里传入具体的 PDF 文件路径
        for key, tensor_list in data.items():
            # 确保 list 中的 tensor 形状一致
            if len(tensor_list) == 0:
                continue

            # if key >=4:
            #     break
            
            # 获取 tensor 的形状
            tensor_shape = tensor_list[0].shape
            num_elements = tensor_list[0].numel()  # 每个 tensor 的元素总数

            # 初始化存储每个元素的值的列表
            element_values = [[] for _ in range(num_elements)]  #len(element_values)是scale*scale,且每个元素是包含32个元素的子list

            # 遍历 list 中的每个 tensor，提取每个元素的值
            for tensor in tensor_list:
                flattened_tensor = tensor.flatten()  # 将 tensor 展平为一维
                for i in range(num_elements):
                    element_values[i].append(flattened_tensor[i].item())

            # 绘制折线图
            # plt.figure(figsize=(10, 6))
            # for i, values in enumerate(element_values):
            #     plt.plot(range(1, len(values) + 1), values, label=f'Element {i + 1}')

            side_length = int(len(element_values) ** 0.5)  # 计算 tensor 的边长
            indices_to_plot = [
                0,  # 左上角
                # side_length - 1,  # 右上角
                # (side_length - 1) * side_length,  # 左下角
                side_length * side_length - 1,  # 右下角
                (side_length // 2) * side_length + (side_length // 2),  # 中心
                (side_length // 2) * side_length + (side_length // 2) + 1  # 中心右侧
            ]

            # 绘制选定像素的折线图
            plt.figure(figsize=(10, 6))
            for i in indices_to_plot:
                if i < len(element_values):
                    # print(f"Plotting element {i} for scale {key}")
                    plt.plot(range(1, len(element_values[i]) + 1), element_values[i], label=f'Element {i}')
                
            plt.title(f'Scale {key} - Tensor Element Trends')
            plt.xlabel('Index in List (1 to 32)')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)

            # 将当前图添加到 PDF 文件
            pdf.savefig()
            plt.close()
            # print(f"Added plot for Scale {key} to PDF")


# 主程序
if __name__ == "__main__":
    # 读取 pkl 文件
    file_path = '/home/lianyaoxiu/lianyaoxiu/Infinity/outputs/loss/testpixel_cos_vintage_insect.pkl'  # 替换为实际路径
    file_path = '/home/lianyaoxiu/lianyaoxiu/Infinity/outputs/loss/testpixel_ratio_vintage_insect.pkl'
    data = read_pkl_file(file_path)

    print(len(data[0]))
    print(data[3][0].shape)
    # 输出 PDF 文件路径
    pdf_filename = '/home/lianyaoxiu/lianyaoxiu/Infinity/outputs/plots/partial_pixel_ratio_plots_output.pdf'  # 替换为实际输出路径

    # 绘制并保存到单个 PDF 文件
    plot_tensors(data, pdf_filename)