import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def average_pooling(arr, target_h, target_w):
    h, w = arr.shape
    block_h = h // target_h
    block_w = w // target_w
    new_h = block_h * target_h
    new_w = block_w * target_w
    arr = arr[:new_h, :new_w]
    return arr.reshape(target_h, block_h, target_w, block_w).mean(axis=(1, 3))

def visualize_attention(data, output_path, title=None):
    # 计算降采样的尺寸
    scale_h = 128
    scale_w = scale_h * data.shape[1] // data.shape[0]
    
    # 计算分界线位置
    qlen = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
    qlen = [x*x for x in qlen]
    qlen = np.cumsum(qlen)[:-1]
    axvlines = [x * scale_w / data.shape[1] for x in qlen]
    
    # 数据预处理和降采样
    data_normalized = data.astype(np.float32)
    data_downsampled = average_pooling(data_normalized, scale_h, scale_w)
    
    # 可视化
    plt.figure(figsize=(30, 8))
    sns.heatmap(data_downsampled, cmap='viridis')
    if title:
        plt.title(title)
    
    # 添加分界线
    for x in axvlines:
        plt.axvline(x=x, color='red', linestyle='--', linewidth=0.5)
    
    # 保存结果
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_folder(input_folder, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历输入文件夹中的所有.npy文件
    for file_path in Path(input_folder).glob('*.npy'):
        try:
            # 加载数据
            data = np.load(str(file_path))
            
            # 解析文件名中的参数
            name_parts = file_path.stem.split('_')  # scores_x_heady
            stage_num = name_parts[1]
            head_num = name_parts[2].replace('head', '')
            
            # 构建输出文件路径
            output_path = os.path.join(output_folder, f"{file_path.stem}.png")
            
            # 可视化并保存
            visualize_attention(
                data, 
                output_path, 
                title=f'Infinity Attention -2 Stage: Layer-{stage_num} Head-{head_num}'
            )
            print(f"处理完成: {file_path.name}")
            
        except Exception as e:
            print(f"处理文件 {file_path.name} 时出错: {str(e)}")

if __name__ == "__main__":
    # 设置输入输出路径
    input_folder = "infi_scores/stage11"
    output_folder = "infi_scores/stage11/vis_results"
    
    # 处理文件夹
    process_folder(f'{input_folder}', output_folder)