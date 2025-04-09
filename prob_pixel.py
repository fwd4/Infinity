### 处理概率 ####

import pickle
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.special import rel_entr  # 用于计算 KL 散度
import os

# 读取 pkl 文件
def read_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# 计算 KL 散度
def calculate_kl_divergence(p, q):
    """
    计算两个概率分布之间的 KL 散度
    p: 第一个概率分布 (numpy array)
    q: 第二个概率分布 (numpy array)
    """
    p = np.clip(p, 1e-10, 1.0)  # 避免 log(0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(rel_entr(p, q))

# 计算 Hamming 距离
def calculate_hamming_distance(arr1, arr2):
    """
    计算两个数组之间的 Hamming 距离
    arr1: 第一个数组 (numpy array)
    arr2: 第二个数组 (numpy array)
    """
    return np.sum(arr1 != arr2) / arr1.size

# 处理每个阶段中每张图像的像素级相似性
def process_sample_similarity_with_last(data, pdf_filename):
    sample_codes_data = data['sample_codes_data']
    stages = sorted(sample_codes_data.keys())  # 获取所有阶段的 key

    with PdfPages(pdf_filename) as pdf:
        for stage in stages:
            stage_data = sample_codes_data[stage]
            last_image = stage_data[-1].squeeze(dim=0).cpu().numpy()    # 最后一张图像的像素概率分布[1, 4, 32]--[4, 32]

            # 创建一个新的页面用于当前阶段
            plt.figure(figsize=(16, 12))
            plt.suptitle(f"Stage {stage} - Pixel-level similarity with the last step", fontsize=16)

            similarities = []
            for step in range(len(stage_data)):  # Iterate through each step
                current_image = stage_data[step].squeeze(dim=0).cpu().numpy()    # Pixel probability distribution of the current image


                # Ensure the number of pixels is consistent
                assert current_image.shape[0] == last_image.shape[0], "Inconsistent number of pixels"

                # Calculate Hamming distance for each pixel  值越大越相似
                pixel_similarities = [
                    1 - calculate_hamming_distance(current_image[j], last_image[j])
                    for j in range(current_image.shape[0])
                ]
                max_similarity = np.max(pixel_similarities)
                min_similarity = np.min(pixel_similarities)

                img_similarity = np.mean(pixel_similarities)  # Calculate the average similarity across all pixels

                similarities.append(img_similarity)
                # Plot the similarity heatmap for the current step
                pixel_similarities_matrix = np.array(pixel_similarities).reshape(int(np.sqrt(len(pixel_similarities))), -1)
                plt.subplot(4, 4, (step % 16) + 1)  # Display up to 8 subplots per page (4 rows x 2 columns)
                plt.imshow(pixel_similarities_matrix, cmap='hot', interpolation='nearest')
                plt.colorbar(label="Average  Hamming distance")
                plt.title(f"Step {step + 1}\nSimilarity: {img_similarity:.4f}\nMax: {max_similarity:.4f}, Min: {min_similarity:.4f}", fontsize=8)
                plt.xlabel("Pixel Column")
                plt.ylabel("Pixel Row")
                plt.grid(False)

                # If the current page is full (8 subplots), save and create a new page
                if (step + 1) % 16 == 0 or step == len(stage_data) - 1:
                    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
                    pdf.savefig()
                    plt.close()
                    if step != len(stage_data) - 1:  # If not the last step, create a new page
                        plt.figure(figsize=(16, 12))
                        plt.suptitle(f"Stage {stage} - Pixel-level similarity with the last step", fontsize=16)

# 处理每个阶段中相邻图像的像素级相似性
def process_sample_similarity_with_adjacent(data, pdf_filename):
    sample_codes_data = data['sample_codes_data']
    stages = sorted(sample_codes_data.keys())  # 获取所有阶段的 key

    with PdfPages(pdf_filename) as pdf:
        for stage in stages:
            stage_data = sample_codes_data[stage]

            # 创建一个新的页面用于当前阶段
            plt.figure(figsize=(16, 12))
            plt.suptitle(f"Stage {stage} - Pixel-level similarity between adjacent steps", fontsize=16)

            for step in range(1, len(stage_data)):  # Iterate through each step starting from the second one
                current_image = stage_data[step].squeeze(dim=0).cpu().numpy()    # Pixel probability distribution of the current image
                previous_image = stage_data[step - 1].squeeze(dim=0).cpu().numpy()  # Pixel probability distribution of the previous image

                # Ensure the number of pixels is consistent
                assert current_image.shape[0] == previous_image.shape[0], "Inconsistent number of pixels"

                # Calculate Hamming distance for each pixel
                pixel_similarities = [
                    1 - calculate_hamming_distance(current_image[j], previous_image[j])
                    for j in range(current_image.shape[0])
                ]
                max_similarity = np.max(pixel_similarities)
                min_similarity = np.min(pixel_similarities)

                img_similarity = np.mean(pixel_similarities)  # Calculate the average similarity across all pixels

                # Plot the similarity heatmap for the current step
                pixel_similarities_matrix = np.array(pixel_similarities).reshape(int(np.sqrt(len(pixel_similarities))), -1)
                plt.subplot(4, 4, (step - 1) % 16 + 1)  # Display up to 16 subplots per page (4 rows x 4 columns)
                plt.imshow(pixel_similarities_matrix, cmap='hot', interpolation='nearest')
                plt.colorbar(label="Average Hamming distance")
                plt.title(f"Step {step}\nSimilarity: {img_similarity:.4f}\nMax: {max_similarity:.4f}, Min: {min_similarity:.4f}", fontsize=8)
                plt.xlabel("Pixel Column")
                plt.ylabel("Pixel Row")
                plt.grid(False)

                # If the current page is full (16 subplots), save and create a new page
                if step % 16 == 0 or step == len(stage_data) - 1:
                    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
                    pdf.savefig()
                    plt.close()
                    if step != len(stage_data) - 1:  # If not the last step, create a new page
                        plt.figure(figsize=(16, 12))
                        plt.suptitle(f"Stage {stage} - Pixel-level similarity between adjacent steps", fontsize=16)

# 处理每个阶段中每张图像与最后一张图像的像素级相似性
def process_prob_similarity_with_last(data, pdf_filename):
    pro_codes_data = data['pro_codes_data']
    stages = sorted(pro_codes_data.keys())  # 获取所有阶段的 key

    with PdfPages(pdf_filename) as pdf:
        for stage in stages:
            stage_data = pro_codes_data[stage]
            last_image = stage_data[-1].reshape(-1, 2).cpu().numpy()    # 最后一张图像的像素概率分布

            # 创建一个新的页面用于当前阶段
            plt.figure(figsize=(16, 12))
            plt.suptitle(f"Stage {stage} - Pixel-level similarity with the last step", fontsize=16)

            for step in range(len(stage_data)):  # Iterate through each step
                current_image = stage_data[step].reshape(-1, 2).cpu().numpy()    # Pixel probability distribution of the current image

                # Ensure the number of pixels is consistent
                assert current_image.shape[0] == last_image.shape[0], "Inconsistent number of pixels"

                # Calculate KL divergence for each pixel
                row_pixel_similarities = [
                    calculate_kl_divergence(current_image[j], last_image[j])
                    for j in range(current_image.shape[0])
                ]


                # Group every 32 rows and calculate the average similarity for each group
                pixel_similarities = [
                    np.mean(row_pixel_similarities[j:j+32]) 
                    for j in range(0, len(row_pixel_similarities), 32)
                ]
                img_similarity = np.mean(pixel_similarities)  # Calculate the average similarity across all groups

                # Find the maximum and minimum values in row_pixel_similarities
                max_similarity = np.max(pixel_similarities)
                min_similarity = np.min(pixel_similarities)

                # Plot the similarity heatmap for the current step
                pixel_similarities_matrix = np.array(pixel_similarities).reshape(int(np.sqrt(len(pixel_similarities))), -1)
                plt.subplot(4, 4, (step % 16) + 1)  # Display up to 8 subplots per page (4 rows x 2 columns)
                plt.imshow(pixel_similarities_matrix, cmap='hot', interpolation='nearest')
                plt.colorbar(label="Average KL Divergence")
                plt.title(f"Step {step + 1}\nSimilarity: {img_similarity:.4f}\nMax: {max_similarity:.4f}, Min: {min_similarity:.4f}", fontsize=8)
                plt.xlabel("Pixel Column")
                plt.ylabel("Pixel Row")
                plt.grid(False)

                # If the current page is full (8 subplots), save and create a new page
                if (step + 1) % 16 == 0 or step == len(stage_data) - 1:
                    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
                    pdf.savefig()
                    plt.close()
                    if step != len(stage_data) - 1:  # If not the last step, create a new page
                        plt.figure(figsize=(16, 12))
                        plt.suptitle(f"Stage {stage} - Pixel-level similarity with the last step", fontsize=16)

# 处理每个阶段中相邻两张图像的像素级相似性
def process_prob_similarity_with_adjacent(data, pdf_filename):
    pro_codes_data = data['pro_codes_data']
    stages = sorted(pro_codes_data.keys())  # 获取所有阶段的 key

    with PdfPages(pdf_filename) as pdf:
        for stage in stages:
            stage_data = pro_codes_data[stage]

            # 创建一个新的页面用于当前阶段
            plt.figure(figsize=(16, 12))
            plt.suptitle(f"Stage {stage} - Pixel-level similarity between adjacent steps", fontsize=16)

            for step in range(1, len(stage_data)):  # Iterate through each step starting from the second one
                current_image = stage_data[step].reshape(-1, 2).cpu().numpy()    # Pixel probability distribution of the current image
                previous_image = stage_data[step - 1].reshape(-1, 2).cpu().numpy()  # Pixel probability distribution of the previous image

                # Ensure the number of pixels is consistent
                assert current_image.shape[0] == previous_image.shape[0], "Inconsistent number of pixels"

                # Calculate KL divergence for each pixel
                row_pixel_similarities = [
                    calculate_kl_divergence(current_image[j], previous_image[j])
                    for j in range(current_image.shape[0])
                ]
                # # Find the maximum and minimum values in row_pixel_similarities
                # max_similarity = np.max(row_pixel_similarities)
                # min_similarity = np.min(row_pixel_similarities)

                # Group every 32 rows and calculate the average similarity for each group
                pixel_similarities = [
                    np.mean(row_pixel_similarities[j:j+32]) 
                    for j in range(0, len(row_pixel_similarities), 32)
                ]
                img_similarity = np.mean(pixel_similarities)  # Calculate the average similarity across all groups

                # Find the maximum and minimum values in row_pixel_similarities
                max_similarity = np.max(pixel_similarities)
                min_similarity = np.min(pixel_similarities)

                # Plot the similarity heatmap for the current step
                pixel_similarities_matrix = np.array(pixel_similarities).reshape(int(np.sqrt(len(pixel_similarities))), -1)
                plt.subplot(4, 4, (step - 1) % 16 + 1)  # Display up to 16 subplots per page (4 rows x 4 columns)
                plt.imshow(pixel_similarities_matrix, cmap='hot', interpolation='nearest')
                plt.colorbar(label="Average KL Divergence")
                plt.title(f"Step {step}\nSimilarity: {img_similarity:.4f}\nMax: {max_similarity:.4f}, Min: {min_similarity:.4f}", fontsize=8)
                plt.xlabel("Pixel Column")
                plt.ylabel("Pixel Row")
                plt.grid(False)


                # If the current page is full (16 subplots), save and create a new page
                if step % 16 == 0 or step == len(stage_data) - 1:
                    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
                    pdf.savefig()
                    plt.close()
                    if step != len(stage_data) - 1:  # If not the last step, create a new page
                        plt.figure(figsize=(16, 12))
                        plt.suptitle(f"Stage {stage} - Pixel-level similarity between adjacent steps", fontsize=16)



# 主程序
if __name__ == "__main__":
    # 读取 pkl 文件
    file_path = '/home/lianyaoxiu/lianyaoxiu/Infinity/outputs/codes/test_pixel_prob_vintage_insect.pkl'  # 替换为实际路径
    data = read_pkl_file(file_path)

    # print(len(data[0]))
    # print(data[3][0].shape)
    # 输出 PDF 文件路径
    pdf_filename_prob = '/home/lianyaoxiu/lianyaoxiu/Infinity/outputs/plots/pixel_prob_sim_plots_output.pdf'  # 替换为实际输出路径
    pdf_filename_sample = '/home/lianyaoxiu/lianyaoxiu/Infinity/outputs/plots/pixel_sample_sim_plots_output.pdf'  # 替换为实际输出路径
    
    pdf_filename_prob_adjacent = '/home/lianyaoxiu/lianyaoxiu/Infinity/outputs/plots/pixel_prob_sim_adjacent_plots_output.pdf'  # 替换为实际输出路径
    pdf_filename_sample_adjacent = '/home/lianyaoxiu/lianyaoxiu/Infinity/outputs/plots/pixel_sample_sim_adjacent_plots_output.pdf'  # 替换为实际输出路径

    process_prob_similarity_with_last(data,pdf_filename_prob)
    # process_sample_similarity_with_last(data,pdf_filename_sample)
    process_prob_similarity_with_adjacent(data,pdf_filename_prob_adjacent)
    # process_sample_similarity_with_adjacent(data,pdf_filename_sample_adjacent)