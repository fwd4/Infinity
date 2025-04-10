import glob
import pickle
import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages


def cosine_similarity(matrix1, matrix2):
    # 将张量展平为一维
    tensor1_flat = torch.tensor(matrix1[0]).flatten()
    tensor2_flat = torch.tensor(matrix2[0]).flatten()
    
    # 计算余弦相似度
    similarity = F.cosine_similarity(tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0), dim=1)
    
    return similarity.item()

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def compare_codes_data(p0_data, p1_data, p2_data):
    similarities_codes_p0_p1 = {}
    similarities_codes_p0_p2 = {}
    for key in p0_data['codes_data'].keys():
        p0_codes = p0_data['codes_data'][key]
        p1_codes = p1_data['codes_data'][key]
        p2_codes = p2_data['codes_data'][key]
        
        similarity_p0_p1 = cosine_similarity(p0_codes, p1_codes)
        similarity_p0_p2 = cosine_similarity(p0_codes, p2_codes)
        
        similarities_codes_p0_p1[key] = similarity_p0_p1
        similarities_codes_p0_p2[key] = similarity_p0_p2
    return similarities_codes_p0_p1, similarities_codes_p0_p2

def compare_summed_codes_data(p0_data, p1_data, p2_data):
    similarities_summed_p0_p1 = {}
    similarities_summed_p0_p2 = {}
    for key in p0_data['summed_codes_data'].keys():
        p0_codes = p0_data['summed_codes_data'][key]
        p1_codes = p1_data['summed_codes_data'][key]
        p2_codes = p2_data['summed_codes_data'][key]
        
        similarity_p0_p1 = cosine_similarity(p0_codes, p1_codes)
        similarity_p0_p2 = cosine_similarity(p0_codes, p2_codes)
        
        similarities_summed_p0_p1[key] = similarity_p0_p1
        similarities_summed_p0_p2[key] = similarity_p0_p2
    return similarities_summed_p0_p1, similarities_summed_p0_p2

def plot_similarities(similarities_codes_p0_p1_catogories, similarities_codes_p0_p2_catogories,
                  similarities_summed_p0_p1_catogories, similarities_summed_p0_p2_catogories, pdf):
    categories = list(similarities_codes_p0_p1_catogories.keys())
    keys = list(similarities_codes_p0_p1_catogories[categories[0]].keys())
    for category in categories:
        plt.figure()
        
        codes_p0_p1_values = similarities_codes_p0_p1_catogories[category]
        codes_p0_p2_values = similarities_codes_p0_p2_catogories[category]
        summed_p0_p1_values = similarities_summed_p0_p1_catogories[category]
        summed_p0_p2_values = similarities_summed_p0_p2_catogories[category]

        plt.plot(keys, [codes_p0_p1_values[key] for key in keys], label='codes p0 vs p1',)
        plt.plot(keys, [codes_p0_p2_values[key] for key in keys], label='codes p0 vs p2')
        plt.plot(keys, [summed_p0_p1_values[key] for key in keys], label='summed p0 vs p1')
        plt.plot(keys, [summed_p0_p2_values[key] for key in keys], label='summed p0 vs p2')
        plt.xlabel('scale')
        plt.ylabel('Cosine Similarity')
        plt.title(f'Similarity for  {category}')
        plt.legend()
        pdf.savefig()  # 保存当前图表到PDF
        plt.close()

categories = [
    "vintage_insect",
    "macro_closeup",
    "3d_school",
    "explore_more",
    "toy_car",
    "fairy_house",
    "cat_fashion",
    "spacefrog_astroduck",
    "miniature_village",
    "corgi_dog",
    "robot_eggplant",
    "perfume_product",
    "mountain_landscape",
]

similarities_codes_p0_p1_catogories = {}
similarities_codes_p0_p2_catogories = {}
similarities_summed_p0_p1_catogories = {}
similarities_summed_p0_p2_catogories = {}
for category in categories:
    # 加载数据
    p0_data = load_data(f'outputs/codes/p0_combined_data_{category}.pkl')
    p1_data = load_data(f'outputs/codes/p1_combined_data_{category}.pkl')
    p2_data = load_data(f'outputs/codes/p2_combined_data_{category}.pkl')

    # 计算相似性
    codes_data_similarity = {}
    summed_codes_data_similarity = {}
    similarities_codes_p0_p1, similarities_codes_p0_p2 = compare_codes_data(p0_data, p1_data, p2_data)
    similarities_summed_p0_p1, similarities_summed_p0_p2 = compare_summed_codes_data(p0_data, p1_data, p2_data)
    similarities_codes_p0_p1_catogories[category] = similarities_codes_p0_p1
    similarities_codes_p0_p2_catogories[category] = similarities_codes_p0_p2
    similarities_summed_p0_p1_catogories[category] = similarities_summed_p0_p1
    similarities_summed_p0_p2_catogories[category] = similarities_summed_p0_p2
with PdfPages('similarities.pdf') as pdf:
    plot_similarities(similarities_codes_p0_p1_catogories, similarities_codes_p0_p2_catogories,
                      similarities_summed_p0_p1_catogories, similarities_summed_p0_p2_catogories, pdf)
# 打印相似性结果
# print(f"Category: {category}")
# for key, similarity in codes_data_similarity.items():
#     print(f"Codes data similarity between {key[0]} and {key[1]}: {similarity}")

# for key, similarity in summed_codes_data_similarity.items():
#     print(f"Summed codes data similarity between {key[0]} and {key[1]}: {similarity}")