import torch  
import numpy as np  
import matplotlib.pyplot as plt  
from matplotlib.colors import LinearSegmentedColormap  
import matplotlib.gridspec as gridspec  
from matplotlib.patches import Patch  
import os  
import pickle  
from matplotlib.backends.backend_pdf import PdfPages  

def normalize_contribution(contributions):

    # 假设 contributions 是已经计算得到的 [4, 4096] 的 tensor  
    norm_contrib = torch.zeros_like(contributions)  # 创建与 contributions 相同形状的零 tensor  

    for i in range(contributions.shape[1]):  # 遍历每一列  
        col_values = contributions[:, i]  
        
        # 检查是否存在小于0的值  
        if (col_values < 0).any():  
            # 将小于0的值对应的归一化结果设置为0  
            # 记录大于0的元素  
            positive_values = col_values[col_values > 0]  
            
            # 对于大于0的元素进行归一化处理  
            if positive_values.numel() > 0:  # 确保存在大于0的元素  
                sum_positive = positive_values.sum()  # 计算大于0的元素总和  
                norm_values = positive_values / sum_positive  # 进行归一化  
                
                # 将归一化后的值放回对应的位置  
                norm_contrib[col_values > 0, i] = norm_values  # 将归一化值赋给大于0的原位置  
        else:  
            # 如果没有小于0的值，进行正常的归一化  
            sum_col = col_values.sum()  # 计算列的总和  
            if sum_col > 0:  # 防止除零  
                norm_contrib[:, i] = col_values / sum_col  # 归一化  
            else:  
                norm_contrib[:, i] = 0  # 如果总和为0，则直接设置为0

    return  norm_contrib

# norm_contrib 现在包含了所需的归一化结果  
def plot_contri(ax, norm_contrib,bin_size=8):

    norm_contrib_np = norm_contrib.reshape(norm_contrib.shape[0], -1).cpu().numpy()   #[2, 64, 64]  

    # 对数据进行聚合，每bin_size列合并为一列  
    n_bins = norm_contrib_np.shape[1] // bin_size  
    binned_data = np.zeros((norm_contrib_np.shape[0], n_bins))  
    
    for i in range(norm_contrib_np.shape[0]):  
        for j in range(n_bins):  
            start_idx = j * bin_size  
            end_idx = (j + 1) * bin_size  
            binned_data[i, j] = np.mean(norm_contrib_np[i, start_idx:end_idx])  # 使用均值聚合  

    # 设置柱状图的参数  
    bar_width = 0.5  # 柱子的宽度  
    x = np.arange(binned_data.shape[1])  # 横坐标  

    # 绘图  
    ax.clear()  # 清空当前坐标轴内容  
    # ax.set(figsize=(12, 6))  # 可选：预设 figsize 如果需要新的图形 

    # # 绘图  
    # plt.figure(figsize=(12, 6))  

    # 颜色和标签的设置  
    colors = ['red', 'blue', 'green', 'purple', 'orange']  
    labels = ['A', 'B', 'C', 'D', 'BASE']  

    bottoms = np.zeros_like(binned_data[0])  
    
    for i in range(binned_data.shape[0]):  
        ax.bar(x, binned_data[i], bottom=bottoms, color=colors[i], width=bar_width, label=labels[i])  
        bottoms += binned_data[i]   

    # 添加标题和标签  
    ax.set_title('Stacked Bar Chart of Norm Contributions')  
    ax.set_xlabel('Index')  
    ax.set_ylabel('Contribution')  
    ax.set_xticks(ticks=x, labels=np.arange(0, binned_data.shape[1]))  # 设置横坐标  
    ax.legend(loc='upper right', fontsize=10)  

    # # 显示图形  
    # plt.tight_layout()

def compute_contribution(A, B, C, D, E, R, base, eps=1e-8):  
    """  
    计算四个矩阵对目标矩阵的贡献度  
    
    参数:  
        A, B, C, D: 输入矩阵, 形状 [ 32, 64, 64]  
        E: 目标矩阵, 形状 [32, 64, 64]  
        base: 基础矩阵, 形状 [32, 64, 64]  
        eps: 数值稳定性参数  
    
    返回:  
        包含各种贡献度指标的字典  
    """  
    # 确保所有输入都是PyTorch张量  
    tensors = [A, B, C, D, E, R, base]  
    for i, t in enumerate(tensors):  
        if not isinstance(t, torch.Tensor):  
            tensors[i] = torch.tensor(t)  
    A, B, C, D, E, R, base = tensors  
    
    # 计算目标差异                  
    v_list = [A, B, C, D, base]              # 四个分量   
    names = ['A', 'B', 'C', 'D', 'BASE']       # 分量名称 
    v_list_global = [R, base]              # 两个分量   
    names_global = ['R', 'BASE']       # 分量名称      
    
    # 展平特征维度  
    # 从 [1, 32, 64, 64] 到 [32, 4096]  
    E_flat = E.reshape(32, -1)  # [32, 4096]  


    # 计算E的范数 (32维特征向量的长度)  
    E_norm = torch.norm(E_flat, dim=0) + eps  # [4096]  
    E_norm_squared = E_norm ** 2              # [4096]  

    # 存储结果  
    contributions = []     # s_j: 标量投影 (原始贡献率)  
    norm_contrib = []      # r_j: 归一化贡献率 (和为1)  


    # 存储结果  
    contributions_global = []     # s_j: 标量投影 (原始贡献率)  
    norm_contrib_global = []      # r_j: 归一化贡献率 (和为1)  


    # 计算每个分量的贡献度  
    for v in v_list:  
        # 展平特征维度  
        v_flat = v.reshape(32, -1)  # [32, 4096]  
        
        # 计算向量投影: <v_j, dE> / ||dE||  
        dot_product = torch.sum(v_flat * E_flat, dim=0)  # [4096]  
        
        # 标量投影 s_j (原始贡献率)  
        scalar_proj = dot_product / E_norm  # [4096]  
        contributions.append(scalar_proj)  
        ratio = dot_product / E_norm_squared  # [4096]  
        norm_contrib.append(ratio)     

    # 计算每个分量的贡献度  
    for v in v_list_global:  
        # 展平特征维度  
        v_flat = v.reshape(32, -1)  # [32, 4096]  
        
        # 计算向量投影: <v_j, dE> / ||dE||  
        dot_product = torch.sum(v_flat * E_flat, dim=0)  # [4096]  
        
        # 标量投影 s_j (原始贡献率)  
        scalar_proj = dot_product / E_norm  # [4096]  
        contributions_global.append(scalar_proj)  
        
        # 归一化贡献率 r_j (各分量和为1)  
        ratio = dot_product / E_norm_squared  # [4096]  
        norm_contrib_global.append(ratio)          
    
    # 将列表转换为张量 [5, 4096]  
    contributions = torch.stack(contributions)  
    norm_contribs = torch.stack(norm_contrib)  
    # norm_contribs = normalize_contribution(contributions) 
    # 将列表转换为张量 [2, 4096] 
    contributions_global = torch.stack(contributions_global)  
    norm_contrib_global = torch.stack(norm_contrib_global)  
    
    # 找出每个像素的主导贡献者  
    max_contrib_indices = torch.argmax(norm_contribs, dim=0)  # [4096]  
    max_contrib_indices_global = torch.argmax(contributions_global, dim=0)  # [4096]  
   
    # 重新整形为空间维度 [64, 64]  
    contributions_map = contributions.reshape(5, 64, 64)  
    norm_contrib_map = norm_contribs.reshape(5, 64, 64)  
    dominant_map = max_contrib_indices.reshape(64, 64)  
    contributions_map_global = contributions_global.reshape(2, 64, 64)  
    norm_contrib_map_global = norm_contrib_global.reshape(2, 64, 64)  
    dominant_map_global = max_contrib_indices_global.reshape(64, 64)  

    # 构建结果字典  
    results = {  
        'raw_contribution': contributions_map,      # 原始贡献率 s_j  
        'normalized_contribution': norm_contrib_map, # 归一化贡献率 r_j  
        'dominant_map': dominant_map,               # 主导分量索引图  
        'component_names': names,                    # 分量名称  
        'raw_contribution_global': contributions_map_global,      # 原始贡献率 s_j  
        'normalized_contribution_global': norm_contrib_map_global, # 归一化贡献率 r_j  
        'dominant_map_global': dominant_map_global,               # 主导分量索引图  
        'component_names_global': names_global                    # 分量名称  
    }  
    
    return results  

def visualize_contributions(results, save_path=None, example_pixel=(32, 32), batch_idx=None, fig=None, category=""):  

    # 提取结果  
    raw_contrib = results['raw_contribution']       # [5, 64, 64]  
    norm_contrib = results['normalized_contribution'] # [5, 64, 64]  
    dominant_map = results['dominant_map']          # [64, 64]  
    names = results['component_names']              # ['A', 'B', 'C', 'D','BASE']  
    raw_contrib_global = results['raw_contribution_global']       # [2, 64, 64]  
    norm_contrib_global = results['normalized_contribution_global'] # [2, 64, 64]  
    dominant_map_global = results['dominant_map_global']          # [64, 64]  
    names_global = results['component_names_global']              # ['R', 'BASE']      

    # 转换为NumPy数组进行可视化  
    raw_contrib_np = raw_contrib.cpu().numpy()
    norm_contrib_np = norm_contrib.cpu().numpy()
    dominant_map_np = dominant_map.cpu().numpy()
   
    raw_contrib_np_global = raw_contrib_global.cpu().numpy()  
    norm_contrib_np_global = norm_contrib_global.cpu().numpy()  
    dominant_map_np_global = dominant_map_global.cpu().numpy()   # [64, 64]  

    ##################### DEBUG #####################
    # 获取值为 0 的索引  
    zero_indices = np.where(norm_contrib_np_global[0] >= 0.4)  
    # zero_indices = np.where(dominant_map_np_global == 0)  
    # 将二维索引转换为元组，表示(x, y)坐标  
    zero_coords = list(zip(zero_indices[0], zero_indices[1]))  
    zero_coords_int = [(int(x), int(y)) for x, y in zero_coords]  

    # 取到norm_contrib_global[0]中对应zero_coords位置的值  
    zero_values = norm_contrib_global[0][zero_indices]
    # 输出结果  
    print("R的取值个数",len(zero_coords))  
    # print("R归一化贡献度:", zero_values)

    # 创建一个空白矩阵，填充为NaN
    masked_map = np.full_like(dominant_map_np, np.nan, dtype=float)
    # 将zero_coords对应的位置填充为dominant_map_np的值
    
    for x, y in zero_coords:
        masked_map[x, y] = dominant_map_np[x, y]
        dominant_value = dominant_map_np[x, y]
        norm_contrib_value = norm_contrib_np[dominant_value, x, y]
        norm_contrib_values = norm_contrib_np[:,x, y]
        norm_contrib_std = np.std(norm_contrib_values)
        # print(f"{int(dominant_value), float(norm_contrib_value), float(norm_contrib_std)}", end='; ')
    

    unique, counts = np.unique(masked_map, return_counts=True)
    count_dict = dict(zip(unique, counts))
    # 转换计数字典中的值为原生整数  
    count_dict_int = {int(k): int(v) for k, v in count_dict.items() if not np.isnan(k)}  
    print("\nCounts of values in dominant_map_np:", count_dict_int)  
    ##################### DEBUG #####################

    # 创建新图形或使用现有图形  
    if fig is None:  
        fig = plt.figure(figsize=(20, 16))  
    else:  
        plt.figure(fig.number)  
        plt.clf()  
        fig.set_size_inches(20, 16)  
    
    # 设置标题  
    batch_title = f"batch {batch_idx}" if batch_idx is not None else ""  
    fig.suptitle(batch_title, fontsize=16)  
    
    gs = gridspec.GridSpec(5, 6, height_ratios=[1, 1, 1, 0.8, 0.8], width_ratios=[1, 1, 1, 1, 1, 1])  
    
    # 颜色映射  
    cmap_contrib = plt.cm.RdBu_r  # 红蓝色图: 红色=正贡献, 蓝色=负贡献  
    cmap_abs = plt.cm.viridis     # 绝对值色图  
    component_colors = ['#FF5733', '#33FF57', '#3357FF', '#F033FF', '#FFD700']  # 分量颜色  
    
    '''
    # 1. 原始贡献率热力图 (s_j) - 第一行  
    for i, name in enumerate(names_global):  
        ax = plt.subplot(gs[0, i])  
        
        # 找到所有热力图的最大/最小值来统一颜色范围  
        vmax = np.max(np.abs(raw_contrib_np_global))  
        vmin = -vmax  
        
        im = ax.imshow(raw_contrib_np_global[i], cmap=cmap_contrib, vmin=vmin, vmax=vmax)  
        ax.set_title(f'{name} Original contribution rate')  
        ax.set_xticks([])  
        ax.set_yticks([])  
        
        # # 为第一个图添加颜色条  
        # if i == 0:  
        plt.colorbar(im, ax=ax, shrink=0.3)  

    for i, name in enumerate(names):  
        ax = plt.subplot(gs[0, i+2])  
        
        # 找到所有热力图的最大/最小值来统一颜色范围  
        vmax = np.max(np.abs(raw_contrib_np))  
        vmin = -vmax  
        
        im = ax.imshow(raw_contrib_np[i], cmap=cmap_contrib, vmin=vmin, vmax=vmax)  
        ax.set_title(f'{name} Original contribution rate')  
        ax.set_xticks([])  
        ax.set_yticks([])  
        
        # # 为第一个图添加颜色条  
        # if i == 0:  
        plt.colorbar(im, ax=ax, shrink=0.3)  
    '''
    for i, name in enumerate(names_global):  
        ax = plt.subplot(gs[0, i])  
        ax1 = plt.subplot(gs[1, i])  
        
        vmax = np.max(np.abs(norm_contrib_np_global))  
        vmin = -vmax   
        # 创建一个掩码数组，初始化为NaN
        filtered_data = np.full_like(norm_contrib_np_global[i], np.nan)
        
        # 获取原始数据的绝对值
        abs_data = np.abs(norm_contrib_np_global[i])
        
        # 计算阈值（绝对值的前10%）
        threshold = np.percentile(abs_data[~np.isnan(abs_data)], 90)
        
        # 只保留绝对值大于阈值的点
        mask = abs_data > threshold
        filtered_data[mask] = norm_contrib_np_global[i][mask]
        
        im = ax.imshow(norm_contrib_np_global[i], cmap=cmap_contrib, vmin=vmin, vmax=vmax)  
        ax.set_title(f'{name} Normalized contribution rate')  
        ax.set_xticks([])  
        ax.set_yticks([])  

        im1 = ax1.imshow(filtered_data, cmap=cmap_contrib, vmin=vmin, vmax=vmax)  
        ax1.set_title(f'Top 10% {name} Normalized contribution rate')  
        ax1.set_xticks([])  
        ax1.set_yticks([])  

        
        # # 为第一个图添加颜色条  
        # if i == 0:  
        plt.colorbar(im, ax=ax, shrink=0.3)  
        plt.colorbar(im1, ax=ax1, shrink=0.3)  

    stacked_image = np.zeros((64, 64, 3))
    colors = np.array([
        [1.0, 0, 0],
        [0, 1.0, 0],
        [0, 0, 1.0],
        [1.0, 1.0, 1.0],
    ]) * 0.5
    for i, name in enumerate(names[:-1]):  
        ax = plt.subplot(gs[0, i+2])  
        ax1 = plt.subplot(gs[1, i+2])  
        
        vmax = np.max(np.abs(norm_contrib_np))  
        vmin = -vmax   
        # import pdb; pdb.set_trace()
        
        im = ax.imshow(norm_contrib_np[i], cmap=cmap_contrib, vmin=vmin, vmax=vmax)  
        ax.set_title(f'{name} Normalized contribution rate')  
        ax.set_xticks([])  
        ax.set_yticks([])  

        # 创建一个掩码数组，初始化为NaN
        filtered_data = np.full_like(norm_contrib_np[i], 0)
        
        # 获取原始数据的绝对值
        abs_data = np.abs(norm_contrib_np[i])
        
        # 计算阈值（绝对值的前10%）
        threshold = np.percentile(abs_data[~np.isnan(abs_data)], 95)
        
        # 只保留绝对值大于阈值的点
        mask = abs_data > threshold
        filtered_data[mask] = 1
        stacked_image += np.expand_dims(filtered_data, 2) * colors[i]
        is_all_zero = np.all(stacked_image == 0, axis=-1)
        stacked_image[is_all_zero] = np.nan

        im1 = ax1.imshow(stacked_image, alpha=1, vmin=vmin, vmax=vmax)  
        stacked_image[is_all_zero] = 0
        ax1.set_title(f'Top 10% {name} Normalized contribution rate')  
        ax1.set_xticks([])  
        ax1.set_yticks([])  
        
        # # 为第一个图添加颜色条  
        # if i == 0:  
        plt.colorbar(im, ax=ax, shrink=0.3)  

    
    # 2.1 主导贡献者地图 - 第2行第一格  
    ax = plt.subplot(gs[2, 0])  
    colors = [component_colors[i] for i in range(len(names_global))]  
    cmap_components = LinearSegmentedColormap.from_list('Components', colors, N=len(names_global))  
    
    im = ax.imshow(dominant_map_np_global, cmap=cmap_components, vmin=0, vmax=len(names_global)-1)  
    ax.set_title('Map of Global_Dominant Contributors')  
    ax.set_xticks([])  
    ax.set_yticks([])  
    # 添加图例  
    legend_elements = [Patch(facecolor=component_colors[i], label=name)   
                      for i, name in enumerate(names_global)]  
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)  

    # 2.2 主导贡献者地图 - 第2行第2格  
    ax = plt.subplot(gs[2, 1])   
    colors = [component_colors[i] for i in range(len(names))]  
    cmap_components = LinearSegmentedColormap.from_list('Components', colors, N=len(names))  
    
    im = ax.imshow(dominant_map_np, cmap=cmap_components, vmin=0, vmax=len(names)-1)  
    ax.set_title('Map of Dominant Contributors')  
    ax.set_xticks([])  
    ax.set_yticks([])  
    # 添加图例  
    legend_elements = [Patch(facecolor=component_colors[i], label=name)   
                      for i, name in enumerate(names)]  
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10) 



    # 2.3 主导贡献者地图 - 第2行第3格  
    ax = plt.subplot(gs[2, 2])   
    colors = [component_colors[i] for i in range(len(names))]  
    cmap_components = LinearSegmentedColormap.from_list('Components', colors, N=len(names))  

    im = ax.imshow(masked_map, cmap=cmap_components, vmin=0, vmax=len(names)-1)  
    ax.set_title('Map of Dominant Contributors (Filtered)')  
    ax.set_xticks([])  
    ax.set_yticks([])  

    # 添加图例  
    legend_elements = [Patch(facecolor=component_colors[i], label=name)   
                      for i, name in enumerate(names)]  
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10) 

    # 2.4 Load and display JPEG image - 第2行第4格
    ax = plt.subplot(gs[2, 3])
    try:
        img = plt.imread(f'/workspace/Infinity/output/infinity_2b_evaluation/mjhq30k_raw/pred/{category}/{batch_idx}.jpg')
        im = ax.imshow(img)
        ax.set_title('Reference Image')
        ax.set_xticks([])
        ax.set_yticks([])
    except:
        print(f"Warning: Could not load reference image {batch_idx}")
    


    # 3.1 贡献者 - 第3行 
    ax = plt.subplot(gs[3, 0:6]) 
    plot_contri(ax, results['normalized_contribution_global'])

    # 4.1 贡献者 - 第4行  
    ax = plt.subplot(gs[4, 0:6]) 
    plot_contri(ax, results['normalized_contribution'])

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为顶部总标题留出空间  
    
    # # 保存图像  
    # if save_path:  
    #     plt.savefig(save_path, dpi=150, bbox_inches='tight')  
    #     print(f"图像已保存至 {save_path}")  
    '''
    # 3.4 主导贡献者地图 - 第三行第一格  
    ax = plt.subplot(gs[2, 3])   
    # 创建一个RGB图像  
    rgb_image = np.zeros((64, 64, 3), dtype=np.float32)  
    
    # 添加每个分量的颜色，按贡献强度加权  
    for i in range(len(names)):  
        # 使用归一化贡献的正部分作为权重  
        weight = np.maximum(norm_contrib_np[i], 0)  
        
        # 为RGB通道添加颜色  
        rgb_values = [int(component_colors[i][j:j+2], 16)/255 for j in range(1, 7, 2)]  
        for c in range(3):  
            rgb_image[:, :, c] += weight * rgb_values[c]  
    
    # # 设置Alpha通道为总贡献的绝对值  
    # total_abs = np.sum(abs_contrib_np, axis=0)  
    # rgb_image[:, :, 3] = total_abs / np.max(total_abs)  
    
    # 确保RGB值在[0,1]范围内  
    rgb_image[:, :, :3] = np.clip(rgb_image[:, :, :3], 0, 1)  
    rgb_masked_map = np.full_like(rgb_image, np.nan, dtype=float)
    # 将zero_coords对应的位置填充为dominant_map_np的值
    for x, y in zero_coords:
        rgb_masked_map[x, y,:] = rgb_image[x, y, :] 

    ax.imshow(rgb_masked_map)  
    ax.set_title('Comprehensive visualization')  
    ax.set_xticks([])  
    ax.set_yticks([])  


    ax = plt.subplot(gs[2, 4]) 
    plot_contri(results['normalized_contribution'])
    plt.title('Stacked Bar Chart')
    '''    
    
    return fig  


def analyze_component_contributions(A, B, C, D, E, R, base, example_pixel=(32, 32), save_path=None, batch_idx=None, fig=None, category=None):  
    """  
    分析矩阵A,B,C,D对E的贡献，并可视化结果  
    
    参数:  
        A, B, C, D: 输入矩阵，形状 [1, 32, 64, 64]  
        E: 目标矩阵，形状 [1, 32, 64, 64]  
        base: 基础矩阵，形状 [1, 32, 64, 64]  
        example_pixel: 示例像素 (y, x) 用于详细分析  
        save_path: 保存图像的路径 (可选)  
        batch_idx: 批次索引，用于显示  
        fig: 现有的Figure对象(用于PDF保存模式)  
    """  
    # 计算贡献度  
    results = compute_contribution(A, B, C, D, E, R, base)  
    
    # 可视化结果  
    fig = visualize_contributions(results, save_path, example_pixel, batch_idx, fig, category=category)  
    
    return results, fig  


def batch_analyze_tensors(tensor_path, output_folder, num_batches=20, example_pixel=(32, 32)):  
    """  
    批量分析张量并生成可视化图像  
    
    参数:  
        tensor_path: 包含张量数据的.pkl文件路径  
        output_folder: 输出文件夹路径  
        num_batches: 要处理的批次数量  
        example_pixel: 示例像素坐标  
    """  
    # 创建输出文件夹  
    os.makedirs(output_folder, exist_ok=True)  
    
    # 加载数据  
    print(f"正在加载数据: {tensor_path}")  
    with open(tensor_path, 'rb') as f:  
        orin_tensor = pickle.load(f)  
    
    # 确保数据是列表且每个元素是元组  
    if not isinstance(orin_tensor, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in orin_tensor):  
        raise ValueError("加载的数据格式不正确，期望为列表，且每个元素是一个包含标号和特征的元组")  
    
    # 限制处理批次数量  
    num_batches = min(num_batches, len(orin_tensor))  
    
    base_name = os.path.basename(tensor_path)  # 获取文件基本名称  
    name_without_extension = base_name.split('.')[0]  # 去掉扩展名  
    pdf_filename = f'{output_folder}/{name_without_extension}_E_comb_all_batches.pdf'  # 创建完整的 PDF 文件名  
    
    # 创建Figure对象以在PDF中复用  
    fig = plt.figure(figsize=(20, 16))  
    
    with PdfPages(pdf_filename) as pdf:  
        # 按批次逐个处理  
        for batch_idx in range(num_batches):  
            print(f"处理批次 {batch_idx}/{num_batches-1}")  
            
            # 提取当前批次的标号和张量  
            batch_label, batch_tensor = orin_tensor[batch_idx]  
            
            # 确保张量形状正确  
            if batch_tensor.shape != (1, 5, 32, 64, 64):  
                raise ValueError(f"批次 {batch_idx} 的张量形状不正确，期望形状为 [1, 5, 32, 64, 64]，实际为 {batch_tensor.shape}")  
            
            # 分割张量，得到A、B、C、D和E  
            A, B, C, D, E = batch_tensor.split(1, dim=1)  # [1, 1, 32, 64, 64]  

            R = A + B + C + D
            # 计算基础矩阵  
            base = E - (A + B + C + D)  
            
            # 删除第二维，将[1, 1, 32, 64, 64]转换为[32, 64, 64]  
            A = A.squeeze()  
            B = B.squeeze()  
            C = C.squeeze()  
            D = D.squeeze()  
            E = E.squeeze()  
            R = R.squeeze()
            base = base.squeeze()  
            
            # 为当前批次创建单独的图像文件  
            individual_save_path = f"{output_folder}/batch_{batch_idx:03d}.png"  
            
            # 分析并生成可视化  
            _, fig = analyze_component_contributions(  
                A, B, C, D, E, R, base,  
                example_pixel=example_pixel,  
                save_path=individual_save_path,  
                batch_idx=batch_label,  
                fig=fig,
                category=name_without_extension  
            )  
            
            pdf.savefig(fig)  
            break
    
    print(f"所有批次处理完成!PDF已保存至: {pdf_filename}")  


if __name__ == "__main__":  
    # 数据路径和输出文件夹  
    folder_path = '/workspace/Infinity/output/infinity_2b_evaluation/mjhq30k_raw/pred'
    output_folder = "resi_contri_E"  
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pkl'):
            tensor_path = os.path.join(folder_path, file_name)
            # 运行批处理分析
            batch_analyze_tensors(
                tensor_path=tensor_path,
                output_folder=output_folder,
                num_batches=200,  # 只处理前200个批次
                example_pixel=(32, 32)  # 默认示例像素
            )