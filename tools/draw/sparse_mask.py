import numpy as np
from scipy import sparse

def create_sparse_matrix(kv_opt: bool):
    # 定义基本参数
    M = 4096
    pix = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
    qlen_raw = [x*x for x in pix]
    qlen = np.cumsum(qlen_raw)
    K = qlen[-1] if not kv_opt else qlen[8] + pix[-1]**2
    
    # 创建全零矩阵
    matrix = np.zeros((M, K), dtype=np.float32)
    
    # 1. 前921列全部设为随机值
    print(qlen[8])
    matrix[:, :qlen[8]] = np.random.normal(0, 0.02, size=(M, qlen[8]))
    
    # 2. 最后4096列以列id为中心，前后512个元素设为随机值
    offset = qlen[8] if kv_opt else qlen[-2]
    for k in range(4096):
        center = k + offset
        start = max(offset, center - 512)
        end = min(K, center + 512)
        matrix[k, start:end] = np.random.normal(0, 0.02, size=(end-start))
    
    return matrix


def plot_and_save(kv_opt):
    # 创建矩阵
    sparse_matrix = create_sparse_matrix(kv_opt)

    # 转换为CSR格式
    csr_matrix = sparse.csr_matrix(sparse_matrix)

    # 保存为npz文件
    sparse.save_npz(f'sparse_matrix_kv_opt{kv_opt}.npz', csr_matrix)

    # 打印基本信息
    print(f"矩阵形状: {sparse_matrix.shape}")
    print(f"非零元素比例: {np.count_nonzero(sparse_matrix) / sparse_matrix.size:.4f}")
    print(f"非零元素数量: {np.count_nonzero(sparse_matrix)}")

    # 可视化稀疏矩阵
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 20))
    plt.spy(sparse_matrix, markersize=0.1)
    plt.title("Sparse Matrix Visualization")
    plt.savefig(f'sparse_matrix_kv_opt{kv_opt}.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_and_save(1)