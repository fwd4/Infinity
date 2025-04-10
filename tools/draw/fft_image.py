import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def tensor_to_frequency_magnitude(tensor):
    """
    将形状为 (H, W, C) 的张量转换为频域并计算幅度图。

    参数:
        tensor (np.ndarray): 输入张量，形状为 (H, W, C)

    返回:
        magnitude_spectrum (np.ndarray): 幅度谱图，形状为 (H, W, C)
    """
    # 对每个通道做二维傅里叶变换
    fft_result = np.fft.fft2(tensor, axes=(0, 1))

    # 将零频率成分移到中心
    #fft_shifted = np.fft.fftshift(fft_result, axes=(0, 1))

    # 计算幅度谱
    magnitude_spectrum = np.abs(fft_result)

    return magnitude_spectrum

# 示例使用
if __name__ == "__main__":
    # 生成一个随机张量，形状为 (H, W, C)
    # H, W, C = 128, 128, 3
    # tensor = np.random.rand(H, W, C)
    tensor = Image.open('/workspace/Infinity/outputs/re_cat_fashion_test.jpg').convert('L')
    magnitude = tensor_to_frequency_magnitude(tensor)
    if len(magnitude.shape) == 2:
        # 切面统计
        u_mean = -np.log(magnitude.mean(axis=0))  # 沿 v 方向平均（列向）
        v_mean = -np.log(magnitude.mean(axis=1))  # 沿 u 方向平均（行向）

        # 可视化
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(u_mean)
        axs[0].set_title('Mean along v-axis (columns)')
        axs[0].set_xlabel('u frequency')
        axs[0].set_ylabel('Mean amplitude')

        axs[1].plot(v_mean)
        axs[1].set_title('Mean along u-axis (rows)')
        axs[1].set_xlabel('v frequency')
        axs[1].set_ylabel('Mean amplitude')
    else:
        for i in range(C):
            plt.subplot(1, C, i + 1)
            plt.imshow(np.log1p(magnitude[:, :, i]), cmap='gray')  # log1p 以便更清晰地显示细节
            plt.title(f'Channel {i}')
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'fft_amplitude.png', dpi=300, bbox_inches='tight')
    #plt.show()
