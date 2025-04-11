import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def create_comparison_matrix(folder1, folder2, output_path):
    # 获取两个文件夹中的图片
    images1 = sorted(list(Path(folder1).glob('*.jpg')))
    images2 = sorted(list(Path(folder2).glob('*.jpg')))
    
    if len(images1) != len(images2):
        raise ValueError("两个文件夹中的图片数量不一致")
    
    # 创建13x2的网格布局
    fig, axes = plt.subplots(13, 2, figsize=(20, 100))
    fig.suptitle('Attention Score Comparison', fontsize=16, y=0.92)
    
    # 填充图片
    for idx, (img1_path, img2_path) in enumerate(zip(images1, images2)):
        # 读取图片
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        # BGR to RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # 显示图片
        axes[idx, 0].imshow(img1)
        axes[idx, 0].set_title(f'Model 1: {img1_path.stem}')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(img2)
        axes[idx, 1].set_title(f'Model 2: {img2_path.stem}')
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    folder1 = "masked_images"
    folder2 = "raw_images"
    output_path = "compare_res.png"
    
    create_comparison_matrix(folder1, folder2, output_path)