import os  
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  
import random
import os
import os.path as osp
import cv2
import numpy as np
import sys
path_to_add = os.path.join(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(path_to_add)  

from infinity.models.basic import scores_
import pickle
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
from tools.run_infinity import *
import pickle
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.special import rel_entr  # 用于计算 KL 散度
import os

model_path = '/share/public_models/Infinity/infinity_2b_reg.pth'   #'/home/model_data/infinity_2b_reg.pth'
vae_path = '/share/public_models/Infinity/infinity_vae_d32reg.pth' #'/home/model_data/infinity_vae_d32reg.pth'
text_encoder_ckpt = '/share/public_models/flan-t5-xl'     #'/home/model_data/flan-t5-xl'
 
import torch
import warnings  
import glob
# 忽略 FutureWarning 类型的警告  
warnings.filterwarnings("ignore", category=FutureWarning)  


# # print("ffff")
# print("vae mode:",vae)

# 读取 pkl 文件
def read_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# 将张量转换为图片
def tensor_to_image(tensor, vae):
    tensor = tensor.squeeze(-3)  # 去掉多余的维度
    img = vae.decode(tensor)
    img = (img + 1) / 2  # 将值归一化到 [0, 1]
    img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8)
    return img[0].cpu().numpy()  # 返回第一个图片并转换为 numpy 格式

# 将张量列表保存为图片到 PDF
def save_tensors_to_pdf(tensor_list, pdf_path, vae, title):
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(title, fontsize=18, y=0.98)
        rows, cols = 2, 4  # 网格布局
        for idx, tensor in enumerate(tensor_list):
            if idx >= rows * cols:  # 限制每页最多显示 rows * cols 张图片
                break
            image = tensor_to_image(tensor, vae)
            ax = fig.add_subplot(rows, cols, idx + 1)
            ax.imshow(image)
            ax.set_title(f"Item {idx}", fontsize=9)
            ax.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    prompts = {
        "vintage_insect": "A highly detailed, photorealistic insect sculpture crafted entirely from vintage 1960s electronic components, including capacitors, resistors, transistors, wires, diodes, solder, and circuit boards. The piece should showcase intricate textures and a retro-futuristic aesthetic.",
        "macro_closeup": "An extreme macro cinematographic close-up shot inspired by Denis Villeneuve's style, focusing on the intricate details of water droplets, ripples, or reflections, with a moody and atmospheric lighting.",
        "3d_school": "A vibrant and creative 3D render designed for the bottom of a mobile application's homepage, featuring a charming miniature school surrounded by tiny children carrying colorful backpacks, set in a playful and imaginative environment.",
        "explore_more": "A stunning and adventurous image featuring the text 'Explore More' in a bold, adventurous font, placed over a scenic hiking trail with lush greenery, towering mountains, and a clear blue sky, evoking a sense of wanderlust.",
        "toy_car": "A close-up, cinematic shot of a diecast toy car in a meticulously crafted diorama setting. The scene is set at night, with warm lights glowing from tiny windows, bokeh effects in the background, and a gentle dusting of snow adding a cozy, wintery ambiance.",
        # "fairy_house": "A photorealistic, ultra-detailed image of a fairy-like house with white walls and pink-tinted windows, surrounded by a lush garden filled with vibrant flowers. The scene should be scenic, cute, and magical, with intricate textures and a soft, dreamy atmosphere.",
        # "cat_fashion": "A hyperrealistic black and white photograph capturing a high-fashion cat runway show, styled in the dramatic, high-contrast fashion of Helmut Newton. The cats should be striking, elegant, and exuding sophistication.",
        # "spacefrog_astroduck": "A photorealistic scene featuring two superheroes, Spacefrog (a dashing green cartoon-like frog with a red cape) and Astroduck (a yellow fuzzy duck, part-robot, with blue/grey armor), standing near a garden pond next to their classic flying saucer, the Tadpole 3000. The scene should be dynamic and vivid.",
        # "miniature_village": "A whimsical and enchanted miniature village bustling with activity, featuring tiny houses, bustling markets, and tiny residents going about their daily lives. The scene should be rich in detail, with a magical and fairy-tale-like atmosphere.",
        # "corgi_dog": "A close-up, high-resolution photograph of a joyful Corgi dog wearing a black hat and round, dark sunglasses. The dog should have a happy expression, with its mouth open and tongue sticking out, exuding excitement and cheerfulness.",
        # "robot_eggplant": "A photorealistic image of a futuristic robot holding a massive eggplant, set against a sunny, natural background with lush greenery and a clear blue sky. The scene should blend technology and nature harmoniously.",
        # # "perfume_product": "A minimalist and highly detailed product photography setup featuring a sleek perfume bottle placed on a white marble table, accompanied by pineapple, coconut, and lime as decorative elements. The scene should be bright, concise, and layered with intricate details, evoking a fresh and luxurious ambiance.",
        # "mountain_landscape": "A breathtaking and picturesque mountainous landscape under a cloudy sky. The mountains are lush and green, dotted with trees and shrubs, while the valley below features a small rural settlement with scattered buildings. The scene should be captured from a high vantage point, offering a sweeping, serene view of the natural beauty."
        # "red_apple_simple": "A single red apple on a wooden table.",  
        # "blue_flower_simple": "A close-up of a blue flower.",  
        # "yellow_banana_simple": "A ripe yellow banana on a white surface.",  
        # "green_grape_simple": "A bunch of green grapes on a wooden table.",  
        # "white_cup_simple": "A white coffee cup filled with coffee.",  
        # "strawberry_simple": "A fresh red strawberry on a light background.",  
        # "orange_simple": "A whole orange on a wooden table.",  
        # "single_pear_simple": "A green pear on a light surface.",  
        # "lime_simple": "A sliced lime on a cutting board.",  
        # "blueberry_simple": "Fresh blueberries on a white background."  
    }

    args = argparse.Namespace(
        pn='1M',
        model_path=model_path,
        cfg_insertion_layer=0,
        vae_type=32,
        vae_path=vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type='infinity_2b',
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=text_encoder_ckpt,
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='/dev/shm',
        checkpoint_type='torch',
        seed=0,
        bf16=1,
        save_file='tmp.jpg',
        enable_model_cache=0
    )

    vae = load_visual_tokenizer(args)

    # for category, prompt in prompts.items():
    #     file_path = osp.join('outputs/codes', f"test_{category}.pkl")
    #     data = read_pkl_file(file_path)

    #     # 保存 test_partial_list 的图片
    #     summed_codes_8 = data['summed_codes_8']
    #     summed_codes_image = tensor_to_image(summed_codes_8, vae)
    #     pdf_path_partial = f'/home/lianyaoxiu/lianyaoxiu/Infinity/outputs/plots1/{osp.basename(file_path).replace(".pkl", "_summed_8.pdf")}'
    #     with PdfPages(pdf_path_partial) as pdf:
    #         fig = plt.figure(figsize=(8, 8))
    #         plt.imshow(summed_codes_image)
    #         plt.title("Summed Codes 8 Images")
    #         plt.axis('off')
    #         pdf.savefig(fig)
    #         plt.close(fig)        

    #     # 保存 summed_codes_8 的图片
    #     summed_codes = data['summed_codes']
    #     summed_codes_image = tensor_to_image(summed_codes, vae)
    #     pdf_path_summed = f'/home/lianyaoxiu/lianyaoxiu/Infinity/outputs/plots1/{osp.basename(file_path).replace(".pkl", "_summed_codes.pdf")}'
    #     with PdfPages(pdf_path_summed) as pdf:
    #         fig = plt.figure(figsize=(8, 8))
    #         plt.imshow(summed_codes_image)
    #         plt.title("Summed Codes 8 Images")
    #         plt.axis('off')
    #         pdf.savefig(fig)
    #         plt.close(fig)

    #     print(f"Processing for file {file_path} completed!")
    file_paths = glob.glob('/root/lianyaoxiu/lianyaoxiu/Infinity/outputs/codes_mtp/test_combined_data_*.pkl')

    # 处理每个文件
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        data = read_pkl_file(file_path)

        # 保存 test_partial_list 的图片
        test_partial_list = data['test_partial_list']  
        pdf_path_partial = f'/root/lianyaoxiu/lianyaoxiu/Infinity/outputs/plots_para/{osp.basename(file_path).replace(".pkl", "_test_partial_list_images.pdf")}'
        save_tensors_to_pdf(test_partial_list, pdf_path_partial, vae, "Test Partial List Images")

        # 保存 summed_codes_8 的图片
        summed_codes_8 = data['summed_codes_para']
        summed_codes_8_image = tensor_to_image(summed_codes_8, vae)
        pdf_path_summed = f'/root/lianyaoxiu/lianyaoxiu/Infinity/outputs/plots_para/{osp.basename(file_path).replace(".pkl", "_summed_codes_para_images.pdf")}'
        with PdfPages(pdf_path_summed) as pdf:
            fig = plt.figure(figsize=(8, 8))
            plt.imshow(summed_codes_8_image)
            plt.title("Summed Codes 8 Images")
            plt.axis('off')
            pdf.savefig(fig)
            plt.close(fig)

        # 计算所有值的总和并保存为图片
        total_sum_tensor = sum(test_partial_list) + data['summed_codes_para']
        total_sum_image = tensor_to_image(total_sum_tensor, vae)
        pdf_path_total = f'/root/lianyaoxiu/lianyaoxiu/Infinity/outputs/plots_para/{osp.basename(file_path).replace(".pkl", "_total_sum_image.pdf")}'
        with PdfPages(pdf_path_total) as pdf:
            fig = plt.figure(figsize=(8, 8))
            plt.imshow(total_sum_image)
            plt.title("Total Sum Image")
            plt.axis('off')
            pdf.savefig(fig)
            plt.close(fig)

        print(f"Processing for file {file_path} completed!")

