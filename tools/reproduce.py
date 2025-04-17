import os  
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  
import random
import os
import os.path as osp
import cv2
import numpy as np
from run_infinity import *
from infinity.models.basic import scores_
import pickle
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

model_path = '/share/public_models/Infinity/infinity_2b_reg.pth'   #'/home/model_data/infinity_2b_reg.pth'
vae_path = '/share/public_models/Infinity/infinity_vae_d32reg.pth' #'/home/model_data/infinity_vae_d32reg.pth'
text_encoder_ckpt = '/share/public_models/flan-t5-xl'     #'/home/model_data/flan-t5-xl'
 
import torch
import warnings  
# 忽略 FutureWarning 类型的警告  
warnings.filterwarnings("ignore", category=FutureWarning)  
# SET
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
    enable_model_cache=0,
    si_list = [9,10,11,12],
    ratio_list = [60,50,0,0],

)

'''
torch.cuda.set_device(0)
model_path='/share/public/public_models/Infinity/infinity_8b_weights'
vae_path='/share/public/public_models/Infinity/infinity_vae_d56_f8_14_patchify.pth'
text_encoder_ckpt = '/share/public/public_models/flan-t5-xl'
args=argparse.Namespace(
    pn='1M',
    model_path=model_path,
    cfg_insertion_layer=0,
    vae_type=14,
    vae_path=vae_path,
    add_lvl_embeding_only_first_block=1,
    use_bit_label=1,
    model_type='infinity_8b',
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0,
    sampling_per_bits=1,
    text_encoder_ckpt=text_encoder_ckpt,
    text_channels=2048,
    apply_spatial_patchify=1,
    h_div_w_template=1.000,
    use_flex_attn=1,
    cache_dir='/dev/shm',
    checkpoint_type='torch_shard',
    seed=0,
    bf16=1,
    save_file='tmp.jpg'
)
'''

# LOAD
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
get_torch_mem_usage()
vae = load_visual_tokenizer(args)
# # print("ffff")
# print("vae mode:",vae)



get_torch_mem_usage()
infinity = load_transformer(vae, args)
get_torch_mem_usage()


prompts = {
    "vintage_insect": "A highly detailed, photorealistic insect sculpture crafted entirely from vintage 1960s electronic components, including capacitors, resistors, transistors, wires, diodes, solder, and circuit boards. The piece should showcase intricate textures and a retro-futuristic aesthetic.",
    # "macro_closeup": "An extreme macro cinematographic close-up shot inspired by Denis Villeneuve's style, focusing on the intricate details of water droplets, ripples, or reflections, with a moody and atmospheric lighting.",
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
# OUTPUT
output_dir = "./outputs/pics_fastvar"
os.makedirs(output_dir, exist_ok=True)

img_cnt = 0
# GEN IMG
for category, prompt in prompts.items():
    cfg = 3
    tau = 0.5
    h_div_w = 1/1 # Aspect Ratio
    #seed = random.randint(0, 10000)
    seed = 42
    enable_positive_prompt = 0

    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    # GEN
    generated_image = gen_one_img(
        infinity,
        vae,
        text_tokenizer,
        text_encoder,
        prompt,
        category=category,
        g_seed=seed,
        gt_leak=0,
        gt_ls_Bl=None,
        cfg_list=cfg,
        tau_list=tau,
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        enable_positive_prompt=enable_positive_prompt,
        verbose=False,
        si_list = args.si_list,
        ratio_list = args.ratio_list,
    )

    # img_cnt+=1
    # if img_cnt == 1:
    #     exit(0)

    # SAVE
    save_pic = True
    if save_pic:
        save_path = osp.join(output_dir, f"{category}-{args.ratio_list}.jpg")
        cv2.imwrite(save_path, generated_image.cpu().numpy())
        print(f"{category} image saved to {save_path}")

# np.save('attn_time.npy', np.array(ATTN_TIME))
