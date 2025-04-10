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

model_path = '/share/public/public_models/Infinity/infinity_2b_reg.pth'   #'/home/model_data/infinity_2b_reg.pth'
vae_path = '/share/public/public_models/Infinity/infinity_vae_d32reg.pth' #'/home/model_data/infinity_vae_d32reg.pth'
text_encoder_ckpt = '/share/public/public_models/Infinity/flan-t5-xl'     #'/home/model_data/flan-t5-xl'
 
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
    enable_model_cache=0

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

######################## TEST ########################
# # Create a non-zero tensor with the desired shape
# summed_codes = torch.zeros(1, 32, 1, 64, 64).cuda()

# # Create a mask for the right upper corner
# mask = torch.ones_like(summed_codes).cuda()
# mask[:, :, :, 32:, 32:] = 0  # Set the right upper corner to zero

# # Apply the mask to summed_codes
# summed_codes = summed_codes * mask

# img = vae.decode(summed_codes.squeeze(-3))
# img = (img + 1) / 2
# img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
# image= img[0]
# save_pic = True
# if save_pic:
#     save_path = osp.join("/home/lianyaoxiu/lianyaoxiu/Infinity", "zeros_vae.jpg")
#     os.makedirs(osp.dirname(save_path), exist_ok=True)

#     succ = cv2.imwrite(save_path, image.cpu().numpy())
#     print(f"Save successful: {succ}")
#     print(f" image saved to {save_path}")
######################## TEST ########################


get_torch_mem_usage()
infinity = load_transformer(vae, args)
get_torch_mem_usage()

# print(infinity)
# PROMPT
# prompts = {
#     "vintage_insect": "Insect made from vintage 1960s electronic components, capacitors, resistors, transistors, wires, diodes, solder, circuitboard.",
#     "macro_closeup": "Denis Villeneuve's extreme macro cinematographic close-up in water.",
#     "3d_school": "A creative 3D image to be placed at the bottom of a mobile application's homepage, depicting a miniature school and children carrying backpacks.",
#     "explore_more": "Create an image with 'Explore More' in an adventurous font over a picturesque hiking trail.",
#     "toy_car": "Close-up shot of a diecast toy car, diorama, night, lights from windows, bokeh, snow.",
#     "fairy_house": "House: white; pink tinted windows; surrounded by flowers; cute; scenic; garden; fairy-like; epic; photography; photorealistic; insanely detailed and intricate; textures; grain; ultra-realistic.",
#     "cat_fashion": "Hyperrealistic black and white photography of cats fashion show in style of Helmut Newton.",
#     "spacefrog_astroduck": "Two superheroes called Spacefrog (a dashing green cartoon-like frog with a red cape) and Astroduck (a yellow fuzzy duck, part-robot, with blue/grey armor), near a garden pond, next to their spaceship, a classic flying saucer, called the Tadpole 3000. Photorealistic.",
#     "miniature_village": "An enchanted miniature village bustling with activity, featuring tiny houses, markets, and residents.",
#     "corgi_dog": "A close-up photograph of a Corgi dog. The dog is wearing a black hat and round, dark sunglasses. The Corgi has a joyful expression, with its mouth open and tongue sticking out, giving an impression of happiness or excitement.",
#     "robot_eggplant": "a robot holding a huge eggplant, sunny nature background",
#     "perfume_product": "Product photography, a perfume placed on a white marble table with pineapple, coconut, lime next to it as decoration, white curtains, full of intricate details, realistic, minimalist, layered gestures in a bright and concise atmosphere, minimalist style.",
#     "mountain_landscape": "The image presents a picturesque mountainous landscape under a cloudy sky. The mountains, blanketed in lush greenery, rise majestically, their slopes dotted with clusters of trees and shrubs. The sky above is a canvas of blue, adorned with fluffy white clouds that add a sense of tranquility to the scene. In the foreground, a valley unfolds, nestled between the towering mountains. It appears to be a rural area, with a few buildings and structures visible, suggesting the presence of a small settlement. The buildings are scattered, blending harmoniously with the natural surroundings. The image is captured from a high vantage point, providing a sweeping view of the valley and the mountains."
# }

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
output_dir = "./outputs/pics3"
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

    # with open(f'outputs/codes/test_partial_pixel_data_{category}.pkl', 'rb') as file:
    #     data = pickle.load(file)

    # pdf_path = osp.join(output_dir, f"combine_{category}.pdf")
    # with PdfPages(pdf_path) as pdf:
    #     for key, value_list in data.items():
    #         fig = plt.figure(figsize=(20, 16))  # 更大的尺寸以适应 8x4 网格  
    #         fig.suptitle(f"Key: {key}", fontsize=18, y=0.98)  
            
    #         # 计算网格布局  
    #         rows = 4  
    #         cols = 8  
    #         for idx, item in enumerate(value_list):
    #             tensor_data = torch.tensor(item)
    #             img = vae.decode(tensor_data.squeeze(-3))
    #             img = (img + 1) / 2
    #             img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
    #             image = img[0]
    #             image = image.cpu().numpy()
    #             ax = fig.add_subplot(rows, cols, idx + 1)  
    #             ax.imshow(image)  
    #             ax.set_title(f"Item {idx}", fontsize=9)  
    #             ax.axis('off')  
            
    #         # 调整子图之间的间距  
    #         plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留出空间  
    #         # 保存当前页面到PDF  
    #         pdf.savefig(fig)  
    #         plt.close(fig)  
    # print(f"All images for category '{category}' saved to {pdf_path}")


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
    )

    # img_cnt+=1
    # if img_cnt == 1:
    #     exit(0)

    # SAVE
    save_pic = False
    if save_pic:
        save_path = osp.join(output_dir, f"{category}_orign.jpg")
        cv2.imwrite(save_path, generated_image.cpu().numpy())
        print(f"{category} image saved to {save_path}")

# np.save('attn_time.npy', np.array(ATTN_TIME))
