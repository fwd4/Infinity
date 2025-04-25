import os  
import torch  
import numpy as np  
from PIL import Image as PImage  
import argparse  
from utils.misc import create_npz_from_sample_folder  

################## 1. Download checkpoints and build models
import os
import torch, torchvision
import random



setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var
import time
MODEL_DEPTH = 30    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30, 36}

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
# download checkpoint
# hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
# vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
# if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
# if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

# build vae, var
FOR_512_px = MODEL_DEPTH == 36
if FOR_512_px:
    patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
else:
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
        flash_if_available=False
    )

vae_ckpt = '/home/model_data/var/vae_ch160v4096z32.pth'
var_ckpt = '/home/model_data/var/var_d30.pth'
# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location=torch.device('cuda')), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location=torch.device('cuda')), strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')

############################# 2. Sample with classifier-free guidance

# set args
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg = 5 #@param {type:"slider", min:1, max:10, step:0.1}
class_labels = (980, 980, 437, 437, 22, 22, 562, 562)  #@param {type:"raw"}
class_labels = (980, 437, 22, 100,980, 437, 22, 100,980, 437, 22, 100,980, 437, 22, 100)  #@param {type:"raw"}
more_smooth = False # True for more smooth output

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')


# sample
B = len(class_labels)
label_B: torch.LongTensor = torch.tensor(class_labels, device=device)

# 解析命令行参数  
parser = argparse.ArgumentParser(description='Generate images for specific class range')  
parser.add_argument('--start_class', type=int, required=True, help='开始类别索引')  
parser.add_argument('--end_class', type=int, required=True, help='结束类别索引（不含）')  
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')  
parser.add_argument('--images_per_class', type=int, default=50, help='每个类别生成的图片数量')  
parser.add_argument('--output_folder', type=str, default='outputs/fid_samples', help='输出文件夹')  
# parser.add_argument('--create_npz', action='store_true', help='是否创建npz文件（仅最后一个进程需要）')  
args = parser.parse_args()  

# 设置设备  
device = f'cuda:0'  
torch.cuda.set_device(0)  

# 设置输出文件夹  
output_folder = args.output_folder  
os.makedirs(output_folder, exist_ok=True)  

# 设置参数  
images_per_class = args.images_per_class  
cfg = 1.5  
top_p = 0.96  
top_k = 900  
more_smooth = False  

# 打印任务信息  
print(f"GPU {args.gpu_id} 处理类别 {args.start_class} 到 {args.end_class-1}，每类 {images_per_class} 张图片")  

# 这里需要加载模型，与原代码保持一致  
# 比如: var = YourModel().to(device)  
# ...  

# 生成图片  
with torch.inference_mode():  
    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):  
        for class_id in range(args.start_class, args.end_class):  
            print(f"GPU {args.gpu_id} 正在处理类别 {class_id}...")  
            label_B = torch.tensor([class_id] * images_per_class, device=device)  
            recon_B3HW = var.autoregressive_infer_cfg(  
                B=images_per_class, label_B=label_B, cfg=cfg, top_k=top_k,   
                top_p=top_p, g_seed=class_id, more_smooth=more_smooth  
            )  
            for i, img in enumerate(recon_B3HW):  
                chw = img.permute(1, 2, 0).mul_(255).cpu().numpy()  
                chw = PImage.fromarray(chw.astype(np.uint8))  
                img_path = os.path.join(output_folder, f'class_{class_id}_img_{i}.png')  
                chw.save(img_path)  
            print(f"GPU {args.gpu_id} 完成类别 {class_id} 的生成")  

print(f"GPU {args.gpu_id} 已完成类别 {args.start_class} 到 {args.end_class-1} 的全部处理")  

# # 只有在指定创建npz时才创建（通常由最后一个完成的进程执行）  
# if args.create_npz:  
#     print(f"创建NPZ文件...")  
#     npz_file_path = 'outputs/fid_samples.npz'  
#     create_npz_from_sample_folder(output_folder, npz_file_path)  
#     print(f"FID samples saved to {npz_file_path}")