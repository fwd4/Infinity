################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import transformers
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var
import time
MODEL_DEPTH = 30    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30, 36}

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
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

vae_ckpt = '/share/public_models/var/vae_ch160v4096z32.pth'
var_ckpt = '/share/public_models/var/var_d30.pth'
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

''' 
#Test Time
# ####"warm up"###########
# with torch.inference_mode():
#     with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
#         for i in range(10):
#             recon_B3HW,total_times,iteration_times,\
#                 iteration_times2,iteration_times1 = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)
            
# with torch.inference_mode():
#     with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
#         total_start = time.perf_counter()
#         recon_B3HW,total_times,iteration_times,\
#             iteration_times2,iteration_times1 = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)
#         torch.cuda.synchronize()
#         total_end = time.perf_counter()
#         print(f"{(total_start-total_end)*1000:.3f}ms")

# for duration in total_times:  
#     print(f"总耗时: {duration:.3f}ms")
# for stage, duration in iteration_times:  
#     print(f"第 {stage} 阶段耗时: {duration:.3f}ms")
# for stage, duration in iteration_times2:  
#     print(f"第 {stage} 阶段耗时: {duration:.3f}ms") 
# for stage, duration in iteration_times1:  
#     print(f"第 {stage} 阶段耗时: {duration:.3f}ms") 

# print(total_times)
# print(iteration_times)
# print(iteration_times2)
# print(iteration_times1)
'''

###"warm up"###########
with torch.inference_mode():
    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
        for i in range(2):
            recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)
            
with torch.inference_mode():
    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
        total_start = time.perf_counter()
        recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)
        # torch.cuda.synchronize()
        total_end = time.perf_counter()
        print(f"TOTAL TIME {(total_end-total_start)*1000:.3f}ms")

print(recon_B3HW.shape)  #torch.Size([8, 3, 256, 256])
chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
chw = PImage.fromarray(chw.astype(np.uint8))
chw.show()
chw.save('outputs/image_mtp_9_[0]1.png')  # 保存为PNG格式
