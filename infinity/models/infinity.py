"""
Definition of Infinity transformer model.
测试每个stage每个block的结果
"""
import matplotlib.pyplot as plt
import math
import random
import time
import pickle
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model
from torch.utils.checkpoint import checkpoint
from PIL import Image
import numpy as np

import infinity.utils.dist as dist
from infinity.utils.dist import for_visualize
from infinity.models.basic import flash_attn_func, flash_fused_op_installed, AdaLNBeforeHead, CrossAttnBlock, SelfAttnBlock, CrossAttention, FastRMSNorm, precompute_rope2d_freqs_grid
from infinity.utils import misc
from infinity.models.flex_attn import FlexAttn
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

import matplotlib.pyplot as plt  
import seaborn as sns  
from matplotlib.backends.backend_pdf import PdfPages  

def get_freq_old(codes, pn, ratio1):  
    flatten_sum = codes.reshape(-1, pn, pn)  
    dc_component = F.avg_pool2d(flatten_sum, pn)  
    dc_diff = torch.norm(flatten_sum - dc_component, dim=0).flatten()  
    total_sz = dc_diff.shape[0]  
    
    # 确保ratio1是降序排列的  
    ratio1 = sorted(ratio1, reverse=True)  
    
    # 创建一个字典来存储每个比例对应的topk索引  
    topk_indices = {}  
    for ratio in ratio1:  
        _, topk = torch.topk(dc_diff, total_sz * ratio // 100)  
        topk_indices[ratio] = set(topk.cpu().numpy())  
    
    # 创建返回的差异集张量列表  
    masks = []  
    device = codes.device  
    
    # 反向处理，从最小比例开始  
    for i in range(len(ratio1)-1, -1, -1):  
        if i == len(ratio1) - 1:  # 最小的比例（例如top5）  
            mask_set = topk_indices[ratio1[i]]  
        else:  
            mask_set = topk_indices[ratio1[i]] - topk_indices[ratio1[i+1]]  
        
        masks.append(torch.tensor(list(mask_set), dtype=torch.long, device=device))  
    
    # 返回所有mask张量，顺序是从小比例到大比例差：top5, top10-top5, top30-top10, top50-top30  
    return tuple(masks)  

def get_freq(codes_list, pn_list, ratio_list):
    """
    计算每个 last_stage_list 中的 top 比例索引，并返回对应的 mask_list。

    参数:
        codes_list: List[Tensor], 每个 Tensor 的形状为 [B, d, h, w]
        pn_list: List[int], 每个对应的分辨率 pn
        ratio_list: List[int], 每个对应的比例，例如 [50, 30, 10, 5]

    返回:
        mask_list: List[Tensor], 每个 Tensor 包含对应比例的索引
    """
    assert len(codes_list) == len(pn_list) == len(ratio_list), "codes_list, pn_list 和 ratio_list 的长度必须相同"

    mask_list = []  # 用于存储每个比例的 mask
    device = codes_list[0].device  # 假设所有张量都在同一个设备上

    for codes, pn, ratio in zip(codes_list, pn_list, ratio_list):
        # 将 codes 重塑为 [-1, pn, pn]
        flatten_sum = codes.reshape(-1, pn, pn)

        # 计算 DC 分量
        dc_component = F.avg_pool2d(flatten_sum, pn)  
        # 计算差异并展平
        dc_diff = torch.norm(flatten_sum - dc_component, dim=0).flatten()
        # 获取总元素数量
        total_sz = dc_diff.numel()

        # 计算当前比例的 top 索引范围
        high_ratio = ratio
        low_ratio = ratio_list[ratio_list.index(ratio) + 1] if ratio_list.index(ratio) + 1 < len(ratio_list) else 0

        # 获取 top_high 和 top_low 的索引
        top_high_indices = torch.topk(dc_diff, total_sz * high_ratio // 100, largest=True, sorted=False).indices
        top_low_indices = torch.topk(dc_diff, total_sz * low_ratio // 100, largest=True, sorted=False).indices

        # 计算 mask（高比例减去低比例）
        mask_set = set(top_high_indices.cpu().numpy()) - set(top_low_indices.cpu().numpy())
        # mask = torch.tensor(list(mask_set), dtype=torch.long, device=device)
        mask = list(mask_set)

        # 将 mask 添加到 mask_list
        mask_list.append(mask)

    return mask_list

def process_and_concat_last_stage(last_stage_list, mask_list):
    """
    处理 last_stage_list 中的每个张量，按照指定步骤操作，并拼接成一个新的张量。

    参数:
        last_stage_list: List[Tensor], 每个张量的形状为 [B, d, 1, h, w]
        mask_list: List[Tensor], 每个张量包含对应的索引

    返回:
        new_last_stage: Tensor, 拼接后的新张量，形状为 [B, total_mask_len, d]
    """
    processed_list = []  # 用于存储处理后的张量

    for last_stage, mask in zip(last_stage_list, mask_list):
        # 1. squeeze(-3) -> [B, d, h, w]
        last_stage = last_stage.squeeze(-3)

        # 2. reshape -> [B, d, h*w]
        B, d, h, w = last_stage.shape
        last_stage = last_stage.reshape(B, d, h * w)

        # 3. 根据 mask 取对应的索引 -> [B, d, mask_len]
        last_stage = last_stage[:, :, mask]

        # 4. 转置为 [B, mask_len, d] 并添加到列表
        last_stage = last_stage.permute(0, 2, 1)  # [B, mask_len, d]
        processed_list.append(last_stage)

    # 5. 拼接所有处理后的张量 -> [B, total_mask_len, d]
    new_last_stage = torch.cat(processed_list, dim=1)

    return new_last_stage

# def get_freq(codes,pn):
#     flatten_sum = codes.reshape(-1, pn, pn)
#     dc_component = F.avg_pool2d(flatten_sum, pn)
#     #import pdb; pdb.set_trace()
#     dc_diff = torch.norm(flatten_sum - dc_component, dim=0).flatten()
#     total_sz = dc_diff.shape[0]
#     _, top50 = torch.topk(dc_diff, total_sz * 50 // 100)
#     _, top15 = torch.topk(dc_diff, total_sz * 15 // 100)
#     _, top5 = torch.topk(dc_diff, total_sz * 5 // 100)
#     _, top100 = torch.topk(dc_diff, total_sz * 100 // 100)
#     mask_minus_1_set = set(top5.cpu().numpy())
#     mask_minus_2_set = set(top15.cpu().numpy()) - set(top5.cpu().numpy())
#     mask_minus_3_set = set(top50.cpu().numpy()) - set(top15.cpu().numpy())
#     device = codes.device  
#     mask_minus_1 = torch.tensor(list(mask_minus_1_set), dtype=torch.long, device=device)  
#     mask_minus_2 = torch.tensor(list(mask_minus_2_set), dtype=torch.long, device=device)  
#     mask_minus_3 = torch.tensor(list(mask_minus_3_set), dtype=torch.long, device=device)  

#     return mask_minus_1, mask_minus_2, mask_minus_3

def cosine_similarity(matrix1, matrix2):  
    # 展平矩阵为1D向量  
    vector1 = matrix1.flatten()  
    vector2 = matrix2.flatten()  
    
    # 计算点积  
    dot_product = torch.dot(vector1, vector2)  

    # 计算向量的范数  
    norm1 = torch.linalg.norm(vector1)  
    norm2 = torch.linalg.norm(vector2)  

    # 计算余弦相似度  
    similarity = dot_product / (norm1 * norm2)  

    return similarity

def cosine_similarity(matrix1, matrix2):  

    # 归一化矩阵的每一行（转换为单位向量）  
    norm1 = torch.linalg.norm(matrix1, dim=1, keepdim=True)  
    norm2 = torch.linalg.norm(matrix2, dim=1, keepdim=True)  
    
    matrix1_normalized = matrix1 / norm1  
    matrix2_normalized = matrix2 / norm2  
    
    # 计算余弦相似度矩阵  
    similarity_matrix = torch.mm(matrix1_normalized, matrix2_normalized.T)  
    
    return similarity_matrix

def compute_diff_ratio(last_stage, diff):
    # last_stage_max = last_stage.max().item()  
    # last_stage_min = last_stage.min().item()
    # last_stage_mean = last_stage.mean().item()  # 计算均值
    # last_stage_median = last_stage.median().item()  # 计算中位数
    # avg_max = diff.max().item()  
    # avg_min = diff.min().item()  
    # diff_mean = diff.mean().item()  # 计算均值
    # diff_median = diff.median().item()  # 计算中位数

    # 计算小于0.1和0.01的百分比  
    total_elements = diff.numel()  
    less_than_01 = (diff < 0.1).sum().item()  
    less_than_005 = (diff < 0.05).sum().item()  
    less_than_001 = (diff < 0.01).sum().item()  

    percent_less_01 = (less_than_01 / total_elements) * 100  
    percent_less_005 = (less_than_005 / total_elements) * 100 
    percent_less_001 = (less_than_001 / total_elements) * 100 

    relative_diff = diff / last_stage.abs()
    less_than_10_percent = (relative_diff < 0.1).sum().item()  
    less_than_5_percent = (relative_diff < 0.05).sum().item()  
    less_than_1_percent = (relative_diff < 0.01).sum().item()  

    percent_less_10 = (less_than_10_percent / total_elements) * 100  
    percent_less_5 = (less_than_5_percent / total_elements) * 100 
    percent_less_1 = (less_than_1_percent / total_elements) * 100 

    
    # print(f"Self Value Max: {last_stage_max:.2e}, Min: {last_stage_min:.2e}, Mean: {last_stage_mean:.2e}, Median: {last_stage_median:.2e}, "
    #    f'Average Difference Max: {avg_max:.2e}, Min: {avg_min:.2e}, Mean: {diff_mean:.2e}, Median: {diff_median:.2e} ,'
    #    f'<10: {percent_less_10:.2f}%, <5: {percent_less_5:.2f}%, <1: {percent_less_1:.2f}% ' 
    #    f'<0.1: {percent_less_01:.2f}%, <0.05: {percent_less_005:.2f}%, <0.01: {percent_less_001:.2f}% ')
    return percent_less_10

def plot_three_heatmaps(diff, title, pdf):  
    """  
    绘制三张并排的热力图, 显示两个batch的差异和平均值  
    """   

    fig, ax = plt.subplots(figsize=(10, 8))  # 创建图形和单个轴   
    # 提取每个矩阵的最大值和最小值  
    # batch1_max = diff_batch1.max().item()  
    # batch1_min = diff_batch1.min().item()  
    # batch2_max = diff_batch2.max().item()  
    # batch2_min = diff_batch2.min().item()  
    avg_max = diff.max().item()  
    avg_min = diff.min().item()  

    # 计算小于0.1和0.01的百分比  
    diff_np = diff.detach().cpu().numpy() 
    total_elements = diff_np.size  
    less_than_01 = np.sum(diff_np < 0.1)  
    less_than_001 = np.sum(diff_np < 0.01)  

    percent_less_01 = (less_than_01 / total_elements) * 100  
    percent_less_001 = (less_than_001 / total_elements) * 100 
    print(f'Average Difference\nMax: {avg_max:.2e}, Min: {avg_min:.2e}, <0.1: {percent_less_01:.2f}%, <0.01: {percent_less_001:.2f}%')
    # # 绘制第一个batch的热力图  
    # sns.heatmap(diff_batch1.detach().cpu().numpy(),   
    #             ax=ax1,   
    #             annot=True,   
    #             fmt='.2e',   
    #             cmap='viridis')  
    # ax1.set_title(f'Batch 1 Difference\nMax: {batch1_max:.2e}, Min: {batch1_min:.2e}')  

    # # 绘制第二个batch的热力图  
    # sns.heatmap(diff_batch2.detach().cpu().numpy(),   
    #             ax=ax2,   
    #             annot=True,   
    #             fmt='.2e',   
    #             cmap='viridis')  
    # ax2.set_title(f'Batch 2 Difference\nMax: {batch2_max:.2e}, Min: {batch2_min:.2e}')  

    

    sns.heatmap(diff_np,     
                ax=ax,  # 指定要绘制的轴  
                annot=False,   
                fmt='.2e',   
                cmap='viridis')  

    ax.set_title(f'Average Difference\nMax: {avg_max:.2e}, Min: {avg_min:.2e}, <0.1: {percent_less_01:.2f}%, <0.01: {percent_less_001:.2f}%'  )  

    # 设置总标题  
    plt.suptitle(title, fontsize=16, y=1.02)  # 使用suptitle，并稍微调整垂直位置  

    plt.tight_layout()   

    # 保存图片  
    pdf.savefig(fig)  
    plt.close()  

def get_torch_mem_usage():
    a, r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    print(f"allocated: {a/1024**3:.2f}GB, reserved: {r/1024**3:.2f}GB")

try:
    from infinity.models.fused_op import fused_ada_layer_norm, fused_ada_rms_norm
except:
    fused_ada_layer_norm, fused_ada_rms_norm = None, None

# ATTN_TIME=[]

from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity

trace_handler = tensorboard_trace_handler(dir_name=f"outputs/profile", use_gzip=False)

class MultiInpIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class TextAttentivePool(nn.Module):
    def __init__(self, Ct5: int, D: int):
        super().__init__()
        self.Ct5, self.D = Ct5, D
        if D > 4096:
            self.head_dim = 64 
        else:
            self.head_dim = 128

        self.num_heads = Ct5 // self.head_dim
        self.ca = CrossAttention(for_attn_pool=True, embed_dim=self.D, kv_dim=Ct5, num_heads=self.num_heads)
    def forward(self, ca_kv):
        return self.ca(None, ca_kv).squeeze(1)

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).reshape(-1, 1, 6, C)   # B16C


class MultipleLayers(nn.Module):
    def __init__(self, ls, num_blocks_in_a_chunk, index):
        super().__init__()
        self.module = nn.ModuleList()
        for i in range(index, index+num_blocks_in_a_chunk):
            self.module.append(ls[i])

    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, checkpointing_full_block=False, rope2d_freqs_grid=None):
        h = x
        for m in self.module:
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(m, h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
            else:
                h = m(h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid)
        return h

class Infinity(nn.Module):
    def __init__(
        self, vae_local,
        text_channels=0, text_maxlen=0,     # text-cond generation
        selecting_idx=None,                 # class-cond generation
        embed_dim=1024, depth=16, num_heads=16, mlp_ratio=4.,   # model's architecture
        drop_rate=0., drop_path_rate=0.,    # drop out and drop path
        norm_eps=1e-6, rms_norm=False,      # norm layer
        shared_aln=False, head_aln=True,    # adaptive norm
        cond_drop_rate=0.1,                 # for classifier-free guidance
        rand_uncond=False,
        cross_attn_layer_scale=-1., nm0=False, tau=1, cos_attn=True, swiglu=False,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        head_depth=1,
        top_p=0.0, top_k=0.0,
        customized_flash_attn=False, fused_mlp=False, fused_norm=False,
        block_chunks=1,
        checkpointing=None,
        pad_to_multiplier=0,
        use_flex_attn=False,
        batch_size=2,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        rope2d_each_sa_layer=0,
        rope2d_normalized_by_hw=0,
        pn=None,
        train_h_div_w_list=None,
        video_frames=1,
        always_training_scales=20,
        apply_spatial_patchify = 0,
        inference_mode=False,
    ):
        # set hyperparameters
        self.C = embed_dim
        self.inference_mode = inference_mode
        self.apply_spatial_patchify = apply_spatial_patchify
        if self.apply_spatial_patchify:
            self.d_vae = vae_local.embed_dim * 4
        else:
            self.d_vae = vae_local.embed_dim
        self.use_bit_label = use_bit_label
        self.codebook_dim = self.d_vae
        self.V = (self.codebook_dim * 2) if self.use_bit_label else vae_local.vocab_size
        self.bit_mask = vae_local.quantizer.lfq.mask if self.use_bit_label else None
        self.Ct5 = text_channels
        self.depth = depth
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.mlp_ratio = mlp_ratio
        self.cond_drop_rate = cond_drop_rate
        self.norm_eps = norm_eps
        self.prog_si = -1
        self.pn = pn
        self.train_h_div_w_list = train_h_div_w_list if train_h_div_w_list else h_div_w_templates
        self.video_frames = video_frames
        self.always_training_scales = always_training_scales

        assert add_lvl_embeding_only_first_block in [0,1]
        self.add_lvl_embeding_only_first_block = add_lvl_embeding_only_first_block
        assert rope2d_each_sa_layer in [0,1]
        self.rope2d_each_sa_layer = rope2d_each_sa_layer
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        print(f'self.codebook_dim: {self.codebook_dim}, self.add_lvl_embeding_only_first_block: {self.add_lvl_embeding_only_first_block}, \
            self.use_bit_label: {self.use_bit_label}, self.rope2d_each_sa_layer: {rope2d_each_sa_layer}, self.rope2d_normalized_by_hw: {self.rope2d_normalized_by_hw}')
        head_up_method = ''
        word_patch_size = 1 if head_up_method in {'', 'no'} else 2
        if word_patch_size > 1:
            assert all(raw_pn % word_patch_size == 0 for raw_pn in raw_scale_schedule), f'raw_scale_schedule={raw_scale_schedule}, not compatible with word_patch_size={word_patch_size}'
        
        self.checkpointing = checkpointing
        self.pad_to_multiplier = max(1, pad_to_multiplier)
        
        customized_kernel_installed = any('Infinity' in arg_name for arg_name in flash_attn_func.__code__.co_varnames)
        self.customized_flash_attn = customized_flash_attn and customized_kernel_installed
        if customized_flash_attn and not customized_kernel_installed:
            import inspect, warnings
            file_path = inspect.getsourcefile(flash_attn_func)
            line_number = inspect.getsourcelines(flash_attn_func)[1]
            info = (
                f'>>>>>> Customized FlashAttention2 is not installed or compiled, but specified in args by --flash=1. Set customized_flash_attn = False. <<<<<<\n'
                f'>>>>>> `flash_attn_func` is in [line {line_number}] [file {file_path}] <<<<<<\n'
                f'>>>>>> {flash_attn_func.__code__.co_varnames=} <<<<<<\n'
            )
            warnings.warn(info, ImportWarning)
            print(info, flush=True)
        
        self.raw_scale_schedule = raw_scale_schedule    # 'raw' means before any patchifying
        self.first_l = 1
        # solve top-p top-k sampling hyperparameters
        self.top_p, self.top_k = max(min(top_p, 1), 0), (round(top_k * self.V) if 0 < top_k < 1 else round(top_k))
        if self.top_p < 1e-5: self.top_p = 0
        if self.top_k >= self.V or self.top_k <= 0: self.top_k = 0
        
        t = torch.zeros(dist.get_world_size(), device=dist.get_device())
        t[dist.get_rank()] = float(flash_fused_op_installed)
        dist.barrier()
        dist.allreduce(t)
        assert round(t.sum().item()) in {0, dist.get_world_size()}, f'flash_fused_op_installed: {t}'
        
        super().__init__()
        self.rng = torch.Generator(device=dist.get_device())
        self.maybe_record_function = nullcontext
        self.text_maxlen = text_maxlen
        self.t2i = text_channels != 0
        
        # [inp & position embedding]
        init_std = math.sqrt(1 / self.C / 3)
        self.norm0_cond = nn.Identity()
        if self.t2i:
            self.selecting_idx = None
            self.num_classes = 0
            self.D = self.C
            
            cfg_uncond = torch.empty(self.text_maxlen, self.Ct5)
            rng = torch.Generator(device='cpu')
            rng.manual_seed(0)
            torch.nn.init.trunc_normal_(cfg_uncond, std=1.2, generator=rng)
            cfg_uncond /= self.Ct5 ** 0.5
            if rand_uncond:
                self.register_buffer('cfg_uncond', cfg_uncond)
            else:
                self.cfg_uncond = nn.Parameter(cfg_uncond)
            
            self.text_norm = FastRMSNorm(self.Ct5, elementwise_affine=True, eps=norm_eps)
            self.text_proj_for_sos = TextAttentivePool(self.Ct5, self.D)
            self.text_proj_for_ca = nn.Sequential(
                nn.Linear(self.Ct5, self.D),
                nn.GELU(approximate='tanh'),
                nn.Linear(self.D, self.D),
            )
        else:   # class-label cond
            if selecting_idx is None:
                num_classes = 1000
                print(f'======= WARNING: selecting_idx not specified, set to 1/{num_classes} @ {dist.get_device()} =======')
                selecting_idx = torch.full((1, num_classes), fill_value=1/num_classes, dtype=torch.float32, device=dist.get_device())
            self.selecting_idx = selecting_idx
            self.num_classes = selecting_idx.shape[-1]
            self.D = self.C
            self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
            nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        if self.rope2d_each_sa_layer:
            rope2d_freqs_grid = precompute_rope2d_freqs_grid(dim=self.C//self.num_heads, dynamic_resolution_h_w=dynamic_resolution_h_w, pad_to_multiplier=self.pad_to_multiplier, rope2d_normalized_by_hw=self.rope2d_normalized_by_hw)
            self.rope2d_freqs_grid = rope2d_freqs_grid
        else:
            raise ValueError(f'self.rope2d_each_sa_layer={self.rope2d_each_sa_layer} not implemented')
        self.lvl_embed = nn.Embedding(15, self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # [input layers] input norm && input embedding
        norm_layer = partial(FastRMSNorm if rms_norm else nn.LayerNorm, eps=norm_eps)
        self.norm0_ve = norm_layer(self.d_vae) if nm0 else nn.Identity()
        self.word_embed = nn.Linear(self.d_vae, self.C)
        
        # [shared adaptive layernorm mapping network]
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        # fused norm
        if fused_norm:
            fused_norm_func = fused_ada_rms_norm if rms_norm else fused_ada_layer_norm
            if fused_norm_func is not None: # pre-compile
                B = 2
                x = torch.randn(B, 1, self.C).requires_grad_(True)
                scale = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                shift = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                # fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale, shift=shift).mean().backward()
                del B, x, scale, shift
        else:
            fused_norm_func = None
        
        # [backbone and head]
        self.use_flex_attn = use_flex_attn
        self.attn_fn_compile_dict = {}
        self.batch_size = batch_size
        if self.use_flex_attn:
            self.attn_fn_compile_dict = self.compile_flex_attn()

        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # dpr means drop path rate (linearly increasing)
        self.unregistered_blocks = []
        for block_idx in range(depth):
            block = (CrossAttnBlock if self.t2i else SelfAttnBlock)(
                embed_dim=self.C, kv_dim=self.D, cross_attn_layer_scale=cross_attn_layer_scale, cond_dim=self.D, act=True, shared_aln=shared_aln, norm_layer=norm_layer,
                num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[block_idx], tau=tau, cos_attn=cos_attn,
                swiglu=swiglu, customized_flash_attn=self.customized_flash_attn, fused_mlp=fused_mlp, fused_norm_func=fused_norm_func,
                checkpointing_sa_only=self.checkpointing == 'self-attn',
                use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            )
            self.unregistered_blocks.append(block)
        
        # [head]
        V = self.V
        if head_aln:
            self.head_nm = AdaLNBeforeHead(self.C, self.D, act=True, norm_layer=norm_layer, fused_norm_func=fused_norm_func)
            self.head = nn.Linear(self.C, V) if head_depth == 1 else nn.Sequential(nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))
        else:
            self.head_nm = MultiInpIdentity()
            self.head = nn.Sequential(norm_layer(self.C), nn.Linear(self.C, V)) if head_depth == 1 else nn.Sequential(norm_layer(self.C), nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))
        
        self.num_block_chunks = block_chunks or 1
        self.num_blocks_in_a_chunk = depth // block_chunks
        print(f"{self.num_blocks_in_a_chunk=}, {depth=}, {block_chunks=}")
        assert self.num_blocks_in_a_chunk * block_chunks == depth
        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(MultipleLayers(self.unregistered_blocks, self.num_blocks_in_a_chunk, i*self.num_blocks_in_a_chunk))
        print(
            f'\n[constructor]  ==== customized_flash_attn={self.customized_flash_attn} (using_flash={sum((b.sa.using_flash if self.t2i else b.attn.using_flash) for b in self.unregistered_blocks)}/{self.depth}), fused_mlp={fused_mlp} (fused_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.unregistered_blocks)}/{self.depth}) ==== \n'
            f'    [Infinity config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}, swiglu={swiglu} num_blocks_in_a_chunk={self.num_blocks_in_a_chunk}\n'
            f'    [drop ratios] drop_rate={drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
    

    def compile_flex_attn(self):
        attn_fn_compile_dict = {}
        for h_div_w in self.train_h_div_w_list:
            h_div_w_template = h_div_w_templates[np.argmin(np.abs(float(h_div_w) - h_div_w_templates))]
            full_scale_schedule = dynamic_resolution_h_w[h_div_w_template][self.pn]['scales']
            if self.inference_mode:
                apply_flex_attn_scales = list(range(1, 1+len(full_scale_schedule)))
                mask_type = "var_infer_mask_with_kv_cache"
                auto_padding = True
            else:
                mask_type = 'var'
                auto_padding = False
                apply_flex_attn_scales = [min(self.always_training_scales, len(full_scale_schedule))]
            #import pdb
            #pdb.set_trace()
            for scales_num in apply_flex_attn_scales:
                print(f'====== apply flex attn hdivw: {h_div_w} scales: {scales_num} ======')
                scale_schedule = full_scale_schedule[:scales_num]
                scale_schedule = [ (min(t, self.video_frames//4+1), h, w) for (t,h, w) in scale_schedule]
                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( pt * ph * pw for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L
                # print(SEQ_L, aligned_L, patchs_nums_tuple)
                attn_fn = FlexAttn(block_scales = patchs_nums_tuple,
                                        mask_type = mask_type,
                                        B = self.batch_size, 
                                        H = self.num_heads,
                                        L = aligned_L,
                                        auto_padding=auto_padding)
                attn_fn_compile_dict[patchs_nums_tuple] = attn_fn

            if self.video_frames > 1: # append image attn_fn when self.video_frames > 1 (namely videos)
                scale_schedule = [ (1, h, w) for (t,h, w) in scale_schedule]
                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( pt * ph * pw for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L
                attn_fn = FlexAttn(block_scales = patchs_nums_tuple,
                                        mask_type = mask_type,
                                        B = self.batch_size, 
                                        H = self.num_heads,
                                        L = aligned_L)
                attn_fn_compile_dict[patchs_nums_tuple] = attn_fn
        return attn_fn_compile_dict
        
    def get_logits(self, h: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        """
        :param h: hidden_state, shaped (B or batch_size, L or seq_len, C or hidden_dim)
        :param cond_BD: shaped (B or batch_size, D or cond_dim)
        :param tau: temperature
        :return: logits, shaped (B or batch_size, V or vocabulary_size)
        """
        with torch.amp.autocast('cuda', enabled=False):
            return self.head(self.head_nm(h.float(), cond_BD.float()))

    def add_lvl_embeding(self, feature, mask_list, scale_ind, scale_schedule, need_to_pad=0):
        bs, seq_len, c = feature.shape
        if mask_list is not None:
            # 假设 mask_list 和 scale_ind 长度相同  
            start_idx = 0  
            for i, mask in enumerate(mask_list):  
                segment_length = len(mask)  
                end_idx = start_idx + segment_length  
                
                # 对 feature 的当前分段应用对应的 scale_ind 值  
                feature[:, start_idx:end_idx] += self.lvl_embed(  
                    scale_ind[i] * torch.ones((bs, segment_length), dtype=torch.int).to(feature.device)  
                )  
                
                start_idx = end_idx  
        
            return feature
        else:
            t_mul_h_mul_w = seq_len
            feature[:, :t_mul_h_mul_w] += self.lvl_embed(scale_ind*torch.ones((bs, t_mul_h_mul_w),dtype=torch.int).to(feature.device))
            return feature
    
    def add_lvl_embeding_for_x_BLC(self, x_BLC, scale_schedule, need_to_pad=0):
        ptr = 0
        x_BLC_list = []
        for scale_ind, patch_t_h_w in enumerate(scale_schedule):
            scale_seq_len = np.array(patch_t_h_w).prod()
            x_BLC_this_scale = x_BLC[:,ptr:ptr+scale_seq_len] # shape: [bs, patch_h*patch_w, c]
            ptr += scale_seq_len
            x_BLC_this_scale = self.add_lvl_embeding(x_BLC_this_scale, scale_ind, scale_schedule)
            x_BLC_list.append(x_BLC_this_scale)
        assert x_BLC.shape[1] == (ptr + need_to_pad), f'{x_BLC.shape[1]} != {ptr} + {need_to_pad}'
        x_BLC_list.append(x_BLC[:,ptr:])
        x_BLC = torch.cat(x_BLC_list, dim=1)
        return x_BLC

    def forward(self, label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]], x_BLC_wo_prefix: torch.Tensor, scale_schedule: List[Tuple[int]],
        cfg_infer=False,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  # returns logits_BLV
        """
        label_B_or_BLT: label_B or (kv_compact, cu_seqlens_k, max_seqlen_k)
        :return: logits BLV, V is vocab_size
        """
        if cfg_infer:
            return self.autoregressive_infer_cfg(label_B_or_BLT=label_B_or_BLT, scale_schedule=scale_schedule, **kwargs)
        
        x_BLC_wo_prefix = x_BLC_wo_prefix.float()       # input should be float32
        B = x_BLC_wo_prefix.shape[0]

        # [1. get input sequence x_BLC]
        with torch.amp.autocast('cuda', enabled=False):
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
            # drop cond
            total = 0
            for le in lens:
                if random.random() < self.cond_drop_rate:
                    kv_compact[total:total+le] = self.cfg_uncond[:le]
                total += le
            must_on_graph = self.cfg_uncond[0, 0] * 0
            kv_compact = self.text_norm(kv_compact).contiguous()
            sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous()    # cond_BD should be float32
            kv_compact = self.text_proj_for_ca(kv_compact).contiguous()
            kv_compact[0, 0] += must_on_graph
            ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
            
            cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()  # gss: gamma, scale, shift; cond_BD_or_gss should be float32
            
            sos = sos.unsqueeze(1).expand(B, 1, -1) + self.pos_start.expand(B, 1, -1)
            x_BLC = torch.cat((sos, self.word_embed(self.norm0_ve(x_BLC_wo_prefix))), dim=1)

            # [1.1. pad the seqlen dim]
            l_end = x_BLC.shape[1]
            need_to_pad = (l_end + self.pad_to_multiplier - 1) // self.pad_to_multiplier * self.pad_to_multiplier - l_end # 0
            
            if self.customized_flash_attn:
                Infinity_visible_kvlen = self.Infinity_visible_kvlen[:l_end]
                Infinity_invisible_qlen = self.Infinity_invisible_qlen[:l_end]
                attn_bias_or_two_vector = (Infinity_visible_kvlen, Infinity_invisible_qlen)
                # todo: solve need_to_pad here
            elif self.use_flex_attn:
                if need_to_pad:
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                assert x_BLC.shape[-1] % 128 == 0, 'x_BLC.shape[-1] % 128 != 0'
                attn_bias_or_two_vector = None
            else:
                d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2],), i) for i, pn in enumerate(scale_schedule)]).view(1, l_end, 1)
                dT = d.transpose(1, 2)    # dT: 11L
                attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
                attn_bias = attn_bias_for_masking[:, :, :l_end, :l_end].contiguous()   # attn_bias: 11LL
                if need_to_pad:
                    attn_bias = F.pad(attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
                    attn_bias[0, 0, l_end:, 0] = 0
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                attn_bias_or_two_vector = attn_bias.type_as(x_BLC).to(x_BLC.device)
        
        if self.use_flex_attn:
            attn_fn = self.attn_fn_compile_dict[tuple(scale_schedule)]
        else:
            attn_fn = None

        # [2. block loop]
        SelfAttnBlock.forward, CrossAttnBlock.forward
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training
        if self.num_block_chunks == 1:
            for i, b in enumerate(self.blocks):
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if checkpointing_full_block:
                    x_BLC = torch.utils.checkpoint.checkpoint(b, x_BLC, cond_BD_or_gss, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, self.rope2d_freqs_grid, use_reentrant=False)
                else:
                    x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid)
        else:
            for i, chunk in enumerate(self.block_chunks): # this path
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                x_BLC = chunk(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, scale_schedule=scale_schedule, checkpointing_full_block=checkpointing_full_block, rope2d_freqs_grid=self.rope2d_freqs_grid)

        # [3. unpad the seqlen dim, and then get logits]
        return self.get_logits(x_BLC[:, :l_end], cond_BD)    # return logits BLV, V is vocab_size

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule=None,
        category=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
        g_seed=None, cfg_list=[], tau_list=[], cfg_sc=3, top_k=0, top_p=0.0,
        returns_vemb=0, ratio_Bl1=None, gumbel=0, norm_cfg=False,
        cfg_exp_k: float=0.0, cfg_insertion_layer=[-5],
        vae_type=0, softmax_merge_topk=-1, ret_img=False,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        inference_mode=False,
        save_img_path=None,
        sampling_per_bits=1,
        verbose=False,
        si_para = 9,
        ratio_list = [50,10,5],
        kv_opt=None
    ):   # returns List[idx_Bl]
        # tt0 = time.time() * 1e3

        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)

        # scale_schedule is used by infinity, vae_scale_schedule is used by vae if there exists a spatial patchify, 
        # we need to convert scale_schedule to vae_scale_schedule by multiply 2 to h and w
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        if any(np.array(cfg_list) != 1):
            bs = 2*B
            if not negative_label_B_or_BLT:
                kv_compact_un = kv_compact.clone()
                total = 0
                for le in lens:
                    kv_compact_un[total:total+le] = (self.cfg_uncond)[:le]
                    total += le
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)
            else:
                kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:]+cu_seqlens_k[-1]), dim=0)
                max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
        else:
            bs = B

        # import pdb
        #pdb.set_trace()

        kv_compact = self.text_norm(kv_compact)
        sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)) # sos shape: [2, 4096]
        kv_compact = self.text_proj_for_ca(kv_compact) # kv_compact shape: [304, 4096]
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(bs, 1, -1)

        with torch.amp.autocast('cuda', enabled=False):
            cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()
        accu_BChw, cur_L, ret = None, 0, []  # current length, list of reconstructed images
        idx_Bl_list, idx_Bld_list = [], []

        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(True)
        
        abs_cfg_insertion_layers = []
        add_cfg_on_logits, add_cfg_on_probs = False, False
        leng = len(self.unregistered_blocks)
        for item in cfg_insertion_layer:
            if item == 0: # add cfg on logits
                add_cfg_on_logits = True
            elif item == 1: # add cfg on probs
                add_cfg_on_probs = True # todo in the future, we may want to add cfg on logits and probs
            elif item < 0: # determine to add cfg at item-th layer's output
                assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={self.num_block_chunks}'
                abs_cfg_insertion_layers.append(leng+item)
            else:
                raise ValueError(f'cfg_insertion_layer: {item} is not valid')
        
        num_stages_minus_1 = len(scale_schedule)-1
        summed_codes = 0
        # print(scale_schedule)
        # get_torch_mem_usage()
        # tt1 = time.time() * 1e3

        # backbone_time = []
        # 用于存储每个scale的block_number和MSE值
        # mse_data = {si: [] for si in range(len(scale_schedule))}
        # diff_data = {block_idx: [] for block_idx in range(len(self.block_chunks)*4)}
        loss_data = {si: [] for si in range(len(scale_schedule))}
        loss_func = 'diff_ratio'
        #skip_list= [[31,30,25,14,29,26],[26,27,28,29,30,31],[16,17,18,19,20,21]]
        #skip_choice = 2
        skip_mode = False
        compute_loss = False
        save_codes = False
        save_para_codes = False
        # with open('skip_list.pkl', 'rb') as f:
        #     skip_list = pickle.load(f)
        profile = False

        # 用于存储每个scale的codes和summed_codes
        # si_para = 9
        codes_data = {si: [] for si in range(len(scale_schedule))}
        summed_codes_data = {si: [] for si in range(len(scale_schedule))}
        partial_codes_data = {si: [] for si in range(len(scale_schedule))}
        test_partial_list = []
        test_partial_list0 = []  
        mask_list = None

        for si, pn in enumerate(scale_schedule):   # si: i-th segment
            # if si>=11:
            #     break
            if si <= si_para:
                if profile:
                    t0 = time.time() * 1e3

                cfg = cfg_list[si]
                if si >= trunk_scale:
                    break

                cur_L += np.array(pn).prod()

                need_to_pad = 0
                attn_fn = None
                if self.use_flex_attn:
                    attn_fn = self.attn_fn_compile_dict.get(tuple(scale_schedule[:(si+1)]), None)

                if profile:
                    torch.cuda.synchronize()
                    t1 = time.time() * 1e3

                layer_idx = 0    
                for block_idx, b in enumerate(self.block_chunks):
                    if self.add_lvl_embeding_only_first_block and block_idx == 0:
                        last_stage = self.add_lvl_embeding(last_stage, mask_list, si, scale_schedule, need_to_pad=need_to_pad)
                        partial_codes_data[si].append(last_stage)
                    if not self.add_lvl_embeding_only_first_block: 
                        last_stage = self.add_lvl_embeding(last_stage, mask_list,si, scale_schedule, need_to_pad=need_to_pad)
                        partial_codes_data[si].append(last_stage)

                    for ii, m in enumerate(b.module):
                        block_number = block_idx * 4 + ii
                        current_stage = last_stage.clone()
                        last_stage = m(x=last_stage, mask_id = mask_list, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid, scale_ind=si, si_para=si_para, kv_opt=kv_opt)
                        partial_codes_data[si].append(last_stage)

                        if compute_loss:
                            if loss_func == 'MSE':                    
                                mse = F.mse_loss(current_stage, last_stage)
                                loss = mse.item()
                            elif loss_func == 'relative_diff':
                                diff = (last_stage - current_stage).abs().reshape(-1,last_stage.shape[-1])  
                                sim = diff.sum()/last_stage.abs().sum()
                                loss = sim
                            elif loss_func == 'cosine_similarity':
                                similarity = cosine_similarity(current_stage[0], last_stage[0])
                                loss = similarity   
                            elif loss_func == 'diff_ratio':
                                diff = (last_stage - current_stage).abs().reshape(-1,last_stage.shape[-1])  
                                loss = compute_diff_ratio(last_stage.abs().reshape(-1,last_stage.shape[-1]), diff)     

                            loss_data[si].append(loss)

                        if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                            last_stage = cfg * last_stage[:B] + (1-cfg) * last_stage[B:]
                            last_stage = torch.cat((last_stage, last_stage), 0)
                            layer_idx += 1                

                if profile:
                    torch.cuda.synchronize()
                    t2 = time.time() * 1e3

                ######################### 1 #############################
                if (cfg != 1) and add_cfg_on_logits:
                    logits_BlV = self.get_logits(last_stage, cond_BD).mul(1/tau_list[si])
                    logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
                else:
                    logits_BlV = self.get_logits(last_stage[:B], cond_BD[:B]).mul(1/tau_list[si])
                
                if self.use_bit_label:
                    tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                    logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
                    idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                    idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
                else:
                    idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                if vae_type != 0:
                    assert returns_vemb
                    if si < gt_leak:
                        idx_Bld = gt_ls_Bl[si]
                    else:
                        assert pn[0] == 1
                        idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1)
                        if self.apply_spatial_patchify:
                            idx_Bld = idx_Bld.permute(0,3,1,2)
                            idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2)
                            idx_Bld = idx_Bld.permute(0,2,3,1)
                        idx_Bld = idx_Bld.unsqueeze(1)

                    idx_Bld_list.append(idx_Bld)
                    codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') 
                ######################### 1 #############################
                
                if si != num_stages_minus_1:
                    summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
                    if si == si_para:
                        summed_codes_para = summed_codes.clone()  # Save summed_codes for stage 8
                        continue
                        
                    last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                    
                    ######################### 2.1 #############################
                    last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
                    if self.apply_spatial_patchify: # patchify operation
                        last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, 4d, h, w]
                    last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
                    last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
                else:
                    summed_codes += codes
                    if si == si_para:
                        summed_codes_para = summed_codes.clone()  # Save summed_codes for stage 8
                        continue

                    ######################### 2.1 #############################

                ######################### 2.2 #############################            

                if si != num_stages_minus_1:
                    last_stage = self.word_embed(self.norm0_ve(last_stage))
                    last_stage = last_stage.repeat(bs//B, 1, 1)
                ######################### 2.2 #############################

                if profile:
                    torch.cuda.synchronize()
                    t3 = time.time() * 1e3
                    print(f"stage {si}, {pn}, all {t3 - t0:.2f}ms, {t1 - t0:.2f}ms, 32block {t2 - t1:.2f}ms, {t3 - t2:.2f}ms")

            if si > si_para:
                assert len(ratio_list) == num_stages_minus_1 - si_para
              
                if profile:
                    torch.cuda.synchronize()
                    t0 = time.time() * 1e3
                last_stage_list = []
                pn_list = []
                scale_list = [i for i in range(si_para+1, num_stages_minus_1+1)]
                for i in range(si,num_stages_minus_1+1,1):
                    last_stage = F.interpolate(summed_codes_para, size=vae_scale_schedule[i], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                    last_stage_list.append(last_stage)
                    pn_list.append(scale_schedule[i][1])
                
                mask_list = get_freq(last_stage_list,pn_list,ratio_list)                 
                com_last_stage = process_and_concat_last_stage(last_stage_list, mask_list)   #[B,com_pruned_seq_len,d] #[1,com_pruned_seq_len,32]

                ######################### 2.2 #############################            
                com_last_stage = self.word_embed(self.norm0_ve(com_last_stage))
                com_last_stage = com_last_stage.repeat(bs//B, 1, 1)
                ######################### 2.2 #############################
                layer_idx = 0
                if profile:
                    torch.cuda.synchronize()
                    t1 = time.time() * 1e3
                

                for block_idx, b in enumerate(self.block_chunks):
                    if self.add_lvl_embeding_only_first_block and block_idx == 0:
                        com_last_stage = self.add_lvl_embeding(com_last_stage, mask_list, scale_list, scale_schedule, need_to_pad=need_to_pad)
                        partial_codes_data[si].append(com_last_stage)
                    if not self.add_lvl_embeding_only_first_block: 
                        com_last_stage = self.add_lvl_embeding(com_last_stage, mask_list, scale_list,scale_schedule, need_to_pad=need_to_pad)
                        partial_codes_data[si].append(com_last_stage)

                    for ii, m in enumerate(b.module):
                        block_number = block_idx * 4 + ii
                        current_stage = com_last_stage.clone()
                        com_last_stage = m(x=com_last_stage, mask_id = mask_list, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=attn_fn, scale_schedule=scale_schedule,
                                               rope2d_freqs_grid=self.rope2d_freqs_grid, scale_ind = scale_list, si_para=si_para, kv_opt=kv_opt)
                        partial_codes_data[si].append(com_last_stage)

                        if compute_loss:
                            if loss_func == 'MSE':                    
                                mse = F.mse_loss(current_stage, last_stage)
                                loss = mse.item()
                            elif loss_func == 'relative_diff':
                                diff = (last_stage - current_stage).abs().reshape(-1,last_stage.shape[-1])  
                                sim = diff.sum()/last_stage.abs().sum()
                                loss = sim
                            elif loss_func == 'cosine_similarity':
                                similarity = cosine_similarity(current_stage[0], last_stage[0])
                                loss = similarity   
                            elif loss_func == 'diff_ratio':
                                diff = (last_stage - current_stage).abs().reshape(-1,last_stage.shape[-1])  
                                loss = compute_diff_ratio(last_stage.abs().reshape(-1,last_stage.shape[-1]), diff)     

                            loss_data[si].append(loss)

                        if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                            last_stage_gather = cfg * last_stage_gather[:B] + (1-cfg) * last_stage_gather[B:]
                            last_stage_gather = torch.cat((last_stage_gather, last_stage_gather), 0)
                            layer_idx += 1                

                
                if profile:
                    torch.cuda.synchronize()
                    t2 = time.time() * 1e3
                ######################### 1 #############################
                last_stage = com_last_stage 
                if (cfg != 1) and add_cfg_on_logits:
                    logits_BlV = self.get_logits(last_stage, cond_BD).mul(1/tau_list[si])
                    logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
                else:
                    logits_BlV = self.get_logits(last_stage[:B], cond_BD[:B]).mul(1/tau_list[si])
                
                if self.use_bit_label:
                    tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                    logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
                    idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                    idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
                else:
                    idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                if vae_type != 0:
                    assert returns_vemb
                    if si < gt_leak:
                        idx_Bld = gt_ls_Bl[si]
                    else:
                        assert pn[0] == 1
                        #######  remove #####
                        # idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1)
                        if self.apply_spatial_patchify:
                            idx_Bld = idx_Bld.permute(0,3,1,2)
                            idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2)
                            idx_Bld = idx_Bld.permute(0,2,3,1)
                        idx_Bld = idx_Bld.unsqueeze(1)  #[1,1,48*48*0.05,32]

                    idx_Bld_list.append(idx_Bld)
                    codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label')  ##[1,1,48*48*0.05,32] -->[1,32,1,48*48*0.05]
                ######################### 1 #############################

                ################## padding 1.1 ##########################
                codes_list = []
                for i in range(len(pn_list)):
                    codes_ = torch.zeros([B,32,1,pn_list[i]**2], device = codes.device,dtype=codes.dtype)
                    codes_list.append(codes_)
                start_id = 0
                for idx, (new_codes, mask) in enumerate(zip(codes_list, mask_list)):  
                    # 将 codes 重塑为 [-1, pn, pn]  
                    new_codes[:, :, :, mask] = codes[:, :, :, start_id:start_id + len(mask)]  
                    new_codes = new_codes.reshape(B, 32, 1, pn_list[idx],pn_list[idx])
                    # 检查是否是最后一轮循环  
                    if idx < len(codes_list) - 1:  # 不处理最后一轮  
                        test_partial_code = F.interpolate(new_codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)  
                        test_partial_list.append(test_partial_code)  
                    else:
                        test_partial_list.append(new_codes)
                    start_id += len(mask)  

                if profile:
                    torch.cuda.synchronize()
                    t3 = time.time() * 1e3
                    print(f"stage {si}, {pn}, all {t3 - t0:.2f}ms, {t1 - t0:.2f}ms, 32block {t2 - t1:.2f}ms, {t3 - t2:.2f}ms")

                break   
        # Save the data to pkl files
        combined_data = {
            'test_partial_list': test_partial_list,
            'summed_codes_para': summed_codes_para
        }
        if save_para_codes:
            with open(f'outputs/codes_mtp/test_combined_data_{category}_50_5_5.pkl', 'wb') as f:
                pickle.dump(combined_data, f)


        # # 将 codes_data 和 summed_codes_data 合并到一个字典中
        # combined_data = {
        #     'partial_codes_data': partial_codes_data,
        #     'codes_data': codes_data,
        #     'summed_codes_data': summed_codes_data
        # }
        # if save_codes:
        #     # 保存 combined_data 到 pkl 文件
        #     with open(f'outputs/codes/test_pixel_partialblock_data_{category}.pkl', 'wb') as f:
        #         pickle.dump(partial_codes_data, f)
        
        # 保存 loss_data 到 pkl 文件
        if compute_loss:
            with open(f'outputs/loss/loss_data_{category}.pkl', 'wb') as f:
                pickle.dump(loss_data, f)

        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(False)

        if not ret_img:
            return ret, idx_Bl_list, []
        
        if vae_type != 0:
            summed_codes = sum(test_partial_list) + summed_codes_para
            img = vae.decode(summed_codes.squeeze(-3))
        else:
            img = vae.viz_from_ms_h_BChw(ret, scale_schedule=scale_schedule, same_shape=True, last_one=True)
        tt3 = time.time() * 1e3

        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
        # print(f"pre: {tt1 - tt0:.2f}ms, backbone: {tt2-tt1:.2f}ms, post{tt3 - tt2:.2f}ms")
        #ATTN_TIME.append(backbone_time)
        return ret, idx_Bl_list, img
    
    @for_visualize
    def vis_key_params(self, ep):
        return
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=False, assign=False):
        for k in state_dict:
            if 'cfg_uncond' in k:
                old, new = state_dict[k], self.cfg_uncond.data
                min_tlen = min(old.shape[0], new.shape[0])
                if min_tlen == old.shape[0]:
                    state_dict[k] = torch.cat((old.to(device=new.device, dtype=new.dtype), new[min_tlen:]))
                else:
                    state_dict[k] = old[:min_tlen]
        
        for buf_name in ('lvl_1L', 'attn_bias_for_masking', 'Infinity_visible_kvlen', 'Infinity_invisible_qlen'):
            state_dict.pop(buf_name, None)
            if hasattr(self, buf_name):
                state_dict[buf_name] = getattr(self, buf_name)
        
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
    
    def special_init(
        self,
        aln_init: float,
        aln_gamma_init: float,
        scale_head: float,
        scale_proj: int,
    ):
        # init head's norm
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(aln_init)    # there's no gamma for head
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        # init head's proj
        if scale_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(scale_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(scale_head)
                self.head[-1].bias.data.zero_()
        
        depth = len(self.unregistered_blocks)
        for block_idx, sab in enumerate(self.unregistered_blocks):
            sab: Union[SelfAttnBlock, CrossAttnBlock]
            # init proj
            scale = 1 / math.sqrt(2*depth if scale_proj == 1 else 2*(1 + block_idx))
            if scale_proj == 1:
                if self.t2i:
                    sab.sa.proj.weight.data.mul_(scale)
                    sab.ca.proj.weight.data.mul_(scale)
                else:
                    sab.attn.proj.weight.data.mul_(scale)
                sab.ffn.fc2.weight.data.mul_(scale)
            # if sab.using_swiglu:
            #     nn.init.ones_(sab.ffn.fcg.bias)
            #     nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            
            # init ada_lin
            if hasattr(sab, 'ada_lin'):
                lin = sab.ada_lin[-1]
                lin.weight.data[:2*self.C].mul_(aln_gamma_init)     # init gamma
                lin.weight.data[2*self.C:].mul_(aln_init)           # init scale and shift
                if hasattr(lin, 'bias') and lin.bias is not None:
                    lin.bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, :2, :].mul_(aln_gamma_init)  # init gamma
                sab.ada_gss.data[:, :, 2:, :].mul_(aln_init)        # init scale and shift
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate}'
    
    def get_layer_id_and_scale_exp(self, para_name: str):
        raise NotImplementedError


def sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)

def sampling_with_top_k_top_p_also_inplace_modifying_probs_(probs_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = probs_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = probs_BlV < probs_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        probs_BlV.masked_fill_(idx_to_remove, 0)
    if top_p > 0:
        sorted_probs, sorted_idx = probs_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_probs.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        probs_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), 0)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    probs_BlV = probs_BlV / probs_BlV.sum(-1, keepdims=True)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(probs_BlV.view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


def get_params_num(d, w, mlp):
    m = round(mlp * w / 256) * 256
    s = d * (w**2 * 8 + w*m * 2)    # sa+ca, mlp
    s += w**2 * 6       # saln
    s += 4096 * w       # pred
    s += 32 * w         # we
    
    Ct5 = 4096
    s += Ct5*w * 4      # T5 attn pool
    s += Ct5*w + w*w    # T5 mlp
    return f'{s/1e9:.2f}B'


TIMM_KEYS = {'img_size', 'pretrained', 'pretrained_cfg', 'pretrained_cfg_overlay', 'global_pool','cache_dir'}

@register_model
def infinity_2b(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_20b(depth=58, embed_dim=4608, num_heads=4608//128, drop_path_rate=0.25, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

# model configuration for scaling Infinity transformer
@register_model
def infinity_layer12(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer16(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer24(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer32(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer40(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer48(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})