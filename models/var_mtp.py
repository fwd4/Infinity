import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2
from torch.nn import functional as F

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
        # 将 codes 重塑为 [B, 32, pn, pn]
        flatten_sum = codes.view(codes.shape[0], -1, pn, pn)  # [B, 32, pn, pn]

        # 计算 DC 分量
        dc_component = F.avg_pool2d(flatten_sum, kernel_size=pn)  # [B, 32, 1, 1]
        # 计算差异并展平
        dc_diff = torch.norm(flatten_sum - dc_component, dim=1).flatten(start_dim=1)  # [B, pn*pn]
        # 获取总元素数量
        total_sz = dc_diff.shape[1]

        # 计算当前比例的 top 索引范围
        high_ratio = ratio
        low_ratio = ratio_list[ratio_list.index(ratio) + 1] if ratio_list.index(ratio) + 1 < len(ratio_list) else 0

        # 获取 top_high 和 top_low 的索引
        top_high_indices = torch.topk(dc_diff, total_sz * high_ratio // 100, dim=1, largest=True, sorted=False).indices  #[B, k]
        top_low_indices = torch.topk(dc_diff, total_sz * low_ratio // 100, dim=1, largest=True, sorted=False).indices  #[B, k]

        # 计算 mask（高比例减去低比例）
        mask_set = [set(high.cpu().numpy()) - set(low.cpu().numpy()) for high, low in zip(top_high_indices, top_low_indices)]  #len= B
        mask = [list(mask_item) for mask_item in mask_set] #len= B

        # 将 mask 添加到 mask_list
        mask_list.append(mask) #[ 第一个分辨率：[[一个img的mask],[一个img的mask]], 另一个分辨率：[[一个img的mask],[一个img的mask]]]

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

    for last_stage, mask in zip(last_stage_list, mask_list):  #[B,32,pn[si+1],pn[si+1]]  [B, k]
        # 1. squeeze(-3) -> [B, d, h, w]
        # last_stage = last_stage.squeeze(-3)

        # 2. reshape -> [B, d, h*w]
        B, d, h, w = last_stage.shape
        last_stage = last_stage.reshape(B, d, h * w)

        # 3. 根据 mask 取对应的索引 -> [B, d, mask_len]
        # 先用列表推导式生成每个批次的结果  
        result_list = [last_stage[b, :, torch.tensor(mask[b], device=last_stage.device)]   
                    for b in range(len(mask))]  

        # 然后将结果堆叠成一个张量，所有批次的掩码长度相同  
        last_stage = torch.stack(result_list)  # 形状: [B, d, mask_length] 

        # 4. 添加到列表
        processed_list.append(last_stage)

    # 5. 拼接所有处理后的张量 -> [B, d, total_mask_len]
    new_last_stage = torch.cat(processed_list, dim=-1)

    return new_last_stage

def process_lvl_pos(lvl_pos, mask_list, patch_nums, si_list, B):
    """
    根据 mask_list 索引 lvl_pos 的值，并返回处理后的张量。

    参数:
        lvl_pos: Tensor, 形状为 [1, 680, 1024]，表示所有 stage 的位置嵌入。
        mask_list: List[List[List[int]]], 三维列表，表示每个 stage 的 mask 索引。
        patch_nums: List[int], 每个 stage 的 patch 数量。
        si_list: List[int], 表示从 si 开始的 stage 索引列表。
        B: int, batch size。

    返回:
        processed_lvl_pos: Tensor, 形状为 [B, len(mask_list[0][0]) + len(mask_list[1][0]), 1024]。
    """
    # 计算每个 stage 的起始位置
    start_list = [sum(pn * pn for pn in patch_nums[:ind]) for ind in si_list]

    processed_list = []  # 用于存储每个 stage 处理后的 lvl_pos

    for stage_idx, (start, mask) in enumerate(zip(start_list, mask_list)):  # 遍历每个 stage
        # 计算当前 stage 的 lvl_pos 范围
        stage_len = patch_nums[si_list[stage_idx]] ** 2
        stage_lvl_pos = lvl_pos[:, start:start + stage_len, :]  # [1, stage_len, 1024]

        # 根据 mask 索引 lvl_pos
        result_list = [stage_lvl_pos[0, torch.tensor(mask[b], device=lvl_pos.device), :] for b in range(B)]
        stage_processed = torch.stack(result_list)  # [B, mask_len, 1024]

        processed_list.append(stage_processed)

    # 拼接所有 stage 的处理结果
    processed_lvl_pos = torch.cat(processed_list, dim=1)  # [B, total_mask_len, 1024]

    return processed_lvl_pos

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))  #[1,1,1024] 随机
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C torch.Size([1, 680, 1024])  随机
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,        
        si_para = 7,
        ratio_list = [60,20],
        kv_opt=None
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        import time
        total_start  = time.perf_counter()
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))   #[16,1024]
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC  #[1, 680, 1024]
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])  #[8,32,16,16]
        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            if si <= si_para:
                ratio = si / self.num_stages_minus_1   #self.num_stages_minus_1 = 9
                # last_L = cur_L
                cur_L += pn*pn
                # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
                cond_BD_or_gss = self.shared_ada_lin(cond_BD)  #[2B,1024]
                x = next_token_map  #[2B,1,1024]
                AdaLNSelfAttn.forward
                for i,b in enumerate(self.blocks):
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)  #torch.Size([2B, 1, 1024])
                logits_BlV = self.get_logits(x, cond_BD)  #torch.Size([2B, 1, 1024])
                
                t = cfg * ratio
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]  #torch.Size([B, 1, 4096])
                
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]  #[B,1,1]-->[B,1]
                if not more_smooth: # this is the default case
                    h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # Embedding(4096, 32) (B, l, Cvae) [8,1,32]
                else:   # not used when evaluating FID/IS/Precision/Recall
                    gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                    h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
                
                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw) #[B,32,16,16],[B,32,pn[si+1],pn[si+1]]
                if si != self.num_stages_minus_1:   # prepare for next stage
                    next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2) #[B,pn*pn,32]
                    next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]  ##[B,pn*pn,1024]
                    next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG  #[2B,pn*pn,1024]
            if si > si_para:
                last_stage_list = []
                pn_list = []
                si_list = [i for i in range(si, self.num_stages_minus_1+1)]

                for i in range(si,self.num_stages_minus_1+1,1):
                    last_stage = F.interpolate(f_hat, size=(self.patch_nums[i], self.patch_nums[i]), mode='area') # [B,32,pn[si+1],pn[si+1]]
                    last_stage_list.append(last_stage)
                    pn_list.append(self.patch_nums[i])

                mask_list = get_freq(last_stage_list, pn_list, ratio_list)
                com_last_stage = process_and_concat_last_stage(last_stage_list, mask_list)  #[B,32,total_mask_len]

                com_last_stage = com_last_stage.view(B, self.Cvae, -1).transpose(1, 2)           #[B,total_mask_len,32]
                com_lvl_pos = process_lvl_pos(lvl_pos, mask_list, self.patch_nums, si_list, B)     #[B,total_mask_len,1024]
                next_token_map = self.word_embed(com_last_stage) + com_lvl_pos              ##[B,total_mask_len,1024]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG  #[2B,total_mask_len,1024]
                
                x = next_token_map
                AdaLNSelfAttn.forward
                for i,b in enumerate(self.blocks):
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)  #torch.Size([2B, total_mask_len, 1024])
                logits_BlV = self.get_logits(x, cond_BD)  #torch.Size([2B, total_mask_len, 1024])

                # Calculate ratios for each stage in si_list
                ratios = [si / self.num_stages_minus_1 for si in si_list]

                # Split logits_BlV[:B] and logits_BlV[B:] into parts based on mask_list lengths
                m = len(si_list)
                split_lengths = [len(mask_list[i][0]) for i in range(m)]  #每个stage中单个batch的mask_list长度
                logits_parts_B = torch.split(logits_BlV[:B], split_lengths, dim=1)
                logits_parts_2B = torch.split(logits_BlV[B:], split_lengths, dim=1)

                # Update logits_BlV by applying the ratio logic to each part
                updated_logits_parts = [(1 + ratio) * part_B - ratio * part_2B
                                         for ratio, part_B, part_2B in zip(ratios, logits_parts_B, logits_parts_2B)]

                logits_BlV = torch.cat(updated_logits_parts, dim=1)  # torch.Size([B, total_mask_len, 4096])

                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]  #[B,total_mask_len,1]-->[B,total_mask_len]
                if not more_smooth: # this is the default case
                    h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # Embedding(4096, 32) (B, l, Cvae) [B,total_mask_len,32]
                else:   # not used when evaluating FID/IS/Precision/Recall
                    gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                    h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
                
                h_BChw = h_BChw.transpose_(1, 2)    #[B, 32 , total_mask_len]
                h_BChw_list = []
                for i in range(len(pn_list)):
                    h_BChw_ = torch.zeros([B,self.Cvae,pn_list[i]**2], device = h_BChw.device,dtype=h_BChw.dtype)
                    h_BChw_list.append(h_BChw_)
                start_id = 0
                for idx, (new_codes, mask) in enumerate(zip(h_BChw_list, mask_list)):  
                    mask_len = len(mask[0]) 
                    # 将 codes 重塑为 [-1, pn, pn]  
                    for b in range(B):  
                        new_codes[b, :, mask[b]] = h_BChw[b, :, start_id:start_id + mask_len]    #[B, 32 , total_mask_len]
                    new_codes = new_codes.reshape(B, self.Cvae, pn_list[idx], pn_list[idx])
                    f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, new_codes)  #f_hat：[B,32,pn[-1],pn[-1]]
                    si = si + 1
                    start_id += mask_len  # 更新起始索引  

                break
            

            

        for b in self.blocks: b.attn.kv_caching(False)
        img_feat = self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
        # torch.cuda.synchronize()
        # total_end = time.perf_counter()
        # print(f"{(total_end-total_start)*1000:.3f}ms")
        return img_feat   # de-normalize, from [-1, 1] to [0, 1]
    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VARHF(VAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )