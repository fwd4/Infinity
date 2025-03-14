"""
Wrap torch's flex attention and handle mess info or potentially refactor
"""
from functools import partial
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask, and_masks, or_masks
    flex_attention_available = True
except ImportError:
    print(f"[Warning] flex attention need pytorch 2.5.0+ but your version is {torch.__version__}")
    flex_attention_available = False

# Import flash_attn's attention
from flash_attn import flash_attn_func                  # q, k, or v: BLHc, ret: BLHc
from flash_attn import flash_attn_varlen_kvpacked_func  # qkv: N3Hc, ret: NHc

from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc

def _causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def _length_to_offsets(lengths, device):
    """Converts a list of lengths to a list of offsets.

    Args:
        lengths: A list of lengths.

    """
    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets

def _generate_var_mask_mod(offsets):
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        offsets: This tensor should be of shape(num_documents + 1)
            this should contain the cumulative counts of document tokens.
            e.g. if you have 3 documents of length 2, 4, 3 then
            offsets = [0, 2, 6, 9]

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.
    """

    def _offsets_to_doc_ids_tensor(offsets):
        device = offsets.device
        counts = offsets[1:] - offsets[:-1]
        return torch.repeat_interleave(
            torch.arange(len(counts), device=device, dtype=torch.int32), counts
        )

    document_id = _offsets_to_doc_ids_tensor(offsets)

    def var_mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_id[q_idx] == document_id[kv_idx]
        causal_mask = _causal_mask(b, h, q_idx, kv_idx)
        return same_doc | causal_mask

    return var_mask_mod

def _generate_var_infer_mask_with_kv_cache(lengths):
    kv_len = sum(lengths)
    def var_mask_mod(b, h, q_idx, kv_idx):
        return kv_idx < kv_len

    return var_mask_mod

pix = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
# pix = [1, 8, 16, 24, 32, 40, 48, 64]
row_aff = [p // 4 for p in pix]
qlen_raw = [x*x for x in pix]
qlen = torch.tensor(np.cumsum(qlen_raw), device='cuda')
pix = torch.tensor(pix, device='cuda')
print(qlen)

MASK_RIGHT=5

def infi_const_mask(b, h, q_idx, kv_idx):
    return kv_idx <= qlen[-MASK_RIGHT-1]

def infi_mask(lengths):
    n_mask_stages = max(0, len(lengths) - len(pix) + MASK_RIGHT)
    last_stage = len(lengths) - 1
    def get_stage_i_mask(i):
        row_affinity = pix[last_stage - i] ** 2 // 8
        def lbound(b, h, q_idx, kv_idx):
            left_distance = kv_idx - qlen[last_stage - i - 1]
            return q_idx * pix[last_stage - i] // pix[last_stage] - left_distance < row_affinity
        def rbound(b, h, q_idx, kv_idx):
            left_distance = kv_idx - qlen[last_stage - i - 1]
            return -q_idx * pix[last_stage - i] // pix[last_stage] + left_distance < row_affinity
        return and_masks(lbound, rbound)
    
    stage_masks = infi_const_mask
    for i in range(n_mask_stages):
        stage_masks = or_masks(stage_masks, get_stage_i_mask(i))
        break
        #stage_masks.append(get_stage_i_mask(i))
    return stage_masks
    #return or_masks(*stage_masks, infi_const_mask)


def infi_mask2(lengths):
    n_mask_stages = 0
    last_stage = len(lengths) - 1
    row_affinity = pix[last_stage] ** 2 // 8
    def lbound(b, h, q_idx, kv_idx):
        left_distance = kv_idx - qlen[-MASK_RIGHT-1]
        return q_idx - left_distance < row_affinity
    def rbound(b, h, q_idx, kv_idx):
        left_distance = kv_idx - qlen[-MASK_RIGHT-1]
        return -q_idx + left_distance < row_affinity
    return or_masks(infi_const_mask, and_masks(lbound, rbound))

# def per_scale_score_mod1(b, h, q_idx, kv_idx):
#     return (q_idx - (kv_idx - qlen[-2])) <= 8*pix[-1]

# def per_scale_score_mod2(b, h, q_idx, kv_idx):
#     return ((kv_idx - qlen[-2]) - q_idx) <= 8*pix[-1]

# def per_scale_score_mod4(b, h, q_idx, kv_idx):
#     return (q_idx * 48 // 64 - (kv_idx - qlen[-3])) <= 6*pix[-3]

# def per_scale_score_mod5(b, h, q_idx, kv_idx):
#     return (- q_idx * 48 // 64 + (kv_idx - qlen[-3])) <= 6*pix[-3]

# def per_scale_score_mod6(b, h, q_idx, kv_idx):
#     return (q_idx * 40 // 64 - (kv_idx - qlen[-4])) <= 5*pix[-4]

# def per_scale_score_mod7(b, h, q_idx, kv_idx):
#     return (- q_idx * 40 // 64 + (kv_idx - qlen[-4])) <= 5*pix[-4]

# def per_scale_score_mod8(b, h, q_idx, kv_idx):
#     return (q_idx * 32 // 64 - (kv_idx - qlen[-5])) <= 4*pix[-5]

# def per_scale_score_mod9(b, h, q_idx, kv_idx):
#     return (- q_idx * 32 // 64 + (kv_idx - qlen[-5])) <= 4*pix[-5]

# def per_scale_score_mod3(b, h, q_idx, kv_idx):
#     return kv_idx <= qlen[-5]
    
# stage13_mod = and_masks(per_scale_score_mod1, per_scale_score_mod2) # 64x64
# stage12_mod = and_masks(per_scale_score_mod4, per_scale_score_mod5) # 48x48
# stage11_mod = and_masks(per_scale_score_mod6, per_scale_score_mod7) # 40x40
# stage10_mod = and_masks(per_scale_score_mod8, per_scale_score_mod9) # 32x32
# infi_mask = or_masks(stage10_mod, stage11_mod, stage12_mod, stage13_mod, per_scale_score_mod3)

class FlexAttn(nn.Module):
    def __init__(
            self, block_scales:list, mask_type:str, B, H, L:int, auto_padding=False
    ):
        """
        :param block_scales: accept VAR's block sizes like [(1,1), (2,2), (3,3)]
        :param mask_type: var/causal
        :param B: batch size
        :param H: heads num
        :param L: sequence length
        """
        super().__init__()
        if not flex_attention_available:
            raise NotImplementedError((f"[Error] flex attention need pytorch 2.5.0+ but your version is {torch.__version__}"))

        self.support_mask_type = ["var", "causal", "var_infer_mask_with_kv_cache"]
        self.auto_padding = auto_padding

        self.flex_attention = torch.compile(flex_attention)

        self.block_scales = block_scales
        self.lengths = [ x * y * z for x,y,z in block_scales]

        self.offsets = _length_to_offsets(self.lengths, device='cuda')

        # if L paded to align 128, block need to cover padding area
        if self.offsets[-1] < L:
            self.offsets = torch.cat((self.offsets, torch.tensor([L], device='cuda')), dim=0)
        
        if mask_type == "var":
            self.mask_mod = _generate_var_mask_mod(self.offsets)
            self.block_mask = create_block_mask(self.mask_mod, B = B, H = H, Q_LEN = L, KV_LEN = L, device = 'cuda', _compile = True)
        elif mask_type == "causal":
            self.mask_mod = _causal_mask
            self.block_mask = create_block_mask(self.mask_mod, B = B, H = H, Q_LEN = L, KV_LEN = L, device = 'cuda', _compile = True)
        elif mask_type == 'var_infer_mask_with_kv_cache':
            self.mask_mod = _generate_var_infer_mask_with_kv_cache(self.lengths)
            print(B, H, L, self.lengths[-2:])
            pad_q = ((self.lengths[-1] + 127) // 128) * 128
            mask = and_masks(infi_mask2(self.lengths), self.mask_mod)
            #mask = self.mask_mod
            self.block_mask = create_block_mask(mask, B = 1, H = 1, Q_LEN = pad_q, KV_LEN = L, device = 'cuda', _compile = True)
            print(f"{self.block_mask}")
        else:
            raise NotImplementedError(f"{mask_type} not supportted in FlexAttn, support type:{self.support_mask_type}")


    def forward(self, q, k, v, scale = None):
        if self.auto_padding:
            q_pad_len = (128 - q.shape[-2] % 128) % 128
            kv_pad_len = (128 - k.shape[-2] % 128) % 128
            q_pad = F.pad(q, (0, 0, 0, q_pad_len))
            k_pad = F.pad(k, (0, 0, 0, kv_pad_len))
            v_pad = F.pad(v, (0, 0, 0, kv_pad_len))
            oup = self.flex_attention(q_pad, k_pad, v_pad, block_mask = self.block_mask, scale = scale)
            if q_pad_len > 0:
                oup = oup[:,:,:-q_pad_len]
        else:
            oup = self.flex_attention(q.to(v.dtype), k.to(v.dtype), v, block_mask = self.block_mask, scale = scale)
        return oup

    def extra_repr(self) -> str:
        tail = ''
        return f'block size:{self.block_scales} {tail}'
