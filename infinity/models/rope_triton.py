import torch  
import triton  
import triton.language as tl  

@triton.jit  
def apply_rotary_kernel(  
    q_ptr, k_ptr,                # 查询和键向量的指针  
    rope_cos_ptr, rope_sin_ptr,  # RoPE 缓存(cos/sin)的指针  
    q_out_ptr, k_out_ptr,        # 输出指针  
    batch_size, seq_len, heads, head_dim, half_head_dim,  # 形状参数  
    stride_qb, stride_qh, stride_ql, stride_qc,           # q的步长  
    stride_kb, stride_kh, stride_kl, stride_kc,           # k的步长  
    stride_rope_l, stride_rope_d,                         # rope_cache的步长  
    stride_q_out_b, stride_q_out_h, stride_q_out_l, stride_q_out_c,  # 输出q的步长  
    stride_k_out_b, stride_k_out_h, stride_k_out_l, stride_k_out_c,  # 输出k的步长  
    BLOCK_SIZE: tl.constexpr,    # 编译时常量，块大小  
):  
    # 获取程序ID  
    pid_b = tl.program_id(0)     # 批次维度  
    pid_h = tl.program_id(1)     # 头维度  
    pid_l = tl.program_id(2)     # 序列长度维度  
    
    # 计算基础偏移量  
    q_base = pid_b * stride_qb + pid_h * stride_qh + pid_l * stride_ql  
    k_base = pid_b * stride_kb + pid_h * stride_kh + pid_l * stride_kl  
    rope_base = pid_l * stride_rope_l  
    
    # 创建用于实部和虚部的偏移量  
    offs_m = tl.arange(0, BLOCK_SIZE // 2)  
    mask_m = offs_m < half_head_dim  
    
    # 处理头部维度的每个块  
    for d_idx in range(0, half_head_dim, BLOCK_SIZE // 2):  
        # 计算当前块的偏移量和掩码  
        offs_d = tl.arange(0, BLOCK_SIZE)  
        block_d = d_idx + offs_d  
        mask_d = block_d < half_head_dim 
        
        # 实部和虚部的索引  
        real_idx = block_d * 2        # 实部索引(偶数位置)  
        imag_idx = block_d * 2 + 1    # 虚部索引(奇数位置)  
        
        # 加载旋转系数  
        cos_ptr = rope_cos_ptr + rope_base + block_d * stride_rope_d  
        sin_ptr = rope_sin_ptr + rope_base + block_d * stride_rope_d  
        cos = tl.load(cos_ptr, mask=mask_d, other=0.0)  
        sin = tl.load(sin_ptr, mask=mask_d, other=0.0)  
        
        # 加载q和k的实部和虚部  
        q_real_ptr = q_ptr + q_base + real_idx * stride_qc  
        q_imag_ptr = q_ptr + q_base + imag_idx * stride_qc  
        k_real_ptr = k_ptr + k_base + real_idx * stride_kc  
        k_imag_ptr = k_ptr + k_base + imag_idx * stride_kc  
        
        q_real = tl.load(q_real_ptr, mask=mask_d, other=0.0)  
        q_imag = tl.load(q_imag_ptr, mask=mask_d, other=0.0)  
        k_real = tl.load(k_real_ptr, mask=mask_d, other=0.0)  
        k_imag = tl.load(k_imag_ptr, mask=mask_d, other=0.0)  
        
        # 应用旋转变换  
        q_real_new = cos * q_real - sin * q_imag  
        q_imag_new = sin * q_real + cos * q_imag  
        k_real_new = cos * k_real - sin * k_imag  
        k_imag_new = sin * k_real + cos * k_imag  
        
        # 存储结果  
        q_out_real_ptr = q_out_ptr + q_base + real_idx * stride_q_out_c  
        q_out_imag_ptr = q_out_ptr + q_base + imag_idx * stride_q_out_c  
        k_out_real_ptr = k_out_ptr + k_base + real_idx * stride_k_out_c  
        k_out_imag_ptr = k_out_ptr + k_base + imag_idx * stride_k_out_c  
        
        tl.store(q_out_real_ptr, q_real_new, mask=mask_d)  
        tl.store(q_out_imag_ptr, q_imag_new, mask=mask_d)  
        tl.store(k_out_real_ptr, k_real_new, mask=mask_d)  
        tl.store(k_out_imag_ptr, k_imag_new, mask=mask_d)  

def apply_rotary_triton(q, k, rope_cache, using_flash=False):  
    """  
    使用Triton实现的RoPE（旋转位置编码）函数  
    
    参数:  
        q: 查询向量，形状为 (batch_size, seq_len, heads, head_dim) 或 (batch_size, heads, seq_len, head_dim)  
        k: 键向量，形状与q相同  
        rope_cache: 包含旋转因子的元组 (cos, sin)，形状均为 (seq_len, head_dim//2)  
        using_flash: 是否使用flash attention格式  
    
    返回:  
        旋转后的q和k  
    """  
    # 处理using_flash的情况  
    if using_flash:  
        q = q.transpose(1, 2)  # (B, L, H, C) -> (B, H, L, C)  
        k = k.transpose(1, 2)
    
    # 获取维度信息  
    batch_size, heads, seq_len, head_dim = q.shape  
    half_head_dim = head_dim // 2  
    
    # 获取rope_cache的cos和sin部分  
    rope_cos = rope_cache[0].view(-1,half_head_dim)  # (seq_len, half_head_dim)
    rope_sin = rope_cache[1].view(-1,half_head_dim)  #.contiguous()
    
    # 创建输出张量  
    q_out = torch.empty_like(q)  
    k_out = torch.empty_like(k)  
    
    # 计算stride信息  
    stride_qb, stride_qh, stride_ql, stride_qc = q.stride()  #shape:[b,h,seq_len,half_head_dim]
    stride_kb, stride_kh, stride_kl, stride_kc = k.stride()  
    stride_rope_l, stride_rope_d = rope_cos.stride()    #shape：torch.Size([seq_len, half_head_dim]) 
    stride_q_out_b, stride_q_out_h, stride_q_out_l, stride_q_out_c = q_out.stride()  
    stride_k_out_b, stride_k_out_h, stride_k_out_l, stride_k_out_c = k_out.stride()  
    
    # 设置块大小  
    BLOCK_SIZE = triton.next_power_of_2(head_dim)  #128
    BLOCK_SIZE = min(BLOCK_SIZE, 256)  # 防止块太大  
    
    # 启动kernel  
    grid = (batch_size, heads, seq_len)  
    apply_rotary_kernel[grid](q,k,rope_cos,rope_sin,q_out,k_out,batch_size,seq_len,heads,head_dim,half_head_dim,stride_qb,stride_qh,stride_ql,stride_qc,stride_kb,stride_kh,stride_kl,stride_kc,stride_rope_l,stride_rope_d,stride_q_out_b,stride_q_out_h,stride_q_out_l,stride_q_out_c,stride_k_out_b,stride_k_out_h,stride_k_out_l,stride_k_out_c,BLOCK_SIZE,)  
    
    # 如果使用flash，转置回原来的维度  
    if using_flash:  
        q_out = q_out.transpose(1, 2)  
        k_out = k_out.transpose(1, 2)  
    
    return q_out, k_out  