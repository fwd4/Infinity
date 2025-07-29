import os
import random
import torch
import os.path as osp
import cv2
import numpy as np
from run_infinity import *
from tqdm import tqdm
import datetime
import yaml
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from pytorch_lightning import seed_everything
import torch



def load_yaml_config(yaml_path):
    """加载YAML配置文件"""
    if not osp.exists(yaml_path):
        print(f"配置文件 {yaml_path} 不存在，将使用默认参数")
        return {}
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """解析命令行参数"""
    # 获取当前文件所在目录
    current_dir = osp.dirname(osp.abspath(__file__))
    # 设置默认配置文件路径为相对于当前文件的路径
    default_config_path = osp.join(current_dir, '../configs/default_config.yaml')
    default_custom_path = osp.join(current_dir, '../configs/custom_config.yaml')
    
    parser = argparse.ArgumentParser(description='Infinity 图像生成')
    parser.add_argument('--config', type=str, default=default_custom_path, help='覆盖默认配置的YAML文件路径')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1, help='Number of processes in distributed training')
    
    return parser.parse_args(), default_config_path


def init_distributed():
    """初始化分布式环境"""
    # Check if this is a distributed run
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # Initialize the process group
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=local_rank,
            device_id=torch.device(f'cuda:{local_rank}')
        )
        print(f"Distributed init: rank={rank}, local_rank={local_rank}, world_size={world_size}, device={torch.cuda.current_device()}")
        return True
    elif cmd_args.local_rank != -1:
        # Fallback to command line args
        torch.cuda.set_device(cmd_args.local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=local_rank,
            device_id=torch.device(f'cuda:{local_rank}')
        )
        print(f"Process {cmd_args.local_rank} initialized on GPU {cmd_args.local_rank}")
        return True
    return False


def setup_gpu(is_distributed, rank):
    """设置GPU设备"""
    if not is_distributed:
        torch.cuda.set_device(0)
        print("Single process mode using GPU 0")
    else:
        # 使用rank作为GPU ID
        torch.cuda.set_device(rank)
        print(f"Process {rank} using GPU {rank}")


def load_models(args):
    """加载模型"""
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    get_torch_mem_usage()
    return text_tokenizer, text_encoder, vae, infinity


def setup_output_dir(gen_kwargs, rank, world_size):
    """设置输出目录"""
    output_dir = f"./outputs/pics_mtp/mtp_{gen_kwargs['si_para']}"
    
    # Generate timestamp on rank 0 and broadcast to all processes
    if world_size > 1:
        if rank == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp_tensor = torch.tensor([int(c) for c in timestamp if c.isdigit()], dtype=torch.long, device='cuda')
        else:
            timestamp_tensor = torch.zeros(14, dtype=torch.long, device='cuda')  # YYYYmmdd_HHMMSS has 14 digits
        
        # Broadcast the timestamp from rank 0 to all processes
        dist.broadcast(timestamp_tensor, 0)
        
        # Convert back to string format
        if rank != 0:
            timestamp_digits = timestamp_tensor.cpu().numpy().astype(str)
            timestamp = "".join(timestamp_digits)
            timestamp = f"{timestamp[:8]}_{timestamp[8:]}"
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_dir = osp.join(output_dir, f"run_{timestamp}")
    
    # Only rank 0 creates the directory
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
    
    # Make sure all processes wait until directory is created
    if world_size > 1:
        dist.barrier()
    
    # Now all processes can use the directory
    return run_dir


def prepare_scale_schedule(h_div_w, pn):
    """准备缩放计划"""
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][pn]['scales']
    # Add visualization code
    schedule_list = [(1, h, w) for (_, h, w) in scale_schedule]
    
    # # Calculate products for visualization
    # products = [a * b * c for a, b, c in schedule_list]
    # indices = range(len(products))

    # moss_products=products[:-3]+[0.35*products[-3],0.1*products[-2],0.05*products[-1]]
    
    # import matplotlib.pyplot as plt
    # # Create bar plot with multiple bars
    # plt.figure(figsize=(12, 6))
    
    # # Calculate bar positions
    # bar_width = 0.35
    # r1 = np.array(indices)
    # r2 = [x + bar_width for x in r1]  # Shift moss_indices bars to the right
    
    # # Create bars
    # plt.bar(r1, products, width=bar_width, color='red', label='VAR-Like', alpha=0.7)
    # plt.bar(r2, moss_products, width=bar_width, color='green', label='MoSs', alpha=0.7)
    
    # # Set font sizes
    # plt.xlabel('Scale Index', fontsize=24)
    # plt.ylabel('Tokens', fontsize=24)
    # plt.title('Scale-wise Token Investment', fontsize=28, pad=18)
    # plt.grid(True, alpha=0.3)
    
    # # Increase legend font size
    # plt.legend(fontsize=24)
    
    # def format_func(value, pos):
    #     return f'{value/1000:.1f}k' if value >= 1000 else str(int(value))
    # from matplotlib.ticker import FuncFormatter
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))
    # # Increase tick label sizes
    # plt.xticks([r + bar_width/2 for r in range(len(indices))], indices, fontsize=18)
    # plt.yticks(fontsize=18)
    
    # # Save the plot
    # plt.savefig('scale_schedule_plot.png', bbox_inches='tight')
    # plt.close()
    # exit(0)
    
    return schedule_list


def cumsum_tensor_lists(tlist):
    """
    Args:
        tlist (list): List of tensors
        
    Returns:
        list: List of cumulative sum tensors
    """
        
    result = []
    cumsum = torch.zeros_like(tlist[0])  # Initialize with zeros tensor of same shape/dtype
    
    for t in tlist:
        cumsum += t
        result.append(cumsum.clone())  # Clone to avoid reference issues
    return result

def generate_images(infinity, vae, text_tokenizer, text_encoder, prompts, gen_kwargs, 
                   run_dir, total_iterations, rank, world_size):
    """生成图像"""
    # 在分布式环境中，每个进程处理一部分数据
    iterations_per_process = (total_iterations + world_size - 1) // world_size
    
    # 将提示词列表转换为列表，以便分片
    prompt_items = list(prompts.items())#[5:6]
    # 设置随机种子，确保不同进程生成不同的图像
    # random.seed(rank + int(time.time()))
    
    with tqdm(total=iterations_per_process, desc=f"GPU {rank} Generating images") as pbar:
        for i in range(iterations_per_process):
            # 计算全局迭代索引
            global_idx = rank * iterations_per_process + i
                
            # 随机选择提示词
            category, prompt = prompt_items[global_idx % len(prompt_items)]
            
            # 设置随机种子（每次迭代都不同）
            # dragon uses 2
            seed_everything(0)
            gen_kwargs['g_seed'] = 42
            
            # 使用**kwargs方式调用gen_one_img
            generated_image, tensors = gen_one_img(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                prompt,
                **gen_kwargs
            )
            '''
            cumsum_tensors = cumsum_tensor_lists(tensors)
            B = cumsum_tensors[-1].permute(0, 2, 3, 4, 1)
            print(len(cumsum_tensors), cumsum_tensors[0].shape, B.shape)
            B = B.reshape(-1, 64*64, 32)
            mask_list = get_freq([B] * 3, [75, 40, 15])
            
            for j, cumsum in enumerate(cumsum_tensors):
                img = vae.decode(cumsum.squeeze(-3))
                img = (img + 1) / 2
                img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))[0]
                img_r = vae.decode(cumsum_tensors[-1].squeeze(-3) - cumsum.squeeze(-3))
                img_r = (img_r + 1) / 2
                img_r = img_r.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))[0]
                
                # Apply masks for the last 3 images
                if j >= len(cumsum_tensors) - 3:
                    mask_idx = j - (len(cumsum_tensors) - 3)  # Index for mask_list
                    img_np = np.zeros((1024, 1024, 4), dtype=np.uint8)
                    img_np[:, :, :3] = img.cpu().numpy()
                    img_np[:, :, 3] = 255
                    
                    # Create a mask for 1024x1024 image
                    h, w = 1024, 1024
                    patch_size = 16
                    
                    # Convert flattened coordinates to 2D coordinates (assuming H/W=64)
                    for coord in range(4096):
                        if coord in mask_list[mask_idx]:
                            continue
                        x = (coord % 64) * patch_size
                        y = (coord // 64) * patch_size
                        
                        # Apply 16x16 mask patch
                        img_np[y:y+patch_size, x:x+patch_size] = [255,255,255,255]
                    
                    
                save_path = osp.join(run_dir, f"re_{category}_gpu{rank}_iter{i}_stage{j}.jpg")
                res_path = osp.join(run_dir, f"re_{category}_gpu{rank}_iter{i}_stage{j}_res.jpg")
                save_np_path = osp.join(run_dir, f"re_{category}_gpu{rank}_iter{i}_stage{j}_np.jpg")
                if not osp.exists(save_path):
                    cv2.imwrite(save_path, img.cpu().numpy())
                    cv2.imwrite(res_path, img_r.cpu().numpy())
                    if j >= len(cumsum_tensors) - 3:
                        #img_np = cv2.resize(img_np, (256, 256), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(save_np_path, img_np)
            '''

            # Collect generated images for grid
            if i % 6 == 0:
                image_batch = []

            image_batch.append(generated_image.cpu().permute(2, 0, 1)[[2,1,0],:,:])

            # Save individual image
            save_path = osp.join(run_dir, f"re_{category}_gpu{rank}_iter{i}.jpg")
            if not osp.exists(save_path):
                cv2.imwrite(save_path, generated_image.cpu().numpy())

            # When 6 images are collected, make a grid and save
            if (i + 1) % 6 == 0 or (i + 1) == iterations_per_process:
                from torchvision.utils import make_grid
                from PIL import Image
                import torchvision

                grid = make_grid(image_batch, nrow=6)
                grid_image = torchvision.transforms.ToPILImage()(grid)

                grid_save_path = osp.join(run_dir, f"grid_gpu{rank}_batch{i//6}.jpg")
                grid_image.save(grid_save_path)
                #cv2.imwrite(grid_save_path, cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR))
                image_batch = []

            # 更新进度条
            pbar.set_postfix({"prompt": category, "gpu": rank})
            pbar.update(1)


def print_statistics(rank):
    """打印统计信息"""
    warpup_ratio = len(COST) // 5
    print(f"rank{rank} imgs {len(COST[warpup_ratio:])}, cost: {np.mean(COST[warpup_ratio:])}, infinity cost={np.mean(INFI_COST[warpup_ratio:])}")


def main():
    """主函数"""
    global cmd_args
    cmd_args, default_config_path = parse_args()
    
    # 加载配置
    config = load_config(default_config_path, cmd_args.config)
    
    # 初始化分布式环境
    is_distributed = init_distributed()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1
    
    # 设置GPU
    setup_gpu(is_distributed, rank)
    
    # 创建Namespace对象
    args = argparse.Namespace(**config.get('init_args', {}))
    # 从配置获取默认生成参数
    gen_kwargs = config.get('gen_kwargs', {}).copy()
    # 准备缩放计划
    gen_kwargs['scale_schedule'] = prepare_scale_schedule(gen_kwargs['h_div_w'], args.pn)
    if 'h_div_w' in gen_kwargs:
        del gen_kwargs['h_div_w']
    
    # 加载模型
    text_tokenizer, text_encoder, vae, infinity = load_models(args)
    
    # 从配置加载提示词
    prompts = config.get('prompts', {})
    if not prompts:
        print("警告: 配置中未找到提示词，请检查配置文件")
    
    # 设置输出目录
    run_dir = setup_output_dir(gen_kwargs, rank, world_size)
    
    # 设置迭代次数
    total_iterations = config.get('total_iterations', 6)
    
    
    # 生成图像
    generate_images(infinity, vae, text_tokenizer, text_encoder, prompts, gen_kwargs, 
                   run_dir, total_iterations, rank, world_size)
    
    # 等待所有进程完成
    if is_distributed:
        dist.barrier()
    
    # 打印统计信息
    print_statistics(rank)
    
    # 清理分布式环境
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
