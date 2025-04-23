import os
import os.path as osp
import hashlib
import time
import argparse
import json
import shutil
import glob
import re
import sys
import yaml
import tqdm

import cv2
import tqdm
import torch
import torch.distributed as dist
import numpy as np
from pytorch_lightning import seed_everything

from infinity.utils.csv_util import load_csv_as_dicts, write_dicts2csv_file
from tools.run_infinity import *
from conf import HF_TOKEN, HF_HOME

# set environment variables
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HF_HOME'] = HF_HOME
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'


def load_yaml_config(yaml_path):
    """加载YAML配置文件"""
    if not osp.exists(yaml_path):
        print(f"配置文件 {yaml_path} 不存在，将使用默认参数")
        return {}
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_config(default_config_path, custom_config_path=None):
    """加载默认配置和自定义配置"""
    # 加载默认配置
    config = load_yaml_config(default_config_path)
    
    # 如果提供了自定义配置，则覆盖默认配置
    if custom_config_path and osp.exists(custom_config_path):
        custom_config = load_yaml_config(custom_config_path)
        # 递归更新配置
        deep_update(config, custom_config)
    
    return config


def deep_update(d, u):
    """递归更新字典"""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def parse_args():
    """解析命令行参数"""
    # 获取当前文件所在目录
    current_dir = osp.dirname(osp.abspath(__file__))
    # 设置默认配置文件路径
    default_config_path = osp.join(current_dir, '../../configs/default_config.yaml')
    default_custom_path = osp.join(current_dir, '../../configs/custom_config.yaml')
    
    parser = argparse.ArgumentParser()
    # add_common_arguments(parser)
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--metadata_file', type=str, default='evaluation/gen_eval/prompts/evaluation_metadata.jsonl')
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0,1,2])
    parser.add_argument('--load_rewrite_prompt_cache', type=int, default=1, choices=[0,1])
    parser.add_argument('--config', type=str, default=default_custom_path, help='覆盖默认配置的YAML文件路径')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    
    return parser.parse_args(), default_config_path


def load_models(args):
    """加载模型"""
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    return text_tokenizer, text_encoder, vae, infinity


def prepare_scale_schedule(h_div_w, pn):
    """准备缩放计划"""
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][pn]['scales']
    return [(1, h, w) for (_, h, w) in scale_schedule]


def init_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # 初始化进程组
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=local_rank,
            device_id=torch.device(f'cuda:{local_rank}')
        )
        print(f"Distributed init: rank={rank}, local_rank={local_rank}, world_size={world_size}, device={torch.cuda.current_device()}")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


if __name__ == '__main__':
    # 初始化分布式环境
    is_distributed, rank, world_size, local_rank = init_distributed()
    
    # 设置GPU设备
    if is_distributed:
        torch.cuda.set_device(local_rank)
    else:
        torch.cuda.set_device(0)
    
    args, default_config_path = parse_args()
    
    # 加载配置
    config = load_config(default_config_path, args.config)
    
    # 将配置中的参数更新到args
    for key, value in config.get('init_args', {}).items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    
    # 确保输出目录存在
    if rank == 0:  # 只有主进程创建目录
        os.makedirs(args.outdir, exist_ok=True)
        
        # 将配置写入输出目录
        config_output_path = os.path.join(args.outdir, "config_used.yaml")
        with open(config_output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    if is_distributed:
        # 等待主进程创建目录
        dist.barrier()
    
    # 获取生成参数
    gen_kwargs = config.get('gen_kwargs', {}).copy()

    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    
    # 在分布式环境中，每个进程处理一部分数据
    if is_distributed:
        # 计算每个进程应处理的数据
        num_items_per_process = len(metadatas) // world_size
        start_idx = rank * num_items_per_process
        end_idx = start_idx + num_items_per_process if rank < world_size - 1 else len(metadatas)
        metadatas = metadatas[start_idx:end_idx]
        print(f"Rank {rank}: Processing {len(metadatas)} prompts from index {start_idx} to {end_idx-1}")
    
    prompt_rewrite_cache_file = osp.join('evaluation/gen_eval', 'prompt_rewrite_cache.json')
    if osp.exists(prompt_rewrite_cache_file):
        with open(prompt_rewrite_cache_file, 'r') as f:
            prompt_rewrite_cache = json.load(f)
    else:
        prompt_rewrite_cache = {}

    if args.model_type == 'flux_1_dev':
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
    elif args.model_type == 'flux_1_dev_schnell':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
    elif 'infinity' in args.model_type:
        # 加载模型
        text_tokenizer, text_encoder, vae, infinity = load_models(args)

        if args.rewrite_prompt:
            from tools.prompt_rewriter import PromptRewriter
            prompt_rewriter = PromptRewriter(system='', few_shot_history=[])

    for index, metadata in enumerate(tqdm.tqdm(metadatas, desc=f"Rank {rank} processing prompts")):
        # 计算全局索引
        global_index = start_idx + index if is_distributed else index
        
        seed_everything(args.seed)  # 每个提示使用不同的种子
        outpath = os.path.join(args.outdir, f"{global_index:0>5}")
        os.makedirs(outpath, exist_ok=True)
        prompt = metadata['prompt']

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        if args.rewrite_prompt:
            old_prompt = prompt
            if args.load_rewrite_prompt_cache and prompt in prompt_rewrite_cache:
                prompt = prompt_rewrite_cache[prompt]
            else:
                refined_prompt = prompt_rewriter.rewrite(prompt)
                prompt_rewrite_cache[prompt] = refined_prompt
                prompt = refined_prompt
            
        images = []
        for sample_j in range(args.n_samples):
            t1 = time.time()
            if args.model_type == 'flux_1_dev':
                image = pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=50,
                    max_sequence_length=512,
                    num_images_per_prompt=1,
                ).images[0]
            elif args.model_type == 'flux_1_dev_schnell':
                image = pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=0.0,
                    num_inference_steps=4,
                    max_sequence_length=256,
                    generator=torch.Generator("cpu").manual_seed(0)
                ).images[0]
            elif args.model_type == 'pixart_sigma':
                image = pipe(prompt).images[0]
            elif 'infinity' in args.model_type:
                # 从配置或参数中获取生成参数
                h_div_w = gen_kwargs.get('h_div_w', 1.0)
                scale_schedule = prepare_scale_schedule(h_div_w, args.pn)
                gen_kwargs['scale_schedule'] = scale_schedule
                gen_kwargs['g_seed'] = None
                
                image, _ = gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, **gen_kwargs)
            else:
                raise ValueError
            t2 = time.time()
            images.append(image)
        
        for i, image in enumerate(images):
            save_file = os.path.join(sample_path, f"{i:05}.jpg")
            if 'infinity' in args.model_type:
                cv2.imwrite(save_file, image.cpu().numpy())
            else:
                image.save(save_file)
    
        # 每个进程单独更新缓存
        # with open(prompt_rewrite_cache_file + f".rank{rank}", 'w') as f:
        #     json.dump(prompt_rewrite_cache, f, indent=2)
    
    # 写入性能日志
    with open(os.path.join(args.outdir, f"perf_rank{rank}.log"), "w") as fp:
        fp.write(f"Rank: {rank}, img_cnt: {len(COST)}, cost: {np.mean(COST[1:])}, infinity cost={np.mean(INFI_COST[1:])}\n")
        fp.write(f"config_file: {args.config}\n")
        fp.write(f"output_dir: {args.outdir}\n")
        fp.write(f"model_type: {args.model_type}\n")
    
    # 等待所有进程完成
    if is_distributed:
        dist.barrier()
        
        # 合并缓存文件（只在主进程执行）
        if rank == 0:
            # 合并性能日志
            all_costs = []
            all_infi_costs = []
            total_images = 0
            
            with open(os.path.join(args.outdir, "combined_perf.log"), "w") as fp:
                fp.write(f"Combined performance across {world_size} GPUs:\n")
                
                for r in range(world_size):
                    perf_file = os.path.join(args.outdir, f"perf_rank{r}.log")
                    if osp.exists(perf_file):
                        with open(perf_file, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                fp.write(f"Rank {r}: {line}")
                                if "img_cnt" in line:
                                    parts = line.split(',')
                                    img_cnt = int(parts[0].split(':')[1].strip())
                                    cost = float(parts[1].split(':')[1].strip())
                                    infi_cost = float(parts[2].split(':')[1].strip())
                                    
                                    total_images += img_cnt
                                    all_costs.append(cost)
                                    all_infi_costs.append(infi_cost)
                
                avg_cost = np.mean(all_costs) if all_costs else 0
                avg_infi_cost = np.mean(all_infi_costs) if all_infi_costs else 0
                
                fp.write(f"\nSummary:\n")
                fp.write(f"Total images: {total_images}\n")
                fp.write(f"Average cost: {avg_cost:.4f}\n")
                fp.write(f"Average infinity cost: {avg_infi_cost:.4f}\n")
                fp.write(f"config_file: {args.config}\n")
                fp.write(f"output_dir: {args.outdir}\n")
                fp.write(f"model_type: {args.model_type}\n")
        
        # 清理分布式环境
        dist.destroy_process_group()
