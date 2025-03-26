import os
import json
import glob
import argparse
import hashlib
import resource
from os import path as osp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import numpy as np
from PIL import Image
from datasets import load_dataset, Dataset

from infinity.utils.dynamic_resolution import h_div_w_templates

# Disable core dumps
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

def compute_hash(text):
    """计算文本的哈希值，用于去重"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_image_size(image_path):
    """获取图片尺寸"""
    try:
        with Image.open(image_path) as img:
            return img.size  # 返回 (width, height)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def find_closest_template(h_div_w):
    """找到最接近的高宽比模板"""
    return h_div_w_templates[np.argmin(np.abs(h_div_w - h_div_w_templates))]

LOCAL_PATH="/root/.cache/huggingface/hub/datasets--takara-ai--image_captions/snapshots/6e741bf8256d607c1fb15d36eaa62dff95cab057/data"
#LOCAL_PATH="/workspace/Infinity/data/processed_dataset/unique_dataset"

def process_dataset(args):
    """处理数据集：下载、去重、分桶"""
    print(f"Loading dataset: {args.dataset_name}")
    
    # 加载HuggingFace数据集
    data_files = {"train": [f"{LOCAL_PATH}/train-00{part:03d}-of-00159.parquet" for part in range(159)]}
    # data_files = {"train": [f"{LOCAL_PATH}/shard-00{part:03d}-of-00100.parquet" for part in range(100)]}
    dataset = load_dataset("parquet", data_files=data_files, split="train")
    #dataset = load_dataset(args.dataset_name, split=args.split, streaming=True).remove_columns(f'{args.image_field}')
    #dataset = load_dataset(args.dataset_name, split="train[:1%]").remove_columns(f'{args.text_field}')
    n_samples = dataset.info.splits[dataset.split].num_examples
    print(f"Dataset stream loading {n_samples} samples")
    
    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # # 创建图片保存目录
    # image_dir = osp.join(output_dir, "images")
    # os.makedirs(image_dir, exist_ok=True)
    
    # # 创建splits目录
    # splits_dir = osp.join(output_dir, "splits")
    # os.makedirs(splits_dir, exist_ok=True)
    
    # 去重处理
    print("Removing duplicates...")
    unique_captions = set()  # 仅用于去重的caption集合
    template_items = defaultdict(list)  # 每个模板的items
    template_shards = defaultdict(int)  # 每个模板已保存的分片数
    duplicates = 0
    
    def save_template_shard(template, items, shard_idx):
        """保存单个模板的一个分片"""
        template_dir = osp.join(output_dir, f"template_{template}")
        os.makedirs(template_dir, exist_ok=True)
        
        template_dataset = Dataset.from_dict({
            "id": [item["id"] for item in items],
            "image": [item["image"] for item in items],
            "caption": [item["caption"] for item in items],
            "width": [item["width"] for item in items],
            "height": [item["height"] for item in items],
            "h_div_w": [item["h_div_w"] for item in items],
        })
        
        shard_path = osp.join(template_dir, f"shard-{shard_idx:05d}.parquet")
        template_dataset.to_parquet(shard_path)
        print(f"Saved template {template} shard {shard_idx} with {len(items)} samples")

    SAMPLES_PER_SHARD = 1000
    current_id = 0
    for item in tqdm(dataset, total=n_samples):
        text = item[args.text_field]
        image = item[args.image_field]
        
        if text not in unique_captions:
            unique_captions.add(text)
            width, height = image.size
            h_div_w = height / width
            h_div_w_template = find_closest_template(h_div_w)
            
            # 保存到对应模板的列表中
            template_items[h_div_w_template].append({
                "id": current_id,
                "image": image,
                "caption": text,
                "width": width,
                "height": height,
                "h_div_w": h_div_w,
            })
            current_id += 1
            
            # 如果当前模板的items达到阈值，保存一个分片
            if len(template_items[h_div_w_template]) >= SAMPLES_PER_SHARD:
                save_template_shard(
                    h_div_w_template, 
                    template_items[h_div_w_template], 
                    template_shards[h_div_w_template]
                )
                template_shards[h_div_w_template] += 1
                template_items[h_div_w_template] = []  # 清空已保存的items
        else:
            duplicates += 1
    
    # 保存剩余的items
    for template, items in template_items.items():
        if items:  # 如果还有未保存的items
            save_template_shard(template, items, template_shards[template])
            template_shards[template] += 1
    
    print(f"\nFound {duplicates} duplicates.")
    print("Template statistics:")
    for template, num_shards in sorted(template_shards.items()):
        total_samples = (num_shards - 1) * SAMPLES_PER_SHARD + len(template_items[template])
        print(f"h/w {template}: {total_samples} samples in {num_shards} shards")
    
    return
    
    # 保存图片并收集元数据
    print("Saving images and collecting metadata...")
    metadata = []
    h_div_w_buckets = defaultdict(list)
    
    with Pool(processes=cpu_count()) as pool:
        for idx, (text_hash, item) in enumerate(tqdm(unique_items.items())):
            # 保存图片
            image_path = osp.join(image_dir, f"{idx}.jpg")
            item["image"].save(image_path)
            
            # 获取图片尺寸
            width, height = item["image"].size
            h_div_w = height / width
            
            # 找到最接近的高宽比模板
            h_div_w_template = find_closest_template(h_div_w)
            
            # 创建元数据
            meta_item = {
                "image_path": image_path,
                "text": item["text"],
                "long_caption": item["text"],  # 使用相同文本作为长描述
                "h_div_w": h_div_w,
                "h_div_w_template": h_div_w_template
            }
            
            metadata.append(meta_item)
            h_div_w_buckets[str(h_div_w_template)].append(meta_item)
    
    # 按高宽比分桶并保存
    print("Creating splits by aspect ratio...")
    for h_div_w_template, items in h_div_w_buckets.items():
        output_file = osp.join(splits_dir, f"{h_div_w_template}_{len(items)}.jsonl")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item) + '\n')
        
        print(f"Created {output_file} with {len(items)} items")
    
    # 保存完整元数据
    with open(osp.join(output_dir, "metadata.jsonl"), 'w', encoding='utf-8') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')
    
    print(f"Processing complete. Results saved to {output_dir}")
    print(f"Total unique samples: {len(metadata)}")
    print("Aspect ratio distribution:")
    for h_div_w_template, items in sorted(h_div_w_buckets.items()):
        print(f"  {h_div_w_template}: {len(items)} samples ({len(items)/len(metadata)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Preprocess HuggingFace dataset for Infinity")
    parser.add_argument("--dataset_name", type=str, default="takara-ai/image_captions", help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--output_dir", type=str, default="/workspace/Infinity/data/processed_dataset", 
                        help="Output directory for processed data")
    parser.add_argument("--text_field", type=str, default="caption", help="Field name for text in the dataset")
    parser.add_argument("--image_field", type=str, default="image", help="Field name for image in the dataset")
    
    args = parser.parse_args()
    process_dataset(args)

if __name__ == "__main__":
    main()