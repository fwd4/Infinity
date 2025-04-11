import os
from os import path as osp
import glob
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi
import argparse
from conf import HF_UPLOAD_TOKEN

ROOT_DIR='/workspace/Infinity'

def upload_filtered_dataset(template, repo_id, token):
    filtered_base_path = f"{ROOT_DIR}/data/enhanced_dataset/template_{template}"
    filtered_shard_files = sorted(glob.glob(f"{filtered_base_path}/shard-*.parquet"))
    
    if not filtered_shard_files:
        raise ValueError(f"No filtered shards found for template {template}")
    
    dataset = load_dataset("parquet", data_files=filtered_shard_files, split="train")
    dataset.push_to_hub(repo_id, private=False, token=token, split=f"h2w_{template}")
    total_filtered = len(dataset)
    print(f"Uploaded all filtered shards ({total_filtered} total samples)")

def main():
    parser = argparse.ArgumentParser(description="Upload filtered dataset to Hugging Face")
    parser.add_argument("h2w", type=str, help="Template value (e.g., '1.0')")
    parser.add_argument("--repo_id", type=str, default="fwd4xl/enhanced_image_captions", help="Hugging Face repository ID")
    parser.add_argument("--token", type=str, default=HF_UPLOAD_TOKEN, help="Hugging Face private token")
    args = parser.parse_args()
    
    upload_filtered_dataset(args.h2w, args.repo_id, args.token)
    print(f"Successfully uploaded filtered dataset h2w_{args.h2w} to {args.repo_id}")

if __name__ == "__main__":
    main()