import os
from os import path as osp
import argparse
import glob
from datasets import load_dataset
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
from io import BytesIO

def scale_image(image, target_size=256):
    """Scale image to have short side = target_size while maintaining aspect ratio"""
    w, h = image.size
    if h < w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

def visualize_samples(dataset, template, shard_id, num_samples=5):
    """Visualize random samples from the dataset with their captions."""
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Create samples directory
    samples_dir = f"/workspace/Infinity/data/enhanced_dataset/template_{template}/samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    def wrap_text(text, width=10):
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            if len(current_line) < width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        return '\n'.join(lines)
    
    for i, idx in enumerate(indices):
        image = dataset['image'][idx]
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        image = scale_image(image)  # Scale image before display
        
        original_caption = wrap_text(dataset['caption'][idx])
        enhanced_caption = wrap_text(dataset['enhanced_caption'][idx])
        
        fig = plt.figure(figsize=(15, 8))
        
        # Left subplot for image
        ax1 = plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis('off')
        
        # Right subplot for captions
        ax2 = plt.subplot(1, 2, 2)
        ax2.text(0, 0.7, original_caption, fontsize=16, color='red', wrap=True, horizontalalignment='left')
        ax2.text(0, 0.2, enhanced_caption, fontsize=16, color='green', wrap=True, horizontalalignment='left')
        ax2.axis('off')
        
        plt.tight_layout()
        output_path = osp.join(samples_dir, f"shard-{shard_id:05d}_sample-{i:02d}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved visualization to {output_path}")

def get_random_shard_id(template):
    shard_pattern = f"/workspace/Infinity/data/enhanced_dataset/template_{template}/shard-*.parquet"
    shard_files = glob.glob(shard_pattern)
    if not shard_files:
        raise ValueError(f"No shards found for template {template}")
    
    random_shard = random.choice(shard_files)
    return int(osp.basename(random_shard).split("-")[1].split(".")[0])

def main():
    parser = argparse.ArgumentParser(description="Visualize samples from enhanced dataset")
    parser.add_argument("template", type=str, help="Template value (e.g., '1.0')")
    parser.add_argument("--shard-id", type=int, help="Shard ID to visualize", default=None)
    args = parser.parse_args()
    
    shard_id = args.shard_id if args.shard_id is not None else get_random_shard_id(args.template)
    
    input_path = f"/workspace/Infinity/data/enhanced_dataset/template_{args.template}/shard-{shard_id:05d}.parquet"
    
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return
    
    dataset = load_dataset("parquet", data_files={"train": input_path}, split="train")
    visualize_samples(dataset, args.template, shard_id)

if __name__ == "__main__":
    main()