import os
import time
from os import path as osp
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import ollama
from ollama import chat
import matplotlib.pyplot as plt

import base64
from io import BytesIO
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from functools import partial
from itertools import cycle
import numpy as np
from datasets import Dataset, concatenate_datasets
import json
import multiprocessing
from opennsfw2 import predict_image
from enum import Enum, auto

SCOLOR = '\033[92m'
ECOLOR = '\033[0m'

def get_paths(template, shard_id):
    base_path = f"/workspace/Infinity/data"
    input_path = f"{base_path}/processed_dataset/template_{template}/shard-{shard_id:05d}.parquet"
    failed_path = f"{base_path}/enhanced_dataset/template_{template}/failed_shard-{shard_id:05d}.parquet"
    output_path = (
        f"{base_path}/enhanced_dataset/template_{template}/shard-{shard_id:05d}.parquet"
    )
    return input_path, failed_path, output_path


def verify_processed_parquet(processed_captions, expected_len, shard_id):
    if len(processed_captions) != expected_len:
        print(f"[Shard-{shard_id:05d}] Warning: Input samples: {expected_len}, Processed samples: {len(processed_captions)}")
        return False

    # Check if all samples have enhanced captions
    empty_captions = [i for i, caption in enumerate(processed_captions) if not caption]
    if empty_captions:
        print(
            f"[Shard-{shard_id:05d}] Warning: Found {len(empty_captions)} empty captions"
        )
        return True
    return True


def gather_samples_to_process(input_dataset, processed_dataset=None):
    """Gather all samples that need processing from both datasets."""
    samples_to_process = []

    if processed_dataset is not None:
        # Find empty captions in processed dataset
        for idx, (image, caption, enhanced) in enumerate(
            zip(
                processed_dataset["image"],
                processed_dataset["caption"],
                processed_dataset["enhanced_caption"],
            )
        ):
            if not enhanced:
                samples_to_process.append(
                    {"idx": idx, "image": image, "caption": caption, "type": "empty"}
                )

        # Find missing samples in input dataset
        processed_keys = set(processed_dataset["caption"])
        for idx, (image, caption) in enumerate(
            zip(input_dataset["image"], input_dataset["caption"])
        ):
            if caption not in processed_keys:
                samples_to_process.append(
                    {"idx": idx, "image": image, "caption": caption, "type": "missing"}
                )
    else:
        # Process all samples from input dataset
        for idx, (image, caption) in enumerate(
            zip(input_dataset["image"], input_dataset["caption"])
        ):
            samples_to_process.append(
                {"idx": idx, "image": image, "caption": caption, "type": "new"}
            )

    return samples_to_process


def update_processed_dataset(
    processed_dataset, input_dataset, processed_samples, sample_type, verbose=False
):
    """Update processed dataset with new results."""
    if sample_type == "empty":
        enhanced_captions = processed_dataset["enhanced_caption"]
        for idx, caption in processed_samples.items():
            enhanced_captions[idx] = caption
        processed_dataset = processed_dataset.map(
            lambda x: {"enhanced_caption": enhanced_captions},
            batched=True,
            batch_size=len(enhanced_captions),
        )
    else:
        # Get indices and captions as lists
        indices = list(processed_samples.keys())
        captions = list(processed_samples.values())

        # Select rows from input dataset using indices
        new_dataset = input_dataset.select(indices)
        # Add enhanced_caption column
        new_dataset = new_dataset.add_column("enhanced_caption", captions)

        if processed_dataset is not None:
            processed_dataset = concatenate_datasets([processed_dataset, new_dataset])
        else:
            processed_dataset = new_dataset
    return processed_dataset


def process_parquet(parquet_path, template, output_dir, verbose=False):
    t0 = time.time()
    shard_id = int(osp.basename(parquet_path).split("-")[1].split(".")[0])
    input_path, failed_path, output_path = get_paths(template, shard_id)

    if verbose:
        print(f"[Shard-{shard_id:05d}] Starting to process")

    # Load datasets
    t1 = time.time()
    input_dataset = load_dataset(
        "parquet", data_files={"train": input_path}, split="train"
    )
    if verbose:
        print(f"[Shard-{shard_id:05d}] Input dataset loaded in {time.time()-t1:.2f}s")

    # Load processed dataset if exists
    t1 = time.time()
    if os.path.exists(output_path):
        if verbose:
            print(
                f"[Shard-{shard_id:05d}] Loading existing processed dataset from {output_path}"
            )
        processed_dataset = load_dataset(
            "parquet", data_files={"train": output_path}, split="train"
        )
        if verbose:
            print(
                f"[Shard-{shard_id:05d}] Processed dataset loaded in {time.time()-t1:.2f}s"
            )
    else:
        if verbose:
            print(f"[Shard-{shard_id:05d}] No existing processed dataset found")
        processed_dataset = None

    # Gather samples
    t1 = time.time()
    samples_to_process = gather_samples_to_process(input_dataset, processed_dataset)
    if verbose:
        print(
            f"[Shard-{shard_id:05d}] Gathered {len(samples_to_process)} samples to process in {time.time()-t1:.2f}s"
        )
    if not samples_to_process:
        if verbose:
            print(f"{SCOLOR}[Shard-{shard_id:05d}] No samples need processing{ECOLOR}")
        return

    # Process samples
    successful_samples = {}
    nsfw_filtered = 0
    error_count = 0
    sample_type = samples_to_process[0]["type"]

    for idx, sample in enumerate(
        tqdm(samples_to_process, desc=f"[Shard-{shard_id:05d}] Processing")
    ):
        t_sample = time.time()
        enhanced_caption, error_code = process_image_and_caption(
            sample["image"], sample["caption"]
        )

        if error_code == ProcessingError.SUCCESS:
            successful_samples[sample["idx"]] = enhanced_caption
        elif error_code == ProcessingError.NSFW_SCORE_HIGH:
            nsfw_filtered += 1
        else:
            error_count += 1

    # Update dataset
    if successful_samples:
        if verbose:
            print(f"\n[Shard-{shard_id:05d}] Updating processed dataset with {len(successful_samples)} new samples")
        t1 = time.time()
        # In process_parquet function, the call remains the same:
        processed_dataset = update_processed_dataset(
            processed_dataset, input_dataset, successful_samples, sample_type, verbose
        )
        if verbose:
            print(f"[Shard-{shard_id:05d}] Dataset updated in memory in {time.time()-t1:.2f}s")

        if verbose:
            print(f"[Shard-{shard_id:05d}] Saving processed dataset to {output_path}")
        t1 = time.time()
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        processed_dataset.to_parquet(output_path)
        if verbose:
            print(f"[Shard-{shard_id:05d}] Dataset saved to disk in {time.time()-t1:.2f}s")

        # Only verify if we have processed samples
        verify_result = verify_processed_parquet(
            list(processed_dataset["enhanced_caption"]), len(input_dataset), shard_id
        )
        if verbose:
            print(f"[Shard-{shard_id:05d}] Verification completed: {'Success' if verify_result else 'Failed'}")

    # Print final statistics
    tt = time.time()
    print(
        f"[Shard-{shard_id:05d}] Processed {len(successful_samples)} samples, filtered {nsfw_filtered} NSFW images, {error_count} errors"
    )
    print(f"{SCOLOR}[Shard-{shard_id:05d}] Total processing time {tt-t0:.2f}s{ECOLOR}")


class ProcessingError(Enum):
    SUCCESS = auto()
    NSFW_SCORE_HIGH = auto()
    NSFW_CHECK_ERROR = auto()
    OLLAMA_TIMEOUT = auto()
    OLLAMA_SERVER_ERROR = auto()
    OLLAMA_OTHER_ERROR = auto()
    UNEXPECTED_ERROR = auto()


def process_image_and_caption(image, caption):
    """Process image for NSFW content and generate enhanced caption if safe."""
    buffered = BytesIO()

    # Convert image to RGB format and resize
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")

    # Resize image maintaining aspect ratio
    w, h = image.size
    scale = min(1024 / min(w, h), 1.0)  # Only scale down, never up
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Save image to buffer for NSFW check and caption generation
    image.save(buffered, format="JPEG")

    # Check NSFW content
    try:
        nsfw_score = predict_image(buffered)
        if nsfw_score > 0.5:
            return "", ProcessingError.NSFW_SCORE_HIGH
    except Exception as e:
        print(f"\nError in NSFW check: {str(e)}")
        return "", ProcessingError.NSFW_CHECK_ERROR
    img_bytes = buffered.getvalue()

    # Prepare message for caption generation
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": caption,
            "images": [img_bytes],
        },
    ]

    try:
        response = chat(model="gemma3:4b", messages=messages, keep_alive="30s")
        result = (
            caption
            if response["done_reason"] == "stop"
            else response["message"]["content"]
        )
        return result, ProcessingError.SUCCESS
    except ollama.ResponseError as e:
        if e.status_code in [408, 504, 524]:
            print(f"\nTimeout error (status {e.status_code}), sleeping 5s")
            time.sleep(5)
            return "", ProcessingError.OLLAMA_TIMEOUT
        elif e.status_code >= 500:
            print(f"\nServer error (status {e.status_code}), sleeping 2s")
            time.sleep(2)
            return "", ProcessingError.OLLAMA_SERVER_ERROR
        else:
            print(f"\nUnexpected error: {str(e)}")
            return "", ProcessingError.OLLAMA_OTHER_ERROR
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return "", ProcessingError.UNEXPECTED_ERROR


def save_sample_with_captions(image, original_caption, enhanced_caption, output_path):
    """Save image and captions as a single figure."""
    plt.rcParams.update({"figure.max_open_warning": 0})
    plt.figure(figsize=(12, 10))

    # 确保图片格式为RGB
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")

    # Convert PIL Image to numpy array for matplotlib
    img_array = np.array(image)
    plt.imshow(img_array)
    plt.axis("off")

    plt.title(
        f"Original: {original_caption}\nEnhanced: {enhanced_caption}",
        wrap=True,
        fontsize=10,
        pad=30,
    )

    plt.subplots_adjust(top=0.85, bottom=0.05)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


SYSTEM_PROMPT = """
You are an expert image captioner who is good at precisely and concisely describe the factual details in an image based on a caption hint. 
Here are the guidelines to generate image description:
- Refine users' caption hint and make it extremely detailed and descriptive, meanwhile stick to the factual details in the image as your first priority. 
- For particularly long users' captions (>50 words), they can be outputted directly without refining. Image descriptions must be between 8-512 words. Extra words will be ignored.
- If the user's prompt requires rendering text, enclose the text with single quotation marks and prefix it with "the text".
- You will only ever output a single image description sentence per user request.
"""


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Process dataset with specified templates"
    )
    parser.add_argument(
        "--templates",
        type=str,
        default="1.0",
        help='Comma-separated template values (e.g., "1.0,0.8,0.6")',
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of worker processes in the pool"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    template_values = [t.strip() for t in args.templates.split(",")]
    base_dir = "/workspace/Infinity/data/processed_dataset"
    output_dir = "/workspace/Infinity/data/enhanced_dataset"
    os.makedirs(output_dir, exist_ok=True)

    parquets = []
    for template_dir in os.listdir(base_dir):
        if not template_dir.startswith("template_"):
            continue
        template_val = template_dir.split("_")[-1]
        if template_val not in template_values:
            continue

        template_path = osp.join(base_dir, template_dir)
        for shard_file in os.listdir(template_path):
            if not shard_file.endswith(".parquet"):
                continue
            parquet_path = osp.join(template_path, shard_file)
            shard_id = int(osp.basename(parquet_path).split("-")[1].split(".")[0])
            parquets.append((template_val, parquet_path, shard_id))

    # Sort by template value first, then shard_id
    parquets.sort(key=lambda x: (x[0], x[2]))
    # Remove shard_id from tuples before passing to pool
    parquets = [(p[0], p[1]) for p in parquets]

    n_parquets = len(parquets)
    parquet_round = args.workers * 3 # n_parquets

    for i in range(0, n_parquets, parquet_round):
        start, end = i, min(n_parquets, i + parquet_round)
        print(f"Working on parquet {start}-{end}, total {n_parquets}")

        with multiprocessing.Pool(processes=args.workers) as pool:
            results = pool.starmap(
                process_parquet,
                [(p[1], p[0], output_dir, args.verbose) for p in parquets[start:end]],
            )

if __name__ == "__main__":
    main()
