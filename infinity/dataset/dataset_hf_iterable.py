import os
from typing import Optional, Union, Dict, Any
from PIL import Image as PImage
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision.transforms.functional import to_tensor
import numpy as np
from datasets import load_dataset, Dataset, IterableDataset as HFIterableDataset
import requests
from io import BytesIO

def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im.add(im).add_(-1)  # normalize to [-1, 1]

class HFIterableDatasetWrapper(IterableDataset):
    def __init__(
        self,
        dataset_name: str,
        image_column: str = "image",
        caption_column: str = "text",
        image_size: int = 1024,
        split: str = "train",
        streaming: bool = True,
        cache_dir: Optional[str] = None,
        **dataset_kwargs: Dict[str, Any]
    ):
        """
        Args:
            dataset_name: Name of the dataset on HuggingFace Hub (e.g., "lambdalabs/pokemon-blip-captions")
            image_column: Name of the column containing image data
            caption_column: Name of the column containing caption data
            image_size: Target size for both height and width
            split: Which split of the dataset to use
            streaming: Whether to stream the dataset or load it entirely into memory
            cache_dir: Directory to cache the downloaded data
            dataset_kwargs: Additional arguments to pass to load_dataset
        """
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            cache_dir=cache_dir,
            **dataset_kwargs
        )
        self.image_column = image_column
        self.caption_column = caption_column
        self.image_size = image_size

    def load_image(self, image_data: Union[str, PImage.Image, Dict]) -> PImage.Image:
        """Handle different types of image data"""
        if isinstance(image_data, PImage.Image):
            return image_data
        elif isinstance(image_data, dict) and "path" in image_data:
            # Handle HF datasets that return image dict
            if not os.path.exists(image_data["path"]):
                raise FileNotFoundError(f"Image file not found: {image_data['path']}")
            return PImage.open(image_data["path"])
        elif isinstance(image_data, str):
            if image_data.startswith(('http://', 'https://')):
                # Handle URLs with timeout and error handling
                try:
                    response = requests.get(image_data, timeout=10)
                    response.raise_for_status()  # Raises an HTTPError for bad responses
                    return PImage.open(BytesIO(response.content))
                except requests.Timeout:
                    raise ValueError(f"Timeout while fetching image from URL: {image_data}")
                except requests.RequestException as e:
                    raise ValueError(f"Failed to fetch image from URL: {image_data}. Error: {str(e)}")
            else:
                # Handle local file paths
                if not os.path.exists(image_data):
                    raise FileNotFoundError(f"Image file not found: {image_data}")
                return PImage.open(image_data)
        else:
            raise ValueError(f"Unsupported image data type: {type(image_data)}")

    def process_image(self, image: PImage.Image) -> torch.Tensor:
        """Process the image to the desired size and format"""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return transform(image, self.image_size, self.image_size)

    def __iter__(self):
        for item in self.dataset:
            try:
                # Get image and caption from the dataset
                image_data = item[self.image_column]
                caption = item[self.caption_column]

                # Load and process image
                image = self.load_image(image_data)
                processed_image = self.process_image(image)

                yield processed_image, caption
            except Exception as e:
                print(f"Error processing item: {e}")
                continue

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HuggingFace Dataset Loader')
    parser.add_argument('--dataset_name', type=str, 
                       default="kakaobrain/coyo-700m",
                       help='Name of the dataset on HuggingFace Hub')
    parser.add_argument('--image_column', type=str, 
                       default="image",
                       help='Name of the column containing image data')
    parser.add_argument('--caption_column', type=str, 
                       default="text",
                       help='Name of the column containing caption data')
    parser.add_argument('--image_size', type=int, 
                       default=512,
                       help='Target size for both height and width')
    
    args = parser.parse_args()
    
    # Example usage with command line arguments
    dataset = HFIterableDatasetWrapper(
        dataset_name=args.dataset_name,
        image_column=args.image_column,
        caption_column=args.caption_column,
        image_size=args.image_size,
        streaming=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
    )

    # Test the dataloader
    for i, (images, captions) in enumerate(dataloader):
        print(f"Batch {i}")
        print(f"Images shape: {images.shape}")
        print(f"Sample caption: {captions[0]}")
        if i >= 2:  # Just test a few batches
            break