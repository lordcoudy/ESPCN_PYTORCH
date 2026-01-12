"""Data loading utilities for ESPCN super-resolution training."""

import os
import random
import zipfile
from os.path import exists, join
from typing import Callable, Optional, Tuple

import torch
import torch.utils.data as data
from PIL import Image
from six.moves import urllib
from torchvision.transforms import InterpolationMode, ToTensor
from torchvision.transforms import functional as F


def worker_init_fn(worker_id: int) -> None:
    """Initialize random seed for each DataLoader worker for reproducibility."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)


def is_image_file(filename: str) -> bool:
    """Check if file is a supported image format."""
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))


def load_img(filepath: str) -> Image.Image:
    """Load image and extract Y (luminance) channel from YCbCr color space."""
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    """Dataset for loading images from a folder with optional transforms.
    
    Args:
        image_dir: Directory containing images
        transform: Optional transform to apply to images
    """
    def __init__(self, image_dir: str, transform: Optional[Callable] = None):
        super().__init__()
        self.image_items = sorted([
            join(image_dir, x) for x in os.listdir(image_dir) 
            if is_image_file(x)
        ])
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = self.image_items[index]
        in_img = load_img(filename)
        target = in_img.copy()

        if self.transform:
            in_img, target = self.transform(in_img, target)

        return in_img, target

    def __len__(self) -> int:
        return len(self.image_items)


class CachedDataset(data.Dataset):
    """Wrapper that pre-loads dataset into RAM for zero I/O overhead.
    
    Best for small datasets (<10GB) where I/O is the bottleneck.
    
    Args:
        dataset: Base dataset to cache
        cache_size: Maximum number of items to cache (None = all)
    """
    def __init__(self, dataset: data.Dataset, cache_size: Optional[int] = None):
        super().__init__()
        self.dataset = dataset
        self.cache = {}
        
        # Pre-load entire dataset or subset
        total = len(dataset) if cache_size is None else min(cache_size, len(dataset))
        from custom_logger import get_logger
        logger = get_logger('data')
        logger.info(f"Pre-caching {total} samples into RAM...")
        
        for i in range(total):
            self.cache[i] = dataset[i]
        
        logger.info(f"Cached {len(self.cache)} samples")
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if index in self.cache:
            return self.cache[index]
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)


def download_bsd500(dest: str = "dataset") -> str:
    """Download and extract BSD500 dataset if not present.
    
    Args:
        dest: Destination directory for dataset
    
    Returns:
        Path to extracted images directory
    """
    output_image_dir = join(dest, "BSDS500/images")
    if not exists(output_image_dir) or len(os.listdir(output_image_dir)) == 0:
        os.makedirs(dest, exist_ok=True)
        url = "https://www.kaggle.com/api/v1/datasets/download/balraj98/berkeley-segmentation-dataset-500-bsds500"
        file_path = join(dest, "berkeley-segmentation-dataset-500-bsds500.zip")
        from custom_logger import get_logger
        logger = get_logger('data')
        logger.info(f"Downloading dataset: {url}")
        urllib.request.urlretrieve(url, file_path)
        if not zipfile.is_zipfile(file_path):
            logger.error("Zipfile doesn't appear to be a zip file")
        logger.info(f"Extracting dataset: {file_path}")
        dir_dest = join(dest, "BSDS500")
        if not exists(dir_dest):
            os.makedirs(dir_dest, exist_ok=True)
        with zipfile.ZipFile(file_path, mode='r') as zip_ref:
            zip_ref.extractall(path=dir_dest)
            if not exists(output_image_dir):
                logger.error("Dataset extraction failed")
        os.remove(file_path)  # Cleanup

    return output_image_dir


def calculate_valid_crop_size(img_size: int, upscale_factor: int) -> int:
    """Calculate crop size that's divisible by upscale factor."""
    return img_size - (img_size % upscale_factor)


class RandomTransform:
    """Transform for training with random augmentations.
    
    Applies random flips, rotation, and creates LR-HR pairs.
    
    Args:
        crop_size: Size to crop images to
        upscale_factor: Factor by which LR is downscaled from HR
        augment: Whether to apply random augmentations (default: True)
    """
    def __init__(self, crop_size: int, upscale_factor: int, augment: bool = True):
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.augment = augment
        self.lr_size = crop_size // upscale_factor

    def __call__(
        self, 
        in_img: Image.Image, 
        target: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                in_img = F.hflip(in_img)
                target = F.hflip(target)

            # Random vertical flip
            if random.random() > 0.5:
                in_img = F.vflip(in_img)
                target = F.vflip(target)

            # Random rotation (small angles to preserve quality)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                in_img = F.rotate(in_img, angle, interpolation=InterpolationMode.BILINEAR)
                target = F.rotate(target, angle, interpolation=InterpolationMode.BILINEAR)

        # Random crop instead of center crop for more variety during training
        if self.augment:
            # Get image size
            w, h = in_img.size
            if w >= self.crop_size and h >= self.crop_size:
                i = random.randint(0, h - self.crop_size)
                j = random.randint(0, w - self.crop_size)
                in_img = F.crop(in_img, i, j, self.crop_size, self.crop_size)
                target = F.crop(target, i, j, self.crop_size, self.crop_size)
            else:
                in_img = F.center_crop(in_img, self.crop_size)
                target = F.center_crop(target, self.crop_size)
        else:
            in_img = F.center_crop(in_img, self.crop_size)
            target = F.center_crop(target, self.crop_size)

        # Create LR-HR pair: downscale input, keep target at original size
        # Use BICUBIC for downscaling (more realistic degradation)
        in_img = F.resize(in_img, self.lr_size, interpolation=InterpolationMode.BICUBIC)
        target = F.resize(target, self.crop_size, interpolation=InterpolationMode.LANCZOS)

        # Convert to tensor
        in_img = ToTensor()(in_img)
        target = ToTensor()(target)

        return in_img, target


class DeterministicTransform(RandomTransform):
    """Transform for validation/testing without random augmentations."""
    
    def __init__(self, crop_size: int, upscale_factor: int):
        super().__init__(crop_size, upscale_factor, augment=False)


def transform(img_size: int, upscale_factor: int, augment: bool = True) -> RandomTransform:
    """Create transform with valid crop size."""
    crop_size = calculate_valid_crop_size(img_size, upscale_factor)
    if augment:
        return RandomTransform(crop_size, upscale_factor)
    return DeterministicTransform(crop_size, upscale_factor)


def get_training_set(upscale_factor: int, img_size: int = 256) -> DatasetFromFolder:
    """Get training dataset with augmentations."""
    root_dir = download_bsd500()
    train_dir = join(root_dir, "train")
    return DatasetFromFolder(train_dir, transform=transform(img_size, upscale_factor, augment=True))


def get_validation_set(upscale_factor: int, img_size: int = 256) -> DatasetFromFolder:
    """Get validation dataset without augmentations."""
    root_dir = download_bsd500()
    val_dir = join(root_dir, "val")
    return DatasetFromFolder(val_dir, transform=transform(img_size, upscale_factor, augment=False))


def get_test_set(upscale_factor: int, img_size: int = 256) -> DatasetFromFolder:
    """Get test dataset without augmentations."""
    root_dir = download_bsd500()
    test_dir = join(root_dir, "test")
    return DatasetFromFolder(test_dir, transform=transform(img_size, upscale_factor, augment=False))