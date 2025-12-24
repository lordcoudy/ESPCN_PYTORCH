import os
import random
import zipfile
from os.path import join, exists

import torch
import torch.utils.data as data
from PIL import Image
from six.moves import urllib
from torchvision.transforms import ToTensor, InterpolationMode
from torchvision.transforms import functional as F


def worker_init_fn(worker_id):
    """Initialize random seed for each DataLoader worker for reproducibility."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)


def is_image_file(filename):
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_items = [join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]
        self.transform = transform

    def __getitem__(self, index):
        filename = self.image_items[index]
        in_img = load_img(filename)
        target = in_img.copy()

        if self.transform:
            in_img, target = self.transform(in_img, target)

        return in_img, target

    def __len__(self):
        return len(self.image_items)


def download_bsd500(dest="dataset"):
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


def calculate_valid_crop_size(img_size, upscale_factor):
    return img_size - (img_size % upscale_factor)


class RandomTransform:
    def __init__(self, crop_size, upscale_factor):
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor

    def __call__(self, in_img, target):
        # Apply random horizontal flip
        if random.random() > 0.5:
            in_img = F.hflip(in_img)
            target = F.hflip(target)

        # Apply random vertical flip
        if random.random() > 0.5:
            in_img = F.vflip(in_img)
            target = F.vflip(target)

        # Apply random rotation
        angle = random.uniform(-10, 10)
        in_img = F.rotate(in_img, angle)
        target = F.rotate(target, angle)

        # Apply center crop and resize
        in_img = F.center_crop(in_img, self.crop_size)
        target = F.center_crop(target, self.crop_size)
        in_img = F.resize(in_img, self.crop_size // self.upscale_factor, interpolation=InterpolationMode.LANCZOS)
        target = F.resize(target, self.crop_size, interpolation=InterpolationMode.LANCZOS)

        # Convert to tensor
        in_img = ToTensor()(in_img)
        target = ToTensor()(target)

        return in_img, target

def transform(img_size, upscale_factor):
    crop_size = calculate_valid_crop_size(img_size, upscale_factor)
    return RandomTransform(crop_size, upscale_factor)


def get_training_set(upscale_factor, img_size=256):
    root_dir = download_bsd500()
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir,
                             transform=transform(img_size, upscale_factor))

def get_validation_set(upscale_factor, img_size=256):
    root_dir = download_bsd500()
    val_dir = join(root_dir, "val")

    return DatasetFromFolder(val_dir,
                             transform=transform(img_size, upscale_factor))

def get_test_set(upscale_factor, img_size=256):
    root_dir = download_bsd500()
    test_dir = join(root_dir, "test")

    return DatasetFromFolder(test_dir,
                             transform=transform(img_size, upscale_factor))