import gzip
import shutil
import tarfile
import os
import zipfile
import random
from os.path import join, exists, basename
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, InterpolationMode
from torchvision.transforms import functional as F
from six.moves import urllib
from torchvision.transforms.v2 import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation


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


def download_bsd300(dest="dataset"):
    # output_image_dir = join(dest, "BSDS300/images")
    output_image_dir = join(dest, "BSDS500/images")

    if not exists(output_image_dir) or len(os.listdir(output_image_dir)) == 0:
        os.makedirs(dest, exist_ok=True)
        # url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        # url = "https://web.archive.org/web/20160306133802/http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
        url = "https://drive.google.com/file/d/1fTBtqwfUVhelz-kE1PkJ0cyeje7dB_zV/view?usp=sharing"
        file_path = join(dest, "BSDS500.tar.gz")

        print("Downloading dataset:", url)
        urllib.request.urlretrieve(url, file_path)

        print("Extracting dataset:", file_path)
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=dest)  # Safer extraction

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
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir,
                             transform=transform(img_size, upscale_factor))

def get_validation_set(upscale_factor, img_size=256):
    root_dir = download_bsd300()
    val_dir = join(root_dir, "val")

    return DatasetFromFolder(val_dir,
                             transform=transform(img_size, upscale_factor))

def get_test_set(upscale_factor, img_size=256):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")

    return DatasetFromFolder(test_dir,
                             transform=transform(img_size, upscale_factor))