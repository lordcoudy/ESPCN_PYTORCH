import tarfile
from os import listdir, makedirs, remove
from os.path import basename, exists, join

import torch.utils.data as data
from PIL import Image
from six.moves import urllib
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Resize, ToTensor)

from utils import measure_time


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, in_transform=None, tgt_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = in_transform
        self.target_transform = tgt_transform

    def __getitem__(self, index):
        in_img = load_img(self.image_filenames[index])
        target = in_img.copy()
        if self.input_transform:
            in_img = self.input_transform(in_img)
        if self.target_transform:
            target = self.target_transform(target)

        return in_img, target

    def __len__(self):
        return len(self.image_filenames)

def download_bsd300(dest= "dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(size= crop_size // upscale_factor, interpolation=InterpolationMode.LANCZOS),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])

def get_training_set(upscale_factor):
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             in_transform =input_transform(crop_size, upscale_factor),
                             tgt_transform =target_transform(crop_size))

def get_test_set(upscale_factor):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             in_transform =input_transform(crop_size, upscale_factor),
                             tgt_transform =target_transform(crop_size))