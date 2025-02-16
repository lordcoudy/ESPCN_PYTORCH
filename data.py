import tarfile
import os
from os.path import join, exists, basename
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, InterpolationMode
from six.moves import urllib


def is_image_file(filename):
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, in_transform=None, tgt_transform=None, rotation=True):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]
        self.image_items = []

        if rotation:
            for filename in self.image_filenames:
                for angle in [0, 90, 180, 270]:
                    self.image_items.append((filename, angle))
        else:
            for filename in self.image_filenames:
                self.image_items.append((filename, 0)) # angle 0 for no rotation

        self.input_transform = in_transform
        self.target_transform = tgt_transform

    def __getitem__(self, index):
        filename, angle = self.image_items[index]
        in_img = load_img(filename)
        target = in_img.copy()

        if angle != 0:
            in_img = in_img.rotate(angle)
            target = target.rotate(angle)

        if self.input_transform:
            in_img = self.input_transform(in_img)
        if self.target_transform:
            target = self.target_transform(target)

        return in_img, target

    def __len__(self):
        return len(self.image_items)


def download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir) or len(os.listdir(output_image_dir)) == 0:
        os.makedirs(dest, exist_ok=True)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        file_path = join(dest, basename(url))

        print("Downloading dataset:", url)
        urllib.request.urlretrieve(url, file_path)

        print("Extracting dataset...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=dest)  # Safer extraction

        os.remove(file_path)  # Cleanup

    return output_image_dir


def calculate_valid_crop_size(img_size, upscale_factor):
    return img_size - (img_size % upscale_factor)


def input_transform(img_size, upscale_factor):
    crop_size = calculate_valid_crop_size(img_size, upscale_factor)
    return Compose([
        CenterCrop(crop_size),  # Adaptive cropping
        Resize(crop_size // upscale_factor, interpolation=InterpolationMode.LANCZOS),  # Better downscaling
        ToTensor(),
    ])


def target_transform(img_size):
    crop_size = calculate_valid_crop_size(img_size, 1)
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor, img_size=256):
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir,
                             in_transform=input_transform(img_size, upscale_factor),
                             tgt_transform=target_transform(img_size))


def get_test_set(upscale_factor, img_size=256):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")

    return DatasetFromFolder(test_dir,
                             in_transform=input_transform(img_size, upscale_factor),
                             tgt_transform=target_transform(img_size))