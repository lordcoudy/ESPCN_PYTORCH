from __future__ import print_function

from os.path import exists

import torch
from PIL import Image
from PIL.Image import Resampling
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import ToPILImage

from custom_logger import get_logger

logger = get_logger('demo')

def run(settings):
    # Load model from file
    model_path = f"{settings.name}_ckp{settings.epoch}.pth"
    model_available = exists(model_path)
    if model_available:
        model = torch.load(model_path, weights_only = False, map_location=settings.device)
        # Training settings
        input_image = settings.input_path
        image = Image.open(input_image).convert('YCbCr')
        y, cb, cr = image.split()
        image_to_tensor = ToTensor()
        input = image_to_tensor(y).unsqueeze(0)
        model = model.to(settings.device)
        input = input.to(settings.device)
        model.eval()
        out = model(input)
        out = out.cpu().squeeze(0).clamp(0, 1)
        out_image_y = ToPILImage()(out)

        out_image_cb = cb.resize(out_image_y.size, Resampling.LANCZOS)
        out_image_cr = cr.resize(out_image_y.size, Resampling.LANCZOS)
        out_image = Image.merge('YCbCr', [out_image_y, out_image_cb, out_image_cr]).convert('RGB')

        out_image.save(f"{settings.name}.png")
        out_image.show(f"{settings.name}.png")
        logger.info(f"Image saved to {settings.name}.png")
    else:
        logger.error(f'{model_path} does not exist')
