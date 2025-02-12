from __future__ import print_function

from os.path import exists

import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import ToPILImage

from utils import measure_time

def run(settings):
    # Load model from file
    model_path = f"{settings.name}_ep[{settings.epoch}].pth"
    model_available = exists(model_path)
    if model_available:
        model = torch.load(model_path, weights_only = False)
        # Training settings
        input_image = settings.input_path
        image = Image.open(input_image).convert('YCbCr')
        y, cb, cr = image.split()
        image_to_tensor = ToTensor()
        input = image_to_tensor(y).unsqueeze(0)
        if torch.cuda.is_available() and settings.cuda:
            model = model.cuda()
            input = input.cuda()
        model.eval()
        with torch.no_grad():
            out = model(input)
        out = out.cpu().squeeze(0).clamp(0, 1)
        out_image_y = ToPILImage()(out)

        out_image_cb = cb.resize(out_image_y.size, Resampling.BICUBIC)
        out_image_cr = cr.resize(out_image_y.size, Resampling.BICUBIC)
        out_image = Image.merge('YCbCr', [out_image_y, out_image_cb, out_image_cr]).convert('RGB')

        out_image.save(f"{settings.name}.png")
        out_image.show(f"{settings.name}.png")
    else:
        print(f'{model_path} does not exist')
