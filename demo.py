from __future__ import print_function

from os.path import exists

import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
from torchvision.transforms import ToTensor

from utils import measure_time

def run(settings):
    # Load model from file
    model_path = f"{settings.name}.pth"
    model_available = exists(model_path)
    if model_available:
        model = torch.load(model_path)
        # Training settings
        input_image = settings.input_path
        image = Image.open(input_image).convert('YCbCr')
        y, cb, cr = image.split()
        image_to_tensor = ToTensor()
        input = image_to_tensor(y).view(1, -1, y.size[1], y.size[0])
        if torch.cuda.is_available() and settings.cuda:
            model = model.cuda()
            input = image_to_tensor(y).view(1, -1, y.size[1], y.size[0]).cuda()
        out = model(input)
        out = out.cpu()
        out_image_y = out[0].detach().numpy()
        out_image_y *= 255.0
        out_image_y = out_image_y.clip(0, 255)
        out_image_y = Image.fromarray(np.uint8(out_image_y[0]), mode = 'L')

        out_image_cb = cb.resize(out_image_y.size, Resampling.LANCZOS)
        out_image_cr = cr.resize(out_image_y.size, Resampling.LANCZOS)
        out_image = Image.merge('YCbCr', [out_image_y, out_image_cb, out_image_cr]).convert('RGB')

        out_image.save(f"{settings.name}.png")
        out_image.show(f"{settings.name}.png")
    else:
        print(f'{model_path} does not exist')
