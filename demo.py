from __future__ import print_function

import os
from os.path import exists, isdir, isfile
import progress.bar

import torch
from PIL import Image
from PIL.Image import Resampling
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import ToPILImage

from custom_logger import get_logger
from utils import measure_time

logger = get_logger('demo')

@measure_time
def model_seq(settings, model, input):
    # Use inference_mode for fastest inference
    with torch.inference_mode():
        out = model(input)
    return out

def process_image(input_image, settings, model, output_path):
    image = Image.open(input_image).convert('YCbCr')
    y, cb, cr = image.split()
    image_to_tensor = ToTensor()
    input = image_to_tensor(y).unsqueeze(0)
    input = input.to(settings.device, non_blocking=True)
    # Apply channels_last memory format if enabled
    if settings.channels_last:
        input = input.to(memory_format=torch.channels_last)
    out = model_seq(settings, model, input)
    out = out.cpu().squeeze(0).clamp(0, 1)
    out_image_y = ToPILImage()(out)
    out_image_cb = cb.resize(out_image_y.size, Resampling.LANCZOS)
    out_image_cr = cr.resize(out_image_y.size, Resampling.LANCZOS)
    out_image = Image.merge('YCbCr', [out_image_y, out_image_cb, out_image_cr]).convert('RGB')
    out_image.save(f"{os.path.join(output_path, os.path.splitext(os.path.basename(input_image))[0])}.png")
    if settings.show_result: out_image.show(f"{os.path.splitext(os.path.basename(input_image))[0]}.png")
    logger.info(f"Image saved to {os.path.join(output_path, os.path.splitext(os.path.basename(input_image))[0])}.png")

def run(settings):
    # Load model from file
    if settings.preload and exists(settings.preload_path):
        model_path = settings.preload_path
    else:
        model_path = f"{settings.name}_ckp{settings.epoch}.pth"
    model_available = exists(model_path)
    if model_available:
        model = torch.load(model_path, weights_only = False, map_location=settings.device)
        model = model.to(settings.device)
        # Apply channels_last memory format if enabled
        if settings.channels_last:
            model = model.to(memory_format=torch.channels_last)
        model.eval()
        output_path = f'{os.path.join(settings.output_path, settings.name)}/'
        os.makedirs(output_path, exist_ok = True)
        if not isfile(settings.input_path):
            if isdir(settings.input_path):
                dir = settings.input_path
            else:
                dir = ("./dataset/BSDS500/images/test/")
            it = 0
            if settings.show_progress_bar:
                bar = progress.bar.IncrementalBar(max = min(settings.cycles, len(os.listdir(dir))),
                                                  suffix = '[%(percent).3f%%] - [%(elapsed).2fs>%(eta).2fs - %(avg).2fs / it]')  # Update bar max
                bar.start()
            for file in os.listdir(dir):
                # Skip non-image files
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                it = it + 1
                if settings.show_progress_bar:
                    bar.bar_prefix = f'Processing image {os.path.basename(file)} [{it}/{min(settings.cycles, len(os.listdir(dir)))}]: '
                    bar.next()
                input_image = os.path.join(dir, file)
                process_image(input_image, settings, model, output_path)
                if it == settings.cycles:
                    break
            if settings.show_progress_bar:
                bar.finish()
        else:
            input_image = settings.input_path
            process_image(input_image, settings, model, output_path)
    else:
        logger.error(f'{model_path} does not exist')
