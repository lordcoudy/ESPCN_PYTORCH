from __future__ import print_function

import os
from os.path import exists, isdir, isfile
from typing import Optional

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
def model_seq(settings, model: torch.nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    """Run model inference with inference_mode optimization."""
    with torch.inference_mode():
        return model(input_tensor)


def process_image(
    input_image: str, 
    settings, 
    model: torch.nn.Module, 
    output_path: str
) -> None:
    """Process a single image through the super-resolution model.
    
    Args:
        input_image: Path to input image
        settings: Settings object
        model: Super-resolution model
        output_path: Directory to save output
    """
    image = Image.open(input_image).convert('YCbCr')
    y, cb, cr = image.split()
    
    # Prepare input tensor
    input_tensor = ToTensor()(y).unsqueeze(0)
    input_tensor = input_tensor.to(settings.device, non_blocking=True)
    
    if settings.channels_last:
        input_tensor = input_tensor.to(memory_format=torch.channels_last)
    
    # Super-resolve
    output = model_seq(settings, model, input_tensor)
    output = output.cpu().squeeze(0).clamp(0, 1)
    
    # Reconstruct color image
    out_image_y = ToPILImage()(output)
    out_image_cb = cb.resize(out_image_y.size, Resampling.LANCZOS)
    out_image_cr = cr.resize(out_image_y.size, Resampling.LANCZOS)
    out_image = Image.merge('YCbCr', [out_image_y, out_image_cb, out_image_cr]).convert('RGB')
    
    # Save result
    output_filename = os.path.splitext(os.path.basename(input_image))[0]
    output_filepath = os.path.join(output_path, f"{output_filename}.png")
    out_image.save(output_filepath)
    
    if settings.show_result:
        out_image.show(f"{output_filename}.png")
    
    logger.info(f"Image saved to {output_filepath}")


def load_model(settings) -> Optional[torch.nn.Module]:
    """Load model from checkpoint with support for both old and new formats.
    
    Args:
        settings: Settings object with model paths
    
    Returns:
        Loaded model or None if not found
    """
    if settings.preload and exists(settings.preload_path):
        model_path = settings.preload_path
    else:
        model_path = f"{settings.name}_ckp{settings.epoch}.pth"
    
    if not exists(model_path):
        logger.error(f'{model_path} does not exist')
        return None
    
    # Try loading - handle both old (full model) and new (state_dict) formats
    checkpoint = torch.load(model_path, weights_only=False, map_location=settings.device)
    
    if isinstance(checkpoint, dict):
        # New format: state dict
        model = settings.create_model()
        model.load_state_dict(checkpoint)
    else:
        # Old format: full model
        model = checkpoint
    
    model = model.to(settings.device)
    
    if settings.channels_last:
        model = model.to(memory_format=torch.channels_last)
    
    model.eval()
    return model

def run(settings) -> None:
    """Run super-resolution demo on input images.
    
    Args:
        settings: Settings object with paths and configuration
    """
    model = load_model(settings)
    if model is None:
        return
    
    output_path = os.path.join(settings.output_path, settings.name)
    os.makedirs(output_path, exist_ok=True)
    
    # Determine input source
    if isfile(settings.input_path):
        # Single image
        logger.info("Demo progress 1/1")
        process_image(settings.input_path, settings, model, output_path)
    else:
        # Directory of images
        if isdir(settings.input_path):
            input_dir = settings.input_path
        else:
            input_dir = "./dataset/BSDS500/images/test/"
        
        # Get list of image files
        image_files = [
            f for f in os.listdir(input_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        max_images = min(settings.cycles, len(image_files))
        
        # Setup progress bar
        bar = None
        if settings.show_progress_bar:
            bar = progress.bar.IncrementalBar(
                max=max_images,
                suffix='[%(percent).3f%%] - [%(elapsed).2fs>%(eta).2fs - %(avg).2fs / it]'
            )
            bar.start()
        
        # Process images
        for idx, filename in enumerate(image_files[:max_images]):
            if bar:
                bar.bar_prefix = f'Processing {filename} [{idx+1}/{max_images}]: '
                bar.next()
            
            input_image = os.path.join(input_dir, filename)
            logger.info(f"Demo progress {idx+1}/{max_images}")
            process_image(input_image, settings, model, output_path)
        
        if bar:
            bar.finish()
