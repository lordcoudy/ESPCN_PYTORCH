import os
import time
from os.path import isdir

import torch
from PIL import Image
from torch.nn.utils import prune
from torchvision.transforms import ToTensor

from custom_logger import get_logger

logger = get_logger('utils')


def sync_device(device):
    """Synchronize device to ensure all operations are complete.
    Critical for accurate timing on async devices (CUDA/MPS)."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def empty_cache(device):
    """Clear device memory cache to prevent OOM errors."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()


def measure_time(func):
    def wrap(*args, **kwargs):
        start = time.time_ns()
        result = func(*args, **kwargs)
        end = time.time_ns()

        from settings import model_dir_i
        os.makedirs(model_dir_i(), exist_ok=True)
        times_dir = os.path.join(model_dir_i(), 'times')
        os.makedirs(times_dir, exist_ok=True)
        with open(f'{times_dir}/time_{func.__name__}.txt', 'a+') as f:
            print(func.__name__, f'{end - start} ns', file=f)
        return result

    return wrap


def calculateLoss(settings, data, target, model):
    # Determine device type for autocast (MPS doesn't fully support autocast, use CPU fallback)
    device_type = str(settings.device.type)
    autocast_device = device_type if device_type in ('cuda', 'cpu') else 'cpu'
    autocast_enabled = settings.mixed_precision and device_type == 'cuda'  # AMP autocast only fully supported on CUDA
    with torch.amp.autocast(device_type=autocast_device, enabled=autocast_enabled):
        output = model(data)
        loss = settings.criterion(output, target)
    return loss

def backPropagate(settings, loss, optimizer):
    # Check if scaler is enabled (scaler object is always truthy)
    if settings.scaler is not None and settings.scaler.is_enabled():
        settings.scaler.scale(loss).backward()
        settings.scaler.step(optimizer)
        settings.scaler.update()
    else:
        loss.backward()
        optimizer.step()

@measure_time
def prune_model(model, amount = 0.2):
    parameters_to_prune = [
        (module, 'weight') for module in model.modules() if isinstance(module, torch.nn.Conv2d)
    ]
    if not parameters_to_prune:
        logger.error("No Conv2D layers found for pruning.")
        return

    prune.global_unstructured(
        parameters=parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

@measure_time
def checkpoint(settings, model, epoch):
    os.makedirs(settings.model_dir, exist_ok = True)
    model_path = f"{os.path.join(settings.model_dir, settings.name)}_ckp{epoch}.pth"
    torch.save(model, model_path)
    logger.debug("Checkpoint saved to {}".format(model_path))

@measure_time
def export_model(settings, model, epoch):
    model.eval()
    if isdir(settings.input_path):
        input_path = f"./dataset/BSDS500/images/test/3063.jpg"
    else:
        input_path = settings.input_path
    img = Image.open(input_path).convert('YCbCr')
    y, cb, cr = img.split()
    img_to_tensor = ToTensor()
    input_tensor = img_to_tensor(y).unsqueeze(0).to(settings.device)
    # Apply channels_last memory format if enabled for consistent tracing
    if settings.channels_last:
        input_tensor = input_tensor.to(memory_format=torch.channels_last)
    traced_script = torch.jit.trace(model, input_tensor)
    os.makedirs(settings.model_dir, exist_ok = True)
    traced_model_path = f"{os.path.join(settings.model_dir, settings.name)}_TRACED_ckp{epoch}.pth"
    traced_script.save(traced_model_path)
    logger.debug("Traced model saved to {}".format(traced_model_path))

def get_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(f"Total parameters: {total_params}")
    logger.debug(f"Trainable parameters: {trainable_params}")

