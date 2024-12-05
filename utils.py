import time

import torch
from torch.nn import functional as f
from torchvision import models
from torchvision.models import VGG16_Weights
from torch.nn.utils import prune
from torch.nn.utils.prune import L1Unstructured
from PIL import Image
from torchvision.transforms import ToTensor

def measure_time(func):
    out = open(f'times\\time_{func.__name__}.txt', 'w').close()
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print(func.__name__, end - start, file = out)
        return result

    return wrap

def mixed_precision(settings, data, target):
    device_type = 'cpu'
    if settings.cuda:
        device_type = 'cuda'
    with torch.amp.autocast(device_type, enabled = (settings.scaler is not None)):
        output = settings.model(data)
        mse = f.mse_loss(output, target)
        percLoss = perceptual_loss(settings.device, output, target)
        loss = mse + 0.1 * percLoss  # Combining losses
        # loss = criterion(output, target)

    if settings.scaler:
        settings.scaler.scale(loss).backward()
        settings.scaler.step(settings.optimizer)
        settings.scaler.update()
    else:
        loss.backward()
        settings.optimizer.step()

    return loss

def perceptual_loss(device, pred, target):
    # Perceptual Loss with VGG16
    vgg = models.vgg16(weights = VGG16_Weights.DEFAULT).features[:16].eval().to(device)
    for param in vgg.parameters():
        param.requires_grad = False
    pred_vgg = vgg(pred.repeat(1, 3, 1, 1))
    target_vgg = vgg(target.repeat(1, 3, 1, 1))
    return f.mse_loss(pred_vgg, target_vgg)

@measure_time
def prune_model(model, amount = 0.2):
    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.conv3, 'weight'),
        (model.conv4, 'weight')
    )
    prune.global_unstructured(
            parameters = parameters_to_prune,
            pruning_method = L1Unstructured,
            amount = amount,
    )

@measure_time
def checkpoint(settings, epoch):
    model_path = (settings.model_path + f"{settings.upscale_factor}x_epoch_{epoch}_optimized-{settings.optimized}_cuda-{settings.cuda}_tuning-{settings.tuning}_pruning-{settings.pruning}_mp-{settings.mixed_precision}_scheduler-{settings.scheduler_enabled}_.pth")
    torch.save(settings.model, model_path)
    print("===> Checkpoint saved to {} >===".format(model_path))

@measure_time
def export_model(settings, epoch):
    settings.model.eval()
    img = Image.open(settings.input_path).convert('YCbCr')
    y, cb, cr = img.split()
    img_to_tensor = ToTensor()
    input_tensor = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])
    if settings.cuda and torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    traced_script = torch.jit.trace(settings.model, input_tensor)
    traced_model_path = (settings.model_path + f"{settings.upscale_factor}x_traced_epoch_{epoch}_optimized-{settings.optimized}_cuda-{settings.cuda}_tuning-{settings.tuning}_pruning-{settings.pruning}_mp-{settings.mixed_precision}_scheduler-{settings.scheduler_enabled}_.pth")

    traced_script.save(traced_model_path)
    print("===> Model exported >===")
    print("===> Traced model saved to {}".format(traced_model_path))