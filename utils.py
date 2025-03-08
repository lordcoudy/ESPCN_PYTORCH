import os
import time

import torch
from PIL import Image
from torch.nn.utils import prune
from torchvision.transforms import ToTensor


def measure_time(func):
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        os.makedirs('times', exist_ok=True)
        with open(f'times/time_{func.__name__}.txt', 'a+') as f:
            print(func.__name__, end - start, file=f)
        return result

    return wrap

def calculateLoss(settings, data, target):
    with torch.amp.autocast(device_type = "cuda" if settings.cuda else "cpu", enabled =settings.mixed_precision):
        output = settings.model(data)
        loss = settings.criterion(output, target)
    return loss

def backPropagate(settings, loss):
    if settings.scaler:
        settings.scaler.scale(loss).backward()
        settings.scaler.step(settings.optimizer)
        settings.scaler.update()
    else:
        loss.backward()
        settings.optimizer.step()
    settings.optimizer.zero_grad()

@measure_time
def prune_model(model, amount = 0.2):
    parameters_to_prune = [
        (module, 'weight') for module in model.modules() if isinstance(module, torch.nn.Conv2d)
    ]
    if not parameters_to_prune:
        print("No Conv2D layers found for pruning.")
        return

    prune.global_unstructured(
        parameters=parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

@measure_time
def checkpoint(settings, model, epoch):
    model_path = f"{settings.name}_ep[{epoch}].pth"
    torch.save(model, model_path)
    print("===> Checkpoint saved to {} >===".format(model_path))

@measure_time
def export_model(settings, model, epoch):
    model.eval()
    img = Image.open(settings.input_path).convert('YCbCr')
    y, cb, cr = img.split()
    img_to_tensor = ToTensor()
    input_tensor = img_to_tensor(y).unsqueeze(0).to(settings.device)
    traced_script = torch.jit.trace(model, input_tensor)
    traced_model_path = f"{settings.name}_ep[{epoch}]_TRACED.pth"
    traced_script.save(traced_model_path)
    print("===> Traced model saved to {}".format(traced_model_path))
