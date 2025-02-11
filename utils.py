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

        print(func.__name__, end - start, end = "\n", file = open(f'times\\time_{func.__name__}.txt', 'a+'))
        return result

    return wrap

def calculateLoss(settings, data, target):
    device_type = 'cpu'
    if settings.cuda:
        device_type = 'cuda'
    with torch.amp.autocast(device_type, enabled =settings.mixed_precision):
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

@measure_time
def prune_model(model, amount = 0.2):
    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.conv3, 'weight'),
    )
    prune.global_unstructured(
            parameters = parameters_to_prune,
            pruning_method = prune.L1Unstructured,
            amount = amount,
    )

@measure_time
def checkpoint(settings, epoch):
    model_path = f"{settings.name}.pth"
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
    traced_model_path = f"{settings.name}_TRACED.pth"
    traced_script.save(traced_model_path)
    print("===> Traced model saved to {}".format(traced_model_path))
