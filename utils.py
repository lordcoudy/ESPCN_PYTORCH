import json
import logging
import time
from logging import Formatter

import torch
from PIL import Image
from torch.nn.utils import prune
from torch.nn.utils.prune import L1Unstructured
from torchvision.transforms import ToTensor


class JsonFormatter(Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger_name": record.name,
            "filename": record.filename,
            "lineno": record.lineno,
            "funcname": record.funcName,
            "message": record.getMessage(), # Important to use getMessage() to handle exceptions
            # Add other relevant fields here as needed (e.g., process ID, thread ID, etc.)
        }
        return json.dumps(log_record)

def measure_time(func):
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print(func.__name__, end - start, end = "\n", file = open(f'times\\time_{func.__name__}.txt', 'a+'))
        return result

    return wrap

# def log(message):
#     # Configure logging
#     json_formatter = JsonFormatter()
#     handler = logging.FileHandler(filename="./ESPyCN.log", mode="a+")  # Or FileHandler for file output
#     handler.setFormatter(json_formatter)
#     logger = logging.getLogger(__name__)
#     logger.addHandler(handler)
#     logger.log(logging.DEBUG, message)


def calculateLoss(settings, data, target):
    settings.optimizer.zero_grad()
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
