from __future__ import print_function
import torch
from PIL import Image
from PIL.Image import Resampling
from torchvision.transforms import ToTensor
from torchvision import transforms

# from torch2trt import torch2trt

import numpy as np

from settings import dictionary

# Load model from file
modelPath = (dictionary['model_path'] + f"{dictionary['upscale_factor']}x_{dictionary['model']}.pth")
model = torch.load(modelPath)

# Training settings
inputImage = dictionary['input_path']
image = Image.open(inputImage).convert('YCbCr')
y, cb, cr = image.split()
imageToTensor = ToTensor()
input = imageToTensor(y).view(1, -1, y.size[1], y.size[0])
if torch.cuda.is_available() and dictionary['cuda']:
    model = model.cuda()
    input = imageToTensor(y).view(1, -1, y.size[1], y.size[0]).cuda()

def run():
    out = model(input)
    out = out.cpu()
    outImageY = out[0].detach().numpy()
    outImageY *= 255.0
    outImageY = outImageY.clip(0, 255)
    outImageY = Image.fromarray(np.uint8(outImageY[0]), mode='L')

    outImageCB = cb.resize(outImageY.size, Resampling.LANCZOS)
    outImageCR = cr.resize(outImageY.size, Resampling.LANCZOS)
    outImage = Image.merge('YCbCr', [outImageY, outImageCB, outImageCR]).convert('RGB')

    outImage.save(dictionary['output_path']+f"{dictionary['upscale_factor']}x_{dictionary['model']}_output.png")
    outImage.show("Upscaled Image")
