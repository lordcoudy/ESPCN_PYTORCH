from __future__ import print_function
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision import transforms

# from torch2trt import torch2trt

import numpy as np

# Load model from file
model = torch.load('2x_espcn_epoch_50.pth')

# Training settings
inputImage = 'E:\\SAVVA\\STUDY\\CUDA\\ESPCN_PYTORCH\\dataset\\BSDS300\\images\\test\\3096.jpg'
image = Image.open(inputImage).convert('YCbCr')
y, cb, cr = image.split()
imageToTensor = ToTensor()
input = imageToTensor(y).view(1, -1, y.size[1], y.size[0])
if torch.cuda.is_available():
    model = model.cuda()
    input = imageToTensor(y).view(1, -1, y.size[1], y.size[0]).cuda()

if __name__ == '__main__':

    out = model(input)
    out = out.cpu()
    outImageY = out[0].detach().numpy()
    outImageY *= 255.0
    outImageY = outImageY.clip(0, 255)
    outImageY = Image.fromarray(np.uint8(outImageY[0]), mode='L')

    outImageCB = cb.resize(outImageY.size, Image.LANCZOS)
    outImageCR = cr.resize(outImageY.size, Image.LANCZOS)
    outImage = Image.merge('YCbCr', [outImageY, outImageCB, outImageCR]).convert('RGB')

    outImage.save("output.jpg")
    print('output image saved')
