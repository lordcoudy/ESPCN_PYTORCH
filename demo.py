from __future__ import print_function
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision import transforms

# from torch2trt import torch2trt

import numpy as np

# Training settings
input_image = 'E:/SAVVA/STUDY/CUDA/ESPCN-PY/pythonProject/dataset/BSDS300/images/test/45096.jpg'
img = Image.open(input_image).convert('YCbCr')
y, cb, cr = img.split()
img_to_tensor = ToTensor()
input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0]).cuda()

out = model(input)
out = out.cpu()
out_img_y = out[0].detach().numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

out_img.save("output.jpg")
print('output image saved')
