import torch
from torch import nn, Tensor
import torch.nn.init as init

import math


class OptimizedESPCN(nn.Module):
    def __init__(self, upscaleFactor=2):
        super(OptimizedESPCN, self).__init__()

        self.fwd = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, (upscaleFactor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscaleFactor)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.fwd(x)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.orthogonal_(module.weight.data,
                                init.calculate_gain('relu'))
