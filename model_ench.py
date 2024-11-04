import torch
from torch import nn, Tensor
import torch.nn.init as init
import torch.nn.functional as F

import math


class OptimizedESPCN(nn.Module):
    def __init__(self, upscaleFactor=2, numChannels=1):
        super(OptimizedESPCN, self).__init__()

        # Feature extraction layers
        self.conv1 = nn.Conv2d(numChannels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Attention mechanism
        self.attn = nn.Conv2d(32, 32, kernel_size=1)

        # Upsampling layers
        self.conv3 = nn.Conv2d(32, numChannels * (upscaleFactor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscaleFactor)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))

        # Attention layer to emphasize spatial importance
        attention_map = torch.sigmoid(self.attn(x))
        x = x * attention_map  # Element-wise attention

        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        x = torch.clamp(x, 0.0, 1.0)  # Clamp output to valid image range
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

class AltESPCN(nn.Module):
    def __init__(self, upscaleFactor, numChannels=1) -> None:
        super(AltESPCN, self).__init__()
        channels = 64
        hiddenChannels = 32
        outChannels = int(numChannels * (upscaleFactor ** 2))

        # Feature mapping
        self.featureMaps = nn.Sequential(
            nn.Conv2d(numChannels, channels, (5, 5), (1, 1), (2, 2)),
            nn.Tanh(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.Tanh(),
            nn.Conv2d(channels, hiddenChannels, (3, 3), (1, 1), (1, 1)),
            nn.Tanh(),
        )

        # Sub-pixel convolution layer
        self.sub_pixel = nn.Sequential(
            nn.Conv2d(hiddenChannels, outChannels, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscaleFactor),
        )

        # Initial model weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data,
                                0.0,
                                math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.featureMaps(x)
        x = self.sub_pixel(x)

        x = torch.clamp_(x, 0.0, 1.0)

        return x