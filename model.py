import torch.nn as nn
import torch.nn.init as init

from utils import measure_time

class ESPCN(nn.Module):
    def __init__(self, upscale_factor=2, num_channels=1):
        super(ESPCN, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = SeparableConv2d(num_channels, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = SeparableConv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = SeparableConv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = SeparableConv2d(32, num_channels * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixelShuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixelShuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.depthwise.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.depthwise.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.depthwise.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.depthwise.weight)

class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=padding, stride=stride)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, bias=bias, stride=stride)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out