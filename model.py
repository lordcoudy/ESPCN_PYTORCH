import torch.nn as nn
import torch.nn.init as init

from utils import measure_time

class ESPCN(nn.Module):
    def __init__(self, upscale_factor=2, num_channels=1):
        super(ESPCN, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = SeparableConv2d(num_channels, 64, 5, 1, 2)
        self.conv2 = SeparableConv2d(64, 64, 3, 1, 1)
        self.conv3 = SeparableConv2d(64, 32, 3, 1, 1)
        self.conv4 = SeparableConv2d(32, num_channels * (upscale_factor ** 2), 3, 1, 1)
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
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 depth_multiplier=1,
        ):
        super().__init__()

        intermediate_channels = in_channels * depth_multiplier
        self.spatialConv = nn.Conv2d(
             in_channels=in_channels,
             out_channels=intermediate_channels,
             kernel_size=kernel_size,
             stride=stride,
             padding=padding,
             dilation=dilation,
             groups=in_channels,
             bias=bias,
             padding_mode=padding_mode
        )
        self.pointConv = nn.Conv2d(
             in_channels=intermediate_channels,
             out_channels=out_channels,
             kernel_size=1,
             stride=1,
             padding=0,
             dilation=1,
             bias=bias,
             padding_mode=padding_mode,
        )

    def forward(self, x):
        return self.pointConv(self.spatialConv(x))
