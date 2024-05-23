import torch
import torch.nn as nn
import torch.nn.init as init

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.bn(self.conv2(out))
        out += residual
        return out

class EnhancedESPCN(nn.Module):
    def __init__(self, upscale_factor, num_channels=1):
        super(EnhancedESPCN, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(num_channels, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.resblock1 = ResidualBlock(64)
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.resblock2 = ResidualBlock(32)
        self.conv4 = nn.Conv2d(32, num_channels * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.resblock1(x)
        x = self.relu(self.conv3(x))
        x = self.resblock2(x)
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        init.orthogonal_(self.conv4.weight)