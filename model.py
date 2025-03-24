import torch.nn as nn
from torch.nn import init

from ICNR import ICNR


class ESPCN(nn.Module):
    def __init__(self, num_classes, upscale_factor, num_channels=1):
        super(ESPCN, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5) 
        self.conv1 = nn.Conv2d(num_channels, 64, (5, 5), padding=(2, 2), padding_mode='reflect')
        self.conv2 = nn.Conv2d(64, 64, (3, 3), padding=(1, 1), padding_mode='reflect')
        self.conv3 = nn.Conv2d(64, 32, (3, 3), padding=(1, 1), padding_mode='reflect')
        self.conv4 = nn.Conv2d(32, num_channels * (upscale_factor**2), (3, 3), padding=(1, 1), padding_mode='reflect')
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.smoothing = nn.Conv2d(num_channels, num_channels, (5, 5), padding=(2, 2), padding_mode='reflect')

        self._initialize_weights(upscale_factor)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.pixel_shuffle(x)
        x = self.smoothing(x)
        return x

    def _initialize_weights(self, upscale_factor):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        weight = ICNR(self.conv4.weight, initializer = nn.init.kaiming_normal_, upscale_factor = upscale_factor)
        self.conv4.weight.data.copy_(weight)
        init.orthogonal_(self.smoothing.weight, 1.0 / (5 * 5))
        init.constant_(self.smoothing.bias, 0.0)