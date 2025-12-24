import torch
import torch.nn as nn
from torch.nn import init

from ICNR import ICNR


class ESPCN(nn.Module):
    VALID_UPSCALE_FACTORS = (2, 3, 4, 8)
    
    def __init__(self, num_classes, upscale_factor, num_channels=1, separable=False, dropout_rate=0.5):
        super(ESPCN, self).__init__()
        
        # Validate upscale_factor
        if upscale_factor not in self.VALID_UPSCALE_FACTORS:
            raise ValueError(
                f"upscale_factor must be one of {self.VALID_UPSCALE_FACTORS}, got {upscale_factor}"
            )
        
        # Use inplace=True for ReLU to save memory (no need to store input for backward)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Conv2d(num_channels, 64, (5, 5), padding=(2, 2), padding_mode='reflect')
        if separable:
            # Depthwise separable convolutions: more efficient on Apple Silicon ANE
            self.conv2_depthwise = nn.Conv2d(64, 64, (3, 3), padding = (1, 1), padding_mode = 'reflect', groups = 64)
            self.conv2_pointwise = nn.Conv2d(64, 64, (1, 1))
            self.conv2 = nn.Sequential(self.conv2_depthwise, self.conv2_pointwise)
            self.conv3_depthwise = nn.Conv2d(64, 64, (3, 3), padding = (1, 1), padding_mode = 'reflect', groups = 64)
            self.conv3_pointwise = nn.Conv2d(64, 32, (1, 1))
            self.conv3 = nn.Sequential(self.conv3_depthwise, self.conv3_pointwise)
        else:
            self.conv2 = nn.Conv2d(64, 64, (3, 3), padding = (1, 1), padding_mode = 'reflect')
            self.conv3 = nn.Conv2d(64, 32, (3, 3), padding=(1, 1), padding_mode='reflect')
        self.conv4 = nn.Conv2d(32, num_channels * (upscale_factor**2), (3, 3), padding=(1, 1), padding_mode='reflect')
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.smoothing = nn.Conv2d(num_channels, num_channels, (5, 5), padding=(2, 2), padding_mode='reflect')

        self._initialize_weights(upscale_factor, separable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optimized forward pass with inplace operations
        # Use functional dropout with explicit training flag for JIT trace compatibility
        x = self.relu(self.conv1(x))
        x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.relu(self.conv2(x))
        x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.relu(self.conv3(x))
        x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv4(x)
        x = self.pixel_shuffle(x)
        x = self.smoothing(x)
        return x

    def _initialize_weights(self, upscale_factor, separable):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        if (separable):
            init.orthogonal_(self.conv2_depthwise.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv2_pointwise.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv3_depthwise.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv3_pointwise.weight, init.calculate_gain('relu'))
        else:
            init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        weight = ICNR(self.conv4.weight, initializer = nn.init.kaiming_normal_, upscale_factor = upscale_factor)
        self.conv4.weight.data.copy_(weight)
        init.orthogonal_(self.smoothing.weight, 1.0 / (5 * 5))
        init.constant_(self.smoothing.bias, 0.0)