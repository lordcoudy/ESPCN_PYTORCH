import torch
import torch.nn as nn
from torch.nn import init


def ICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, \
        ("The size of the first dimension: "
         f"tensor.shape[0] = {tensor.shape[0]}"
         " is not divisible by square of upscale_factor: "
         f"upscale_factor = {upscale_factor}")
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared,
                             *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)

class ESPCN(nn.Module):
    def __init__(self, num_classes, upscale_factor, num_channels=1):
        super(ESPCN, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(num_channels, 64, (5, 5), padding=(2, 2), padding_mode='reflect')
        self.conv2 = nn.Conv2d(64, 64, (3, 3), padding=(1, 1), padding_mode='reflect')
        self.conv3 = nn.Conv2d(64, 32, (3, 3), padding=(1, 1), padding_mode='reflect')
        self.conv4 = nn.Conv2d(32, num_channels * (upscale_factor**2), (3, 3), padding=(1, 1), padding_mode='reflect')
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.smoothing = nn.Conv2d(num_channels, num_channels, (5, 5), padding=(2, 2), padding_mode='reflect')
        self.sharpening = nn.Conv2d(num_channels, num_channels, (3, 3), padding=(1, 1), bias = False)

        self._initialize_weights(upscale_factor)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.pixel_shuffle(x)
        x = self.smoothing(x)
        x = self.sharpening(x)
        return x

    def _initialize_weights(self, upscale_factor):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        weight = ICNR(self.conv4.weight, initializer = nn.init.kaiming_normal_, upscale_factor = upscale_factor)
        self.conv4.weight.data.copy_(weight)
        nn.init.constant_(self.smoothing.weight, 1.0 / (5 * 5))
        nn.init.constant_(self.smoothing.bias, 0.0)
        kernel = torch.tensor([[-0.25, -0.25, -0.25],
                               [-0.25, 3., -0.25],
                               [-0.25, -0.25, -0.25]])
        kernel = kernel.view(1,1,3,3).repeat(1, 1, 1, 1)
        with torch.no_grad():
            self.sharpening.weight = nn.Parameter(kernel)