import torch.nn as nn
from torch.nn import init
import torch

def icnr_init(tensor, upscale_factor=2, init=nn.init.kaiming_normal_):
    new_shape = (tensor.size(0) // (upscale_factor ** 2),) + tensor.size()[1:]
    subkernel = torch.zeros(new_shape, device=tensor.device)
    init(subkernel)
    subkernel = subkernel.repeat(upscale_factor ** 2, 1, 1, 1)
    with torch.no_grad():
        tensor.copy_(subkernel)
    return tensor

class ESPCN(nn.Module):
    def __init__(self, num_classes, upscale_factor, num_channels = 1):
        super(ESPCN, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(num_channels, 64, (5, 5), padding=(2, 2), padding_mode='reflect')
        self.conv2 = nn.Conv2d(64, 64, (3, 3), padding=(1, 1), padding_mode='reflect')
        self.conv3 = nn.Conv2d(64, 32, (3, 3), padding=(1, 1), padding_mode='reflect')
        self.conv4 = nn.Conv2d(32, num_channels * (upscale_factor ** 2),(3, 3), padding = (1, 1), padding_mode = 'reflect')
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights(upscale_factor)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self, upscale_factor):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)
        icnr_init(self.conv4.weight.data, upscale_factor)
