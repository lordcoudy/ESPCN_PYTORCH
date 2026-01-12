from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from ICNR import ICNR


class ESPCN(nn.Module):
    """Efficient Sub-Pixel Convolutional Neural Network for image super-resolution.
    
    Architecture follows the original ESPCN paper with modern optimizations:
    - Optional depthwise separable convolutions for efficiency
    - Configurable dropout for regularization
    - ICNR initialization for sub-pixel convolution
    - Reflect padding to reduce border artifacts
    
    Args:
        upscale_factor: Upscaling factor (2, 3, 4, or 8)
        num_channels: Number of input/output channels (default: 1 for Y channel)
        separable: Use depthwise separable convolutions (default: False)
        dropout_rate: Dropout probability during training (default: 0.0)
        use_bn: Use batch normalization (default: False)
    """
    VALID_UPSCALE_FACTORS = (2, 3, 4, 8)
    
    def __init__(
        self, 
        upscale_factor: int, 
        num_channels: int = 1, 
        separable: bool = False, 
        dropout_rate: float = 0.0,
        use_bn: bool = False,
        # Keep num_classes for backward compatibility (ignored)
        num_classes: Optional[int] = None
    ):
        super().__init__()
        
        # Validate upscale_factor
        if upscale_factor not in self.VALID_UPSCALE_FACTORS:
            raise ValueError(
                f"upscale_factor must be one of {self.VALID_UPSCALE_FACTORS}, got {upscale_factor}"
            )
        
        self.upscale_factor = upscale_factor
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.separable = separable
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(num_channels, 64, 5, padding=2, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        
        if separable:
            # Depthwise separable convolutions: more efficient on Apple Silicon ANE
            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect', groups=64),
                nn.Conv2d(64, 64, 1)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect', groups=64),
                nn.Conv2d(64, 32, 1)
            )
        else:
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect')
            self.conv3 = nn.Conv2d(64, 32, 3, padding=1, padding_mode='reflect')
        
        self.bn2 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        self.bn3 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        
        # Sub-pixel upsampling layer
        self.conv4 = nn.Conv2d(32, num_channels * (upscale_factor ** 2), 3, padding=1, padding_mode='reflect')
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        # Optional smoothing layer for reducing checkerboard artifacts
        self.smoothing = nn.Conv2d(num_channels, num_channels, 5, padding=2, padding_mode='reflect')

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional dropout during training.
        
        Note: Inplace operations are disabled during training when dropout is enabled,
        as they can cause gradient computation issues, especially on MPS devices.
        """
        # Disable inplace when: (1) using MPS device, or (2) training with dropout
        use_inplace = (x.device.type != 'mps') and not (self.dropout_rate > 0 and self.training)
        
        # Feature extraction with ReLU activation
        x = F.relu(self.bn1(self.conv1(x)), inplace=use_inplace)
        if self.dropout_rate > 0 and self.training:
            x = F.dropout(x, p=self.dropout_rate, training=True, inplace=False)
        
        x = F.relu(self.bn2(self.conv2(x)), inplace=use_inplace)
        if self.dropout_rate > 0 and self.training:
            x = F.dropout(x, p=self.dropout_rate, training=True, inplace=False)
        
        x = F.relu(self.bn3(self.conv3(x)), inplace=use_inplace)
        if self.dropout_rate > 0 and self.training:
            x = F.dropout(x, p=self.dropout_rate, training=True, inplace=False)
        
        # Sub-pixel upsampling
        x = self.pixel_shuffle(self.conv4(x))
        
        # Apply smoothing to reduce artifacts
        x = self.smoothing(x)
        return x

    def _initialize_weights(self) -> None:
        """Initialize weights using orthogonal init for conv layers and ICNR for sub-pixel layer."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Skip the sub-pixel conv (initialized separately with ICNR)
                if module is self.conv4:
                    continue
                # Skip smoothing layer (use small initialization)
                if module is self.smoothing:
                    init.orthogonal_(module.weight, gain=1.0 / 25.0)
                    if module.bias is not None:
                        init.zeros_(module.bias)
                    continue
                # Standard conv layers
                init.orthogonal_(module.weight, gain=init.calculate_gain('relu'))
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                init.ones_(module.weight)
                init.zeros_(module.bias)
        
        # ICNR initialization for sub-pixel convolution
        weight = ICNR(self.conv4.weight, initializer=init.kaiming_normal_, upscale_factor=self.upscale_factor)
        self.conv4.weight.data.copy_(weight)
        if self.conv4.bias is not None:
            init.zeros_(self.conv4.bias)