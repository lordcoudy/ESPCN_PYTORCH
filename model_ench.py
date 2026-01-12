from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from model import ESPCN


class Classifier(nn.Module):
    """ResNet18-based classifier for object-aware super-resolution routing.
    
    Args:
        num_classes: Number of object classes to classify
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Adapt for single-channel (Y luminance) input
        num_channels = 1
        original_conv = self.base_model.conv1
        self.base_model.conv1 = nn.Conv2d(
            num_channels, 
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        # Initialize with mean of original RGB weights
        with torch.no_grad():
            self.base_model.conv1.weight.copy_(
                original_conv.weight.mean(dim=1, keepdim=True)
            )
        
        # Replace classifier head
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class logits (not argmax, for training flexibility)."""
        return self.base_model(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted class indices."""
        return self.forward(x).argmax(dim=1)


class ObjectAwareESPCN(nn.Module):
    """Object-aware ESPCN with class-specific super-resolution networks.
    
    Uses a classifier to route input images to specialized ESPCN networks
    based on detected object class.
    
    Args:
        num_classes: Number of object classes
        upscale_factor: Upscaling factor for ESPCN networks
        num_channels: Number of input/output channels (default: 1)
        separable: Use depthwise separable convolutions in ESPCN
    """
    def __init__(
        self, 
        num_classes: int, 
        upscale_factor: int, 
        num_channels: int = 1, 
        separable: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.classifier = Classifier(num_classes)
        self.espcn_networks = nn.ModuleList([
            ESPCN(
                upscale_factor=upscale_factor,
                num_channels=num_channels,
                separable=separable
            ) for _ in range(num_classes)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route images through class-specific ESPCN networks.
        
        Uses a weighted sum approach for JIT-compatibility instead of
        dynamic indexing with Python loops.
        """
        batch_size = x.size(0)
        
        # Get class predictions (no gradient for classifier during SR training)
        with torch.no_grad():
            class_logits = self.classifier(x)  # [B, num_classes]
            class_probs = torch.zeros_like(class_logits)
            class_indices = class_logits.argmax(dim=1)  # [B]
            # Create one-hot encoding for hard routing
            class_probs.scatter_(1, class_indices.unsqueeze(1), 1.0)  # [B, num_classes]
        
        # Process through all ESPCN networks and combine with one-hot weights
        # This is JIT-traceable since we don't use .item() or Python indexing
        sr_output = None
        for class_id in range(self.num_classes):
            # Get output from this class's ESPCN
            class_sr = self.espcn_networks[class_id](x)  # [B, C, H, W]
            
            # Weight by class membership (0 or 1 for hard routing)
            weight = class_probs[:, class_id].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
            weighted_sr = class_sr * weight
            
            if sr_output is None:
                sr_output = weighted_sr
            else:
                sr_output = sr_output + weighted_sr
        
        return sr_output
    
    def forward_with_class(self, x: torch.Tensor, class_id: int) -> torch.Tensor:
        """Super-resolve using a specific ESPCN network.
        
        Useful for inference when class is already known.
        """
        return self.espcn_networks[class_id](x)
