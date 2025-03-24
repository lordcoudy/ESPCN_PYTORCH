import torch
import torch.nn as nn
from torchvision import models

from model import ESPCN


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_channels = 1
        num_filters = self.base_model.conv1.out_channels
        kernel_size = self.base_model.conv1.kernel_size
        stride = self.base_model.conv1.stride
        padding = self.base_model.conv1.padding
        conv1 = torch.nn.Conv2d(num_channels, num_filters, kernel_size = kernel_size, stride = stride, padding = padding)
        original_weights = self.base_model.conv1.weight.data.mean(dim = 1, keepdim = True)
        conv1.weight.data = original_weights.repeat(1, num_channels, 1, 1)
        self.base_model.conv1 = conv1
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x).argmax(dim = 1)


class ObjectAwareESPCN(nn.Module):
    def __init__(self, num_classes, upscale_factor, num_channels = 1):
        super(ObjectAwareESPCN, self).__init__()
        self.classifier = Classifier(num_classes)
        self.espcn_networks = nn.ModuleList([ESPCN(num_classes=1, upscale_factor=upscale_factor, num_channels=num_channels) for _ in range(num_classes)])

    def forward(self, x):
        class_ids = self.classifier(x)  # Shape [batch_size]
        sr_outputs = []
        from custom_logger import get_logger
        logger = get_logger('data')
        logger.info(f"Class IDs: {class_ids}")
        for i in range(x.size(0)):
            sr_out = self.espcn_networks[class_ids[i]](x[i].unsqueeze(0))
            sr_outputs.append(sr_out)
        return torch.cat(sr_outputs, dim=0)
