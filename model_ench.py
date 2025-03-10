import torch
import torch.nn as nn
from torchvision import models

from model import ESPCN


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Extract the first conv layer's parameters
        num_channels = 1
        num_filters = self.base_model.conv1.out_channels
        kernel_size = self.base_model.conv1.kernel_size
        stride = self.base_model.conv1.stride
        padding = self.base_model.conv1.padding
        conv1 = torch.nn.Conv2d(num_channels, num_filters, kernel_size = kernel_size, stride = stride, padding = padding)
        # Initialize the new conv1 layer's weights by averaging the pretrained weights across the channel dimension
        original_weights = self.base_model.conv1.weight.data.mean(dim = 1, keepdim = True)
        # Expand the averaged weights to the number of input channels of the new dataset
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
        # Classify the object
        class_probs = self.classifier(x)
        class_id = class_probs[0].item()
        # log(f"Class ID: {class_id}")
        print(f"Class ID: {class_id}")
        # Select the corresponding ESPCN network
        sr_output = self.espcn_networks[class_id](x)
        return sr_output
