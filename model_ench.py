class EnhancedESPCN(nn.Module):
    def __init__(self, upscaleFactor=2, numChannels=1):
        super(EnhancedESPCN, self).__init__()

        # First block with channel attention
        self.conv1 = nn.Sequential(
            nn.Conv2d(numChannels, 64, (5, 5), (1, 1), (2, 2)),
            nn.BatchNorm2d(64),
            nn.PReLU(),  # Parametric ReLU for adaptive learning
            ChannelAttention(64),  # Add channel attention
            nn.Dropout2d(p=0.2)
        )

        # Second block with enhanced residual connection
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            ChannelAttention(64),
            nn.Dropout2d(p=0.2)
        )

        # Third block with dense connection
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 32, (3, 3), (1, 1), (1, 1)),  # 128 input channels for concatenation
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout2d(p=0.2)
        )

        self.conv4 = nn.Conv2d(32, numChannels * (upscaleFactor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixelShuffle = nn.PixelShuffle(upscaleFactor)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)

        # Dense connection: concatenate features
        dense_out = torch.cat([conv1_out, conv2_out], dim=1)
        conv3_out = self.conv3(dense_out)

        x = self.conv4(conv3_out)
        return self.pixelShuffle(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
