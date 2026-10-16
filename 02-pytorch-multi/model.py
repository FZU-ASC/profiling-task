"""A residual CNN, deliberately different from the single-file MLP."""
import torch
from torch import nn
from torch.profiler import record_function


class ResidualBlock(nn.Module):
    def __init__(self, channels, name):
        super().__init__()
        self.name = name
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        with record_function(self.name):
            with record_function("conv_path"):
                residual = torch.relu(self.bn1(self.conv1(x)))
                residual = self.bn2(self.conv2(residual))
            with record_function("residual_add"):
                return torch.relu(x + residual)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(
            ResidualBlock(16, "residual_block_1"),
            nn.AvgPool2d(2),
            ResidualBlock(16, "residual_block_2"),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(16, 4)

    def forward(self, x):
        with record_function("stem"):
            x = self.stem(x)
        with record_function("backbone"):
            x = self.blocks(x)
        with record_function("classification_head"):
            return self.head(self.pool(x).flatten(1))
