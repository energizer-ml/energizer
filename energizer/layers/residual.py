from tkinter import N
import energizer
from energizer.neural_network import Module
from energizer.tensor import Tensor
import numpy as np


class ResidualBlock(Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = energizer.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = energizer.BatchNorm2d(channels)
        self.conv2 = energizer.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = energizer.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = energizer.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = energizer.ReLU()(out)
        return out


class BottleneckBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.expansion = 4
        reduce_channels = out_channels // self.expansion

        self.conv1 = energizer.Conv2d(
            in_channels, reduce_channels, kernel_size=1, bias=False
        )
        self.bn1 = energizer.BatchNorm2d(reduce_channels)

        self.conv2 = energizer.Conv2d(
            reduce_channels,
            reduce_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.bn2 = energizer.BatchNorm2d(reduce_channels)

        self.conv3 = energizer.Conv2d(
            reduce_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = energizer.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = energizer.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
            self.shortcut_bn = energizer.BatchNorm2d(out_channels)
        else:
            self.shortcut = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = energizer.ReLU()(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = energizer.ReLU()(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)
            identity = self.shortcut_bn(identity)

        out += identity
        out = energizer.ReLU()(out)

        return out
