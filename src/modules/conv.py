import torch
from torch import nn


from typing import Tuple


class ConvActNorm(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: str = "same",
        bias: bool = True,
        activation: str = "ELU",
    ):
        super().__init__()
        activation_cls = getattr(nn, activation)

        self.layers = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            activation_cls(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: str = "same",
        bias: bool = True,
        activation: str = "ELU",
        pool_kernel_size: Tuple[int, int] = (2, 2),
        pool_type: str = "AvgPool2d",
    ):
        super().__init__()

        pool_cls = getattr(nn, pool_type)

        self.layers = nn.Sequential(
            ConvActNorm(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                activation=activation,
            ),
            ConvActNorm(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                activation=activation,
            ),
            pool_cls(kernel_size=pool_kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x
