from typing import Tuple
from omegaconf import DictConfig
import torch

from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from ..modules.conv import ConvBlock


class PANNLike(nn.Module):
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
        n_classes: int,
        use_activation_checkpoint: bool = False,
    ):
        super().__init__()

        self._make_conv_blocks(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            activation=activation,
            pool_kernel_size=pool_kernel_size,
            pool_type=pool_type,
        )

        act_cls = getattr(nn, activation)

        self.classif = nn.Sequential(
            nn.GroupNorm(1, out_channels[-1]),
            nn.Linear(out_channels[-1], out_channels[-1]),
            act_cls(),
            nn.Linear(out_channels[-1], n_classes),
        )

        self.use_activation_checkpoint = use_activation_checkpoint

    @staticmethod
    def from_config(config: DictConfig):
        return PANNLike(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            bias=config.bias,
            activation=config.activation,
            pool_kernel_size=config.pool_kernel_size,
            pool_type=config.pool_type,
            n_classes=config.n_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        _summary_

        Args:
            x (torch.Tensor): input spectrogram, shape (batch_size, n_channels, n_mels, time)

        Returns:
            torch.Tensor: logit tensor, shape (batch_size, n_classes)
        """

        # (batch_size, out_channels[-1], n_mels // 2^K, time // 2^K)
        if self.use_activation_checkpoint:
            z = checkpoint_sequential(self.conv_blocks, len(self.conv_blocks), x)
        else:
            z = self.conv_blocks(x)

        z = torch.mean(z, dim=[-1, -2])  # (batch_size, out_channels[-1])

        logits = self.classif(z)  # (batch_size, n_classes)

        return logits

    def _make_conv_blocks(
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
        conv_blocks = []

        in_chan = in_channels

        for out_chan in out_channels:
            conv_blocks.append(
                ConvBlock(
                    in_channels=in_chan,
                    out_channels=out_chan,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    activation=activation,
                    pool_kernel_size=pool_kernel_size,
                    pool_type=pool_type,
                )
            )

            in_chan = out_chan

        self.conv_blocks = nn.Sequential(*conv_blocks)


if __name__ == "__main__":
    x = torch.randn(2, 1, 128, 128)
    model = PANNLike(
        in_channels=1,
        out_channels=[32, 64, 128, 256],
        n_classes=10,
        use_activation_checkpoint=True,
    )

    print(model(x).shape)
    print(model)
