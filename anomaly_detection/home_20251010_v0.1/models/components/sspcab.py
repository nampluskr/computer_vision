###########################################################
# anomalib/src/anomalib/models/components/layers/sspcab.py
###########################################################

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class AttentionModule(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 8) -> None:
        super().__init__()

        out_channels = in_channels // reduction_ratio
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, in_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # reduce feature map to 1d vector through global average pooling
        avg_pooled = inputs.mean(dim=(2, 3))

        # squeeze and excite
        act = self.fc1(avg_pooled)
        act = F.relu(act)
        act = self.fc2(act)
        act = F.sigmoid(act)

        # multiply with input
        return inputs * act.view(act.shape[0], act.shape[1], 1, 1)


class SSPCAB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 1,
        dilation: int = 1,
        reduction_ratio: int = 8,
    ) -> None:
        super().__init__()

        self.pad = kernel_size + dilation
        self.crop = kernel_size + 2 * dilation + 1

        self.masked_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.masked_conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.masked_conv3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.masked_conv4 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
        )

        self.attention_module = AttentionModule(
            in_channels=in_channels,
            reduction_ratio=reduction_ratio,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # compute masked convolution
        padded = F.pad(inputs, (self.pad,) * 4)
        masked_out = torch.zeros_like(inputs)
        masked_out += self.masked_conv1(padded[..., : -self.crop, : -self.crop])
        masked_out += self.masked_conv2(padded[..., : -self.crop, self.crop :])
        masked_out += self.masked_conv3(padded[..., self.crop :, : -self.crop])
        masked_out += self.masked_conv4(padded[..., self.crop :, self.crop :])

        # apply channel attention module
        return self.attention_module(masked_out)