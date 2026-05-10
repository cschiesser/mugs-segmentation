"""Classical U-Net for binary segmentation.

Based on Ronneberger et al. (2015), "U-Net: Convolutional Networks for
Biomedical Image Segmentation". Re-implemented from scratch (no pretrained
weights, per project rules).

Configurable knobs:
    - base_channels: width of the first encoder stage. Each subsequent stage
      doubles. Used for the small/large config comparison required by the
      project specification.
    - depth: number of down-sampling stages. Total stages = depth + 1
      (encoder) + depth (decoder) + 1 bottleneck.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscale by 2 with max-pool, then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Up(nn.Module):
    """Upscale by 2, concatenate with skip, then DoubleConv.

    Bilinear upsampling + 1x1 conv is used (lighter than transpose conv and
    avoids checkerboard artefacts).
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
        )
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Spatial sizes can differ by 1 pixel after odd-sized inputs, so pad.
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        # Channel widths for each stage.
        widths = [base_channels * (2 ** i) for i in range(depth + 1)]

        # Encoder.
        self.input_conv = DoubleConv(in_channels, widths[0])
        self.downs = nn.ModuleList(
            [Down(widths[i], widths[i + 1]) for i in range(depth)]
        )

        # Decoder. Each Up takes (in_ch from below, skip_ch from encoder).
        self.ups = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(Up(in_ch=widths[i], skip_ch=widths[i - 1], out_ch=widths[i - 1]))

        # 1x1 to logits.
        self.output_conv = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.input_conv(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        # ``skips`` now contains [s0, s1, ..., s_depth]; the last entry is the
        # bottleneck and is consumed first by the decoder.
        x = skips.pop()
        for up in self.ups:
            x = up(x, skips.pop())
        return self.output_conv(x)
