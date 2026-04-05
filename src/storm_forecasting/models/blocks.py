from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(out_ch: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, out_ch)
    while out_ch % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, out_ch)


def conv_block(in_ch: int, out_ch: int, dropout_p: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        _group_norm(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout_p > 0.0:
        layers.append(nn.Dropout2d(dropout_p))
    layers.extend(
        [
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            _group_norm(out_ch),
            nn.ReLU(inplace=True),
        ]
    )
    if dropout_p > 0.0:
        layers.append(nn.Dropout2d(dropout_p))
    return nn.Sequential(*layers)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.block = conv_block(in_ch, out_ch, dropout_p=dropout_p)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.block(x)
        pooled = self.pool(features)
        return features, pooled


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = conv_block(in_ch + skip_ch, out_ch, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)
