from __future__ import annotations

import torch
import torch.nn as nn

from storm_forecasting.models.blocks import Down, Up, conv_block
from storm_forecasting.models.convlstm import ConvLSTM


class ConvLSTMUNetSeq2Seq(nn.Module):
    """ConvLSTM bottleneck U-Net for direct multi-step VIL forecasting.

    Input:
        (B, Tin, 1, H, W)
    Output:
        (B, Tout, 1, H, W)

    The encoder processes each input frame independently with shared weights.
    A ConvLSTM encodes the latent sequence. A second ConvLSTM decodes future
    latent states from zeros, and each future latent is mapped back to an image
    using the final input frame's skip connections.
    """

    def __init__(
        self,
        base: int = 16,
        bottleneck_ch: int = 128,
        tin: int = 12,
        tout: int = 12,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.tin = tin
        self.tout = tout

        self.down1 = Down(1, base, dropout_p=dropout_p)
        self.down2 = Down(base, base * 2, dropout_p=dropout_p)
        self.down3 = Down(base * 2, base * 4, dropout_p=dropout_p)

        self.pool_bn = nn.MaxPool2d(2)
        self.bn = conv_block(base * 4, bottleneck_ch, dropout_p=dropout_p)

        self.enc_lstm = ConvLSTM(bottleneck_ch, bottleneck_ch)
        self.dec_lstm = ConvLSTM(bottleneck_ch, bottleneck_ch)

        self.up3 = Up(bottleneck_ch, base * 4, base * 4, dropout_p=dropout_p)
        self.up2 = Up(base * 4, base * 2, base * 2, dropout_p=dropout_p)
        self.up1 = Up(base * 2, base, base, dropout_p=dropout_p)
        self.out = nn.Conv2d(base, 1, kernel_size=1)

    def encode_frame(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        s1, p1 = self.down1(x)
        s2, p2 = self.down2(p1)
        s3, p3 = self.down3(p2)
        z = self.pool_bn(p3)
        z = self.bn(z)
        return z, (s1, s2, s3)

    def decode_frame(
        self,
        z: torch.Tensor,
        skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        s1, s2, s3 = skips
        x = self.up3(z, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.out(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, _, _, _ = x.shape
        if steps != self.tin:
            raise ValueError(f"Expected Tin={self.tin}, got {steps}")

        latent_seq = []
        skips_last = None
        for step in range(self.tin):
            z, skips = self.encode_frame(x[:, step])
            latent_seq.append(z)
            skips_last = skips

        if skips_last is None:
            raise RuntimeError("No encoder skip features were produced.")

        latent_seq = torch.stack(latent_seq, dim=1)
        _, state = self.enc_lstm(latent_seq)

        zeros = torch.zeros(
            batch_size,
            self.tout,
            latent_seq.size(2),
            latent_seq.size(3),
            latent_seq.size(4),
            device=x.device,
        )
        latent_future, _ = self.dec_lstm(zeros, state=state)

        outputs = []
        for step in range(self.tout):
            outputs.append(self.decode_frame(latent_future[:, step], skips_last))
        return torch.stack(outputs, dim=1)
