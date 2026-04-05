from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(
            in_ch + hid_ch,
            4 * hid_ch,
            kernel_size=kernel_size,
            padding=padding,
        )

    def init_state(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hid_ch, height, width, device=device)
        c = torch.zeros(batch_size, self.hid_ch, height, width, device=device)
        return h, c

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hid_ch, kernel_size=kernel_size)

    def forward(
        self,
        x_seq: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Process a sequence with shape (B, T, C, H, W)."""
        bsz, steps, _, height, width = x_seq.shape
        if state is None:
            state = self.cell.init_state(bsz, height, width, x_seq.device)

        outputs = []
        h_t, c_t = state
        for step in range(steps):
            h_t, c_t = self.cell(x_seq[:, step], (h_t, c_t))
            outputs.append(h_t)

        return torch.stack(outputs, dim=1), (h_t, c_t)
