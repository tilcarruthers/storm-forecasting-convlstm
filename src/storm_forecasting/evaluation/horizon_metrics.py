from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def mae_per_horizon(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_enabled: bool,
    tout: int,
) -> np.ndarray:
    model.eval()
    sums = np.zeros(tout, dtype=np.float64)
    count = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
            y_hat = model(x)

        abs_err = (y_hat - y).abs().mean(dim=(0, 2, 3, 4))
        sums += abs_err.detach().cpu().numpy()
        count += 1

    return sums / max(count, 1)


def plot_horizon_metric(
    values: np.ndarray,
    title: str = "MAE per Forecast Horizon",
    ylabel: str = "MAE",
    save_path: str | Path | None = None,
) -> None:
    horizons = np.arange(1, len(values) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(horizons, values, marker="o")
    plt.xlabel("Forecast step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()
