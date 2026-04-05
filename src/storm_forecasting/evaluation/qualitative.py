from __future__ import annotations

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def predict_example(
    model: torch.nn.Module,
    dataset,
    index: int,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    model.eval()
    x, y, meta = dataset[index]
    x_batch = x.unsqueeze(0).to(device)

    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
        y_hat = model(x_batch)

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    y_hat_np = y_hat[0].detach().cpu().numpy()
    return x_np, y_np, y_hat_np, meta


def plot_example(
    x: np.ndarray,
    y: np.ndarray,
    y_hat: np.ndarray,
    timesteps: tuple[int, ...] = (0, 3, 6, 9, 11),
    save_path: str | Path | None = None,
) -> None:
    rows = len(timesteps)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, step in enumerate(timesteps):
        axes[row, 0].imshow(x[min(step, x.shape[0] - 1), 0], vmin=0, vmax=1)
        axes[row, 0].set_title(f"Input step {min(step, x.shape[0] - 1) + 1}")
        axes[row, 1].imshow(y[step, 0], vmin=0, vmax=1)
        axes[row, 1].set_title(f"Target t+{step + 1}")
        axes[row, 2].imshow(y_hat[step, 0], vmin=0, vmax=1)
        axes[row, 2].set_title(f"Prediction t+{step + 1}")
        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_compact_panel(
    y: np.ndarray,
    y_hat: np.ndarray,
    horizons: tuple[int, ...] = (1, 4, 8, 12),
    save_path: str | Path | None = None,
) -> None:
    fig, axes = plt.subplots(2, len(horizons), figsize=(3 * len(horizons), 6))
    for col, horizon in enumerate(horizons):
        idx = horizon - 1
        axes[0, col].imshow(y[idx, 0], vmin=0, vmax=1)
        axes[0, col].set_title(f"GT t+{horizon}")
        axes[1, col].imshow(y_hat[idx, 0], vmin=0, vmax=1)
        axes[1, col].set_title(f"Pred t+{horizon}")
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_pred_gt_gif(
    y: np.ndarray,
    y_hat: np.ndarray,
    out_path: str | Path,
    fps: int = 2,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for step in range(y.shape[0]):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(y[step, 0], vmin=0, vmax=1)
        axes[0].set_title(f"GT t+{step + 1}")
        axes[1].imshow(y_hat[step, 0], vmin=0, vmax=1)
        axes[1].set_title(f"Pred t+{step + 1}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(out_path, frames, fps=fps)
    return out_path
