from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).abs().mean().item())


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(((pred - target) ** 2).mean().item())


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(np.sqrt(mse(pred, target)))


def weighted_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    thresholds: tuple[float, ...] = (0.25, 0.5, 0.75),
    weights: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
) -> float:
    if len(weights) != len(thresholds) + 1:
        raise ValueError("weights must be one longer than thresholds")

    weight_map = torch.full_like(target, weights[0])
    for threshold, weight in zip(thresholds, weights[1:]):
        weight_map = torch.where(target >= threshold, torch.full_like(weight_map, weight), weight_map)

    abs_err = (pred - target).abs()
    return float((abs_err * weight_map).sum().item() / weight_map.sum().clamp_min(1e-12).item())


def mean_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """Average SSIM over (B, T, C, H, W)."""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    values: list[float] = []
    for batch_idx in range(pred_np.shape[0]):
        for step_idx in range(pred_np.shape[1]):
            pred_frame = pred_np[batch_idx, step_idx, 0]
            target_frame = target_np[batch_idx, step_idx, 0]
            values.append(
                ssim(
                    target_frame,
                    pred_frame,
                    data_range=data_range,
                )
            )
    return float(np.mean(values)) if values else float("nan")


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    compute_ssim: bool = False,
    weighted_mae_cfg: dict | None = None,
) -> dict[str, float]:
    results = {
        "mae": mae(pred, target),
        "mse": mse(pred, target),
        "rmse": rmse(pred, target),
    }
    if weighted_mae_cfg is not None:
        thresholds = tuple(weighted_mae_cfg.get("thresholds", (0.25, 0.5, 0.75)))
        weights = tuple(weighted_mae_cfg.get("weights", (1.0, 2.0, 3.0, 4.0)))
        results["weighted_mae"] = weighted_mae(pred, target, thresholds=thresholds, weights=weights)
    if compute_ssim:
        results["ssim"] = mean_ssim(pred, target)
    return results
