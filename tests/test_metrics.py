import torch

from storm_forecasting.evaluation.metrics import compute_metrics


def test_compute_metrics_basic() -> None:
    pred = torch.zeros(2, 12, 1, 16, 16)
    target = torch.ones(2, 12, 1, 16, 16)
    metrics = compute_metrics(pred, target, compute_ssim=False)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert metrics["mae"] > 0
