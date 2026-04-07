from __future__ import annotations

import torch


def enable_dropout(model: torch.nn.Module) -> None:
    """Enable dropout layers during evaluation for MC dropout."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout | torch.nn.Dropout2d | torch.nn.Dropout3d):
            module.train()


@torch.no_grad()
def mc_dropout_predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_samples: int,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, torch.Tensor]:
    model.eval()
    enable_dropout(model)

    preds = []
    for _ in range(n_samples):
        with torch.amp.autocast(
            device_type=device.type, enabled=amp_enabled and device.type == "cuda"
        ):
            pred = model(x.to(device))
        preds.append(pred.unsqueeze(0))

    samples = torch.cat(preds, dim=0)
    mean = samples.mean(dim=0)
    variance = samples.var(dim=0, unbiased=False)
    return {"samples": samples, "mean": mean, "variance": variance}
