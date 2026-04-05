from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_single_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping in {path}, got {type(data)!r}")
    return data


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config with optional _base_ inheritance."""
    path = Path(config_path).resolve()
    raw = _load_single_yaml(path)

    bases = raw.pop("_base_", [])
    if isinstance(bases, (str, Path)):
        bases = [bases]

    config: dict[str, Any] = {}
    for base in bases:
        base_path = (path.parent / base).resolve()
        config = deep_merge_dicts(config, load_config(base_path))

    config = deep_merge_dicts(config, raw)
    return config
