from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when a config file or key structure is invalid."""


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
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = _load_single_yaml(path)

    bases = raw.pop("_base_", [])
    if isinstance(bases, str | Path):
        bases = [bases]

    config: dict[str, Any] = {}
    for base in bases:
        base_path = (path.parent / base).resolve()
        config = deep_merge_dicts(config, load_config(base_path))

    config = deep_merge_dicts(config, raw)
    return config


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config keys into dotted paths for tabular export."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, dotted))
        else:
            flat[dotted] = value
    return flat


def save_resolved_config(config: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def save_flat_config_csv(config: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = flatten_config(config)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        for key in sorted(flat):
            writer.writerow([key, flat[key]])
    return path


def require_config_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid config section: {section}")
    return value
