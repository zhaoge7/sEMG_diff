from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config."""
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must contain a mapping at the top level: {path}")
    return cfg


def ensure_project_dirs(cfg: dict[str, Any]) -> None:
    """Create writable project directories declared in the config."""
    project = cfg.get("project", {})
    for key in ("processed_dir", "results_dir"):
        value = project.get(key)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)


def get_seed(cfg: dict[str, Any]) -> int:
    return int(cfg.get("splits", {}).get("seed", 2026))
