from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .paths import repo_root, resolve_path


def load_config(path: str | Path) -> dict[str, Any]:
    root = repo_root()
    p = resolve_path(path, root)
    if p is None or not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(p)
    cfg["_root"] = str(root)
    return cfg


def save_config_snapshot(cfg: dict[str, Any], path: str | Path) -> None:
    clean = deepcopy(cfg)
    clean.pop("_config_path", None)
    clean.pop("_root", None)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, sort_keys=False, allow_unicode=True)
