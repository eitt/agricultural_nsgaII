from __future__ import annotations

from pathlib import Path


def repo_root(start: str | Path | None = None) -> Path:
    """Return the repository root, identified by pyproject.toml."""
    if start is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start).resolve()
    candidates = [start_path, *start_path.parents]
    for p in candidates:
        if (p / "pyproject.toml").exists():
            return p
    # Installed editable package: src/agriopt/paths.py -> repo root is parents[2].
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError("Could not locate repository root containing pyproject.toml")


def resolve_path(value: str | Path | None, root: Path | None = None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return (root or repo_root()) / p
