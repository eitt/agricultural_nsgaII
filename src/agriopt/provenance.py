from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return None


def environment_metadata(root: Path, input_files: Iterable[Path] = ()) -> dict:
    files = {}
    for p in input_files:
        if p and p.exists() and p.is_file():
            files[str(p.relative_to(root) if p.is_relative_to(root) else p)] = {
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(root),
        "inputs": files,
    }


def write_json(data: dict | list, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
