from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_matrix_csv(path: str | Path) -> pd.DataFrame:
    """Read a wide matrix CSV and remove common date/index columns from the data matrix."""
    p = Path(path)
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Empty CSV: {p}")
    first = str(df.columns[0]).strip().lower()
    if first in {"week", "date", "fecha", "timestamp", "unnamed: 0"}:
        idx = df.iloc[:, 0]
        df = df.iloc[:, 1:].copy()
        try:
            df.index = pd.to_datetime(idx)
        except Exception:
            df.index = idx
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.isna().all().any():
        bad = df.columns[df.isna().all()].tolist()
        raise ValueError(f"Columns with no numeric observations in {p}: {bad}")
    return df


def read_demand_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Empty demand CSV: {p}")
    first = str(df.columns[0]).strip().lower()
    if first in {"week", "date", "fecha", "timestamp", "unnamed: 0"}:
        idx = df.iloc[:, 0]
        df = df.iloc[:, 1:].copy()
        try:
            df.index = pd.to_datetime(idx)
        except Exception:
            df.index = idx
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
