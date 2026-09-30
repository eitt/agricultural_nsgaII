from __future__ import annotations

from pathlib import Path
import ast
import warnings

import numpy as np
import pandas as pd

from .config import load_config
from .forecasting import backtest_dataframe, forecast_full_series
from .io_utils import read_matrix_csv
from .parameters import CROP_ORDER
from .paths import repo_root, resolve_path
from .provenance import environment_metadata, write_json


def build_forecasts(config_path: str | Path) -> dict[str, str]:
    cfg = load_config(config_path)
    root = repo_root()
    data_cfg = cfg.get("data", {})
    fc_cfg = cfg.get("forecast", {})
    history_path = resolve_path(data_cfg.get("price_history"), root)
    if history_path is None or not history_path.exists():
        raise FileNotFoundError(
            f"Price history not found: {history_path}. Restore the bundled input matrix or supply a valid configuration path."
        )
    prices = read_matrix_csv(history_path)
    missing = [p for p in CROP_ORDER if p not in prices.columns]
    if missing:
        raise ValueError(
            "Price history does not contain every canonical crop required by the benchmark: "
            + ", ".join(missing)
            + ". Supply the canonical crop columns listed in agriopt.parameters.CROP_ORDER."
        )
    prices = prices[CROP_ORDER].dropna(how="all")
    # Short gaps can occur when markets do not report a product every week.
    prices = prices.interpolate(limit=4, limit_direction="both").ffill().bfill()
    if prices.isna().any().any():
        bad = prices.columns[prices.isna().any()].tolist()
        raise ValueError(f"Unresolved missing values in price history for: {bad}")

    test_size = int(fc_cfg.get("test_size", 52))
    horizon = int(fc_cfg.get("horizon", 104))
    metric = str(fc_cfg.get("selection_metric", "smape")).lower()
    metrics, _ = backtest_dataframe(prices, test_size=test_size)
    if metric not in metrics.columns:
        raise ValueError(f"Unknown forecast selection metric {metric!r}; available: {metrics.columns.tolist()}")
    best = metrics.loc[metrics.groupby("product")[metric].idxmin()].copy().set_index("product")

    forecasts: dict[str, np.ndarray] = {}
    meta: list[dict] = []
    for product in CROP_ORDER:
        row = best.loc[product]
        model = str(row["model"])
        order = row.get("arima_order", None)
        if isinstance(order, str) and order not in {"", "None", "nan"}:
            try:
                order = ast.literal_eval(order)
            except Exception:
                order = None
        if model == "GP":
            order = None
        pred, uncertainty = forecast_full_series(prices[product].to_numpy(float), horizon, model, order)
        forecasts[product] = pred
        meta.append(
            {
                "product": product,
                "selected_model": model,
                f"cv_{metric}": float(row[metric]),
                "cv_rmse": float(row["rmse"]),
                "mean_predictive_scale": float(np.nanmean(uncertainty)),
            }
        )

    if isinstance(prices.index, pd.DatetimeIndex) and len(prices.index):
        start = prices.index.max() + pd.Timedelta(days=7)
        idx = pd.date_range(start=start, periods=horizon, freq="W-MON")
    else:
        idx = pd.RangeIndex(horizon, name="week")
    forecast_df = pd.DataFrame(forecasts, index=idx)
    forecast_df.index.name = "week"

    output_dir = resolve_path(fc_cfg.get("output_dir", "outputs/forecast"), root)
    assert output_dir is not None
    processed = output_dir
    interim = output_dir
    processed.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)
    forecast_path = resolve_path(data_cfg.get("price_forecast", "data/processed/price_forecast_104w.csv"), root)
    assert forecast_path is not None
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(forecast_path, index_label="week")
    metrics.to_csv(interim / "forecast_backtest.csv", index=False)
    pd.DataFrame(meta).to_csv(interim / "forecast_model_selection.csv", index=False)
    write_json(
        environment_metadata(root, [history_path])
        | {"test_size": test_size, "horizon": horizon, "selection_metric": metric},
        processed / "forecast_provenance.json",
    )
    return {
        "forecast": str(forecast_path),
        "backtest": str(interim / "forecast_backtest.csv"),
        "selection": str(interim / "forecast_model_selection.csv"),
    }
