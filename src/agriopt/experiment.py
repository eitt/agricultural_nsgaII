from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import time
import traceback
import warnings

import numpy as np
import pandas as pd

from .config import load_config, save_config_snapshot
from .exact_scip import epsilon_constraint_front, save_exact_results
from .hybrid_nsga2 import run_hybrid_nsga2, save_hybrid_results
from .instances import CATALOG, make_instance
from .io_utils import read_demand_csv, read_matrix_csv
from .nsga2 import run_nsga2, save_nsga2_results
from .paths import repo_root, resolve_path
from .productivity import load_productivity
from .provenance import environment_metadata, write_json


def _new_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def resolve_run_dir(run: str | Path, root: Path | None = None) -> Path:
    root = root or repo_root()
    if str(run).lower() == "latest":
        marker = root / "results" / "LATEST"
        if not marker.exists():
            raise FileNotFoundError("results/LATEST does not exist; run an experiment first")
        run = marker.read_text(encoding="utf-8").strip()
    p = Path(run)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "results":
        return root / p
    return root / "results" / p


def _load_inputs(cfg: dict, root: Path):
    dcfg = cfg.get("data", {})
    prod_path = resolve_path(dcfg.get("productivity"), root)
    hist_path = resolve_path(dcfg.get("price_history"), root)
    fc_path = resolve_path(dcfg.get("price_forecast"), root)
    if prod_path is None or not prod_path.exists():
        raise FileNotFoundError(f"Productivity data not found: {prod_path}")
    if hist_path is None or not hist_path.exists():
        raise FileNotFoundError(f"Price history not found: {hist_path}")
    if fc_path is None or not fc_path.exists():
        raise FileNotFoundError(f"Price forecast not found: {fc_path}")
    productivity = load_productivity(prod_path)
    history = read_matrix_csv(hist_path)
    forecast = read_matrix_csv(fc_path)
    demand = None
    demand_path = resolve_path(dcfg.get("demand"), root)
    if demand_path is not None:
        if demand_path.exists():
            demand = read_demand_csv(demand_path)
        else:
            warnings.warn(f"Configured demand file does not exist: {demand_path}; proceeding without a market cap")
    return productivity, history, forecast, demand, [p for p in [prod_path, hist_path, fc_path, demand_path] if p]


def _method_directory(run_dir: Path, instance: str, method: str, seed: int) -> Path:
    return run_dir / instance / method / f"seed_{seed}"


def _write_method_metadata(out: Path, **kwargs) -> None:
    write_json(kwargs, out / "metadata.json")


def _completed_run(out: Path, method: str) -> dict | None:
    meta_path = out / "metadata.json"
    expected = {
        "exact": out / "exact_front_scip.csv",
        "nsga2": out / "nsga2_front.csv",
        "nsga2_qp": out / "nsga2_qp_front.csv",
        "hnsga2": out / "hnsga2_front.csv",
    }.get(method)
    if not meta_path.exists() or expected is None or not expected.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return meta if meta.get("status") == "ok" else None


def run_experiment(
    config_path: str | Path,
    *,
    run_id: str | None = None,
    continue_on_error: bool = True,
    instances_override: list[str] | None = None,
    methods_override: list[str] | None = None,
    seeds_override: list[int] | None = None,
    force_exact_instances: bool = False,
    resume: bool = True,
) -> Path:
    cfg = load_config(config_path)
    root = repo_root()
    prefix = str(cfg.get("project", {}).get("run_id", "run"))
    rid = run_id or _new_run_id(prefix)
    run_dir = root / "results" / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    (root / "results" / "LATEST").write_text(rid, encoding="utf-8")
    save_config_snapshot(cfg, run_dir / "config.yaml")

    productivity, history, forecast, demand, input_files = _load_inputs(cfg, root)
    manifest = environment_metadata(root, input_files)
    manifest.update({"run_id": rid, "config": str(cfg.get("_config_path"))})
    write_json(manifest, run_dir / "manifest.json")

    ecfg = cfg.get("experiment", {})
    methods = methods_override or [str(x) for x in ecfg.get("methods", ["nsga2"])]
    instances = instances_override or [str(x) for x in ecfg.get("instances", CATALOG["instance"].tolist())]
    seeds = seeds_override or [int(x) for x in ecfg.get("seeds", [int(cfg.get("project", {}).get("seed", 2026))])]
    exact_instances = set(str(x) for x in ecfg.get("exact_instances", []))
    if force_exact_instances:
        exact_instances.update(instances)
    base_seed = int(cfg.get("project", {}).get("seed", 2026))

    catalog = CATALOG.set_index("instance")
    unknown = [x for x in instances if x not in catalog.index]
    if unknown:
        raise ValueError(f"Unknown benchmark instances: {unknown}")

    run_rows: list[dict] = []
    for instance in instances:
        row = catalog.loc[instance].copy()
        row["instance"] = instance
        data, instance_meta = make_instance(row, productivity, forecast, history, demand=demand)
        write_json(instance_meta, run_dir / instance / "instance_metadata.json")

        if "exact" in methods and instance in exact_instances:
            method = "exact"
            seed = base_seed
            out = _method_directory(run_dir, instance, method, seed)
            out.mkdir(parents=True, exist_ok=True)
            prior = _completed_run(out, method) if resume else None
            if prior is not None:
                run_rows.append({"instance": instance, "method": method, "seed": seed, "status": "ok",
                                 "runtime_s": float(prior.get("runtime_s", float("nan"))),
                                 "path": str(out.relative_to(run_dir)), "error": None, "resumed": True})
            else:
                t0 = time.perf_counter()
                try:
                    exact_cfg = ecfg.get("exact", {})
                    points = int(exact_cfg.get("points", 11))
                    time_map = exact_cfg.get("time_limit_by_instance", {}) or {}
                    time_limit = float(time_map.get(instance, exact_cfg.get("time_limit", 3600)))
                    sols = epsilon_constraint_front(
                        data,
                        np.linspace(0.0, 1.0, max(2, points)),
                        time_limit=time_limit,
                        mip_gap=float(exact_cfg.get("mip_gap", 1e-4)),
                        threads=int(exact_cfg.get("threads", 1)),
                        seed=seed,
                        strong_pareto_tiebreak=True,
                    )
                    save_exact_results(sols, out)
                    status = "ok"
                    error = None
                except Exception as exc:
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                    (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    if not continue_on_error:
                        raise
                runtime = time.perf_counter() - t0
                _write_method_metadata(out, instance=instance, method=method, seed=seed, status=status,
                                       runtime_s=runtime, error=error)
                run_rows.append({"instance": instance, "method": method, "seed": seed, "status": status,
                                 "runtime_s": runtime, "path": str(out.relative_to(run_dir)), "error": error,
                                 "resumed": False})

        for seed in seeds:
            for method in [m for m in methods if m != "exact"]:
                out = _method_directory(run_dir, instance, method, seed)
                out.mkdir(parents=True, exist_ok=True)
                prior = _completed_run(out, method) if resume else None
                if prior is not None:
                    run_rows.append({"instance": instance, "method": method, "seed": seed, "status": "ok",
                                     "runtime_s": float(prior.get("runtime_s", float("nan"))),
                                     "path": str(out.relative_to(run_dir)), "error": None, "resumed": True})
                    continue
                t0 = time.perf_counter()
                error = None
                try:
                    if method == "nsga2":
                        mcfg = ecfg.get("nsga2", {})
                        result = run_nsga2(
                            data,
                            population_size=int(mcfg.get("population", 80)),
                            generations=int(mcfg.get("generations", 150)),
                            seed=seed,
                        )
                        save_nsga2_results(result, out)
                    elif method in {"nsga2_qp", "hnsga2"}:
                        hcfg = ecfg.get("hybrid", {})
                        mode = "full" if method == "nsga2_qp" else "surrogate"
                        result = run_hybrid_nsga2(
                            data,
                            population_size=int(hcfg.get("population", 80)),
                            generations=int(hcfg.get("generations", 150)),
                            seed=seed,
                            mode=mode,
                            refinement_fraction=float(hcfg.get("refinement_fraction", 0.25)),
                            warmup_generations=int(hcfg.get("warmup_generations", 5)),
                            min_training_samples=int(hcfg.get("min_training_samples", 30)),
                            refinement_backend=str(hcfg.get("backend", "scip")),
                            refinement_time_limit=float(hcfg.get("refinement_time_limit", 60)),
                            refinement_threads=int(ecfg.get("exact", {}).get("threads", 1)),
                        )
                        save_hybrid_results(result, out, prefix="nsga2_qp" if mode == "full" else "hnsga2")
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    status = "ok"
                except Exception as exc:
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                    (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    if not continue_on_error:
                        raise
                runtime = time.perf_counter() - t0
                _write_method_metadata(out, instance=instance, method=method, seed=seed, status=status,
                                       runtime_s=runtime, error=error)
                run_rows.append({"instance": instance, "method": method, "seed": seed, "status": status,
                                 "runtime_s": runtime, "path": str(out.relative_to(run_dir)), "error": error, "resumed": False})

    pd.DataFrame(run_rows).to_csv(run_dir / "run_index.csv", index=False)
    return run_dir
