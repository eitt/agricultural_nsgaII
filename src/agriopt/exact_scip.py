from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import csv
import math
import numpy as np

from .model import ProblemData


@dataclass(frozen=True)
class Event:
    idx: int
    lot: int
    crop: int
    sow: int
    harvest: int


@dataclass
class ExactSolution:
    epsilon_fraction: float
    income_mcop: float
    risk: float
    status: str
    mip_gap: float
    solve_time_s: float
    nodes: int
    events: list[dict]


def _import_scip():
    try:
        from pyscipopt import Model, quicksum
    except ImportError as exc:
        raise RuntimeError(
            "SCIP exact optimization requires PySCIPOpt. Install it with `pip install pyscipopt`; "
            "the PyPI wheels currently bundle SCIP on the supported platforms."
        ) from exc
    return Model, quicksum


def enumerate_events(data: ProblemData) -> list[Event]:
    events: list[Event] = []
    idx = 0
    for l in range(data.L):
        for k in range(data.K):
            if data.eligible is not None and not bool(data.eligible[l, k]):
                continue
            for harvest in data.candidate_harvests(k):
                h = int(harvest)
                sow = data.sow_from_harvest(k, h)
                if data.event_yield(l, k, sow) <= 0:
                    continue
                events.append(Event(idx=idx, lot=l, crop=k, sow=sow, harvest=h))
                idx += 1
    return events


def _configure_scip(model, *, verbose: bool, time_limit: Optional[float], mip_gap: float,
                    threads: int, seed: int) -> None:
    if not verbose:
        model.hideOutput()
    if time_limit is not None:
        model.setRealParam("limits/time", float(time_limit))
    model.setRealParam("limits/gap", float(mip_gap))
    if threads and threads > 0:
        try:
            model.setIntParam("parallel/maxnthreads", int(threads))
        except Exception:
            pass
    try:
        model.setIntParam("randomization/randomseedshift", int(seed))
    except Exception:
        pass


def build_scip_model(
    data: ProblemData,
    *,
    objective: str,
    income_floor: float | None = None,
    risk_cap: float | None = None,
    verbose: bool = False,
    time_limit: Optional[float] = None,
    mip_gap: float = 1e-4,
    threads: int = 1,
    seed: int = 2026,
):
    """Build the compact bi-objective crop-planning model in SCIP.

    The risk objective is represented through an epigraph variable and a convex quadratic
    constraint, allowing SCIP to solve the model as a mixed-integer quadratically constrained
    program while the model objective remains linear.
    """
    Model, quicksum = _import_scip()
    if objective not in {"max_income", "min_risk"}:
        raise ValueError("objective must be 'max_income' or 'min_risk'")

    events = enumerate_events(data)
    model = Model("agricultural_production_decisions")
    _configure_scip(model, verbose=verbose, time_limit=time_limit, mip_gap=mip_gap,
                    threads=threads, seed=seed)

    y = {}
    z = {}
    for e in events:
        y[e.idx] = model.addVar(vtype="B", name=f"y_{e.lot}_{e.crop}_{e.harvest}")
        z[e.idx] = model.addVar(lb=0.0, ub=float(data.areas[e.lot]), vtype="C",
                                name=f"z_{e.lot}_{e.crop}_{e.harvest}")
        model.addCons(z[e.idx] <= float(data.areas[e.lot]) * y[e.idx],
                      name=f"link_{e.idx}")

    # Occupancy cliques provide a strong conflict representation without pairwise Big-M logic.
    occupancy: dict[tuple[int, int], list[int]] = {}
    same_family: dict[tuple[int, object, int], list[int]] = {}
    for e in events:
        for w in data.active_weeks(e.crop, e.harvest, include_setup=True):
            occupancy.setdefault((e.lot, int(w)), []).append(e.idx)
        if data.same_family_extra_rest > 0:
            fam = data.botanical_family[e.crop]
            for w in data.same_family_active_weeks(e.crop, e.harvest):
                same_family.setdefault((e.lot, fam, int(w)), []).append(e.idx)

    for (l, w), ids in occupancy.items():
        if len(ids) > 1:
            model.addCons(quicksum(y[i] for i in ids) <= 1, name=f"occupancy_{l}_{w}")
    for (l, fam, w), ids in same_family.items():
        if len(ids) > 1:
            safe_fam = str(fam).replace(" ", "_")
            model.addCons(quicksum(y[i] for i in ids) <= 1,
                          name=f"family_{l}_{safe_fam}_{w}")

    by_crop_time: dict[tuple[int, int], list[Event]] = {
        (k, t): [] for k in range(data.K) for t in range(data.T)
    }
    for e in events:
        by_crop_time[(e.crop, e.harvest)].append(e)

    production_expr = {}
    exposure = {}
    for k in range(data.K):
        for t in range(data.T):
            evs = by_crop_time[(k, t)]
            if evs:
                prod = quicksum(float(data.event_yield(e.lot, k, e.sow)) * z[e.idx] for e in evs)
            else:
                prod = 0.0
            production_expr[(k, t)] = prod
            ex = model.addVar(lb=0.0, vtype="C", name=f"exposure_{k}_{t}")
            exposure[(k, t)] = ex
            coeff = float(data.prices[k, t] / data.monetary_scale)
            model.addCons(ex == coeff * prod, name=f"exposure_def_{k}_{t}")

    if data.demand is not None:
        for g in range(data.G):
            crops = np.flatnonzero(data.sales_group == g).tolist()
            for t in range(data.T):
                model.addCons(
                    quicksum(production_expr[(k, t)] for k in crops) <= float(data.demand[g, t]),
                    name=f"demand_{g}_{t}",
                )

    income_expr = quicksum(exposure[(k, t)] for k in range(data.K) for t in range(data.T))

    if income_floor is not None:
        model.addCons(income_expr >= float(income_floor), name="epsilon_income_floor")

    # Direct mean-variance representation; Sigma is symmetrized and PSD-regularized in ProblemData.
    risk_terms = []
    sigma = data.covariance
    for t in range(data.T):
        for k in range(data.K):
            c = float(sigma[k, k])
            if abs(c) > 1e-16:
                risk_terms.append(c * exposure[(k, t)] * exposure[(k, t)])
            for j in range(k + 1, data.K):
                c = 2.0 * float(sigma[k, j])
                if abs(c) > 1e-16:
                    risk_terms.append(c * exposure[(k, t)] * exposure[(j, t)])
    risk_expr = quicksum(risk_terms) if risk_terms else 0.0

    risk_var = None
    if objective == "min_risk":
        risk_var = model.addVar(lb=0.0, vtype="C", name="portfolio_risk")
        model.addCons(risk_var >= risk_expr, name="risk_epigraph")
        model.setObjective(risk_var, "minimize")
    else:
        if risk_cap is not None:
            model.addCons(risk_expr <= float(risk_cap), name="risk_cap")
        model.setObjective(income_expr, "maximize")

    return model, events, y, z, exposure, income_expr, risk_expr, risk_var


def _has_solution(model) -> bool:
    try:
        return int(model.getNSols()) > 0
    except Exception:
        return model.getBestSol() is not None


def _extract(data: ProblemData, events: list[Event], z, exposure, model) -> tuple[float, float, list[dict]]:
    sol = model.getBestSol()
    if sol is None:
        return math.nan, math.nan, []

    e_mat = np.zeros((data.K, data.T), dtype=float)
    for k in range(data.K):
        for t in range(data.T):
            e_mat[k, t] = float(model.getSolVal(sol, exposure[(k, t)]))
    income = float(e_mat.sum())
    risk = float(sum(e_mat[:, t] @ data.covariance @ e_mat[:, t] for t in range(data.T)))

    rows: list[dict] = []
    for e in events:
        area = float(model.getSolVal(sol, z[e.idx]))
        if area <= 1e-8:
            continue
        yld = float(data.event_yield(e.lot, e.crop, e.sow))
        qty = yld * area
        rows.append({
            "lot": e.lot,
            "farm": int(data.farm_ids[e.lot]),
            "municipality": data.municipalities[e.lot],
            "crop": e.crop,
            "crop_name": data.crop_names[e.crop],
            "sow": e.sow,
            "harvest": e.harvest,
            "area_m2": area,
            "yield_kg_m2": yld,
            "production_kg": qty,
            "price_cop_kg": float(data.prices[e.crop, e.harvest]),
        })
    return income, risk, rows


def _solve_metadata(model) -> tuple[str, float, float, int]:
    status = str(model.getStatus())
    try:
        gap = float(model.getGap()) if _has_solution(model) else math.nan
    except Exception:
        gap = math.nan
    try:
        solve_time = float(model.getSolvingTime())
    except Exception:
        solve_time = math.nan
    try:
        nodes = int(model.getNNodes())
    except Exception:
        nodes = -1
    return status, gap, solve_time, nodes


def _solve_single(
    data: ProblemData,
    *,
    objective: str,
    income_floor: float | None = None,
    risk_cap: float | None = None,
    verbose: bool = False,
    time_limit: Optional[float] = None,
    mip_gap: float = 1e-4,
    threads: int = 1,
    seed: int = 2026,
):
    built = build_scip_model(
        data,
        objective=objective,
        income_floor=income_floor,
        risk_cap=risk_cap,
        verbose=verbose,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        seed=seed,
    )
    model, events, y, z, exposure, income_expr, risk_expr, risk_var = built
    model.optimize()
    status, gap, solve_time, nodes = _solve_metadata(model)
    income, risk, rows = _extract(data, events, z, exposure, model) if _has_solution(model) else (math.nan, math.nan, [])
    return {
        "model": model,
        "status": status,
        "gap": gap,
        "solve_time": solve_time,
        "nodes": nodes,
        "income": income,
        "risk": risk,
        "events": rows,
    }


def epsilon_constraint_front(
    data: ProblemData,
    income_fractions: Iterable[float] = np.linspace(0.0, 1.0, 11),
    *,
    verbose: bool = False,
    time_limit: Optional[float] = None,
    mip_gap: float = 1e-4,
    threads: int = 1,
    seed: int = 2026,
    strong_pareto_tiebreak: bool = True,
) -> list[ExactSolution]:
    """Generate a bi-objective reference front with SCIP and epsilon-constraint.

    Each epsilon point is solved in a fresh SCIP model, which avoids stateful model changes after
    transformation and makes the run robust across PySCIPOpt versions.  When requested, a second
    lexicographic solve maximizes income subject to the attained risk level, producing a strongly
    efficient representative whenever the risk solve has a feasible incumbent.
    """
    max_income = _solve_single(
        data, objective="max_income", verbose=verbose, time_limit=time_limit,
        mip_gap=mip_gap, threads=threads, seed=seed,
    )
    if not np.isfinite(max_income["income"]):
        raise RuntimeError(f"SCIP could not obtain an income-maximizing incumbent; status={max_income['status']}")
    income_max = float(max_income["income"])

    out: list[ExactSolution] = []
    for j, eta_raw in enumerate(income_fractions):
        eta = float(eta_raw)
        if not 0.0 <= eta <= 1.0:
            raise ValueError("income_fractions must lie in [0,1]")
        floor = eta * income_max
        r = _solve_single(
            data, objective="min_risk", income_floor=floor, verbose=verbose,
            time_limit=time_limit, mip_gap=mip_gap, threads=threads, seed=seed + j + 1,
        )
        if not np.isfinite(r["risk"]):
            out.append(ExactSolution(eta, math.nan, math.nan, r["status"], r["gap"],
                                     r["solve_time"], r["nodes"], []))
            continue

        if strong_pareto_tiebreak:
            tol = max(1e-9, 1e-7 * max(1.0, abs(float(r["risk"]))))
            tiebreak = _solve_single(
                data, objective="max_income", income_floor=floor,
                risk_cap=float(r["risk"]) + tol, verbose=verbose,
                time_limit=time_limit, mip_gap=mip_gap, threads=threads,
                seed=seed + 1000 + j,
            )
            if np.isfinite(tiebreak["income"]):
                r = tiebreak

        out.append(ExactSolution(
            eta,
            float(r["income"]),
            float(r["risk"]),
            str(r["status"]),
            float(r["gap"]),
            float(r["solve_time"]),
            int(r["nodes"]),
            list(r["events"]),
        ))
    return out


def save_exact_results(solutions: list[ExactSolution], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "exact_front_scip.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "solution_id", "income_fraction", "income_mcop", "risk", "status",
            "mip_gap", "solve_time_s", "nodes",
        ])
        for i, s in enumerate(solutions):
            w.writerow([
                i, s.epsilon_fraction, s.income_mcop, s.risk, s.status,
                s.mip_gap, s.solve_time_s, s.nodes,
            ])
    fields = [
        "solution_id", "lot", "farm", "municipality", "crop", "crop_name", "sow",
        "harvest", "area_m2", "yield_kg_m2", "production_kg", "price_cop_kg",
    ]
    with (out_dir / "exact_events_scip.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, s in enumerate(solutions):
            for row in s.events:
                w.writerow({"solution_id": i, **row})
