from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional
import numpy as np
from scipy.optimize import minimize

from .model import ProblemData, objectives_from_production


@dataclass(frozen=True)
class ScheduleEvent:
    idx: int
    lot: int
    slot: int
    crop: int
    sow: int
    harvest: int


@dataclass
class RefinementResult:
    success: bool
    status: str
    income_mcop: float
    risk: float
    events: list[dict]
    solve_time_s: float = math.nan


def schedule_from_genes(data: ProblemData, crops: np.ndarray, waits: np.ndarray) -> list[ScheduleEvent]:
    """Decode only discrete schedule decisions while preserving agronomic sequencing."""
    events: list[ScheduleEvent] = []
    idx = 0
    for l in range(data.L):
        prev_crop: Optional[int] = None
        prev_harvest = -int(data.setup_weeks)
        for slot in range(crops.shape[1]):
            k = int(crops[l, slot])
            if k < 0:
                continue
            if data.eligible is not None and not bool(data.eligible[l, k]):
                continue
            extra = 0
            if prev_crop is not None and data.botanical_family[prev_crop] == data.botanical_family[k]:
                extra = int(data.same_family_extra_rest)
            earliest = max(0, prev_harvest + int(data.setup_weeks) + extra)
            target = earliest + int(max(0, waits[l, slot]))
            sow = data.next_allowed_sow(k, target)
            if sow is None:
                continue
            harvest = int(sow + data.maturity[k])
            if harvest >= data.T:
                continue
            if data.event_yield(l, k, sow) <= 0:
                continue
            events.append(ScheduleEvent(idx, l, slot, k, sow, harvest))
            idx += 1
            prev_crop = k
            prev_harvest = harvest
    return events


def _schedule_matrices(data: ProblemData, schedule: list[ScheduleEvent]):
    n = len(schedule)
    c = np.zeros(n, dtype=float)  # income per m2 in MCOP
    ub = np.zeros(n, dtype=float)
    yld = np.zeros(n, dtype=float)
    for i, e in enumerate(schedule):
        yld[i] = data.event_yield(e.lot, e.crop, e.sow)
        c[i] = yld[i] * float(data.prices[e.crop, e.harvest]) / float(data.monetary_scale)
        ub[i] = float(data.areas[e.lot])

    # Q satisfies risk = a'Qa for area vector a.
    Q = np.zeros((n, n), dtype=float)
    for i, ei in enumerate(schedule):
        for j in range(i, n):
            ej = schedule[j]
            if ei.harvest != ej.harvest:
                continue
            q = c[i] * c[j] * float(data.covariance[ei.crop, ej.crop])
            Q[i, j] = q
            Q[j, i] = q
    Q = 0.5 * (Q + Q.T)
    mineig = float(np.linalg.eigvalsh(Q).min()) if n else 0.0
    if mineig < -1e-10:
        Q += np.eye(n) * (-mineig + 1e-10)

    A = []
    b = []
    if data.demand is not None:
        for g in range(data.G):
            for t in range(data.T):
                row = np.zeros(n, dtype=float)
                for i, e in enumerate(schedule):
                    if e.harvest == t and int(data.sales_group[e.crop]) == g:
                        row[i] = yld[i]
                if np.any(row):
                    A.append(row)
                    b.append(float(data.demand[g, t]))
    return c, Q, ub, yld, (np.vstack(A) if A else np.zeros((0, n))), np.asarray(b, dtype=float)


def _events_from_area(data: ProblemData, schedule: list[ScheduleEvent], area: np.ndarray) -> tuple[float, float, list[dict]]:
    production = np.zeros((data.K, data.T), dtype=float)
    rows: list[dict] = []
    for i, e in enumerate(schedule):
        a = float(area[i])
        if a <= 1e-8:
            continue
        yld = float(data.event_yield(e.lot, e.crop, e.sow))
        qty = yld * a
        production[e.crop, e.harvest] += qty
        rows.append({
            "lot": e.lot,
            "slot": e.slot,
            "farm": int(data.farm_ids[e.lot]),
            "municipality": data.municipalities[e.lot],
            "crop": e.crop,
            "crop_name": data.crop_names[e.crop],
            "sow": e.sow,
            "harvest": e.harvest,
            "area_m2": a,
            "yield_kg_m2": yld,
            "production_kg": qty,
            "price_cop_kg": float(data.prices[e.crop, e.harvest]),
        })
    income, risk = objectives_from_production(data, production)
    return income, risk, rows


def baseline_area_for_schedule(data: ProblemData, schedule: list[ScheduleEvent], area_frac: np.ndarray) -> np.ndarray:
    area = np.zeros(len(schedule), dtype=float)
    residual = None if data.demand is None else data.demand.copy()
    for i, e in enumerate(schedule):
        a = float(data.areas[e.lot] * np.clip(area_frac[e.lot, e.slot], 0.0, 1.0))
        yld = float(data.event_yield(e.lot, e.crop, e.sow))
        if residual is not None:
            g = int(data.sales_group[e.crop])
            rem = max(0.0, float(residual[g, e.harvest]))
            a = min(a, rem / yld if yld > 0 else 0.0)
            residual[g, e.harvest] -= yld * a
        area[i] = max(0.0, a)
    return area


def evaluate_schedule_baseline(data: ProblemData, schedule: list[ScheduleEvent], area_frac: np.ndarray):
    area = baseline_area_for_schedule(data, schedule, area_frac)
    return (*_events_from_area(data, schedule, area), area)


def refine_schedule_slsqp(
    data: ProblemData,
    schedule: list[ScheduleEvent],
    baseline_area: np.ndarray,
    *,
    income_floor: float | None = None,
    maxiter: int = 500,
    ftol: float = 1e-10,
) -> RefinementResult:
    """Convex continuous area refinement used as a solver-independent development fallback."""
    import time
    t0 = time.perf_counter()
    if not schedule:
        return RefinementResult(True, "EMPTY", 0.0, 0.0, [], time.perf_counter() - t0)
    c, Q, ub, yld, A, b = _schedule_matrices(data, schedule)
    x0 = np.clip(np.asarray(baseline_area, dtype=float), 0.0, ub)
    if income_floor is None:
        income_floor = float(c @ x0)

    cons = [{"type": "ineq", "fun": lambda x, c=c, f=float(income_floor): float(c @ x - f),
             "jac": lambda x, c=c: c}]
    if A.shape[0]:
        cons.append({"type": "ineq", "fun": lambda x, A=A, b=b: b - A @ x,
                     "jac": lambda x, A=A, b=b: -A})

    def fun(x):
        return float(x @ Q @ x)

    def jac(x):
        return 2.0 * (Q @ x)

    res = minimize(fun, x0, jac=jac, bounds=[(0.0, float(u)) for u in ub], constraints=cons,
                   method="SLSQP", options={"maxiter": int(maxiter), "ftol": float(ftol), "disp": False})
    if not res.success:
        inc0, risk0, rows0 = _events_from_area(data, schedule, x0)
        return RefinementResult(False, f"SLSQP:{res.message}", inc0, risk0, rows0, time.perf_counter() - t0)
    income, risk, rows = _events_from_area(data, schedule, np.asarray(res.x, dtype=float))
    return RefinementResult(True, "SLSQP:OPTIMAL", income, risk, rows, time.perf_counter() - t0)


def refine_schedule_scip(
    data: ProblemData,
    schedule: list[ScheduleEvent],
    baseline_area: np.ndarray,
    *,
    income_floor: float | None = None,
    time_limit: float | None = None,
    gap: float = 1e-7,
    verbose: bool = False,
    threads: int = 1,
) -> RefinementResult:
    """Convex QP area refinement using SCIP through PySCIPOpt."""
    import time
    t0 = time.perf_counter()
    try:
        from pyscipopt import Model, quicksum
    except ImportError as exc:
        raise RuntimeError("SCIP refinement requires `pip install pyscipopt`.") from exc

    if not schedule:
        return RefinementResult(True, "EMPTY", 0.0, 0.0, [], time.perf_counter() - t0)
    c, Q, ub, yld, A, b = _schedule_matrices(data, schedule)
    x0 = np.clip(np.asarray(baseline_area, dtype=float), 0.0, ub)
    if income_floor is None:
        income_floor = float(c @ x0)

    m = Model("conditional_area_refinement")
    if not verbose:
        m.hideOutput()
    if time_limit is not None:
        m.setRealParam("limits/time", float(time_limit))
    m.setRealParam("limits/gap", float(gap))
    try:
        m.setIntParam("parallel/maxnthreads", int(threads))
    except Exception:
        pass

    a = [m.addVar(lb=0.0, ub=float(ub[i]), vtype="C", name=f"a_{i}") for i in range(len(schedule))]
    income_expr = quicksum(float(c[i]) * a[i] for i in range(len(schedule)))
    m.addCons(income_expr >= float(income_floor), name="income_floor")
    for r in range(A.shape[0]):
        nz = np.flatnonzero(np.abs(A[r]) > 0)
        m.addCons(quicksum(float(A[r, i]) * a[i] for i in nz) <= float(b[r]), name=f"demand_{r}")

    risk_terms = []
    for i in range(len(schedule)):
        if abs(Q[i, i]) > 1e-16:
            risk_terms.append(float(Q[i, i]) * a[i] * a[i])
        for j in range(i + 1, len(schedule)):
            if abs(Q[i, j]) > 1e-16:
                risk_terms.append(2.0 * float(Q[i, j]) * a[i] * a[j])
    risk_expr = quicksum(risk_terms) if risk_terms else 0.0
    q = m.addVar(lb=0.0, vtype="C", name="risk")
    m.addCons(q >= risk_expr, name="risk_epigraph")
    m.setObjective(q, "minimize")
    m.optimize()

    sol = m.getBestSol()
    if sol is None:
        inc0, risk0, rows0 = _events_from_area(data, schedule, x0)
        return RefinementResult(False, f"SCIP:{m.getStatus()}", inc0, risk0, rows0, time.perf_counter() - t0)
    area = np.array([float(m.getSolVal(sol, a[i])) for i in range(len(schedule))], dtype=float)
    income, risk, rows = _events_from_area(data, schedule, area)
    return RefinementResult(True, f"SCIP:{m.getStatus()}", income, risk, rows, time.perf_counter() - t0)


def apply_refined_events_to_area_frac(data: ProblemData, area_frac: np.ndarray, events: list[dict]) -> None:
    """Write refined continuous decisions back into the chromosome for inheritance."""
    area_frac[:] = 0.0
    for e in events:
        l = int(e["lot"])
        slot = int(e.get("slot", 0))
        if 0 <= l < area_frac.shape[0] and 0 <= slot < area_frac.shape[1]:
            area_frac[l, slot] = float(np.clip(float(e["area_m2"]) / data.areas[l], 0.0, 1.0))
