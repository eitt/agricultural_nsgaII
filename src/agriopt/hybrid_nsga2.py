from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import csv
import math
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .model import ProblemData
from .nsga2 import (
    Chromosome, NSGA2Result, infer_max_slots, initialize_population, decode, rank_and_crowding,
    tournament, crossover, mutate, environmental_selection,
)
from .refinement import (
    schedule_from_genes, baseline_area_for_schedule, refine_schedule_scip,
    refine_schedule_slsqp, apply_refined_events_to_area_frac,
)


@dataclass
class HybridResult(NSGA2Result):
    refinement_log: list[dict] | None = None
    surrogate_log: list[dict] | None = None


def _feature_vector(data: ProblemData, ind: Chromosome, generation: int, generations: int) -> np.ndarray:
    events = ind.events or []
    if events:
        crops = np.array([int(e["crop"]) for e in events], dtype=int)
        prices = np.array([float(e["price_cop_kg"]) for e in events], dtype=float)
        yields = np.array([float(data.event_yield(int(e["lot"]), int(e["crop"]), int(e["sow"]))) for e in events])
        areas = np.array([float(e["area_m2"]) for e in events], dtype=float)
        fams = {str(data.botanical_family[k]) for k in crops.tolist()}
        farms = {int(data.farm_ids[int(e["lot"])]) for e in events}
        municipalities = {data.municipalities[int(e["lot"])] for e in events}
        vol = np.sqrt(np.maximum(0.0, np.diag(data.covariance)[crops]))
        util = float(np.mean([areas[i] / data.areas[int(events[i]["lot"])] for i in range(len(events))]))
        return np.array([
            len(events), len(set(crops.tolist())), len(fams), util,
            float(np.mean(yields)), float(np.max(yields)),
            float(np.mean(prices)), float(np.max(prices)),
            float(np.mean(vol)), float(np.max(vol)),
            len(farms), len(municipalities),
            float(ind.income_mcop), float(ind.risk),
            float(ind.rank if ind.rank < 10**8 else 0),
            float(ind.crowding if np.isfinite(ind.crowding) else 1e3),
            float(generation / max(1, generations)),
        ], dtype=float)
    return np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        float(ind.income_mcop), float(ind.risk), 0, 0,
        float(generation / max(1, generations)),
    ], dtype=float)


def _refine_individual(
    data: ProblemData,
    ind: Chromosome,
    *,
    backend: str,
    time_limit: float | None,
    threads: int,
) -> tuple[float, dict]:
    schedule = schedule_from_genes(data, ind.crops, ind.waits)
    if not schedule:
        return 0.0, {"attempted": True, "success": True, "useful": False, "status": "EMPTY", "gain": 0.0}
    baseline_area = baseline_area_for_schedule(data, schedule, ind.area_frac)
    income0, risk0 = float(ind.income_mcop), float(ind.risk)
    if backend == "scip":
        rr = refine_schedule_scip(
            data, schedule, baseline_area, income_floor=income0,
            time_limit=time_limit, verbose=False, threads=threads,
        )
    elif backend == "slsqp":
        rr = refine_schedule_slsqp(data, schedule, baseline_area, income_floor=income0)
    else:
        raise ValueError("refinement backend must be 'scip' or 'slsqp'")

    # A conditional refinement is accepted only when it preserves income and does not worsen risk.
    tol_i = 1e-8 * max(1.0, abs(income0))
    tol_r = 1e-8 * max(1.0, abs(risk0))
    accepted = rr.success and rr.income_mcop >= income0 - tol_i and rr.risk <= risk0 + tol_r
    if accepted:
        ind.income_mcop = float(rr.income_mcop)
        ind.risk = float(rr.risk)
        ind.objectives = np.array([-ind.income_mcop, ind.risk], dtype=float)
        ind.events = [dict(x) for x in rr.events]
        apply_refined_events_to_area_frac(data, ind.area_frac, rr.events)
    relative_risk_gain = max(0.0, (risk0 - rr.risk) / max(1e-12, abs(risk0))) if np.isfinite(risk0) else 0.0
    relative_income_gain = max(0.0, (rr.income_mcop - income0) / max(1e-12, abs(income0))) if np.isfinite(income0) else 0.0
    gain = float(relative_risk_gain + relative_income_gain)
    useful = bool(accepted and gain > 1e-5)
    return gain, {
        "attempted": True,
        "success": bool(rr.success),
        "accepted": bool(accepted),
        "useful": useful,
        "status": rr.status,
        "gain": gain,
        "income_before": income0,
        "income_after": float(rr.income_mcop),
        "risk_before": risk0,
        "risk_after": float(rr.risk),
        "solve_time_s": float(rr.solve_time_s),
    }


def run_hybrid_nsga2(
    data: ProblemData,
    *,
    population_size: int = 80,
    generations: int = 150,
    seed: int = 2026,
    max_slots: Optional[int] = None,
    max_wait: int = 4,
    crossover_probability: float = 0.90,
    mode: str = "surrogate",  # surrogate or full
    refinement_fraction: float = 0.25,
    exploration_share: float = 0.20,
    warmup_generations: int = 5,
    min_training_samples: int = 30,
    retrain_every: int = 2,
    refinement_backend: str = "scip",
    refinement_time_limit: float | None = None,
    refinement_threads: int = 1,
) -> HybridResult:
    """Run NSGA-II with conditional exact/convex continuous refinement.

    `mode='full'` refines every offspring and provides the matheuristic ablation without
    surrogate gating.  `mode='surrogate'` trains an ExtraTrees regression surrogate on realized
    refinement gains and allocates a fixed QP budget to the most promising offspring while
    retaining a random exploration share.
    """
    if mode not in {"surrogate", "full"}:
        raise ValueError("mode must be 'surrogate' or 'full'")
    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if population_size % 2:
        population_size += 1
    rng = np.random.default_rng(seed)
    max_slots = infer_max_slots(data) if max_slots is None else int(max_slots)

    pop = initialize_population(data, rng, population_size, max_slots, max_wait)
    rank_and_crowding(pop)

    X_train: list[np.ndarray] = []
    y_train: list[float] = []
    surrogate: ExtraTreesRegressor | None = None
    history: list[dict] = []
    ref_log: list[dict] = []
    surrogate_log: list[dict] = []

    for gen in range(generations):
        children: list[Chromosome] = []
        while len(children) < population_size:
            p1, p2 = tournament(pop, rng), tournament(pop, rng)
            if rng.random() < crossover_probability:
                c1, c2 = crossover(p1, p2, rng)
            else:
                c1, c2 = p1.clone(), p2.clone()
                c1.objectives = c2.objectives = None
            mutate(c1, data, rng, max_wait)
            mutate(c2, data, rng, max_wait)
            children.append(decode(data, c1))
            if len(children) < population_size:
                children.append(decode(data, c2))

        # Cheap ranks/crowding become surrogate features before expensive refinement.
        rank_and_crowding(children)
        features = np.vstack([_feature_vector(data, c, gen, generations) for c in children])

        if mode == "full":
            selected = np.arange(population_size, dtype=int)
            scores = np.full(population_size, np.nan)
        else:
            budget = max(1, int(math.ceil(refinement_fraction * population_size)))
            can_train = len(y_train) >= min_training_samples and np.std(y_train) > 1e-12
            if can_train and (surrogate is None or gen % max(1, retrain_every) == 0):
                surrogate = ExtraTreesRegressor(
                    n_estimators=200, min_samples_leaf=2, random_state=seed + gen,
                    n_jobs=-1,
                )
                surrogate.fit(np.vstack(X_train), np.asarray(y_train, dtype=float))
            if gen < warmup_generations or surrogate is None:
                selected = rng.choice(population_size, size=min(budget, population_size), replace=False)
                scores = np.full(population_size, np.nan)
            else:
                scores = surrogate.predict(features)
                n_random = min(budget, max(1, int(round(exploration_share * budget))))
                n_exploit = max(0, budget - n_random)
                exploit = np.argsort(scores)[::-1][:n_exploit]
                remaining = np.setdiff1d(np.arange(population_size), exploit, assume_unique=False)
                explore = rng.choice(remaining, size=min(n_random, len(remaining)), replace=False)
                selected = np.unique(np.concatenate([exploit, explore])).astype(int)

        selected_set = set(int(i) for i in selected.tolist())
        useful = 0
        qp_time = 0.0
        for i, child in enumerate(children):
            if i not in selected_set:
                continue
            feat = features[i].copy()
            gain, rec = _refine_individual(
                data, child, backend=refinement_backend,
                time_limit=refinement_time_limit, threads=refinement_threads,
            )
            X_train.append(feat)
            y_train.append(gain)
            useful += int(bool(rec.get("useful", False)))
            qp_time += float(rec.get("solve_time_s", 0.0))
            ref_log.append({"generation": gen + 1, "child": i, "mode": mode, **rec})

        pop = environmental_selection(pop + children, population_size)
        fronts = rank_and_crowding(pop)
        f0 = [pop[i] for i in fronts[0]]
        history.append({
            "generation": gen + 1,
            "front_size": len(f0),
            "max_income_mcop": max(x.income_mcop for x in f0),
            "min_risk": min(x.risk for x in f0),
            "refinement_calls": len(selected_set),
            "useful_refinements": useful,
            "refinement_hit_rate": useful / max(1, len(selected_set)),
            "refinement_time_s": qp_time,
        })
        surrogate_log.append({
            "generation": gen + 1,
            "training_samples": len(y_train),
            "surrogate_active": surrogate is not None,
            "mean_predicted_gain": float(np.nanmean(scores)) if np.isfinite(scores).any() else math.nan,
        })

    fronts = rank_and_crowding(pop)
    pareto_raw = [pop[i] for i in fronts[0]]
    pareto_raw.sort(key=lambda x: (x.income_mcop, x.risk))
    pareto: list[Chromosome] = []
    seen: set[tuple[float, float]] = set()
    for ind in pareto_raw:
        key = (round(float(ind.income_mcop), 10), round(float(ind.risk), 10))
        if key not in seen:
            seen.add(key)
            pareto.append(ind)
    return HybridResult(
        pareto=pareto, population=pop, history=history, max_slots=max_slots,
        refinement_log=ref_log, surrogate_log=surrogate_log,
    )


def save_hybrid_results(result: HybridResult, out_dir: str | Path, prefix: str = "hnsga2") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{prefix}_front.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["solution_id", "income_mcop", "risk", "n_events"])
        for i, ind in enumerate(result.pareto):
            w.writerow([i, ind.income_mcop, ind.risk, len(ind.events or [])])
    event_fields = ["solution_id", "lot", "farm", "municipality", "slot", "crop", "crop_name", "sow", "harvest", "area_m2", "yield_kg_m2", "production_kg", "price_cop_kg"]
    with (out_dir / f"{prefix}_events.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=event_fields)
        w.writeheader()
        for i, ind in enumerate(result.pareto):
            for row in ind.events or []:
                w.writerow({"solution_id": i, **row})
    if result.history:
        with (out_dir / f"{prefix}_history.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(result.history[0].keys()))
            w.writeheader(); w.writerows(result.history)
    if result.refinement_log:
        keys = sorted({k for row in result.refinement_log for k in row})
        with (out_dir / f"{prefix}_refinement_log.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(result.refinement_log)
    if result.surrogate_log:
        with (out_dir / f"{prefix}_surrogate_log.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(result.surrogate_log[0].keys()))
            w.writeheader(); w.writerows(result.surrogate_log)
