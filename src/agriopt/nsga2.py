from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import csv
import math
import numpy as np

from .model import ProblemData, objectives_from_production


@dataclass
class Chromosome:
    crops: np.ndarray   # (L,S), -1 means unused slot
    waits: np.ndarray   # (L,S), nonnegative integer waiting weeks
    area_frac: np.ndarray  # (L,S), [0,1]
    objectives: Optional[np.ndarray] = None  # minimization vector [-income, risk]
    income_mcop: float = np.nan
    risk: float = np.nan
    events: Optional[list[dict]] = None
    rank: int = 10**9
    crowding: float = 0.0

    def clone(self) -> "Chromosome":
        return Chromosome(
            crops=self.crops.copy(),
            waits=self.waits.copy(),
            area_frac=self.area_frac.copy(),
            objectives=None if self.objectives is None else self.objectives.copy(),
            income_mcop=self.income_mcop,
            risk=self.risk,
            events=None if self.events is None else [dict(x) for x in self.events],
            rank=self.rank,
            crowding=self.crowding,
        )


@dataclass
class NSGA2Result:
    pareto: list[Chromosome]
    population: list[Chromosome]
    history: list[dict]
    max_slots: int


def infer_max_slots(data: ProblemData) -> int:
    min_cycle = int(np.min(data.maturity)) + int(data.setup_weeks)
    return max(2, int(math.ceil(data.T / max(1, min_cycle))) + 2)


def random_chromosome(
    data: ProblemData,
    rng: np.random.Generator,
    max_slots: int,
    max_wait: int,
    empty_probability: float = 0.10,
) -> Chromosome:
    crops = rng.integers(0, data.K, size=(data.L, max_slots), endpoint=False)
    empty = rng.random((data.L, max_slots)) < empty_probability
    crops = crops.astype(int)
    crops[empty] = -1
    waits = rng.integers(0, max_wait + 1, size=(data.L, max_slots), endpoint=False).astype(int)
    # Bias away from zero area but retain enough continuous variation.
    area_frac = rng.beta(2.0, 1.4, size=(data.L, max_slots))
    return Chromosome(crops=crops, waits=waits, area_frac=area_frac)


def initialize_population(
    data: ProblemData,
    rng: np.random.Generator,
    population_size: int,
    max_slots: int,
    max_wait: int,
) -> list[Chromosome]:
    """Create the common initial population used by every evolutionary variant.

    The zero-production boundary point is retained because it is a natural feasible
    extreme.  No method receives an additional hand-crafted high-production point.
    Keeping this logic in one function also guarantees identical random-number
    consumption when methods start from the same seed.
    """
    population = [
        decode(data, random_chromosome(data, rng, max_slots, max_wait))
        for _ in range(population_size)
    ]
    zero = Chromosome(
        crops=np.full((data.L, max_slots), -1, dtype=int),
        waits=np.zeros((data.L, max_slots), dtype=int),
        area_frac=np.zeros((data.L, max_slots), dtype=float),
    )
    population[0] = decode(data, zero)
    return population


def decode(data: ProblemData, ind: Chromosome) -> Chromosome:
    production = np.zeros((data.K, data.T), dtype=float)
    residual_demand = None if data.demand is None else data.demand.copy()
    events: list[dict] = []

    for l in range(data.L):
        prev_crop: Optional[int] = None
        prev_harvest = -data.setup_weeks

        for s in range(ind.crops.shape[1]):
            k = int(ind.crops[l, s])
            if k < 0:
                continue
            if data.eligible is not None and not bool(data.eligible[l,k]):
                continue

            extra = 0
            if prev_crop is not None and data.botanical_family[prev_crop] == data.botanical_family[k]:
                extra = int(data.same_family_extra_rest)
            earliest = max(0, prev_harvest + int(data.setup_weeks) + extra)
            target = earliest + int(ind.waits[l, s])
            sow = data.next_allowed_sow(k, target)
            if sow is None:
                continue
            harvest = int(sow + data.maturity[k])
            if harvest >= data.T:
                continue

            frac = float(np.clip(ind.area_frac[l, s], 0.0, 1.0))
            area = float(data.areas[l] * frac)
            yld = data.event_yield(l,k,sow)
            if area <= 1e-10 or yld <= 0:
                continue

            if residual_demand is not None:
                g = int(data.sales_group[k])
                rem = max(0.0, float(residual_demand[g, harvest]))
                area = min(area, rem / yld)
                if area <= 1e-10:
                    continue

            qty = yld * area
            production[k, harvest] += qty
            if residual_demand is not None:
                residual_demand[int(data.sales_group[k]), harvest] -= qty

            events.append({
                "lot": l,
                "slot": s,
                "crop": k,
                "crop_name": data.crop_names[k],
                "sow": sow,
                "harvest": harvest,
                "area_m2": area,
                "production_kg": qty,
                "price_cop_kg": float(data.prices[k, harvest]),
            })
            prev_crop = k
            prev_harvest = harvest

    income, risk = objectives_from_production(data, production)
    ind.income_mcop = income
    ind.risk = risk
    ind.objectives = np.array([-income, risk], dtype=float)
    ind.events = events
    return ind


def dominates(a: np.ndarray, b: np.ndarray, tol: float = 1e-12) -> bool:
    return bool(np.all(a <= b + tol) and np.any(a < b - tol))


def fast_non_dominated_sort(pop: list[Chromosome]) -> list[list[int]]:
    n = len(pop)
    dominated: list[list[int]] = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    first: list[int] = []

    for p in range(n):
        op = pop[p].objectives
        if op is None:
            raise ValueError("population contains unevaluated individual")
        for q in range(p + 1, n):
            oq = pop[q].objectives
            if dominates(op, oq):
                dominated[p].append(q)
                domination_count[q] += 1
            elif dominates(oq, op):
                dominated[q].append(p)
                domination_count[p] += 1

    for p in range(n):
        if domination_count[p] == 0:
            pop[p].rank = 0
            first.append(p)

    fronts: list[list[int]] = []
    current = first
    rank = 0
    while current:
        fronts.append(current)
        nxt: list[int] = []
        for p in current:
            for q in dominated[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    pop[q].rank = rank + 1
                    nxt.append(q)
        rank += 1
        current = nxt
    return fronts


def assign_crowding(pop: list[Chromosome], front: list[int]) -> None:
    if not front:
        return
    for i in front:
        pop[i].crowding = 0.0
    if len(front) <= 2:
        for i in front:
            pop[i].crowding = math.inf
        return

    obj = np.array([pop[i].objectives for i in front], dtype=float)
    m = obj.shape[1]
    for j in range(m):
        order = np.argsort(obj[:, j])
        pop[front[int(order[0])]].crowding = math.inf
        pop[front[int(order[-1])]].crowding = math.inf
        lo = float(obj[order[0], j])
        hi = float(obj[order[-1], j])
        span = hi - lo
        if span <= 1e-15:
            continue
        for r in range(1, len(front) - 1):
            idx = front[int(order[r])]
            if math.isinf(pop[idx].crowding):
                continue
            prev_v = float(obj[order[r - 1], j])
            next_v = float(obj[order[r + 1], j])
            pop[idx].crowding += (next_v - prev_v) / span


def rank_and_crowding(pop: list[Chromosome]) -> list[list[int]]:
    fronts = fast_non_dominated_sort(pop)
    for f in fronts:
        assign_crowding(pop, f)
    return fronts


def tournament(pop: list[Chromosome], rng: np.random.Generator) -> Chromosome:
    i, j = rng.integers(0, len(pop), size=2)
    a, b = pop[int(i)], pop[int(j)]
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    if a.crowding > b.crowding:
        return a
    if b.crowding > a.crowding:
        return b
    return a if rng.random() < 0.5 else b


def crossover(a: Chromosome, b: Chromosome, rng: np.random.Generator) -> tuple[Chromosome, Chromosome]:
    mask = rng.random(a.crops.shape) < 0.5
    c1_crop = np.where(mask, a.crops, b.crops)
    c2_crop = np.where(mask, b.crops, a.crops)

    mask_w = rng.random(a.waits.shape) < 0.5
    c1_wait = np.where(mask_w, a.waits, b.waits)
    c2_wait = np.where(mask_w, b.waits, a.waits)

    alpha = rng.random(a.area_frac.shape)
    c1_area = alpha * a.area_frac + (1.0 - alpha) * b.area_frac
    c2_area = alpha * b.area_frac + (1.0 - alpha) * a.area_frac
    return (
        Chromosome(c1_crop.copy(), c1_wait.copy(), np.clip(c1_area, 0.0, 1.0)),
        Chromosome(c2_crop.copy(), c2_wait.copy(), np.clip(c2_area, 0.0, 1.0)),
    )


def mutate(
    ind: Chromosome,
    data: ProblemData,
    rng: np.random.Generator,
    max_wait: int,
    p_crop: float = 0.04,
    p_wait: float = 0.04,
    p_area: float = 0.08,
    area_sigma: float = 0.15,
) -> None:
    crop_mask = rng.random(ind.crops.shape) < p_crop
    if np.any(crop_mask):
        # -1 plus K crop choices.
        new = rng.integers(-1, data.K, size=int(crop_mask.sum()), endpoint=False)
        ind.crops[crop_mask] = new

    wait_mask = rng.random(ind.waits.shape) < p_wait
    if np.any(wait_mask):
        delta = rng.integers(-2, 3, size=int(wait_mask.sum()))
        vals = ind.waits[wait_mask] + delta
        ind.waits[wait_mask] = np.clip(vals, 0, max_wait)

    area_mask = rng.random(ind.area_frac.shape) < p_area
    if np.any(area_mask):
        ind.area_frac[area_mask] += rng.normal(0.0, area_sigma, size=int(area_mask.sum()))
        ind.area_frac[:] = np.clip(ind.area_frac, 0.0, 1.0)

    ind.objectives = None
    ind.events = None


def environmental_selection(combined: list[Chromosome], n_keep: int) -> list[Chromosome]:
    fronts = rank_and_crowding(combined)
    selected: list[Chromosome] = []
    for front in fronts:
        if len(selected) + len(front) <= n_keep:
            selected.extend(combined[i] for i in front)
        else:
            rem = n_keep - len(selected)
            order = sorted(front, key=lambda i: combined[i].crowding, reverse=True)
            selected.extend(combined[i] for i in order[:rem])
            break
    return selected


def run_nsga2(
    data: ProblemData,
    *,
    population_size: int = 80,
    generations: int = 150,
    seed: int = 2026,
    max_slots: Optional[int] = None,
    max_wait: int = 4,
    crossover_probability: float = 0.90,
) -> NSGA2Result:
    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if population_size % 2:
        population_size += 1
    rng = np.random.default_rng(seed)
    max_slots = infer_max_slots(data) if max_slots is None else int(max_slots)

    pop = initialize_population(data, rng, population_size, max_slots, max_wait)
    rank_and_crowding(pop)
    history: list[dict] = []

    for gen in range(generations):
        children: list[Chromosome] = []
        while len(children) < population_size:
            p1 = tournament(pop, rng)
            p2 = tournament(pop, rng)
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

        pop = environmental_selection(pop + children, population_size)
        fronts = rank_and_crowding(pop)
        f0 = [pop[i] for i in fronts[0]]
        history.append({
            "generation": gen + 1,
            "front_size": len(f0),
            "max_income_mcop": max(x.income_mcop for x in f0),
            "min_risk": min(x.risk for x in f0),
        })

    fronts = rank_and_crowding(pop)
    pareto_raw = [pop[i] for i in fronts[0]]
    pareto_raw.sort(key=lambda x: (x.income_mcop, x.risk))
    # One representative per objective pair; NSGA-II can retain genetically distinct
    # schedules that map to the same decoded point (especially the zero-production extreme).
    pareto: list[Chromosome] = []
    seen: set[tuple[float, float]] = set()
    for ind in pareto_raw:
        key = (round(float(ind.income_mcop), 10), round(float(ind.risk), 10))
        if key not in seen:
            seen.add(key)
            pareto.append(ind)
    return NSGA2Result(pareto=pareto, population=pop, history=history, max_slots=max_slots)


def save_nsga2_results(result: NSGA2Result, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "nsga2_front.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["solution_id", "income_mcop", "risk", "n_events"])
        for i, ind in enumerate(result.pareto):
            w.writerow([i, ind.income_mcop, ind.risk, len(ind.events or [])])

    fields = ["solution_id", "lot", "slot", "crop", "crop_name", "sow", "harvest", "area_m2", "production_kg", "price_cop_kg"]
    with (out_dir / "nsga2_events.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, ind in enumerate(result.pareto):
            for row in ind.events or []:
                w.writerow({"solution_id": i, **row})

    with (out_dir / "nsga2_history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["generation", "front_size", "max_income_mcop", "min_risk"])
        w.writeheader()
        w.writerows(result.history)
