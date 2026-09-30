from __future__ import annotations

from collections import defaultdict
import numpy as np

from .model import ProblemData, objectives_from_production


def validate_events(data: ProblemData, events: list[dict], tol: float = 1e-7) -> dict:
    """Validate a decoded/event-list solution against the compact model logic."""
    errors: list[str] = []
    production = np.zeros((data.K, data.T), dtype=float)
    by_lot: dict[int, list[dict]] = defaultdict(list)

    for e in events:
        l = int(e["lot"]); k = int(e["crop"]); s = int(e["sow"]); h = int(e["harvest"])
        area = float(e["area_m2"])
        if not (0 <= l < data.L and 0 <= k < data.K):
            errors.append(f"invalid indices in event {e}")
            continue
        if h != s + int(data.maturity[k]):
            errors.append(f"maturity mismatch lot={l}, crop={k}, sow={s}, harvest={h}")
        if h >= data.T or s < 0:
            errors.append(f"event outside horizon: {e}")
        if not data.can_sow(k, s):
            errors.append(f"forbidden sow week: {e}")
        if area < -tol or area > float(data.areas[l]) + tol:
            errors.append(f"area bound violated: {e}")
        qty = float(data.event_yield(l, k, s) * area)
        production[k, h] += qty
        by_lot[l].append(e)

    for l, seq in by_lot.items():
        seq = sorted(seq, key=lambda x: int(x["sow"]))
        for a, b in zip(seq, seq[1:]):
            ka, kb = int(a["crop"]), int(b["crop"])
            rest = data.setup_weeks
            if data.botanical_family[ka] == data.botanical_family[kb]:
                rest += data.same_family_extra_rest
            if int(b["sow"]) < int(a["harvest"]) + rest:
                errors.append(f"overlap/rest violation lot={l}: {a} -> {b}")

    if data.demand is not None:
        for g in range(data.G):
            crops = np.flatnonzero(data.sales_group == g)
            agg = production[crops].sum(axis=0)
            bad = np.flatnonzero(agg > data.demand[g] + tol)
            for t in bad:
                errors.append(f"demand exceeded group={g}, week={int(t)}")

    income, risk = objectives_from_production(data, production)
    return {"ok": not errors, "errors": errors, "income_mcop": income, "risk": risk, "production": production}
