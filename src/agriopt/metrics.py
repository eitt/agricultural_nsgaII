from __future__ import annotations
import numpy as np


def as_minimization_front(income, risk):
    income = np.asarray(income, dtype=float)
    risk = np.asarray(risk, dtype=float)
    return np.column_stack([-income, risk])


def nondominated(points: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    if p.ndim != 2:
        raise ValueError("points must be a two-dimensional array")
    if p.shape[0] == 0:
        return p.copy()
    p = np.unique(p[np.all(np.isfinite(p), axis=1)], axis=0)
    if p.shape[1] == 2:
        order = np.lexsort((p[:, 1], p[:, 0]))
        ordered = p[order]
        keep = np.zeros(len(ordered), dtype=bool)
        best_second = np.inf
        for i, second in enumerate(ordered[:, 1]):
            if second < best_second - tol:
                keep[i] = True
                best_second = second
        return ordered[keep]
    keep = np.ones(len(p), dtype=bool)
    for i in range(len(p)):
        if not keep[i]:
            continue
        for j in range(len(p)):
            if i == j or not keep[i]:
                continue
            if np.all(p[j] <= p[i] + tol) and np.any(p[j] < p[i] - tol):
                keep[i] = False
    return p[keep]


def reference_scaling(reference: np.ndarray):
    ref = nondominated(np.asarray(reference, dtype=float))
    if len(ref) == 0:
        raise ValueError("reference front must contain at least one finite point")
    ideal = np.nanmin(ref, axis=0)
    nadir = np.nanmax(ref, axis=0)
    span = np.where(nadir - ideal > 1e-12, nadir - ideal, 1.0)
    return ideal, span


def pooled_empirical_reference(fronts: list[np.ndarray]) -> np.ndarray:
    """Return the nondominated union of finite, duplicate-free run fronts."""
    usable = [np.asarray(front, dtype=float) for front in fronts if np.asarray(front).size]
    if not usable:
        return np.empty((0, 2), dtype=float)
    return nondominated(np.vstack(usable))


def normalize(points: np.ndarray, ideal: np.ndarray, span: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=float) - ideal) / span


def hypervolume_2d(points: np.ndarray, reference_point=(1.1, 1.1)) -> float:
    p = nondominated(np.asarray(points, dtype=float))
    rp = np.asarray(reference_point, dtype=float)
    p = p[np.all(p <= rp, axis=1)]
    if len(p) == 0:
        return 0.0
    p = p[np.argsort(p[:, 0])]
    hv = 0.0
    current_y = rp[1]
    for x, y in p:
        if y < current_y:
            hv += max(0.0, rp[0] - x) * max(0.0, current_y - y)
            current_y = y
    return float(hv)


def igd_plus(approximation: np.ndarray, reference: np.ndarray) -> float:
    a = nondominated(np.asarray(approximation, dtype=float))
    r = nondominated(np.asarray(reference, dtype=float))
    if len(a) == 0 or len(r) == 0:
        return float('nan')
    vals = []
    for rr in r:
        d = np.sqrt(np.sum(np.maximum(a - rr, 0.0) ** 2, axis=1))
        vals.append(float(np.min(d)))
    return float(np.mean(vals))


def additive_epsilon(approximation: np.ndarray, reference: np.ndarray) -> float:
    a = nondominated(np.asarray(approximation, dtype=float))
    r = nondominated(np.asarray(reference, dtype=float))
    if len(a) == 0 or len(r) == 0:
        return float('nan')
    return float(max(min(np.max(aa - rr) for aa in a) for rr in r))
