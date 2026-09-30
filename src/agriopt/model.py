from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ProblemData:
    crop_names: list[str]
    areas: np.ndarray              # m2 by lot
    farm_ids: np.ndarray           # integer farm for each lot
    municipalities: list[str]      # municipality per lot
    yields: np.ndarray             # L x K baseline kg/m2
    prices: np.ndarray             # K x T COP/kg
    covariance: np.ndarray         # K x K covariance of log returns
    maturity: np.ndarray           # K weeks
    botanical_family: np.ndarray   # K labels/codes
    sales_group: np.ndarray        # K ints, zero based
    sow_allowed: np.ndarray        # K x T bool
    eligible: np.ndarray | None = None          # L x K empirical eligibility
    yield_by_sow_week: np.ndarray | None = None # L x K x T kg/m2
    setup_weeks: int = 1
    same_family_extra_rest: int = 1
    demand: np.ndarray | None = None             # G x T kg
    monetary_scale: float = 1e6                  # objective unit: million COP

    def __post_init__(self) -> None:
        self.areas = np.asarray(self.areas, dtype=float)
        self.farm_ids = np.asarray(self.farm_ids, dtype=int)
        self.yields = np.asarray(self.yields, dtype=float)
        self.prices = np.asarray(self.prices, dtype=float)
        self.covariance = np.asarray(self.covariance, dtype=float)
        self.maturity = np.asarray(self.maturity, dtype=int)
        self.botanical_family = np.asarray(self.botanical_family, dtype=object)
        self.sales_group = np.asarray(self.sales_group, dtype=int)
        self.sow_allowed = np.asarray(self.sow_allowed, dtype=bool)
        if self.eligible is not None:
            self.eligible = np.asarray(self.eligible, dtype=bool)
        if self.yield_by_sow_week is not None:
            self.yield_by_sow_week = np.asarray(self.yield_by_sow_week, dtype=float)
        if self.demand is not None:
            self.demand = np.asarray(self.demand, dtype=float)

        if len(self.crop_names) != self.prices.shape[0]:
            raise ValueError("crop_names and prices must have the same crop dimension")
        if self.yields.shape != (self.L, self.K):
            raise ValueError(f"yields must have shape {(self.L, self.K)}, received {self.yields.shape}")
        if self.sow_allowed.shape != (self.K, self.T):
            raise ValueError(f"sow_allowed must have shape {(self.K, self.T)}, received {self.sow_allowed.shape}")
        if self.covariance.shape != (self.K, self.K):
            raise ValueError(f"covariance must have shape {(self.K, self.K)}, received {self.covariance.shape}")
        if len(self.maturity) != self.K or len(self.botanical_family) != self.K or len(self.sales_group) != self.K:
            raise ValueError("maturity, botanical_family, and sales_group must have one entry per crop")
        if len(self.farm_ids) != self.L or len(self.municipalities) != self.L:
            raise ValueError("farm_ids and municipalities must have one entry per lot")
        if self.eligible is not None and self.eligible.shape != (self.L, self.K):
            raise ValueError(f"eligible must have shape {(self.L, self.K)}")
        if self.yield_by_sow_week is not None and self.yield_by_sow_week.shape != (self.L, self.K, self.T):
            raise ValueError(f"yield_by_sow_week must have shape {(self.L, self.K, self.T)}")
        if self.demand is not None and self.demand.shape[1] != self.T:
            raise ValueError("demand must have T columns")
        if np.any(self.areas <= 0):
            raise ValueError("lot areas must be strictly positive")
        if np.any(self.maturity <= 0):
            raise ValueError("maturity times must be strictly positive")
        if self.setup_weeks < 0 or self.same_family_extra_rest < 0:
            raise ValueError("rest parameters must be nonnegative")
        if self.monetary_scale <= 0:
            raise ValueError("monetary_scale must be positive")

        # Numerical symmetry is useful for both the exact MIQCP and objective evaluation.
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        mineig = float(np.linalg.eigvalsh(self.covariance).min())
        if mineig < -1e-10:
            # A tiny diagonal shift preserves empirical dependence while ensuring a convex risk model.
            self.covariance = self.covariance + np.eye(self.K) * (-mineig + 1e-10)

    @property
    def L(self) -> int:
        return int(len(self.areas))

    @property
    def K(self) -> int:
        return int(len(self.crop_names))

    @property
    def T(self) -> int:
        return int(self.prices.shape[1])

    @property
    def G(self) -> int:
        if self.demand is not None:
            return int(self.demand.shape[0])
        return int(self.sales_group.max()) + 1 if self.sales_group.size else 0

    def can_sow(self, k: int, sow: int) -> bool:
        return bool(0 <= sow < self.T and self.sow_allowed[k, sow])

    def event_yield(self, l: int, k: int, sow: int) -> float:
        if self.yield_by_sow_week is not None:
            return float(self.yield_by_sow_week[l, k, sow])
        return float(self.yields[l, k])

    def next_allowed_sow(self, k: int, start: int) -> int | None:
        if start >= self.T:
            return None
        start = max(0, int(start))
        idx = np.flatnonzero(self.sow_allowed[k, start:])
        return None if len(idx) == 0 else int(start + idx[0])

    def candidate_harvests(self, k: int) -> np.ndarray:
        sow = np.flatnonzero(self.sow_allowed[k])
        harvest = sow + int(self.maturity[k])
        return harvest[harvest < self.T]

    def sow_from_harvest(self, k: int, harvest: int) -> int:
        return int(harvest - self.maturity[k])

    def active_weeks(self, k: int, harvest: int, include_setup: bool = True) -> range:
        """Weeks during which a lot is unavailable because crop k occupies it.

        A crop planted at s and harvested at h=s+n occupies [s,h), while the common
        preparation interval occupies the following setup_weeks.  This convention agrees
        with the decoder, where the next crop can start at h + setup_weeks.
        """
        sow = self.sow_from_harvest(k, harvest)
        end = harvest + (self.setup_weeks if include_setup else 0)
        return range(max(0, sow), min(self.T, end))

    def same_family_active_weeks(self, k: int, harvest: int) -> range:
        """Expanded occupancy used only among crops in the same botanical family."""
        sow = self.sow_from_harvest(k, harvest)
        end = harvest + self.setup_weeks + self.same_family_extra_rest
        return range(max(0, sow), min(self.T, end))


def objectives_from_production(data: ProblemData, production: np.ndarray) -> tuple[float, float]:
    production = np.asarray(production, dtype=float)
    if production.shape != (data.K, data.T):
        raise ValueError(f"production must have shape {(data.K, data.T)}")
    exposure = (production * data.prices) / float(data.monetary_scale)
    income = float(exposure.sum())
    risk = 0.0
    for t in range(data.T):
        e = exposure[:, t]
        risk += float(e @ data.covariance @ e)
    return income, risk
