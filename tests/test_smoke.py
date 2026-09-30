from pathlib import Path

import numpy as np
import pandas as pd

from agriopt.instances import CATALOG, make_instance
from agriopt.io_utils import read_matrix_csv
from agriopt.nsga2 import infer_max_slots, initialize_population, run_nsga2
from agriopt.hybrid_nsga2 import run_hybrid_nsga2
from agriopt.productivity import load_productivity
from agriopt.validation import validate_events

ROOT = Path(__file__).resolve().parents[1]


def build_i01():
    prod = load_productivity(ROOT / "data/seed/productividad_actualizada.csv")
    hist = read_matrix_csv(ROOT / "data/seed/precios_historicos_2013_2017.csv")
    fc = read_matrix_csv(ROOT / "data/seed/demo_price_forecast_104w.csv")
    return make_instance(CATALOG.iloc[0], prod, fc, hist)[0]


def test_instance_and_nsga2():
    data = build_i01()
    assert (data.L, data.K, data.T) == (2, 5, 52)
    res = run_nsga2(data, population_size=10, generations=2, seed=11)
    assert res.pareto
    assert all(validate_events(data, ind.events or [])["ok"] for ind in res.pareto)


def test_evolutionary_variants_share_identical_initialization():
    data = build_i01()
    max_slots = infer_max_slots(data)
    rng_a = np.random.default_rng(2026)
    rng_b = np.random.default_rng(2026)
    pop_a = initialize_population(data, rng_a, 10, max_slots, 4)
    pop_b = initialize_population(data, rng_b, 10, max_slots, 4)
    for left, right in zip(pop_a, pop_b):
        assert (left.crops == right.crops).all()
        assert (left.waits == right.waits).all()
        assert (left.area_frac == right.area_frac).all()
    assert (pop_a[0].crops == -1).all()
    assert (pop_a[0].area_frac == 0.0).all()
    assert rng_a.random() == rng_b.random()
    baseline = run_nsga2(data, population_size=10, generations=0, seed=2026, max_slots=max_slots)
    for expected, actual in zip(pop_a, baseline.population):
        assert (expected.crops == actual.crops).all()
        assert (expected.waits == actual.waits).all()
        assert (expected.area_frac == actual.area_frac).all()


def test_hybrid_slsqp():
    data = build_i01()
    res = run_hybrid_nsga2(
        data,
        population_size=10,
        generations=2,
        seed=12,
        mode="surrogate",
        refinement_fraction=0.3,
        warmup_generations=1,
        min_training_samples=3,
        refinement_backend="slsqp",
    )
    assert res.pareto
    assert res.refinement_log
