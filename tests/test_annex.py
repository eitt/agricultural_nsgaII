from pathlib import Path
import json
import numpy as np
import pytest
import yaml
import agriopt
from agriopt.instances import CATALOG, make_instance
from agriopt.productivity import load_productivity
from agriopt.io_utils import read_matrix_csv
from agriopt.frozen_instances import load_frozen_instance
from agriopt.exact_scip import enumerate_events, build_scip_model, _extract
from agriopt.validation import validate_events
from agriopt.model import objectives_from_production
from agriopt.hybrid_nsga2 import run_hybrid_nsga2

ROOT = Path(__file__).resolve().parents[1]

def test_package_is_self_contained():
    assert Path(agriopt.__file__).resolve().is_relative_to(ROOT)
    forbidden = ['reporting.py', 'data_pipeline.py', 'dane_sipsa.py']
    assert not any((ROOT / 'src/agriopt' / name).exists() for name in forbidden)

@pytest.mark.parametrize('name', CATALOG.instance.tolist())
def test_frozen_instance_identical_to_builder(name):
    frozen = load_frozen_instance(ROOT / 'instances' / name)
    prod = load_productivity(ROOT / 'data/seed/productividad_actualizada.csv')
    history = read_matrix_csv(ROOT / 'data/processed/prices_weekly.csv')
    forecast = read_matrix_csv(ROOT / 'data/processed/price_forecast_104w.csv')
    rebuilt, _ = make_instance(CATALOG[CATALOG.instance.eq(name)].iloc[0], prod, forecast, history)
    for field, expected in vars(rebuilt).items():
        actual = getattr(frozen, field)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected
    assert np.linalg.eigvalsh(frozen.covariance).min() >= -1e-10
    assert frozen.demand is None

def test_published_controls():
    cfg = yaml.safe_load((ROOT / 'configs/publication_core.yaml').read_text())
    e = cfg['experiment']
    assert e['seeds'] == list(range(2026, 2036))
    assert e['instances'] == CATALOG.instance.tolist()[:4]
    assert e['methods'] == ['exact', 'nsga2', 'nsga2_qp', 'hnsga2']
    assert e['nsga2'] == {'population': 80, 'generations': 150}
    assert e['hybrid']['backend'] == 'slsqp'
    assert e['hybrid']['refinement_fraction'] == .25
    assert e['hybrid']['warmup_generations'] == 5
    assert e['hybrid']['min_training_samples'] == 30
    assert e['exact']['threads'] == 4 and e['exact']['points'] == 11
    assert cfg['data']['demand'] is None

def test_full_refinement_feasible():
    data = load_frozen_instance(ROOT / 'instances/I01_micro')
    result = run_hybrid_nsga2(data, population_size=10, generations=2,
                             seed=2026, mode='full', refinement_backend='slsqp')
    assert result.pareto
    for individual in result.pareto:
        validation = validate_events(data, individual.events or [])
        assert validation['ok'], validation['errors']

def test_exact_feasibility_and_objective_consistency():
    pytest.importorskip('pyscipopt')
    data = load_frozen_instance(ROOT / 'instances/I01_micro')
    model, events, _, z, exposure, *_ = build_scip_model(
        data, objective='max_income', time_limit=10, threads=1, seed=2026)
    assert events and len(events) == len(enumerate_events(data))
    model.optimize()
    assert model.getNSols() > 0
    income, risk, rows = _extract(data, events, z, exposure, model)
    validation = validate_events(data, rows, tol=1e-5)
    assert validation['ok'], validation['errors']
    np.testing.assert_allclose([income, risk],
        objectives_from_production(data, validation['production']), rtol=1e-6, atol=1e-6)
