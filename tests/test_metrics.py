import numpy as np

import pytest

from agriopt.metrics import (
    additive_epsilon,
    hypervolume_2d,
    igd_plus,
    nondominated,
    pooled_empirical_reference,
    reference_scaling,
)


def test_metrics_identical_front():
    p = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
    assert hypervolume_2d(p, (1.1, 1.1)) > 0
    assert igd_plus(p, p) == 0
    assert additive_epsilon(p, p) == 0
    assert len(nondominated(p)) == 3


def test_front_cleaning_removes_duplicates_and_nonfinite_rows():
    points = np.array([
        [0.1, 0.9], [0.1, 0.9], [0.5, 0.5], [0.8, 0.8], [np.nan, 0.2], [np.inf, 0.1],
    ])
    cleaned = nondominated(points)
    assert cleaned.shape == (2, 2)
    assert np.all(np.isfinite(cleaned))


def test_empty_singleton_and_zero_span_reference_behaviour():
    empty = np.empty((0, 2))
    assert nondominated(empty).shape == (0, 2)
    assert hypervolume_2d(empty) == 0.0
    assert np.isnan(igd_plus(empty, np.array([[0.0, 0.0]])))
    with pytest.raises(ValueError):
        reference_scaling(empty)
    ideal, span = reference_scaling(np.array([[2.0, 3.0]]))
    assert np.array_equal(ideal, np.array([2.0, 3.0]))
    assert np.array_equal(span, np.ones(2))


def test_pooled_reference_is_nondominated_union():
    pooled = pooled_empirical_reference([
        np.array([[0.0, 1.0], [0.5, 0.5]]),
        np.array([[0.5, 0.5], [1.0, 0.0], [0.8, 0.8]]),
    ])
    assert pooled.shape == (3, 2)
    assert {tuple(row) for row in pooled} == {(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)}
