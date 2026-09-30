import numpy as np
import pytest
from agriopt.forecasting import forecast_one, forecast_full_series, _metrics

def test_three_forecasting_candidates_and_holdout():
    t = np.arange(70)
    y = np.exp(7 + .002*t + .1*np.sin(2*np.pi*t/52))
    results, predictions = forecast_one(y, 'test_crop', test_size=12)
    assert {r.model for r in results} == {'ARIMA','GP','ARIMA_GP'}
    np.testing.assert_array_equal(predictions['actual'], y[-12:])
    for r in results:
        assert np.isfinite([r.mae,r.rmse,r.smape,r.mase]).all()
        pred = predictions[r.model]
        assert pred.shape == (12,) and (pred > 0).all()
        refit, scale = forecast_full_series(y, 5, r.model, r.arima_order)
        assert refit.shape == scale.shape == (5,)
        assert np.isfinite(refit).all() and (refit > 0).all()

def test_forecasting_error_metrics():
    metrics = _metrics(np.array([10.,20.]), np.array([10.,20.]), np.array([8.,9.,10.]))
    assert all(v == 0 for v in metrics.values())

def test_nonpositive_prices_rejected():
    with pytest.raises(ValueError, match='positive'):
        forecast_one(np.zeros(70), 'invalid', test_size=12)
