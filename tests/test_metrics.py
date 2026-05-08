import numpy as np
import pandas as pd
import pytest

from src.rl.metrics import calculate_all_metrics, mdd, sharpe_ratio

_EXPECTED_METRIC_KEYS = {
    "cumulative_return",
    "cagr",
    "annualized_volatility",
    "var_95",
    "cvar_95",
    "mdd",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "alpha",
    "beta",
    "information_ratio",
}


@pytest.fixture
def sample_series():
    rng = np.random.default_rng(42)
    portfolio = pd.Series(rng.normal(0.0005, 0.015, 252))
    benchmark = pd.Series(rng.normal(0.0003, 0.010, 252))
    return portfolio, benchmark


def test_sharpe_ratio_is_finite(sample_series):
    portfolio, _ = sample_series
    assert np.isfinite(sharpe_ratio(portfolio))


def test_mdd_is_non_negative(sample_series):
    portfolio, _ = sample_series
    assert mdd(portfolio) >= 0.0


def test_calculate_all_metrics_returns_12_keys(sample_series):
    portfolio, benchmark = sample_series
    result = calculate_all_metrics(portfolio, benchmark)
    assert len(result) == 12


def test_calculate_all_metrics_has_expected_keys(sample_series):
    portfolio, benchmark = sample_series
    result = calculate_all_metrics(portfolio, benchmark)
    assert set(result.keys()) == _EXPECTED_METRIC_KEYS


def test_calculate_all_metrics_all_finite(sample_series):
    portfolio, benchmark = sample_series
    result = calculate_all_metrics(portfolio, benchmark)
    for key, val in result.items():
        assert np.isfinite(val), f"{key} 값이 유한하지 않음: {val}"
