"""MVO 포트폴리오 테스트.

실제 데이터 파일 없이 합성 데이터로 run_mvo()의 핵심 경로를 검증한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rl.mvo import (
    MAX_WEIGHT,
    _optimize_min_variance,
    _snap_to_trading_days,
    run_mvo,
    run_mvo_all_windows,
)
from src.rl.metrics import calculate_all_metrics

# ── 공통 픽스처 ───────────────────────────────────────────────────────────────

_ASSETS = ["SPY", "QQQ", "TLT", "GLD", "EEM"]


def _make_returns(start: str, end: str, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="B")
    return pd.DataFrame(
        rng.normal(0.0003, 0.008, (len(idx), len(_ASSETS))),
        index=idx,
        columns=_ASSETS,
    )


@pytest.fixture
def full_returns() -> pd.DataFrame:
    """2020-01-01 ~ 2024-12-31 전체 구간 합성 수익률."""
    return _make_returns("2020-01-01", "2024-12-31")


# ── _optimize_min_variance ────────────────────────────────────────────────────

def test_optimize_weights_sum_to_one():
    """최적 비중의 합이 1이어야 한다."""
    rng = np.random.default_rng(0)
    n = len(_ASSETS)
    data = rng.normal(0, 0.01, (100, n))
    cov = np.cov(data.T)
    w = _optimize_min_variance(cov, n)
    assert abs(w.sum() - 1.0) < 1e-6


def test_optimize_weights_non_negative():
    """모든 비중이 0 이상이어야 한다 (공매도 금지)."""
    rng = np.random.default_rng(1)
    n = len(_ASSETS)
    data = rng.normal(0, 0.01, (200, n))
    cov = np.cov(data.T)
    w = _optimize_min_variance(cov, n)
    assert np.all(w >= -1e-8)


def test_optimize_weights_respect_max_weight():
    """개별 자산 비중이 MAX_WEIGHT(40%)를 초과하지 않아야 한다."""
    rng = np.random.default_rng(2)
    n = len(_ASSETS)
    data = rng.normal(0, 0.01, (200, n))
    cov = np.cov(data.T)
    w = _optimize_min_variance(cov, n)
    assert np.all(w <= MAX_WEIGHT + 1e-6)


def test_optimize_min_variance_lower_than_equal_weight():
    """MVO 포트폴리오 분산이 equal-weight보다 작거나 같아야 한다."""
    rng = np.random.default_rng(3)
    n = len(_ASSETS)
    data = rng.normal(0, 0.01, (300, n))
    cov = np.cov(data.T)
    w_mvo = _optimize_min_variance(cov, n)
    w_eq = np.ones(n) / n
    var_mvo = float(w_mvo @ cov @ w_mvo)
    var_eq = float(w_eq @ cov @ w_eq)
    assert var_mvo <= var_eq + 1e-8


# ── _snap_to_trading_days ─────────────────────────────────────────────────────

def test_snap_returns_only_trading_days():
    """스냅 결과가 실제 거래일 인덱스 내 날짜만 포함해야 한다."""
    trading = pd.date_range("2024-01-01", "2024-12-31", freq="B")
    cal = pd.date_range("2024-01-01", "2024-12-31", freq="ME")
    snapped = _snap_to_trading_days(cal, trading)
    assert set(snapped).issubset(set(trading))


def test_snap_is_monotonic():
    """스냅 결과가 단조 증가해야 한다."""
    trading = pd.date_range("2024-01-01", "2024-12-31", freq="B")
    cal = pd.date_range("2024-01-01", "2024-12-31", freq="ME")
    snapped = _snap_to_trading_days(cal, trading)
    assert snapped.is_monotonic_increasing


# ── run_mvo ───────────────────────────────────────────────────────────────────

def test_run_mvo_returns_series(full_returns):
    """run_mvo가 pd.Series를 반환해야 한다."""
    result = run_mvo(full_returns, "2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31")
    assert isinstance(result, pd.Series)


def test_run_mvo_index_within_test_period(full_returns):
    """반환된 Series의 인덱스가 테스트 기간 내 거래일이어야 한다."""
    result = run_mvo(full_returns, "2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31")
    assert result.index.min() >= pd.Timestamp("2023-01-01")
    assert result.index.max() <= pd.Timestamp("2023-12-31")


def test_run_mvo_non_empty(full_returns):
    """테스트 기간 데이터가 있으면 Series가 비어 있지 않아야 한다."""
    result = run_mvo(full_returns, "2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31")
    assert len(result) > 0


def test_run_mvo_all_finite(full_returns):
    """반환된 수익률이 모두 유한한 값이어야 한다."""
    result = run_mvo(full_returns, "2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31")
    assert np.all(np.isfinite(result.values))


def test_run_mvo_empty_on_no_data(full_returns):
    """테스트 기간에 데이터가 없으면 빈 Series를 반환해야 한다."""
    result = run_mvo(full_returns, "2020-01-01", "2020-12-31", "2030-01-01", "2030-12-31")
    assert result.empty


def test_run_mvo_compatible_with_calculate_all_metrics(full_returns):
    """MVO 수익률이 calculate_all_metrics에 정상 입력되어야 한다."""
    mvo = run_mvo(full_returns, "2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31")
    benchmark = full_returns.loc["2023-01-01":"2023-12-31"]["SPY"]
    common = mvo.index.intersection(benchmark.index)
    metrics = calculate_all_metrics(mvo.loc[common], benchmark.loc[common])
    assert "sharpe_ratio" in metrics
    assert "mdd" in metrics
    assert len(metrics) == 12


# ── run_mvo_all_windows ───────────────────────────────────────────────────────

_WINDOWS = [
    {"name": "w3", "train_start": "2020-01-01", "train_end": "2023-12-31",
     "test_start": "2024-01-01", "test_end": "2024-12-31"},
]


def test_run_mvo_all_windows_returns_dict(full_returns):
    """run_mvo_all_windows가 윈도우 이름을 키로 하는 dict를 반환해야 한다."""
    result = run_mvo_all_windows(full_returns, _WINDOWS)
    assert isinstance(result, dict)
    assert "w3" in result
    assert isinstance(result["w3"], pd.Series)
