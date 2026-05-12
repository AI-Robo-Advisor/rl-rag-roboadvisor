"""Walk-Forward 백테스트 프레임워크 테스트.

모델 파일이 없는 환경(CI/로컬)에서도 equal-weight fallback 경로로
모든 공개 API가 동작함을 검증한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rl.backtest import (
    SAFEGUARD_THRESHOLD,
    WINDOWS,
    _detect_safeguard_events,
    run_window_backtest,
    run_all_windows,
    run_stress_test,
)

# ── 공통 픽스처 ───────────────────────────────────────────────────────────────

_ASSETS = ["A", "B", "C"]


def _make_returns(start: str, end: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="B")
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, (len(idx), len(_ASSETS))),
        index=idx,
        columns=_ASSETS,
    )


def _make_features(returns: pd.DataFrame) -> pd.DataFrame:
    """returns 인덱스와 동일한 features DataFrame 생성 (equal-weight 경로에서만 사용)."""
    rng = np.random.default_rng(99)
    cols = {}
    for a in _ASSETS:
        cols[f"{a}_return"] = rng.normal(0, 0.01, len(returns))
        cols[f"{a}_RSI"] = rng.uniform(0.3, 0.7, len(returns))
        cols[f"{a}_MACD_signal"] = rng.normal(0, 0.1, len(returns))
    return pd.DataFrame(cols, index=returns.index)


@pytest.fixture
def wide_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """2020-01-01 ~ 2025-12-31 전체 구간 — 4개 윈도우 + 스트레스 구간 모두 포함."""
    ret = _make_returns("2020-01-01", "2025-12-31")
    feat = _make_features(ret)
    return ret, feat


# ── _detect_safeguard_events ──────────────────────────────────────────────────

def _dated_series(values: np.ndarray, start: str = "2022-01-03") -> pd.Series:
    """DatetimeIndex가 있는 Series 생성 헬퍼 (_detect_safeguard_events 입력 형식)."""
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def test_safeguard_no_event_on_flat_series():
    """수익률 0 시리즈 → Safe-Guard 이벤트 없음."""
    events = _detect_safeguard_events(_dated_series(np.zeros(100)))
    assert events == []


def test_safeguard_triggered_on_deep_drawdown():
    """15% 초과 MDD 구간에서 이벤트가 1개 이상 탐지되어야 한다."""
    fall = np.full(50, -0.004)     # 누적 MDD ≈ 18%
    recover = np.full(100, 0.005)  # 이후 반등
    events = _detect_safeguard_events(_dated_series(np.concatenate([fall, recover])))
    assert len(events) >= 1
    assert events[0]["drawdown_at_trigger"] >= SAFEGUARD_THRESHOLD


def test_safeguard_event_has_required_keys():
    """이벤트 dict에 triggered_at·drawdown_at_trigger·resumed_at 키가 있어야 한다."""
    fall = np.full(50, -0.004)
    recover = np.full(100, 0.005)
    events = _detect_safeguard_events(_dated_series(np.concatenate([fall, recover])))
    assert len(events) >= 1
    assert {"triggered_at", "drawdown_at_trigger", "resumed_at"} <= set(events[0])


def test_safeguard_resumed_at_none_when_not_recovered():
    """회복 없이 종료 → resumed_at이 None이어야 한다."""
    events = _detect_safeguard_events(_dated_series(np.full(100, -0.005)))
    assert len(events) == 1
    assert events[-1]["resumed_at"] is None


# ── run_window_backtest ───────────────────────────────────────────────────────

def test_run_window_backtest_equal_weight_fallback(wide_data, tmp_path, monkeypatch):
    """모델 파일이 없으면 equal-weight fallback으로 결과를 반환한다."""
    import src.rl.backtest as bt

    monkeypatch.setattr(bt, "MODELS_DIR", tmp_path)  # 빈 디렉터리 → 모델 없음

    ret, feat = wide_data
    metrics, portfolio, weights_df = run_window_backtest(WINDOWS[2], ret, feat, "return")

    assert isinstance(portfolio, pd.Series)
    assert isinstance(weights_df, pd.DataFrame)
    assert len(portfolio) > 0


def test_run_window_backtest_metrics_has_required_keys(wide_data, tmp_path, monkeypatch):
    """metrics dict에 성과 지표 12개 + 메타 4개 키가 포함되어야 한다."""
    import src.rl.backtest as bt

    monkeypatch.setattr(bt, "MODELS_DIR", tmp_path)

    ret, feat = wide_data
    metrics, _, _ = run_window_backtest(WINDOWS[0], ret, feat, "sharpe")

    for key in ("sharpe_ratio", "mdd", "cagr", "window", "reward", "test_start", "test_end"):
        assert key in metrics, f"키 누락: {key}"


def test_run_window_backtest_metadata_matches_window(wide_data, tmp_path, monkeypatch):
    """metrics의 window·reward·test_start 값이 입력 window와 일치해야 한다."""
    import src.rl.backtest as bt

    monkeypatch.setattr(bt, "MODELS_DIR", tmp_path)

    ret, feat = wide_data
    metrics, _, _ = run_window_backtest(WINDOWS[1], ret, feat, "mdd")

    assert metrics["window"] == "w2"
    assert metrics["reward"] == "mdd"
    assert metrics["test_start"] == WINDOWS[1]["test_start"]


# ── run_all_windows ───────────────────────────────────────────────────────────

def test_run_all_windows_returns_4_rows(wide_data, tmp_path, monkeypatch):
    """4개 윈도우 실행 후 4행 DataFrame을 반환해야 한다."""
    import src.rl.backtest as bt

    monkeypatch.setattr(bt, "_load_data", lambda: wide_data)
    monkeypatch.setattr(bt, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(bt, "RESULTS_DIR", tmp_path)

    result = run_all_windows(reward="return")
    assert len(result) == 4


def test_run_all_windows_saves_csv(wide_data, tmp_path, monkeypatch):
    """backtest_{reward}.csv가 RESULTS_DIR에 저장되어야 한다."""
    import src.rl.backtest as bt

    monkeypatch.setattr(bt, "_load_data", lambda: wide_data)
    monkeypatch.setattr(bt, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(bt, "RESULTS_DIR", tmp_path)

    run_all_windows(reward="mdd")
    assert (tmp_path / "backtest_mdd.csv").exists()


def test_run_all_windows_saves_weights_parquet(wide_data, tmp_path, monkeypatch):
    """weights_{reward}.parquet가 RESULTS_DIR에 저장되어야 한다."""
    import src.rl.backtest as bt

    monkeypatch.setattr(bt, "_load_data", lambda: wide_data)
    monkeypatch.setattr(bt, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(bt, "RESULTS_DIR", tmp_path)

    run_all_windows(reward="sharpe")
    assert (tmp_path / "weights_sharpe.parquet").exists()


def test_run_all_windows_csv_has_required_columns(wide_data, tmp_path, monkeypatch):
    """저장된 CSV에 date·episode_return 컬럼이 있어야 한다."""
    import src.rl.backtest as bt

    monkeypatch.setattr(bt, "_load_data", lambda: wide_data)
    monkeypatch.setattr(bt, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(bt, "RESULTS_DIR", tmp_path)

    run_all_windows(reward="return")
    df = pd.read_csv(tmp_path / "backtest_return.csv")
    assert "date" in df.columns
    assert "episode_return" in df.columns


# ── run_stress_test ───────────────────────────────────────────────────────────

def test_run_stress_test_returns_required_keys(wide_data, tmp_path, monkeypatch):
    """스트레스 테스트 결과에 api_spec.md의 필수 키가 모두 있어야 한다."""
    import src.rl.backtest as bt

    monkeypatch.setattr(bt, "_load_data", lambda: wide_data)
    monkeypatch.setattr(bt, "MODELS_DIR", tmp_path)

    result = run_stress_test(reward="return")
    required = {
        "period", "metrics", "benchmark_metrics",
        "safeguard_events", "dates", "ppo_cum", "bm_cum", "drawdown",
    }
    assert required <= set(result)


def test_run_stress_test_period_is_rate_hike(wide_data, tmp_path, monkeypatch):
    """period가 2022-01-01 ~ 2022-12-31 금리 충격 구간이어야 한다."""
    import src.rl.backtest as bt

    monkeypatch.setattr(bt, "_load_data", lambda: wide_data)
    monkeypatch.setattr(bt, "MODELS_DIR", tmp_path)

    result = run_stress_test(reward="return")
    assert result["period"]["start"] == "2022-01-01"
    assert result["period"]["end"] == "2022-12-31"


def test_run_stress_test_arrays_same_length(wide_data, tmp_path, monkeypatch):
    """dates·ppo_cum·bm_cum·drawdown 배열 길이가 동일하고 0보다 커야 한다."""
    import src.rl.backtest as bt

    monkeypatch.setattr(bt, "_load_data", lambda: wide_data)
    monkeypatch.setattr(bt, "MODELS_DIR", tmp_path)

    result = run_stress_test(reward="return")
    n = len(result["dates"])
    assert n > 0
    assert len(result["ppo_cum"]) == n
    assert len(result["bm_cum"]) == n
    assert len(result["drawdown"]) == n


def test_run_stress_test_ppo_cum_starts_near_one(wide_data, tmp_path, monkeypatch):
    """누적 포트폴리오 가치는 1.0 근방에서 시작해야 한다."""
    import src.rl.backtest as bt

    monkeypatch.setattr(bt, "_load_data", lambda: wide_data)
    monkeypatch.setattr(bt, "MODELS_DIR", tmp_path)

    result = run_stress_test(reward="return")
    assert abs(result["ppo_cum"][0] - 1.0) < 0.05
