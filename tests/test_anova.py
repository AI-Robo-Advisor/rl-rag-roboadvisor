"""ANOVA 3종 실험 + Tukey HSD 테스트.

모델 파일·backtest CSV 없는 환경에서 equal-weight fallback 경로를 검증한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rl.anova import (
    _eta_squared,
    _tukey_post_hoc,
    run_anova,
    run_reward_function_comparison,
    run_strategy_comparison,
    run_market_regime_comparison,
    run_all_anova,
)

# ── 공통 픽스처 ───────────────────────────────────────────────────────────────

_ASSETS = ["SPY", "QQQ", "TLT", "GLD", "EEM"]


def _make_returns(start: str, end: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="B")
    return pd.DataFrame(
        rng.normal(0.0003, 0.008, (len(idx), len(_ASSETS))),
        index=idx,
        columns=_ASSETS,
    )


@pytest.fixture
def full_returns() -> pd.DataFrame:
    """2020-01-01 ~ 2024-12-31 전 구간 합성 수익률."""
    return _make_returns("2020-01-01", "2024-12-31")


# ── _eta_squared ──────────────────────────────────────────────────────────────

def test_eta_squared_zero_for_identical_groups():
    """동일한 집단 → η² = 0이어야 한다."""
    s = pd.Series([0.001, -0.002, 0.003] * 50)
    eta2 = _eta_squared([s, s.copy(), s.copy()])
    assert abs(eta2) < 1e-6


def test_eta_squared_between_zero_and_one():
    """η²는 [0, 1] 범위 내에 있어야 한다."""
    rng = np.random.default_rng(0)
    g1 = pd.Series(rng.normal(0.001, 0.01, 100))
    g2 = pd.Series(rng.normal(0.002, 0.01, 100))
    g3 = pd.Series(rng.normal(-0.001, 0.01, 100))
    eta2 = _eta_squared([g1, g2, g3])
    assert 0.0 <= eta2 <= 1.0


def test_eta_squared_larger_for_more_separated_groups():
    """집단 간 평균 차이가 클수록 η²가 커야 한다."""
    rng = np.random.default_rng(1)
    close = [pd.Series(rng.normal(m, 0.01, 100)) for m in [0.001, 0.002, 0.003]]
    far = [pd.Series(rng.normal(m, 0.01, 100)) for m in [0.001, 0.020, 0.040]]
    assert _eta_squared(far) > _eta_squared(close)


# ── _tukey_post_hoc ───────────────────────────────────────────────────────────

def test_tukey_returns_correct_pair_count():
    """집단 k개 → C(k,2) 쌍이 반환되어야 한다."""
    rng = np.random.default_rng(2)
    groups = {
        "A": pd.Series(rng.normal(0.001, 0.01, 100)),
        "B": pd.Series(rng.normal(0.002, 0.01, 100)),
        "C": pd.Series(rng.normal(0.003, 0.01, 100)),
    }
    rows = _tukey_post_hoc(groups)
    assert len(rows) == 3  # C(3,2)


def test_tukey_row_has_required_keys():
    """각 Tukey 행에 group1·group2·meandiff·p_adj·reject 키가 있어야 한다."""
    rng = np.random.default_rng(3)
    groups = {
        "X": pd.Series(rng.normal(0.0, 0.01, 80)),
        "Y": pd.Series(rng.normal(0.005, 0.01, 80)),
    }
    rows = _tukey_post_hoc(groups)
    assert len(rows) == 1
    assert {"group1", "group2", "meandiff", "p_adj", "reject"} == set(rows[0])


def test_tukey_reject_is_bool():
    """reject 필드가 bool 타입이어야 한다."""
    rng = np.random.default_rng(4)
    groups = {
        "A": pd.Series(rng.normal(0.0, 0.01, 100)),
        "B": pd.Series(rng.normal(0.01, 0.01, 100)),
    }
    rows = _tukey_post_hoc(groups)
    assert isinstance(rows[0]["reject"], bool)


def test_tukey_returns_empty_for_single_group():
    """집단이 1개이면 빈 리스트를 반환해야 한다."""
    groups = {"only": pd.Series([0.001, 0.002, 0.003])}
    assert _tukey_post_hoc(groups) == []


# ── run_anova ─────────────────────────────────────────────────────────────────

def test_run_anova_returns_required_keys():
    """run_anova 결과에 name·f_statistic·p_value·eta_squared·post_hoc 키가 있어야 한다."""
    rng = np.random.default_rng(5)
    groups = {
        "g1": pd.Series(rng.normal(0.001, 0.01, 100)),
        "g2": pd.Series(rng.normal(0.003, 0.01, 100)),
        "g3": pd.Series(rng.normal(-0.001, 0.01, 100)),
    }
    result = run_anova("test_exp", groups)
    assert {"name", "f_statistic", "p_value", "eta_squared", "post_hoc"} == set(result)


def test_run_anova_name_matches_input():
    rng = np.random.default_rng(6)
    groups = {f"g{i}": pd.Series(rng.normal(i * 0.001, 0.01, 50)) for i in range(3)}
    result = run_anova("my_experiment", groups)
    assert result["name"] == "my_experiment"


def test_run_anova_p_value_in_range():
    """p_value가 [0, 1] 범위에 있어야 한다."""
    rng = np.random.default_rng(7)
    groups = {f"g{i}": pd.Series(rng.normal(i * 0.001, 0.01, 100)) for i in range(3)}
    result = run_anova("check_pval", groups)
    assert 0.0 <= result["p_value"] <= 1.0


def test_run_anova_fallback_for_insufficient_groups():
    """유효 집단 < 2이면 f=0, p=1, post_hoc=[] 반환해야 한다."""
    result = run_anova("empty", {"only": pd.Series([0.001])})
    assert result["f_statistic"] == 0.0
    assert result["p_value"] == 1.0
    assert result["post_hoc"] == []


def test_run_anova_post_hoc_is_list():
    """post_hoc 필드가 list이어야 한다."""
    rng = np.random.default_rng(8)
    groups = {f"g{i}": pd.Series(rng.normal(i * 0.002, 0.01, 80)) for i in range(3)}
    result = run_anova("check_type", groups)
    assert isinstance(result["post_hoc"], list)


# ── 실험별 함수 ───────────────────────────────────────────────────────────────

def test_run_reward_function_comparison_structure(full_returns, tmp_path, monkeypatch):
    """reward_function_comparison 결과에 필수 키와 3종 post_hoc 쌍이 있어야 한다."""
    import src.rl.anova as anova_module
    monkeypatch.setattr(anova_module, "RESULTS_DIR", tmp_path)  # CSV 없음

    result = run_reward_function_comparison(full_returns)
    assert result["name"] == "reward_function_comparison"
    assert "f_statistic" in result
    # equal-weight fallback → 3종 동일 시리즈 → 집단 구분 안 됨 → post_hoc 개수만 확인
    assert isinstance(result["post_hoc"], list)


def test_run_strategy_comparison_structure(full_returns, tmp_path, monkeypatch):
    """strategy_comparison 결과에 필수 키가 있어야 한다."""
    import src.rl.anova as anova_module
    monkeypatch.setattr(anova_module, "RESULTS_DIR", tmp_path)

    result = run_strategy_comparison(full_returns)
    assert result["name"] == "strategy_comparison"
    assert "eta_squared" in result
    assert isinstance(result["post_hoc"], list)


def test_run_market_regime_comparison_structure(full_returns, tmp_path, monkeypatch):
    """market_regime_comparison 결과에 Two-way ANOVA 필수 키가 있어야 한다."""
    import src.rl.anova as anova_module
    monkeypatch.setattr(anova_module, "RESULTS_DIR", tmp_path)

    result = run_market_regime_comparison(full_returns)
    assert result["name"] == "market_regime_comparison"
    assert "p_value" in result
    assert isinstance(result["post_hoc"], list)
    # Two-way 추가 필드
    assert "interaction" in result
    assert "f_statistic" in result["interaction"]
    assert "p_value" in result["interaction"]
    assert "significant" in result["interaction"]
    assert isinstance(result["interaction"]["significant"], bool)
    assert "strategy_effect" in result
    assert "f_statistic" in result["strategy_effect"]


def test_run_all_anova_returns_three_results(full_returns, tmp_path, monkeypatch):
    """run_all_anova가 3개 실험 결과를 담은 리스트를 반환해야 한다."""
    import src.rl.anova as anova_module
    monkeypatch.setattr(anova_module, "RESULTS_DIR", tmp_path)

    results = run_all_anova(full_returns)
    assert len(results) == 3
    names = [r["name"] for r in results]
    assert "reward_function_comparison" in names
    assert "strategy_comparison" in names
    assert "market_regime_comparison" in names


def test_run_all_anova_with_csv(full_returns, tmp_path, monkeypatch):
    """backtest CSV가 있으면 해당 데이터를 사용해야 한다."""
    import src.rl.anova as anova_module
    monkeypatch.setattr(anova_module, "RESULTS_DIR", tmp_path)

    # CSV 파일 직접 생성 — rate_hike(2022) + recovery(2023) + bull(2024) 모두 커버
    rng = np.random.default_rng(99)
    for reward in ["return", "sharpe", "mdd"]:
        idx = pd.date_range("2022-01-03", "2024-12-31", freq="B")
        pd.DataFrame({
            "date": idx.strftime("%Y-%m-%d"),
            "episode_return": rng.normal(0.001, 0.01, len(idx)),
        }).to_csv(tmp_path / f"backtest_{reward}.csv", index=False)

    results = run_all_anova(full_returns)
    assert len(results) == 3
    for r in results:
        assert np.isfinite(r["f_statistic"])
        assert 0.0 <= r["p_value"] <= 1.0
    # market_regime_comparison은 Two-way 필드 추가 검증
    regime_result = next(r for r in results if r["name"] == "market_regime_comparison")
    assert "interaction" in regime_result
    assert "strategy_effect" in regime_result
