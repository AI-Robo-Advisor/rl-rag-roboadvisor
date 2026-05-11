"""ANOVA 3종 실험 + Tukey HSD 사후 검정.

실험 구성:
    1. reward_function_comparison: 보상함수 3종(return/sharpe/mdd) 성과 비교
    2. strategy_comparison: PPO vs MVO vs 동일비중 비교
    3. market_regime_comparison: 시장 국면별(bull/bear/crisis) 성과 비교

국면 정의 (Walk-Forward 윈도우 기반):
    bear   = 2022        (W1 test, 금리 인상기)
    bull   = 2023 + 2024 (W2·W3 test, 회복장·AI 랠리)
    crisis = 2020-02-01 ~ 2020-05-31 (코로나 폭락, OOS 스트레스 구간)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from src.rl.backtest import WINDOWS
from src.rl.mvo import run_mvo_all_windows

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/results")

# 국면별 날짜 구간 (multiple slices → concat)
_REGIME_SLICES: dict[str, list[tuple[str, str]]] = {
    "bear":   [("2022-01-01", "2022-12-31")],
    "bull":   [("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31")],
    "crisis": [("2020-02-01", "2020-05-31")],
}


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _eta_squared(groups: list[pd.Series]) -> float:
    """일원 ANOVA 효과 크기 η² 계산.

    Args:
        groups: 집단별 로그수익률 Series 리스트.

    Returns:
        η² 값 (0 ~ 1). SS_total = 0이면 0.0 반환.
    """
    all_values = np.concatenate([g.values for g in groups])
    grand_mean = all_values.mean()

    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_values - grand_mean) ** 2)

    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def _tukey_post_hoc(groups: dict[str, pd.Series]) -> list[dict]:
    """Tukey HSD 사후 검정 수행.

    Args:
        groups: {그룹명: 수익률 Series} dict.

    Returns:
        쌍별 결과 목록. 각 항목: group1, group2, meandiff, p_adj, reject.
        집단 수 < 2이면 빈 리스트 반환.
    """
    if len(groups) < 2:
        return []

    endog = np.concatenate([v.values for v in groups.values()])
    labels = np.concatenate(
        [np.full(len(v), name) for name, v in groups.items()]
    )

    tukey = pairwise_tukeyhsd(endog, labels, alpha=0.05)

    rows = []
    for row in tukey._results_table.data[1:]:
        g1, g2, meandiff, p_adj, *_, reject = row
        rows.append({
            "group1": str(g1),
            "group2": str(g2),
            "meandiff": round(float(meandiff), 6),
            "p_adj": round(float(p_adj), 6),
            "reject": bool(reject),
        })
    return rows


def _load_backtest_series(reward: str, returns: pd.DataFrame) -> pd.Series:
    """data/results/backtest_{reward}.csv 로드. 없으면 equal-weight fallback.

    Args:
        reward: 보상함수 종류 ("return" | "sharpe" | "mdd").
        returns: fallback 계산용 전체 기간 raw 로그수익률.

    Returns:
        테스트 기간(4개 윈도우 합산) 일별 포트폴리오 로그수익률 Series.
    """
    csv_path = RESULTS_DIR / f"backtest_{reward}.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
            return df["episode_return"].dropna()
        except Exception as exc:
            logger.warning("backtest_%s.csv 로드 실패: %s — equal-weight fallback", reward, exc)

    logger.info("backtest_%s.csv 없음 — equal-weight fallback 적용", reward)
    ew = returns.mean(axis=1)
    parts = [
        ew.loc[w["test_start"]:w["test_end"]]
        for w in WINDOWS
        if not ew.loc[w["test_start"]:w["test_end"]].empty
    ]
    return pd.concat(parts) if parts else pd.Series(dtype=float)


def _slice_series(series: pd.Series, slices: list[tuple[str, str]]) -> pd.Series:
    parts = [series.loc[s:e] for s, e in slices if not series.loc[s:e].empty]
    return pd.concat(parts) if parts else pd.Series(dtype=float)


# ── 공개 API ──────────────────────────────────────────────────────────────────

def run_anova(experiment: str, groups: dict[str, pd.Series]) -> dict:
    """일원 ANOVA + Tukey HSD 수행.

    Args:
        experiment: 실험 이름.
        groups: {그룹명: 일별 로그수익률 Series} dict (2개 이상).

    Returns:
        api_spec.md AnovaResult 포맷 dict:
        name, f_statistic, p_value, eta_squared, post_hoc(list).
        유효 집단이 2개 미만이면 f_statistic=0, p_value=1로 반환.
    """
    valid = {k: v.dropna() for k, v in groups.items() if len(v.dropna()) >= 2}

    if len(valid) < 2:
        logger.warning("[%s] 유효 집단 부족 — 더미 결과 반환", experiment)
        return {
            "name": experiment,
            "f_statistic": 0.0,
            "p_value": 1.0,
            "eta_squared": 0.0,
            "post_hoc": [],
        }

    f_stat, p_val = f_oneway(*[v.values for v in valid.values()])
    eta2 = _eta_squared(list(valid.values()))
    post_hoc = _tukey_post_hoc(valid)

    return {
        "name": experiment,
        "f_statistic": round(float(f_stat) if np.isfinite(f_stat) else 0.0, 6),
        "p_value": round(float(p_val) if np.isfinite(p_val) else 1.0, 6),
        "eta_squared": round(eta2, 6),
        "post_hoc": post_hoc,
    }


def run_reward_function_comparison(returns: pd.DataFrame) -> dict:
    """실험 1: 보상함수 3종(return / sharpe / mdd) 성과 비교.

    Args:
        returns: 전체 기간 raw 로그수익률 DataFrame.

    Returns:
        run_anova() 포맷 dict.
    """
    groups = {
        f"PPO-{r}": _load_backtest_series(r, returns)
        for r in ["return", "sharpe", "mdd"]
    }
    return run_anova("reward_function_comparison", groups)


def run_strategy_comparison(returns: pd.DataFrame) -> dict:
    """실험 2: PPO vs MVO vs 동일비중 성과 비교.

    PPO: backtest_return.csv (없으면 equal-weight fallback).
    MVO: mvo.py run_mvo_all_windows (4개 윈도우 합산).
    동일비중: returns.mean(axis=1) 테스트 기간 합산.

    Args:
        returns: 전체 기간 raw 로그수익률 DataFrame.

    Returns:
        run_anova() 포맷 dict.
    """
    # PPO
    ppo_series = _load_backtest_series("return", returns)

    # MVO
    mvo_by_window = run_mvo_all_windows(returns, WINDOWS)
    mvo_parts = [s for s in mvo_by_window.values() if not s.empty]
    mvo_series = pd.concat(mvo_parts) if mvo_parts else pd.Series(dtype=float)

    # 동일비중
    ew = returns.mean(axis=1)
    ew_parts = [
        ew.loc[w["test_start"]:w["test_end"]]
        for w in WINDOWS
        if not ew.loc[w["test_start"]:w["test_end"]].empty
    ]
    ew_series = pd.concat(ew_parts) if ew_parts else pd.Series(dtype=float)

    groups = {"PPO": ppo_series, "MVO": mvo_series, "동일비중": ew_series}
    return run_anova("strategy_comparison", groups)


def run_market_regime_comparison(returns: pd.DataFrame) -> dict:
    """실험 3: 시장 국면별(bull / bear / crisis) PPO 성과 비교.

    각 국면의 수익률은 backtest_return.csv에서 날짜로 슬라이싱.
    crisis(코로나 구간)는 backtest CSV에 없으므로 equal-weight 사용.

    Args:
        returns: 전체 기간 raw 로그수익률 DataFrame.

    Returns:
        run_anova() 포맷 dict.
    """
    # PPO 결과 로드 (없으면 equal-weight)
    ppo_all = _load_backtest_series("return", returns)
    ew = returns.mean(axis=1)

    def _regime_series(slices: list[tuple[str, str]], source: pd.Series) -> pd.Series:
        return _slice_series(source, slices)

    groups: dict[str, pd.Series] = {}
    for regime, slices in _REGIME_SLICES.items():
        # bear/bull은 backtest 기간(2022~2025)에 포함 → ppo_all 사용
        # crisis(2020)는 backtest 기간 외 → equal-weight로 fallback
        source = ppo_all if regime != "crisis" else ew
        groups[regime] = _regime_series(slices, source)

    return run_anova("market_regime_comparison", groups)


def run_all_anova(returns: pd.DataFrame) -> list[dict]:
    """3종 ANOVA 실험 전체 실행.

    Args:
        returns: 전체 기간 raw 로그수익률 DataFrame.

    Returns:
        [reward_function_comparison, strategy_comparison, market_regime_comparison] 순서.
    """
    return [
        run_reward_function_comparison(returns),
        run_strategy_comparison(returns),
        run_market_regime_comparison(returns),
    ]
