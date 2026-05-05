"""
포트폴리오 성과 지표 계산 모듈.

12개 지표 함수와 통합 함수 calculate_all_metrics()를 제공합니다.
모든 함수는 일별 로그수익률 pd.Series를 입력으로 받습니다.
실제 계산 로직은 Sprint 3에서 완성 예정이며, 현재는 뼈대(stub) 구현입니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
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
    "calculate_all_metrics",
]

_TRADING_DAYS = 252


# ─────────────────────────────────────────────
# 수익률
# ─────────────────────────────────────────────


def cumulative_return(returns: pd.Series) -> float:
    """
    누적 수익률을 계산합니다.

    Args:
        returns: 일별 로그수익률 시계열.

    Returns:
        누적 수익률 (예: 0.25 = 25%).
    """
    if returns.empty:
        return 0.0
    return float(np.expm1(returns.sum()))


def cagr(returns: pd.Series, years: float) -> float:
    """
    연평균 복합 성장률(CAGR)을 계산합니다.

    Args:
        returns: 일별 로그수익률 시계열.
        years: 기간(년). 0이면 0.0 반환.

    Returns:
        CAGR (예: 0.12 = 12%).
    """
    if returns.empty or years <= 0:
        return 0.0
    total = cumulative_return(returns)
    return float((1 + total) ** (1 / years) - 1)


# ─────────────────────────────────────────────
# 리스크
# ─────────────────────────────────────────────


def annualized_volatility(returns: pd.Series) -> float:
    """
    연환산 변동성(표준편차)을 계산합니다.

    Args:
        returns: 일별 로그수익률 시계열.

    Returns:
        연환산 변동성 (예: 0.18 = 18%).
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(_TRADING_DAYS))


def var_95(returns: pd.Series) -> float:
    """
    역사적 시뮬레이션 기반 95% VaR을 계산합니다 (손실은 양수).

    Args:
        returns: 일별 로그수익률 시계열.

    Returns:
        VaR (예: 0.02 = 하루 최대 손실 2%).
    """
    if returns.empty:
        return 0.0
    return float(-np.percentile(returns, 5))


def cvar_95(returns: pd.Series) -> float:
    """
    95% CVaR(Expected Shortfall)을 계산합니다 (손실은 양수).

    Args:
        returns: 일별 로그수익률 시계열.

    Returns:
        CVaR (예: 0.03 = 평균 꼬리 손실 3%).
    """
    if returns.empty:
        return 0.0
    cutoff = np.percentile(returns, 5)
    tail = returns[returns <= cutoff]
    if tail.empty:
        return 0.0
    return float(-tail.mean())


def mdd(returns: pd.Series) -> float:
    """
    최대 낙폭(Maximum Drawdown)을 계산합니다 (양수로 반환).

    Args:
        returns: 일별 로그수익률 시계열.

    Returns:
        MDD (예: 0.30 = 최대 30% 하락).
    """
    if returns.empty:
        return 0.0
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(-drawdown.min())


# ─────────────────────────────────────────────
# 위험조정 수익률
# ─────────────────────────────────────────────


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """
    샤프 비율을 계산합니다.

    Args:
        returns: 일별 로그수익률 시계열.
        rf: 일별 무위험 수익률 (기본값 0).

    Returns:
        연환산 샤프 비율.
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    excess = returns - rf
    vol = excess.std(ddof=1)
    if vol == 0:
        return 0.0
    return float((excess.mean() / vol) * np.sqrt(_TRADING_DAYS))


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """
    소르티노 비율을 계산합니다 (하방 편차만 사용).

    Args:
        returns: 일별 로그수익률 시계열.
        rf: 일별 무위험 수익률 (기본값 0).

    Returns:
        연환산 소르티노 비율.
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    excess = returns - rf
    downside = excess[excess < 0]
    if downside.empty:
        return 0.0
    downside_std = downside.std(ddof=1)
    if downside_std == 0:
        return 0.0
    return float((excess.mean() / downside_std) * np.sqrt(_TRADING_DAYS))


def calmar_ratio(returns: pd.Series) -> float:
    """
    칼마 비율(CAGR / MDD)을 계산합니다.

    Args:
        returns: 일별 로그수익률 시계열.

    Returns:
        칼마 비율. MDD가 0이면 0.0 반환.
    """
    if returns.empty:
        return 0.0
    years = len(returns) / _TRADING_DAYS
    _cagr = cagr(returns, years)
    _mdd = mdd(returns)
    if _mdd == 0:
        return 0.0
    return float(_cagr / _mdd)


# ─────────────────────────────────────────────
# 상대 성과
# ─────────────────────────────────────────────


def alpha(returns: pd.Series, benchmark: pd.Series) -> float:
    """
    젠센 알파(Jensen's Alpha)를 계산합니다.

    Args:
        returns: 포트폴리오 일별 로그수익률.
        benchmark: 벤치마크 일별 로그수익률.

    Returns:
        연환산 알파.
    """
    if returns.empty or benchmark.empty:
        return 0.0
    aligned_r, aligned_b = returns.align(benchmark, join="inner")
    if len(aligned_r) < 2:
        return 0.0
    _beta = beta(aligned_r, aligned_b)
    return float((aligned_r.mean() - _beta * aligned_b.mean()) * _TRADING_DAYS)


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """
    포트폴리오 베타를 계산합니다.

    Args:
        returns: 포트폴리오 일별 로그수익률.
        benchmark: 벤치마크 일별 로그수익률.

    Returns:
        베타. 벤치마크 분산이 0이면 0.0 반환.
    """
    if returns.empty or benchmark.empty:
        return 0.0
    aligned_r, aligned_b = returns.align(benchmark, join="inner")
    if len(aligned_r) < 2:
        return 0.0
    cov_matrix = np.cov(aligned_r.values, aligned_b.values)
    bench_var = float(cov_matrix[1, 1])
    if bench_var == 0:
        return 0.0
    return float(cov_matrix[0, 1] / bench_var)


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    """
    정보 비율(Information Ratio)을 계산합니다.

    Args:
        returns: 포트폴리오 일별 로그수익률.
        benchmark: 벤치마크 일별 로그수익률.

    Returns:
        연환산 정보 비율. 추적오차가 0이면 0.0 반환.
    """
    if returns.empty or benchmark.empty:
        return 0.0
    aligned_r, aligned_b = returns.align(benchmark, join="inner")
    if len(aligned_r) < 2:
        return 0.0
    active = aligned_r - aligned_b
    te = active.std(ddof=1)
    if te == 0:
        return 0.0
    return float((active.mean() / te) * np.sqrt(_TRADING_DAYS))


# ─────────────────────────────────────────────
# 통합 함수
# ─────────────────────────────────────────────


def calculate_all_metrics(
    returns: pd.Series,
    benchmark: pd.Series,
) -> dict[str, float]:
    """
    12개 성과 지표를 한 번에 계산해 dict로 반환합니다.

    Args:
        returns: 포트폴리오 일별 로그수익률.
        benchmark: 벤치마크 일별 로그수익률 (alpha/beta/IR 계산에 사용).

    Returns:
        지표명 → float 매핑 dict. 키 12개 고정:
        cumulative_return, cagr, annualized_volatility, var_95, cvar_95,
        mdd, sharpe_ratio, sortino_ratio, calmar_ratio, alpha, beta,
        information_ratio.
    """
    years = len(returns) / _TRADING_DAYS if not returns.empty else 1.0
    return {
        "cumulative_return":     cumulative_return(returns),
        "cagr":                  cagr(returns, years),
        "annualized_volatility": annualized_volatility(returns),
        "var_95":                var_95(returns),
        "cvar_95":               cvar_95(returns),
        "mdd":                   mdd(returns),
        "sharpe_ratio":          sharpe_ratio(returns),
        "sortino_ratio":         sortino_ratio(returns),
        "calmar_ratio":          calmar_ratio(returns),
        "alpha":                 alpha(returns, benchmark),
        "beta":                  beta(returns, benchmark),
        "information_ratio":     information_ratio(returns, benchmark),
    }
