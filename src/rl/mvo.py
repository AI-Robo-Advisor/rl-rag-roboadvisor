"""평균-분산 최적화(MVO) 포트폴리오 — PPO 비교 기준선.

설정:
    - 공분산 추정: 과거 252거래일 롤링 윈도우
    - 제약: 비중 합 = 1, 공매도 금지(w_i ≥ 0), 개별 자산 최대 40%
    - 리밸런싱: 월 1회 (월말 기준 실제 거래일로 스냅)
    - 최적화: scipy.optimize.minimize SLSQP (최소 분산 목적)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

LOOKBACK: int = 252
MAX_WEIGHT: float = 0.40


# ── 최적화 ────────────────────────────────────────────────────────────────────

def _optimize_min_variance(cov: np.ndarray, n_assets: int) -> np.ndarray:
    """최소 분산 포트폴리오 비중 계산.

    Args:
        cov: (n_assets × n_assets) 공분산 행렬.
        n_assets: 자산 수.

    Returns:
        합이 1인 비중 배열. 최적화 실패 시 equal-weight 반환.
    """
    w0 = np.ones(n_assets) / n_assets

    result = minimize(
        fun=lambda w: float(w @ cov @ w),
        x0=w0,
        method="SLSQP",
        bounds=[(0.0, MAX_WEIGHT)] * n_assets,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if result.success:
        w = np.clip(result.x, 0.0, MAX_WEIGHT)
        return w / w.sum()

    logger.warning("MVO 최적화 실패: %s — equal-weight fallback", result.message)
    return w0


def _snap_to_trading_days(
    cal_dates: pd.DatetimeIndex,
    trading_index: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """캘린더 월말 날짜를 실제 거래일(당일 또는 직전 거래일)로 스냅.

    Args:
        cal_dates: pd.date_range로 생성한 캘린더 날짜.
        trading_index: 실제 거래일 인덱스.

    Returns:
        거래일로 스냅된 DatetimeIndex (중복 제거).
    """
    positions = trading_index.searchsorted(cal_dates, side="right") - 1
    valid = positions[positions >= 0]
    return trading_index[valid].unique()


# ── 공개 API ──────────────────────────────────────────────────────────────────

def run_mvo(
    returns: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    rebalance_freq: str = "ME",
) -> pd.Series:
    """MVO 포트폴리오 테스트 기간 일별 로그수익률 계산.

    Walk-Forward 설계를 따라 train 기간 이전 데이터를 공분산 추정에
    활용하지 않도록, 각 리밸런싱일 기준 최근 LOOKBACK(252) 거래일만 사용한다.

    Args:
        returns: 전체 기간 raw 로그수익률 DataFrame (data/processed/returns.parquet).
        train_start: 학습 기간 시작일 (문자열 "YYYY-MM-DD").
        train_end: 학습 기간 종료일.
        test_start: 테스트 기간 시작일.
        test_end: 테스트 기간 종료일.
        rebalance_freq: pandas offset 문자열, 기본 "ME" (월말 리밸런싱).

    Returns:
        테스트 기간 일별 포트폴리오 로그수익률 Series.
        데이터가 없으면 빈 Series 반환.
    """
    test_ret = returns.loc[test_start:test_end]
    if test_ret.empty:
        logger.warning("테스트 기간 데이터 없음: %s ~ %s", test_start, test_end)
        return pd.Series(dtype=float)

    n_assets = len(returns.columns)

    # 월말 캘린더 날짜 → 실제 거래일 스냅
    cal_rebal = pd.date_range(test_start, test_end, freq=rebalance_freq)
    rebal_dates = _snap_to_trading_days(cal_rebal, test_ret.index)

    # 테스트 첫 거래일 반드시 포함
    first_day = test_ret.index[[0]]
    rebal_dates = first_day.append(rebal_dates).unique().sort_values()

    # 리밸런싱일별 최적 비중 계산
    weight_map: dict[pd.Timestamp, np.ndarray] = {}
    for date in rebal_dates:
        # 해당 날짜까지의 최근 LOOKBACK 거래일 창
        history = returns.loc[:date].iloc[:-1].iloc[-LOOKBACK:]
        if len(history) < 30:
            weight_map[date] = np.ones(n_assets) / n_assets
            logger.debug("데이터 부족 (%s, %d행) — equal-weight", date.date(), len(history))
            continue
        cov = history.cov().values
        weight_map[date] = _optimize_min_variance(cov, n_assets)

    # 일별 포트폴리오 수익률 산출
    portfolio_returns: list[float] = []
    current_weights = np.ones(n_assets) / n_assets

    for date, row in test_ret.iterrows():
        if date in weight_map:
            current_weights = weight_map[date]
        portfolio_returns.append(float(np.dot(current_weights, row.values)))

    return pd.Series(portfolio_returns, index=test_ret.index)


def run_mvo_all_windows(
    returns: pd.DataFrame,
    windows: list[dict[str, str]],
    rebalance_freq: str = "ME",
) -> dict[str, pd.Series]:
    """Walk-Forward 윈도우 전체에 대해 MVO 수익률 시계열 계산.

    Args:
        returns: 전체 기간 raw 로그수익률 DataFrame.
        windows: backtest.WINDOWS 형식의 윈도우 목록.
        rebalance_freq: pandas offset 문자열.

    Returns:
        {"w1": Series, "w2": Series, ...} 형식 dict.
    """
    return {
        w["name"]: run_mvo(
            returns,
            w["train_start"], w["train_end"],
            w["test_start"],  w["test_end"],
            rebalance_freq=rebalance_freq,
        )
        for w in windows
    }
