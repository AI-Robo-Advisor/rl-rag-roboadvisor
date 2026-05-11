"""Walk-Forward 백테스트 프레임워크 + 코로나 구간 스트레스 테스트.

Walk-Forward 윈도우 4개(W1~W3 + Final Holdout):
    W1:    train 2018~2021 / test 2022 (금리 충격)
    W2:    train 2019~2022 / test 2023 (회복장)
    W3:    train 2020~2023 / test 2024 (AI 랠리)
    Final: train 2021~2024 / test 2025 (최신 구간)

스트레스 테스트: 코로나 구간(2020-02-01 ~ 2020-05-31)을 OOS로 활용.
Final Holdout 모델(학습 2021~2024)이 코로나 미학습 구간에서 Safe-Guard가
정상 작동하는지 검증한다.

Safe-Guard 임계값 15% 근거:
    코로나 KOSPI MDD 최대 38% — 15%는 그 절반 이하의 조기 경보 수준.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.rl.metrics import calculate_all_metrics

logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────────────────────────────

WINDOWS: list[dict[str, str]] = [
    {
        "name": "w1",
        "train_start": "2018-01-01", "train_end": "2021-12-31",
        "test_start":  "2022-01-01", "test_end":  "2022-12-31",
    },
    {
        "name": "w2",
        "train_start": "2019-01-01", "train_end": "2022-12-31",
        "test_start":  "2023-01-01", "test_end":  "2023-12-31",
    },
    {
        "name": "w3",
        "train_start": "2020-01-01", "train_end": "2023-12-31",
        "test_start":  "2024-01-01", "test_end":  "2024-12-31",
    },
    {
        "name": "final",
        "train_start": "2021-01-01", "train_end": "2024-12-31",
        "test_start":  "2025-01-01", "test_end":  "2025-12-31",
    },
]

REWARD_TYPES: list[str] = ["return", "sharpe", "mdd"]

RETURNS_PATH = Path("data/processed/returns.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
RESULTS_DIR = Path("data/results")
MODELS_DIR = Path("models")

BENCHMARK_TICKER = "SPY"
SAFEGUARD_THRESHOLD: float = 0.15
STRESS_START = "2020-02-01"
STRESS_END = "2020-05-31"


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = pd.read_parquet(RETURNS_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    return returns, features


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[start:end]


def _benchmark(returns: pd.DataFrame) -> pd.Series:
    if BENCHMARK_TICKER in returns.columns:
        return returns[BENCHMARK_TICKER]
    return returns.iloc[:, 0]


def _equal_weight_returns(returns: pd.DataFrame) -> pd.Series:
    return returns.mean(axis=1)


def _run_model(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    model_path: Path,
    reward: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """PPO 모델로 포트폴리오 수익률·비중 시계열 계산.

    Args:
        returns: 테스트 기간 raw 로그수익률.
        features: 테스트 기간 정규화 피처.
        model_path: PPO 모델 파일 경로.
        reward: 보상함수 종류 ("return" | "sharpe" | "mdd").

    Returns:
        (포트폴리오 일별 수익률 Series, 자산별 비중 DataFrame)
        모델 파일 없으면 equal-weight fallback.
    """
    n_assets = len(returns.columns)

    if not model_path.exists():
        logger.warning("모델 파일 없음 (%s) — equal-weight fallback", model_path)
        weights = pd.DataFrame(
            np.full((len(returns), n_assets), 1.0 / n_assets),
            index=returns.index,
            columns=returns.columns,
        )
        return _equal_weight_returns(returns), weights

    try:
        from stable_baselines3 import PPO
        from src.rl.env import PortfolioEnv

        model = PPO.load(str(model_path))
        env = PortfolioEnv(returns, features, reward_type=reward)
        obs, _ = env.reset()

        portfolio_returns: list[float] = []
        weight_rows: list[np.ndarray] = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            portfolio_returns.append(float(info.get("portfolio_return", 0.0)))
            weight_rows.append(
                info.get("weights", np.ones(n_assets) / n_assets)
            )
            done = terminated or truncated

        idx = returns.index[:len(portfolio_returns)]
        return (
            pd.Series(portfolio_returns, index=idx),
            pd.DataFrame(weight_rows, index=idx, columns=returns.columns),
        )

    except Exception as exc:
        logger.error("모델 실행 오류 (%s): %s — equal-weight fallback", model_path, exc)
        weights = pd.DataFrame(
            np.full((len(returns), n_assets), 1.0 / n_assets),
            index=returns.index,
            columns=returns.columns,
        )
        return _equal_weight_returns(returns), weights


def _detect_safeguard_events(portfolio: pd.Series) -> list[dict[str, Any]]:
    """MDD ≥ SAFEGUARD_THRESHOLD 구간에서 Safe-Guard 발동·재개 이벤트 탐지.

    Args:
        portfolio: 일별 포트폴리오 로그수익률.

    Returns:
        발동 이벤트 목록. 각 항목: triggered_at, drawdown_at_trigger, resumed_at.
    """
    cumulative = np.exp(portfolio.cumsum().values)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max

    events: list[dict[str, Any]] = []
    in_guard = False
    triggered_at: str | None = None
    dd_at_trigger: float = 0.0

    for date, dd in zip(portfolio.index, drawdown):
        date_str = date.strftime("%Y-%m-%d")
        if not in_guard and abs(dd) >= SAFEGUARD_THRESHOLD:
            in_guard = True
            triggered_at = date_str
            dd_at_trigger = round(abs(dd), 4)
        elif in_guard and abs(dd) < SAFEGUARD_THRESHOLD * 0.5:
            events.append({
                "triggered_at": triggered_at,
                "drawdown_at_trigger": dd_at_trigger,
                "resumed_at": date_str,
            })
            in_guard = False

    if in_guard:
        events.append({
            "triggered_at": triggered_at,
            "drawdown_at_trigger": dd_at_trigger,
            "resumed_at": None,
        })

    return events


# ── 공개 API ──────────────────────────────────────────────────────────────────

def run_window_backtest(
    window: dict[str, str],
    returns: pd.DataFrame,
    features: pd.DataFrame,
    reward: str,
) -> tuple[dict[str, Any], pd.Series, pd.DataFrame]:
    """단일 Walk-Forward 윈도우 백테스트.

    Args:
        window: WINDOWS 항목 (name, train_start, train_end, test_start, test_end).
        returns: 전체 기간 raw 로그수익률.
        features: 전체 기간 정규화 피처.
        reward: 보상함수 종류.

    Returns:
        (성과 지표 dict, 포트폴리오 수익률 Series, 비중 DataFrame)
    """
    test_ret = _slice(returns, window["test_start"], window["test_end"])
    test_feat = _slice(features, window["test_start"], window["test_end"])
    bench = _benchmark(test_ret)
    model_path = MODELS_DIR / f"ppo_{reward}_{window['name']}.zip"

    portfolio, weights_df = _run_model(test_ret, test_feat, model_path, reward)

    common = portfolio.index.intersection(bench.index)
    metrics = calculate_all_metrics(portfolio.loc[common], bench.loc[common])
    metrics.update({
        "window": window["name"],
        "reward": reward,
        "test_start": window["test_start"],
        "test_end": window["test_end"],
    })

    logger.info(
        "[%s][%s] sharpe=%.3f  mdd=%.3f  cagr=%.3f",
        reward, window["name"],
        metrics["sharpe_ratio"], metrics["mdd"], metrics["cagr"],
    )
    return metrics, portfolio, weights_df


def run_all_windows(reward: str = "return") -> pd.DataFrame:
    """4개 Walk-Forward 윈도우 순차 실행 후 data/results/ 에 저장.

    Args:
        reward: 보상함수 종류.

    Returns:
        윈도우별 성과 지표 DataFrame (4행).
    """
    returns, features = _load_data()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []
    all_portfolio: list[pd.Series] = []
    all_weights: list[pd.DataFrame] = []

    for window in WINDOWS:
        metrics, portfolio, weights_df = run_window_backtest(
            window, returns, features, reward
        )
        all_metrics.append(metrics)
        all_portfolio.append(portfolio)
        all_weights.append(weights_df)

    combined_portfolio = pd.concat(all_portfolio)
    pd.DataFrame({
        "date": combined_portfolio.index.strftime("%Y-%m-%d"),
        "episode_return": combined_portfolio.values,
    }).to_csv(RESULTS_DIR / f"backtest_{reward}.csv", index=False)

    pd.concat(all_weights).to_parquet(RESULTS_DIR / f"weights_{reward}.parquet")

    logger.info(
        "저장 완료: data/results/backtest_%s.csv, weights_%s.parquet", reward, reward
    )
    return pd.DataFrame(all_metrics)


def run_all_rewards() -> dict[str, pd.DataFrame]:
    """3종 보상함수 전체 백테스트 실행.

    Returns:
        {"return": DataFrame, "sharpe": DataFrame, "mdd": DataFrame}
    """
    return {reward: run_all_windows(reward) for reward in REWARD_TYPES}


def run_stress_test(reward: str = "return") -> dict[str, Any]:
    """코로나 구간(2020-02-01 ~ 2020-05-31) OOS 스트레스 테스트.

    Final Holdout 모델(학습 2021~2024)이 코로나를 학습하지 않았으므로
    가장 순수한 OOS 검증이 가능하다.

    Args:
        reward: 사용할 모델의 보상함수 종류.

    Returns:
        api_spec.md StressTestResponse 포맷 dict.
    """
    returns, features = _load_data()
    stress_ret = _slice(returns, STRESS_START, STRESS_END)
    stress_feat = _slice(features, STRESS_START, STRESS_END)
    bench = _benchmark(stress_ret)

    model_path = MODELS_DIR / f"ppo_{reward}_final.zip"
    portfolio, _ = _run_model(stress_ret, stress_feat, model_path, reward)

    common = portfolio.index.intersection(bench.index)
    port = portfolio.loc[common]
    bm = bench.loc[common]

    metrics = calculate_all_metrics(port, bm)
    bench_metrics = calculate_all_metrics(bm, bm)
    safeguard_events = _detect_safeguard_events(port)

    port_cum = np.exp(port.cumsum().values)
    bm_cum = np.exp(bm.cumsum().values)
    running_max = np.maximum.accumulate(port_cum)
    drawdown = (port_cum - running_max) / running_max

    return {
        "period": {"start": STRESS_START, "end": STRESS_END},
        "metrics": {k: round(v, 6) for k, v in metrics.items() if isinstance(v, float)},
        "benchmark_metrics": {k: round(v, 6) for k, v in bench_metrics.items() if isinstance(v, float)},
        "safeguard_events": safeguard_events,
        "dates": [d.strftime("%Y-%m-%d") for d in port.index],
        "ppo_cum": [round(float(v), 6) for v in port_cum],
        "bm_cum": [round(float(v), 6) for v in bm_cum],
        "drawdown": [round(float(v), 6) for v in drawdown],
    }
