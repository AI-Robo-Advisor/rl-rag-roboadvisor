"""MDD 페널티 보상의 lambda 민감도 실험 (과제 명세 §5-3 line 86).

W1 윈도우(학습 2018~2021 / 테스트 2022)에 대해 lambda_mdd ∈ {0.5, 1.0, 2.0, 5.0}
4개 모델을 학습하고, 각 모델의 학습 곡선(monitor.csv)과 OOS 백테스트 지표를
data/results/lambda_sweep/ 아래 저장한다.

산출물:
    models/ppo_mdd_w1_lambda{X}.zip                              (4개)
    logs/lambda_sweep/monitor_lambda{X}.monitor.csv              (4개)
    data/results/lambda_sweep/metrics_summary.csv                 (4행)
    data/results/lambda_sweep/lambda_tradeoff.png                 (수익률-MDD 산점도)
    data/results/lambda_sweep/training_curves.png                 (4개 학습 곡선 패널)

실행:
    PYTHONPATH=. python scripts/lambda_sweep_w1.py
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.rl.env import PortfolioEnv
from src.rl.metrics import calculate_all_metrics
from src.rl.train_walkforward import (
    LOOKBACK,
    TOTAL_TIMESTEPS,
    WINDOWS,
    align_data,
    apply_normalization,
    load_risk_data,
    load_training_data,
    normalize_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LAMBDAS: list[float] = [0.5, 1.0, 2.0, 5.0]
WINDOW_NAME = "w1"
REWARD_TYPE = "mdd"
BENCHMARK_TICKER = "SPY"

MODELS_DIR = Path("models")
RESULTS_DIR = Path("data/results/lambda_sweep")
LOGS_DIR = Path("logs/lambda_sweep")


def _model_path(lam: float) -> Path:
    """학습 모델 경로 (lambda 값을 소수점 둘째자리까지 인코딩)."""
    tag = f"{lam:.2f}".replace(".", "p")
    return MODELS_DIR / f"ppo_mdd_w1_lambda{tag}.zip"


def _monitor_path(lam: float) -> Path:
    tag = f"{lam:.2f}".replace(".", "p")
    return LOGS_DIR / f"monitor_lambda{tag}.monitor.csv"


def train_lambda(
    lam: float,
    returns_df: pd.DataFrame,
    raw_features_df: pd.DataFrame,
    risk_df: pd.DataFrame | None,
    total_timesteps: int,
) -> Path:
    """단일 lambda 값으로 W1 mdd 모델 1개를 학습한다.

    Args:
        lam: lambda_mdd 값.
        returns_df: 전체 기간 raw 로그수익률.
        raw_features_df: 전체 기간 raw 피처.
        risk_df: 일별 risk_vectors. None이면 risk=0.
        total_timesteps: PPO 총 학습 스텝.

    Returns:
        저장된 모델 zip 경로.
    """
    window = WINDOWS[WINDOW_NAME]
    train_returns = returns_df.loc[window["train_start"] : window["train_end"]].copy()
    train_raw = raw_features_df.loc[window["train_start"] : window["train_end"]].copy()
    train_returns, train_raw = align_data(train_returns, train_raw, WINDOW_NAME)

    train_features, _ = normalize_features(train_raw, WINDOW_NAME)
    risk_series = (
        risk_df.loc[window["train_start"] : window["train_end"]]
        if risk_df is not None
        else None
    )

    env = PortfolioEnv(
        returns_df=train_returns,
        features_df=train_features,
        lookback=LOOKBACK,
        reward_type=REWARD_TYPE,
        lambda_mdd=lam,
        risk_series=risk_series,
    )
    monitor_csv = _monitor_path(lam)
    monitor_csv.parent.mkdir(parents=True, exist_ok=True)
    # Monitor에 filename(확장자 제외)을 주면 ".monitor.csv"가 자동 append되므로
    # ".monitor.csv" suffix를 한 번만 적용하려면 str(path)[:-len(".monitor.csv")]
    monitor_prefix = str(monitor_csv).replace(".monitor.csv", "")
    env = Monitor(env, filename=monitor_prefix)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        seed=42,
    )

    logger.info("[lambda=%.2f] 학습 시작 (%d steps)", lam, total_timesteps)
    model.learn(total_timesteps=total_timesteps)
    model_path = _model_path(lam)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path).replace(".zip", ""))
    logger.info("[lambda=%.2f] 학습 완료 → %s", lam, model_path)
    return model_path


def backtest_lambda(
    lam: float,
    model_path: Path,
    returns_df: pd.DataFrame,
    raw_features_df: pd.DataFrame,
    risk_df: pd.DataFrame | None,
) -> dict:
    """학습된 모델을 W1 테스트 구간에서 평가하여 12개 지표를 반환한다.

    Args:
        lam: lambda_mdd 값 (결과에 라벨로 저장).
        model_path: 학습된 PPO zip 경로.
        returns_df: 전체 raw 로그수익률.
        raw_features_df: 전체 raw 피처.
        risk_df: risk_vectors.

    Returns:
        backtest_metrics.csv 호환 dict + {"lambda": lam}.
    """
    window = WINDOWS[WINDOW_NAME]
    train_returns = returns_df.loc[window["train_start"] : window["train_end"]].copy()
    train_raw = raw_features_df.loc[window["train_start"] : window["train_end"]].copy()
    train_returns, train_raw = align_data(train_returns, train_raw, WINDOW_NAME)
    _, stats = normalize_features(train_raw, WINDOW_NAME)

    test_returns = returns_df.loc[window["test_start"] : window["test_end"]].copy()
    test_raw = raw_features_df.loc[window["test_start"] : window["test_end"]].copy()
    test_returns, test_raw = align_data(test_returns, test_raw, WINDOW_NAME)
    test_features = apply_normalization(test_raw, stats)

    risk_series = (
        risk_df.loc[window["test_start"] : window["test_end"]]
        if risk_df is not None
        else None
    )

    env = PortfolioEnv(
        returns_df=test_returns,
        features_df=test_features,
        lookback=LOOKBACK,
        reward_type=REWARD_TYPE,
        lambda_mdd=lam,
        risk_series=risk_series,
    )

    model = PPO.load(str(model_path))
    obs, _ = env.reset()
    daily_returns: list[float] = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if "net_return" in info:
            daily_returns.append(float(info["net_return"]))

    test_dates = test_returns.index[LOOKBACK : LOOKBACK + len(daily_returns)]
    portfolio = pd.Series(daily_returns, index=test_dates)
    bench = test_returns[BENCHMARK_TICKER].loc[portfolio.index]

    metrics = calculate_all_metrics(portfolio, bench)
    metrics["lambda"] = lam
    metrics["window"] = WINDOW_NAME
    metrics["reward"] = REWARD_TYPE
    metrics["test_start"] = window["test_start"]
    metrics["test_end"] = window["test_end"]
    metrics["n_test_days"] = len(portfolio)
    logger.info(
        "[lambda=%.2f] OOS cum=%.4f  MDD=%.4f  Sharpe=%.3f  Alpha=%.3f",
        lam,
        metrics["cumulative_return"],
        metrics["mdd"],
        metrics["sharpe_ratio"],
        metrics["alpha"],
    )
    return metrics


def plot_tradeoff(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """누적수익률 vs MDD 트레이드오프 산점도 (lambda 라벨 포함)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(
        metrics_df["mdd"] * 100,
        metrics_df["cumulative_return"] * 100,
        s=120,
        c=metrics_df["lambda"],
        cmap="viridis",
        edgecolor="black",
        zorder=3,
    )
    for _, row in metrics_df.iterrows():
        ax.annotate(
            f"λ={row['lambda']:.1f}",
            (row["mdd"] * 100, row["cumulative_return"] * 100),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=10,
        )

    ax.axvline(15.0, color="red", linestyle="--", alpha=0.6, label="Safe-Guard 15%")
    ax.set_xlabel("MDD (%)")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("W1 (2022 rate-hike) — MDD penalty λ tradeoff")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Tradeoff plot saved → %s", out_path)


def plot_training_curves(out_path: Path, smooth_window: int = 20) -> None:
    """4개 lambda 모델의 monitor.csv에서 episode reward 곡선을 그린다."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for lam in LAMBDAS:
        csv_path = _monitor_path(lam)
        if not csv_path.exists():
            logger.warning("monitor 파일 없음 → 건너뜀: %s", csv_path)
            continue
        df = pd.read_csv(csv_path, skiprows=1)  # 첫 줄은 메타데이터 헤더
        if "r" not in df.columns or df.empty:
            continue
        cumsteps = df["l"].cumsum()
        smoothed = df["r"].rolling(smooth_window, min_periods=1).mean()
        ax.plot(cumsteps, smoothed, label=f"λ={lam:.1f}", linewidth=1.6)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(f"Episode Reward (rolling mean, window={smooth_window})")
    ax.set_title("W1 PPO-mdd Training Curves by λ")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved → %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="W1 mdd reward lambda sweep")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help=f"PPO 총 학습 스텝 (기본 {TOTAL_TIMESTEPS}).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="학습 건너뛰고 기존 모델로 백테스트·플롯만 실행.",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    returns_df, raw_features_df = load_training_data()
    risk_df = load_risk_data()

    metrics_rows: list[dict] = []
    for lam in LAMBDAS:
        model_path = _model_path(lam)
        if not args.skip_train or not model_path.exists():
            train_lambda(lam, returns_df, raw_features_df, risk_df, args.timesteps)
        else:
            logger.info("[lambda=%.2f] --skip-train: 기존 모델 사용 → %s", lam, model_path)
        metrics_rows.append(
            backtest_lambda(lam, model_path, returns_df, raw_features_df, risk_df)
        )

    metrics_df = pd.DataFrame(metrics_rows)
    summary_path = RESULTS_DIR / "metrics_summary.csv"
    metrics_df.to_csv(summary_path, index=False)
    logger.info("Metrics summary saved → %s", summary_path)

    # 마크다운 요약 (보고서 첨부용)
    md_lines = [
        "# Lambda Sweep Summary (W1)",
        "",
        f"- timesteps: {args.timesteps}",
        f"- benchmark: {BENCHMARK_TICKER}",
        "",
        "| λ | cum_return | CAGR | ann_vol | MDD | Sharpe | Sortino | Calmar | Alpha | Beta |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in metrics_df.sort_values("lambda").iterrows():
        md_lines.append(
            f"| {r['lambda']:.1f} | {r['cumulative_return']:.4f} | {r['cagr']:.4f} | "
            f"{r['annualized_volatility']:.4f} | {r['mdd']:.4f} | {r['sharpe_ratio']:.3f} | "
            f"{r['sortino_ratio']:.3f} | {r['calmar_ratio']:.3f} | "
            f"{r['alpha']:.3f} | {r['beta']:.3f} |"
        )
    (RESULTS_DIR / "metrics_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    plot_tradeoff(metrics_df, RESULTS_DIR / "lambda_tradeoff.png")
    plot_training_curves(RESULTS_DIR / "training_curves.png")

    logger.info("=" * 60)
    logger.info("Lambda sweep 완료 — 산출물 디렉토리: %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
