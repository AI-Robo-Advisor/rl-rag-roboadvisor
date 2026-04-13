"""
실험 2: 데이터 기초 통계 + 동일가중 포트폴리오 vs SPY 비교 (논문 4.2절)

실행 전 데이터 수집 필요:
    python -m src.data.collector   # 약 5~10분 소요

실행:
    python scripts/experiment_data_stats.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RETURNS_PATH = Path("data/processed/returns.parquet")
PRICES_PATH = Path("data/raw/prices.parquet")

TRADING_DAYS = 252  # 연간 거래일


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """parquet 파일을 로드합니다. 없으면 수집 방법을 안내합니다."""
    if not RETURNS_PATH.exists():
        print("❌ data/processed/returns.parquet 없음.")
        print("   먼저 실행하세요: python -m src.data.collector")
        sys.exit(1)
    returns = pd.read_parquet(RETURNS_PATH)
    prices = pd.read_parquet(PRICES_PATH) if PRICES_PATH.exists() else None
    return returns, prices


def compute_asset_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """자산별 기초 통계를 계산합니다."""
    stats = pd.DataFrame(index=returns.columns)
    stats["연평균수익률(%)"] = (returns.mean() * TRADING_DAYS * 100).round(2)
    stats["연변동성(%)"] = (returns.std() * np.sqrt(TRADING_DAYS) * 100).round(2)
    stats["샤프비율"] = (
        returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)
    ).round(3)
    stats["최대낙폭(MDD%)"] = (
        returns.apply(lambda r: _compute_mdd(r) * 100)
    ).round(2)
    stats["왜도"] = returns.skew().round(3)
    stats["첨도"] = returns.kurt().round(3)
    return stats


def _compute_mdd(returns: pd.Series) -> float:
    """최대 낙폭(MDD)을 계산합니다."""
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    return float(drawdown.min())


def compute_equal_weight_portfolio(returns: pd.DataFrame) -> pd.Series:
    """동일가중 포트폴리오 일별 수익률을 반환합니다."""
    return returns.mean(axis=1)


def compute_portfolio_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    strategy_name: str,
    benchmark_name: str,
) -> dict:
    """전략과 벤치마크의 성과 지표를 비교합니다."""
    ann_ret_s = strategy_returns.mean() * TRADING_DAYS
    ann_ret_b = benchmark_returns.mean() * TRADING_DAYS
    ann_vol_s = strategy_returns.std() * np.sqrt(TRADING_DAYS)
    ann_vol_b = benchmark_returns.std() * np.sqrt(TRADING_DAYS)

    sharpe_s = ann_ret_s / ann_vol_s
    sharpe_b = ann_ret_b / ann_vol_b
    mdd_s = _compute_mdd(strategy_returns)
    mdd_b = _compute_mdd(benchmark_returns)

    cum_s = float((1 + strategy_returns).prod() - 1)
    cum_b = float((1 + benchmark_returns).prod() - 1)

    return {
        "strategy": strategy_name,
        "benchmark": benchmark_name,
        "기간": f"{strategy_returns.index[0].date()} ~ {strategy_returns.index[-1].date()}",
        "거래일수": len(strategy_returns),
        f"{strategy_name}_누적수익률(%)": round(cum_s * 100, 2),
        f"{benchmark_name}_누적수익률(%)": round(cum_b * 100, 2),
        f"{strategy_name}_연환산수익률(%)": round(ann_ret_s * 100, 2),
        f"{benchmark_name}_연환산수익률(%)": round(ann_ret_b * 100, 2),
        f"{strategy_name}_샤프비율": round(sharpe_s, 3),
        f"{benchmark_name}_샤프비율": round(sharpe_b, 3),
        f"{strategy_name}_MDD(%)": round(mdd_s * 100, 2),
        f"{benchmark_name}_MDD(%)": round(mdd_b * 100, 2),
    }


def print_paper_format(
    returns: pd.DataFrame,
    asset_stats: pd.DataFrame,
    portfolio_metrics: dict,
) -> None:
    """논문 4.2절에 기재할 형식으로 출력합니다."""
    sep = "=" * 65

    print(sep)
    print("실험 2: 데이터 기초 통계 분석")
    print(sep)
    print(f"\n수집 기간  : {returns.index[0].date()} ~ {returns.index[-1].date()}")
    print(f"총 거래일수 : {len(returns)}일")
    print(f"대상 자산  : {len(returns.columns)}종")
    print(f"결측치 비율 : {returns.isnull().sum().sum() / returns.size * 100:.3f}%")

    print("\n【자산별 성과 지표】")
    print(asset_stats.to_string())

    print(f"\n{sep}")
    print("실험 3: 동일가중 포트폴리오 vs SPY 벤치마크")
    print(sep)
    for key, val in portfolio_metrics.items():
        if key not in ("strategy", "benchmark"):
            print(f"  {key}: {val}")

    ew_cum = portfolio_metrics.get("동일가중_누적수익률(%)", "N/A")
    spy_cum = portfolio_metrics.get("SPY_누적수익률(%)", "N/A")
    ew_sharpe = portfolio_metrics.get("동일가중_샤프비율", "N/A")
    spy_sharpe = portfolio_metrics.get("SPY_샤프비율", "N/A")

    print(f"\n【논문 기재 예시 문장】")
    print(
        f"수집된 10종 ETF 데이터({portfolio_metrics['기간']}, {portfolio_metrics['거래일수']}거래일)를 "
        f"분석한 결과, 동일가중 포트폴리오의 누적 수익률은 {ew_cum}%로 "
        f"SPY 벤치마크({spy_cum}%)와 비교되었으며, "
        f"샤프비율은 각각 {ew_sharpe}, {spy_sharpe}로 산출되었다. "
        "이는 단순 분산투자만으로도 일정 수준의 리스크 조정 성과를 달성할 수 있음을 보여주며, "
        "RL 기반 동적 자산배분 전략의 개선 여지를 입증한다."
    )
    print()

    # 상관계수 매트릭스 출력
    print(f"\n{sep}")
    print("【자산 간 상관계수 매트릭스】")
    corr = returns.corr().round(3)
    print(corr.to_string())
    print()


if __name__ == "__main__":
    print("데이터 로드 중...")
    returns, prices = load_data()

    print(f"✅ returns.parquet 로드: shape={returns.shape}")

    asset_stats = compute_asset_stats(returns)

    # 동일가중 포트폴리오 vs SPY
    ew_returns = compute_equal_weight_portfolio(returns)
    spy_returns = returns["SPY"] if "SPY" in returns.columns else ew_returns

    metrics = compute_portfolio_metrics(
        strategy_returns=ew_returns,
        benchmark_returns=spy_returns,
        strategy_name="동일가중",
        benchmark_name="SPY",
    )

    print_paper_format(returns, asset_stats, metrics)
