from pathlib import Path

import numpy as np
import pandas as pd


RETURNS_PATH = Path("data/processed/returns.parquet")
OUTPUT_PATH = Path("data/results/safeguard_threshold_analysis.csv")

PERIODS = {
    "covid_crisis_2020": ("2020-02-01", "2020-05-31"),
    "rate_hike_2022": ("2022-01-01", "2022-12-31"),
}

BENCHMARKS = ["SPY", "069500"]


def calculate_mdd(log_returns: pd.Series) -> float:
    """로그수익률 시계열의 최대 낙폭(MDD)을 계산합니다."""
    cumulative = np.exp(log_returns.cumsum())
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return abs(drawdown.min())


def calculate_volatility(log_returns: pd.Series) -> float:
    """로그수익률 시계열의 연환산 변동성을 계산합니다."""
    return log_returns.std() * np.sqrt(252)


def calculate_var_95(log_returns: pd.Series) -> float:
    """로그수익률 시계열의 95% VaR를 손실 기준 양수로 계산합니다."""
    return abs(np.percentile(log_returns, 5))


def main() -> None:
    returns = pd.read_parquet(RETURNS_PATH)
    returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index()

    rows = []

    for period_name, (start, end) in PERIODS.items():
        period_returns = returns.loc[start:end]

        targets = {
            "equal_weight": period_returns.mean(axis=1),
        }

        for ticker in BENCHMARKS:
            if ticker in period_returns.columns:
                targets[ticker] = period_returns[ticker]

        for name, series in targets.items():
            rows.append(
                {
                    "period": period_name,
                    "asset": name,
                    "start": start,
                    "end": end,
                    "mdd": calculate_mdd(series),
                    "annualized_volatility": calculate_volatility(series),
                    "var_95": calculate_var_95(series),
                }
            )

    result = pd.DataFrame(rows)
    print(result)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"\n저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()