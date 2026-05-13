import numpy as np
import pandas as pd


RETURNS_PATH = "data/processed/returns.parquet"

PERIODS = {
    "covid_crisis_2020": ("2020-02-01", "2020-05-31"),
    "rate_hike_2022": ("2022-01-01", "2022-12-31"),
}

BENCHMARKS = ["SPY", "069500"]


def calculate_mdd(log_returns: pd.Series) -> float:
    cumulative = np.exp(log_returns.cumsum())
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return abs(drawdown.min())


def calculate_volatility(log_returns: pd.Series) -> float:
    return log_returns.std() * np.sqrt(252)


def calculate_var_95(log_returns: pd.Series) -> float:
    return abs(np.percentile(log_returns, 5))


def main():
    returns = pd.read_parquet(RETURNS_PATH)
    returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index()

    rows = []

    for period_name, (start, end) in PERIODS.items():
        period_returns = returns.loc[start:end]

        equal_weight = period_returns.mean(axis=1)

        targets = {
            "equal_weight": equal_weight,
        }

        for ticker in BENCHMARKS:
            if ticker in period_returns.columns:
                targets[ticker] = period_returns[ticker]

        for name, series in targets.items():
            rows.append({
                "period": period_name,
                "asset": name,
                "start": start,
                "end": end,
                "mdd": calculate_mdd(series),
                "annualized_volatility": calculate_volatility(series),
                "var_95": calculate_var_95(series),
            })

    result = pd.DataFrame(rows)
    print(result)

    result.to_csv("data/results/safeguard_threshold_analysis.csv", index=False)
    print("\n저장 완료: data/results/safeguard_threshold_analysis.csv")


if __name__ == "__main__":
    main()