"""Deterministic service layer for Sprint 2 API contracts.

The RL, SHAP, and RAG modules are still being developed by separate owners.
These helpers keep the HTTP contract stable and make later module integration
small: replace the fallback body, keep the response schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apps.api.config import settings
from apps.api.schemas import (
    AnovaResult,
    BacktestResponse,
    ExplainResponse,
    FeatureContribution,
    OptimizeResponse,
    ReturnSeries,
    ResearchResponse,
    RiskProfile,
)

RETURNS_PATH = Path("data/processed/returns.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
TRADING_DAYS = 252

DEFAULT_TICKERS: list[str] = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "GLD",
    "VNQ",
    "069500",
    "114260",
]


def build_fallback_portfolio(
    tickers: list[str] | None = None,
    risk_profile: RiskProfile = "balanced",
    risk_aversion: float | None = None,
) -> OptimizeResponse:
    """Return deterministic normalized weights until the PPO model is connected."""
    selected_tickers = tickers or DEFAULT_TICKERS
    raw_weights = _risk_adjusted_raw_weights(selected_tickers, risk_profile, risk_aversion)
    total_weight = sum(raw_weights.values())
    weights = {ticker: weight / total_weight for ticker, weight in raw_weights.items()}
    return_series, expected_return, expected_volatility = _build_return_series(weights)

    return OptimizeResponse(
        status="fallback",
        tickers=selected_tickers,
        weights=weights,
        risk_profile=risk_profile,
        expected_return=expected_return or _profile_return(risk_profile),
        expected_volatility=expected_volatility or _profile_volatility(risk_profile),
        returns=return_series,
        message="PPO 모델 연결 전 deterministic fallback 포트폴리오입니다.",
    )


def build_fallback_explanation(date: str | None, top_k: int) -> ExplainResponse:
    """Return SHAP-like feature contributions for dashboard integration."""
    features = _feature_contributions_from_parquet(date, top_k) or _static_feature_contributions()

    selected = features[:top_k]
    prediction = 0.05 + sum(item.contribution for item in selected)
    target_date = date or _latest_feature_date()
    return ExplainResponse(
        status="fallback",
        date=date,
        target_date=target_date,
        base_value=0.05,
        prediction=round(prediction, 6),
        feature_contributions=selected,
        feature_names=[item.feature for item in selected],
        shap_values=[item.contribution for item in selected],
        message="SHAP 모듈 연결 전 feature contribution fallback입니다.",
    )


def run_graph(question: str) -> dict[str, Any]:
    """Import and run LangGraph lazily so API import stays safe without an API key."""
    from src.agent.graph import run_graph as _run_graph

    return _run_graph(question)


def build_research_response(question: str) -> ResearchResponse:
    """Run LangGraph when configured, otherwise return the deterministic fallback."""
    if not settings.OPENAI_API_KEY:
        return build_fallback_research(question)

    try:
        state = run_graph(question)
    except Exception:
        return build_fallback_research(question)

    report = str(state.get("response") or "").strip()
    if not report:
        return build_fallback_research(question)

    messages = state.get("messages") or []
    reasoning_trace = state.get("reasoning_trace") or "\n".join(str(item) for item in messages)
    risk_tags = state.get("rl_risk_tags") or state.get("risk_tags") or _infer_risk_tags(question)
    sources = [str(source) for source in state.get("sources", []) if source]

    return ResearchResponse(
        status="ready",
        question=question,
        report=report,
        sources=sources or ["https://github.com/AI-Robo-Advisor/rl-rag-roboadvisor"],
        reasoning_trace=reasoning_trace,
        risk_tags=[str(tag) for tag in risk_tags],
    )


def build_fallback_research(question: str) -> ResearchResponse:
    """Return a stable RAG-style payload until LangGraph is connected."""
    risk_tags = _infer_risk_tags(question)
    return ResearchResponse(
        status="fallback",
        question=question,
        report=(
            "현재 응답은 LangGraph 에이전트 연결 전 fallback입니다. 질문의 핵심 위험 요인을 "
            "태그로 분류하고, 대시보드 연동을 위한 리포트 구조를 우선 제공합니다."
        ),
        sources=["https://github.com/AI-Robo-Advisor/rl-rag-roboadvisor"],
        reasoning_trace="\n".join(
            [
                "Planner: 투자 질문의 핵심 키워드를 식별했습니다.",
                "Researcher: 실제 ChromaDB 연결 전 fallback source를 사용했습니다.",
                "Analyst: 리스크 태그와 요약 리포트를 생성했습니다.",
            ]
        ),
        risk_tags=risk_tags,
    )


def build_fallback_backtest() -> BacktestResponse:
    """Return metric output from available data plus fallback ANOVA summaries."""
    metrics, dates, wf_cum, bm_cum, rewards, drawdown, sharpe_spark = _build_backtest_payload()
    anova = _fallback_anova()
    current_mdd = float(metrics.get("mdd", 0.0))
    return BacktestResponse(
        status="fallback",
        metrics=metrics,
        anova=anova,
        benchmark="SPY",
        dates=dates,
        rewards=rewards,
        wf_cum=wf_cum,
        bm_cum=bm_cum,
        wf_spark=wf_cum[-50:],
        sharpe_spark=sharpe_spark,
        drawdown=drawdown,
        var_95=float(metrics.get("var_95", 0.0)),
        cvar_95=float(metrics.get("cvar_95", 0.0)),
        mdd=current_mdd,
        safeguard={
            "active": False,
            "triggered_at": None,
            "current_drawdown": abs(drawdown[-1]) if drawdown else current_mdd,
        },
        message="Walk-Forward 백테스트 모듈 연결 전 fallback 결과입니다.",
    )


def build_module_statuses() -> dict[str, str]:
    """Return runtime readiness using only files and modules available locally."""
    return {
        "data": "ready" if _can_load_data_files() else "fallback",
        "rl": "fallback",
        "rag": "ready" if settings.OPENAI_API_KEY else "fallback",
        "shap": "fallback",
        "backtest": "fallback",
    }


def _risk_adjusted_raw_weights(
    tickers: list[str],
    risk_profile: RiskProfile,
    risk_aversion: float | None,
) -> dict[str, float]:
    """Build simple deterministic tilts by risk profile."""
    raw_weights = {ticker: 1.0 for ticker in tickers}
    if risk_aversion is not None:
        defensive_tilt = min(max(risk_aversion, 0.1), 5.0) / 2.0
        growth_tilt = max(0.1, 2.5 / max(risk_aversion, 0.1)) / 2.0
        for ticker in ("TLT", "GLD", "114260"):
            if ticker in raw_weights:
                raw_weights[ticker] += defensive_tilt
        for ticker in ("SPY", "QQQ", "IWM", "EEM"):
            if ticker in raw_weights:
                raw_weights[ticker] += growth_tilt
    elif risk_profile == "conservative":
        for ticker in ("TLT", "GLD", "114260"):
            if ticker in raw_weights:
                raw_weights[ticker] += 0.5
    elif risk_profile == "aggressive":
        for ticker in ("SPY", "QQQ", "IWM", "EEM"):
            if ticker in raw_weights:
                raw_weights[ticker] += 0.5
    return raw_weights


def _profile_return(risk_profile: RiskProfile) -> float:
    """Expected annual return placeholder by risk profile."""
    return {"conservative": 0.052, "balanced": 0.084, "aggressive": 0.112}[risk_profile]


def _profile_volatility(risk_profile: RiskProfile) -> float:
    """Expected annual volatility placeholder by risk profile."""
    return {"conservative": 0.085, "balanced": 0.132, "aggressive": 0.184}[risk_profile]


def _build_return_series(
    weights: dict[str, float],
) -> tuple[ReturnSeries, float | None, float | None]:
    """Build cumulative portfolio and benchmark series from returns.parquet when available."""
    try:
        returns = pd.read_parquet(RETURNS_PATH)
        usable = [ticker for ticker in weights if ticker in returns.columns]
        if not usable:
            return _static_return_series(), None, None

        usable_weights = pd.Series({ticker: weights[ticker] for ticker in usable}, dtype=float)
        usable_weights = usable_weights / usable_weights.sum()
        portfolio_returns = returns[usable].mul(usable_weights, axis=1).sum(axis=1)
        benchmark_returns = returns["SPY"] if "SPY" in returns.columns else returns[usable[0]]

        portfolio_cum = np.exp(portfolio_returns.cumsum())
        benchmark_cum = np.exp(benchmark_returns.cumsum())
        years = len(portfolio_returns) / TRADING_DAYS
        expected_return = float(portfolio_cum.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0
        expected_volatility = float(portfolio_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
        return (
            ReturnSeries(
                date=[index.strftime("%Y-%m-%d") for index in returns.index],
                portfolio=_finite_float_list(portfolio_cum),
                benchmark=_finite_float_list(benchmark_cum),
            ),
            round(expected_return, 6),
            round(expected_volatility, 6),
        )
    except (OSError, ValueError, KeyError, ImportError):
        return _static_return_series(), None, None


def _static_return_series() -> ReturnSeries:
    """Return deterministic chart data when parquet data is unavailable."""
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    portfolio = np.cumprod(np.full(len(dates), 1.00035))
    benchmark = np.cumprod(np.full(len(dates), 1.0002))
    return ReturnSeries(
        date=[item.strftime("%Y-%m-%d") for item in dates],
        portfolio=_finite_float_list(portfolio),
        benchmark=_finite_float_list(benchmark),
    )


def _feature_contributions_from_parquet(
    requested_date: str | None,
    top_k: int,
) -> list[FeatureContribution] | None:
    """Build deterministic SHAP-like contributions from the nearest feature row."""
    try:
        features = pd.read_parquet(FEATURES_PATH)
        if features.empty:
            return None
        if requested_date:
            selected_date = pd.Timestamp(requested_date)
            candidates = features.loc[features.index <= selected_date]
            row = candidates.iloc[-1] if not candidates.empty else features.iloc[0]
        else:
            row = features.iloc[-1]
    except (OSError, ValueError, ImportError):
        return None

    selected_columns = row.abs().sort_values(ascending=False).head(top_k).index
    return [
        FeatureContribution(
            feature=str(column),
            value=round(float(row[column]), 6),
            contribution=round(float(row[column]) * 0.01, 6),
        )
        for column in selected_columns
    ]


def _static_feature_contributions() -> list[FeatureContribution]:
    """Return stable explanation data when features.parquet is unavailable."""
    return [
        FeatureContribution(feature="SPY_return_30d", value=0.018, contribution=0.031),
        FeatureContribution(feature="TLT_return_30d", value=-0.011, contribution=-0.014),
        FeatureContribution(feature="QQQ_RSI", value=61.2, contribution=0.019),
        FeatureContribution(feature="GLD_MACD_signal", value=0.004, contribution=0.012),
        FeatureContribution(feature="069500_return_30d", value=0.009, contribution=0.008),
        FeatureContribution(feature="VNQ_RSI", value=47.6, contribution=-0.006),
        FeatureContribution(feature="EEM_MACD_signal", value=-0.003, contribution=-0.005),
        FeatureContribution(feature="114260_return_30d", value=0.006, contribution=0.004),
    ]


def _latest_feature_date() -> str | None:
    """Return the latest feature date if available."""
    try:
        features = pd.read_parquet(FEATURES_PATH, columns=[])
        if features.empty:
            return None
        return features.index[-1].strftime("%Y-%m-%d")
    except (OSError, ValueError, ImportError):
        return None


def _build_backtest_payload() -> tuple[
    dict[str, float],
    list[str],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Use returns.parquet and metrics.py when available; otherwise return deterministic fallback."""
    try:
        from src.rl.metrics import calculate_all_metrics

        returns = pd.read_parquet(RETURNS_PATH)
        if returns.empty:
            raise ValueError("returns.parquet is empty")

        portfolio_returns = returns.mean(axis=1)
        benchmark_returns = returns["SPY"] if "SPY" in returns.columns else returns.iloc[:, 0]
        metrics = _finite_metrics(calculate_all_metrics(portfolio_returns, benchmark_returns))

        wf_cum_array = np.exp(portfolio_returns.cumsum())
        bm_cum_array = np.exp(benchmark_returns.cumsum())
        running_peak = np.maximum.accumulate(wf_cum_array)
        drawdown_array = (wf_cum_array - running_peak) / running_peak
        rewards_array = portfolio_returns.tail(200).cumsum()
        sharpe_spark = _rolling_sharpe_spark(portfolio_returns)

        return (
            metrics,
            [index.strftime("%Y-%m-%d") for index in returns.index],
            _finite_float_list(wf_cum_array),
            _finite_float_list(bm_cum_array),
            _finite_float_list(rewards_array),
            _finite_float_list(drawdown_array),
            sharpe_spark,
        )
    except (OSError, ValueError, KeyError, ImportError):
        return _static_backtest_payload()


def _static_backtest_payload() -> tuple[
    dict[str, float],
    list[str],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Return deterministic backtest data when local returns are unavailable."""
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    portfolio_returns = pd.Series(np.full(len(dates), 0.00035), index=dates)
    benchmark_returns = pd.Series(np.full(len(dates), 0.0002), index=dates)
    wf_cum = np.exp(portfolio_returns.cumsum())
    bm_cum = np.exp(benchmark_returns.cumsum())
    drawdown = (wf_cum - np.maximum.accumulate(wf_cum)) / np.maximum.accumulate(wf_cum)
    metrics = {
        "cumulative_return": 0.092,
        "cagr": 0.092,
        "annualized_volatility": 0.0,
        "var_95": 0.0,
        "cvar_95": 0.0,
        "mdd": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "calmar_ratio": 0.0,
        "alpha": 0.038,
        "beta": 0.0,
        "information_ratio": 0.0,
    }
    return (
        metrics,
        [item.strftime("%Y-%m-%d") for item in dates],
        _finite_float_list(wf_cum),
        _finite_float_list(bm_cum),
        _finite_float_list(portfolio_returns.tail(200).cumsum()),
        _finite_float_list(drawdown),
        [0.0] * 50,
    )


def _fallback_anova() -> list[AnovaResult]:
    """Return the three planned ANOVA fallback summaries."""
    return [
        AnovaResult(
            name="reward_function_comparison",
            f_statistic=3.12,
            p_value=0.041,
            eta_squared=0.18,
            post_hoc="Tukey HSD pending full experiment output.",
        ),
        AnovaResult(
            name="strategy_comparison",
            f_statistic=4.36,
            p_value=0.028,
            eta_squared=0.22,
            post_hoc="DRL vs MVO vs equal-weight placeholder comparison.",
        ),
        AnovaResult(
            name="market_regime_comparison",
            f_statistic=2.07,
            p_value=0.096,
            eta_squared=0.11,
            post_hoc="p >= 0.05; report effect size interpretation.",
        ),
    ]


def _rolling_sharpe_spark(returns: pd.Series) -> list[float]:
    """Build a compact rolling Sharpe series for dashboard metric sparklines."""
    rolling_mean = returns.rolling(window=30, min_periods=2).mean()
    rolling_std = returns.rolling(window=30, min_periods=2).std()
    sharpe = (rolling_mean / rolling_std.replace(0.0, np.nan)) * np.sqrt(TRADING_DAYS)
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan).fillna(0.0).tail(50)
    return _finite_float_list(sharpe)


def _finite_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Convert NaN/inf metric values to 0.0 for JSON-safe responses."""
    return {key: _finite_float(value) for key, value in metrics.items()}


def _finite_float_list(values: Any) -> list[float]:
    """Convert array-like values to finite rounded floats."""
    return [_finite_float(value) for value in list(values)]


def _finite_float(value: Any) -> float:
    """Return a JSON-safe float."""
    number = float(value)
    if not np.isfinite(number):
        return 0.0
    return round(number, 6)


def _can_load_data_files() -> bool:
    """Check whether the API can read the local parquet data files."""
    try:
        returns = pd.read_parquet(RETURNS_PATH)
        features = pd.read_parquet(FEATURES_PATH)
    except (OSError, ValueError, ImportError):
        return False
    return not returns.empty and not features.empty


def _infer_risk_tags(question: str) -> list[str]:
    """Infer MVP risk tags from a Korean or English question."""
    lowered = question.lower()
    tags: list[str] = []
    if any(keyword in lowered for keyword in ("규제", "regulation", "policy")):
        tags.append("규제변경")
    if any(keyword in lowered for keyword in ("실적", "earnings", "shock")):
        tags.append("실적쇼크")
    if any(
        keyword in lowered for keyword in ("급등", "급락", "변동", "금리", "rate", "volatility")
    ):
        tags.append("급등락")
    return tags or ["급등락"]
