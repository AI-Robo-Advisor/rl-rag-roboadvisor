"""Deterministic service layer for Sprint 2 API contracts.

The RL, SHAP, and RAG modules are still being developed by separate owners.
These helpers keep the HTTP contract stable and make later module integration
small: replace the fallback body, keep the response schema.
"""

from __future__ import annotations

from apps.api.schemas import (
    AnovaResult,
    BacktestResponse,
    ExplainResponse,
    FeatureContribution,
    OptimizeResponse,
    ResearchResponse,
    RiskProfile,
    Source,
)

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
) -> OptimizeResponse:
    """Return deterministic normalized weights until the PPO model is connected."""
    selected_tickers = tickers or DEFAULT_TICKERS
    raw_weights = _risk_adjusted_raw_weights(selected_tickers, risk_profile)
    total_weight = sum(raw_weights.values())
    weights = {ticker: weight / total_weight for ticker, weight in raw_weights.items()}

    return OptimizeResponse(
        status="fallback",
        tickers=selected_tickers,
        weights=weights,
        risk_profile=risk_profile,
        expected_return=0.084 if risk_profile == "balanced" else _profile_return(risk_profile),
        expected_volatility=(
            0.132 if risk_profile == "balanced" else _profile_volatility(risk_profile)
        ),
        message="PPO 모델 연결 전 deterministic fallback 포트폴리오입니다.",
    )


def build_fallback_explanation(date: str | None, top_k: int) -> ExplainResponse:
    """Return SHAP-like feature contributions for dashboard integration."""
    features = [
        FeatureContribution(feature="SPY_return_30d", value=0.018, contribution=0.031),
        FeatureContribution(feature="TLT_return_30d", value=-0.011, contribution=-0.014),
        FeatureContribution(feature="QQQ_RSI", value=61.2, contribution=0.019),
        FeatureContribution(feature="GLD_MACD_signal", value=0.004, contribution=0.012),
        FeatureContribution(feature="069500_return_30d", value=0.009, contribution=0.008),
        FeatureContribution(feature="VNQ_RSI", value=47.6, contribution=-0.006),
        FeatureContribution(feature="EEM_MACD_signal", value=-0.003, contribution=-0.005),
        FeatureContribution(feature="114260_return_30d", value=0.006, contribution=0.004),
    ]

    selected = features[:top_k]
    prediction = 0.05 + sum(item.contribution for item in selected)
    return ExplainResponse(
        status="fallback",
        date=date,
        base_value=0.05,
        prediction=round(prediction, 6),
        feature_contributions=selected,
        message="SHAP 모듈 연결 전 feature contribution fallback입니다.",
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
        sources=[
            Source(
                title="MVP 로보어드바이저 명세",
                url="https://github.com/AI-Robo-Advisor/rl-rag-roboadvisor",
                published_at=None,
            )
        ],
        reasoning_trace=[
            "Planner: 투자 질문의 핵심 키워드를 식별했습니다.",
            "Researcher: 실제 ChromaDB 연결 전 fallback source를 사용했습니다.",
            "Analyst: 리스크 태그와 요약 리포트를 생성했습니다.",
        ],
        risk_tags=risk_tags,
    )


def build_fallback_backtest() -> BacktestResponse:
    """Return the MVP metric and ANOVA contract with deterministic values."""
    metrics = {
        "cumulative_return": 0.184,
        "cagr": 0.087,
        "annualized_volatility": 0.132,
        "var_95": -0.021,
        "cvar_95": -0.034,
        "max_drawdown": -0.118,
        "sharpe_ratio": 0.74,
        "sortino_ratio": 1.08,
        "calmar_ratio": 0.74,
        "alpha": 0.031,
        "beta": 0.86,
        "information_ratio": 0.42,
    }
    anova = [
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
    return BacktestResponse(
        status="fallback",
        metrics=metrics,
        anova=anova,
        benchmark="SPY",
        message="Walk-Forward 백테스트 모듈 연결 전 fallback 결과입니다.",
    )


def _risk_adjusted_raw_weights(tickers: list[str], risk_profile: RiskProfile) -> dict[str, float]:
    """Build simple deterministic tilts by risk profile."""
    raw_weights = {ticker: 1.0 for ticker in tickers}
    if risk_profile == "conservative":
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
