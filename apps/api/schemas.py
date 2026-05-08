"""Pydantic schemas for the FastAPI backend."""

from typing import Literal

from pydantic import BaseModel, Field

RiskProfile = Literal["conservative", "balanced", "aggressive"]
EndpointStatus = Literal["ready", "fallback", "unavailable"]


class ApiStatus(BaseModel):
    """Runtime API settings exposed by the health endpoint."""

    host: str
    port: int
    log_level: str


class HealthResponse(BaseModel):
    """Health response with integration readiness."""

    status: Literal["ok"]
    api: ApiStatus
    modules: dict[str, EndpointStatus]


class OptimizeRequest(BaseModel):
    """Portfolio optimization request."""

    tickers: list[str] | None = Field(default=None, min_length=1)
    risk_profile: RiskProfile = "balanced"


class OptimizeResponse(BaseModel):
    """Portfolio optimization response."""

    status: EndpointStatus
    tickers: list[str]
    weights: dict[str, float]
    risk_profile: RiskProfile
    expected_return: float
    expected_volatility: float
    message: str


class ExplainRequest(BaseModel):
    """SHAP explanation request."""

    date: str | None = None
    top_k: int = Field(default=8, ge=1, le=20)


class FeatureContribution(BaseModel):
    """One feature contribution for a SHAP-like explanation."""

    feature: str
    value: float
    contribution: float


class ExplainResponse(BaseModel):
    """SHAP-style explanation response."""

    status: EndpointStatus
    date: str | None
    base_value: float
    prediction: float
    feature_contributions: list[FeatureContribution]
    message: str


class ResearchRequest(BaseModel):
    """RAG research request."""

    question: str = Field(min_length=1)


class ResearchResponse(BaseModel):
    """Agentic RAG response contract."""

    status: EndpointStatus
    question: str
    report: str
    sources: list[str]
    reasoning_trace: str
    risk_tags: list[str]


class AnovaResult(BaseModel):
    """ANOVA summary row."""

    name: str
    f_statistic: float
    p_value: float
    eta_squared: float
    post_hoc: str


class BacktestResponse(BaseModel):
    """Backtest metrics and statistical validation response."""

    status: EndpointStatus
    metrics: dict[str, float]
    anova: list[AnovaResult]
    benchmark: str
    message: str
