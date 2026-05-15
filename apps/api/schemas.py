"""Pydantic schemas for the FastAPI backend."""

from typing import Literal

from pydantic import BaseModel, Field

RiskProfile = Literal["conservative", "balanced", "aggressive"]
EndpointStatus = Literal["ready", "fallback", "unavailable"]
BacktestWindow = Literal["w1", "w2", "w3", "final"]


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
    risk_aversion: float | None = Field(default=None, gt=0)


class ReturnSeries(BaseModel):
    """Chart-ready cumulative return series."""

    date: list[str]
    portfolio: list[float]
    benchmark: list[float]


class OptimizeResponse(BaseModel):
    """Portfolio optimization response."""

    status: EndpointStatus
    elapsed_ms: float = 0.0
    timed_out: bool = False
    tickers: list[str]
    weights: dict[str, float]
    risk_profile: RiskProfile
    expected_return: float
    expected_volatility: float
    returns: ReturnSeries
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
    elapsed_ms: float = 0.0
    timed_out: bool = False
    date: str | None
    target_date: str | None
    base_value: float
    prediction: float
    feature_contributions: list[FeatureContribution]
    feature_names: list[str]
    shap_values: list[float]
    message: str


class ResearchRequest(BaseModel):
    """RAG research request."""

    question: str = Field(min_length=1)


class ResearchResponse(BaseModel):
    """Agentic RAG response contract."""

    status: EndpointStatus
    elapsed_ms: float = 0.0
    timed_out: bool = False
    question: str
    report: str
    sources: list[str]
    reasoning_trace: str
    risk_tags: list[str]


class TukeyRow(BaseModel):
    """One Tukey HSD comparison row."""

    group1: str
    group2: str
    meandiff: float
    p_adj: float
    reject: bool


class InteractionStats(BaseModel):
    """Two-way ANOVA interaction summary."""

    f_statistic: float
    p_value: float
    significant: bool


class StrategyEffectStats(BaseModel):
    """Two-way ANOVA strategy main effect summary."""

    f_statistic: float
    p_value: float


class AnovaResult(BaseModel):
    """ANOVA summary row."""

    name: str
    f_statistic: float
    p_value: float
    eta_squared: float
    post_hoc: list[TukeyRow]
    interaction: InteractionStats | None = None
    strategy_effect: StrategyEffectStats | None = None


class SafeguardState(BaseModel):
    """Safe-Guard runtime state for backtest summaries."""

    active: bool
    triggered_at: str | None
    current_drawdown: float


class BacktestResponse(BaseModel):
    """Backtest metrics and statistical validation response."""

    status: EndpointStatus
    elapsed_ms: float = 0.0
    timed_out: bool = False
    metrics: dict[str, float]
    anova: list[AnovaResult]
    benchmark: str
    dates: list[str]
    rewards: list[float]
    wf_cum: list[float]
    bm_cum: list[float]
    wf_spark: list[float]
    sharpe_spark: list[float]
    drawdown: list[float]
    var_95: float
    cvar_95: float
    mdd: float
    safeguard: SafeguardState
    message: str
