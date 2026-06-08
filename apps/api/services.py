"""Deterministic service layer for Sprint 2 API contracts.

The RL, SHAP, and RAG modules are still being developed by separate owners.
These helpers keep the HTTP contract stable and make later module integration
small: replace the fallback body, keep the response schema.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator

import numpy as np
import pandas as pd
from pydantic import ValidationError

from apps.api.config import settings
from apps.api.observability import log_e2e_event
from apps.api.schemas import (
    AnovaResult,
    BacktestResponse,
    BacktestWindow,
    ExplainResponse,
    FeatureContribution,
    InteractionStats,
    OptimizeResponse,
    ReasoningEvent,
    ResearchResponse,
    RiskSignal,
    ReturnSeries,
    RiskProfile,
    SafeguardState,
    StrategyEffectStats,
    TrainCurveResponse,
    TukeyRow,
)
from src.agent.risk_tags import RL_RISK_TAGS, apply_decay

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

RETURNS_PATH = Path("data/processed/returns.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
RAW_FEATURES_PATH = Path("data/processed/raw_features.parquet")
SCALERS_DIR = Path("data/processed/scalers")
SHAP_ARTIFACT_PATH = Path("data/results/shap_explanations.json")
ANOVA_RESULTS_PATH = Path("data/results/anova_results.json")
UNIFIED_EVENTS_PATH = Path("data/processed/unified_events.parquet")
PPO_MODEL_PATH = Path("models/ppo_sharpe_final_risk.zip")
TRADING_DAYS = 252
PPO_TIMEOUT_SECONDS = 4.75
SHAP_TIMEOUT_SECONDS = 4.75
RESEARCH_TIMEOUT_SECONDS = 4.5
REASONING_WINDOW_DAYS = 3
_PPO_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_SHAP_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_RESEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_STREAM_EVENT_ALLOWLIST = {
    "on_chain_start",
    "on_chain_end",
    "on_chat_model_stream",
    "on_tool_start",
    "on_tool_end",
}
WINDOW_PERIODS: dict[BacktestWindow, tuple[str, str]] = {
    "w1": ("2022-01-01", "2022-12-31"),
    "w2": ("2023-01-01", "2023-12-31"),
    "w3": ("2024-01-01", "2024-12-31"),
    "final": ("2025-01-01", "2025-12-31"),
}

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

_TAG_TO_SEVERITY_COL: dict[str, str] = {tag: tag for tag in RL_RISK_TAGS}
_RISK_FEATURE_TO_TAG: dict[str, str] = {
    **{f"risk_{tag}": tag for tag in RL_RISK_TAGS},
    "risk_macro": "macro_rate_risk",
    "risk_equity": "equity_market_risk",
    "risk_geo": "geopolitical_fx_risk",
}


def _elapsed_ms(start: float) -> float:
    """Return elapsed wall-clock time in milliseconds."""
    return round((perf_counter() - start) * 1000, 3)


@lru_cache(maxsize=1)
def _load_returns() -> pd.DataFrame:
    """Load processed returns once per API process."""
    return pd.read_parquet(RETURNS_PATH)


@lru_cache(maxsize=1)
def _load_features() -> pd.DataFrame:
    """Load processed features once per API process."""
    return pd.read_parquet(FEATURES_PATH)


@lru_cache(maxsize=1)
def _load_raw_features() -> pd.DataFrame:
    """RL 경로 전용 — walk-forward 정규화 전 raw features."""
    return pd.read_parquet(RAW_FEATURES_PATH)


@lru_cache(maxsize=4)
def _load_window_scaler(window_name: str) -> tuple[pd.Series, pd.Series]:
    """윈도우별 train mean/std 로드 (data/processed/scalers/{window}_feature_stats.json)."""
    path = SCALERS_DIR / f"{window_name}_feature_stats.json"
    with path.open(encoding="utf-8") as f:
        stats = json.load(f)
    return pd.Series(stats["mean"]), pd.Series(stats["std"])


def _normalize_with_scaler(raw: pd.DataFrame, window_name: str) -> pd.DataFrame:
    """window_name train scaler로 raw_features를 Z-score 정규화."""
    mean, std = _load_window_scaler(window_name)
    missing = set(raw.columns) - set(mean.index)
    if missing:
        raise ValueError(f"Scaler에 없는 피처: {missing}")
    std_safe = std.replace(0.0, 1e-8)
    return (raw - mean) / std_safe


_UNIFIED_EVENTS_CACHE: tuple[Path, int, pd.DataFrame] | None = None


def build_fallback_portfolio(
    tickers: list[str] | None = None,
    risk_profile: RiskProfile = "balanced",
    risk_aversion: float | None = None,
    risk_tags: list[str] | None = None,
    *,
    elapsed_ms: float = 0.0,
    timed_out: bool = False,
) -> OptimizeResponse:
    """Return deterministic normalized weights until the PPO model is connected."""
    selected_tickers = tickers or DEFAULT_TICKERS
    raw_weights = _risk_adjusted_raw_weights(selected_tickers, risk_profile, risk_aversion)
    total_weight = sum(raw_weights.values())
    weights = {ticker: weight / total_weight for ticker, weight in raw_weights.items()}
    return_series, expected_return, expected_volatility = _build_return_series(weights)

    return OptimizeResponse(
        status="fallback",
        elapsed_ms=elapsed_ms,
        timed_out=timed_out,
        tickers=selected_tickers,
        weights=weights,
        risk_profile=risk_profile,
        expected_return=expected_return or _profile_return(risk_profile),
        expected_volatility=expected_volatility or _profile_volatility(risk_profile),
        returns=return_series,
        message=(
            "PPO 모델 연결 전 deterministic fallback 포트폴리오입니다. "
            "리스크 태그는 PPO ready 경로에서 관측 벡터로 반영됩니다."
            if risk_tags
            else "PPO 모델 연결 전 deterministic fallback 포트폴리오입니다."
        ),
    )


def build_portfolio_response(
    tickers: list[str] | None = None,
    risk_profile: RiskProfile = "balanced",
    risk_aversion: float | None = None,
    risk_tags: list[str] | None = None,
    risk_signals: list[RiskSignal] | None = None,
) -> OptimizeResponse:
    """Return PPO portfolio weights when available, otherwise fallback weights."""
    start = perf_counter()
    selected_tickers = tickers or DEFAULT_TICKERS
    selected_risk_tags = risk_tags or []
    try:
        weights = _predict_ppo_weights_with_timeout(
            selected_tickers,
            selected_risk_tags,
            risk_signals,
        )
    except Exception as exc:
        return build_fallback_portfolio(
            selected_tickers,
            risk_profile,
            risk_aversion,
            selected_risk_tags,
            elapsed_ms=_elapsed_ms(start),
            timed_out=isinstance(exc, TimeoutError),
        )

    if set(weights) != set(selected_tickers):
        return build_fallback_portfolio(
            selected_tickers,
            risk_profile,
            risk_aversion,
            selected_risk_tags,
            elapsed_ms=_elapsed_ms(start),
        )

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return build_fallback_portfolio(
            selected_tickers,
            risk_profile,
            risk_aversion,
            selected_risk_tags,
            elapsed_ms=_elapsed_ms(start),
        )

    normalized = {ticker: weight / total_weight for ticker, weight in weights.items()}
    return_series, expected_return, expected_volatility = _build_return_series(normalized)
    return OptimizeResponse(
        status="ready",
        elapsed_ms=_elapsed_ms(start),
        timed_out=False,
        tickers=selected_tickers,
        weights=normalized,
        risk_profile=risk_profile,
        expected_return=expected_return or _profile_return(risk_profile),
        expected_volatility=expected_volatility or _profile_volatility(risk_profile),
        returns=return_series,
        message="PPO 모델 기반 포트폴리오 비중입니다.",
    )


def build_fallback_explanation(
    date: str | None,
    top_k: int,
    *,
    elapsed_ms: float = 0.0,
    timed_out: bool = False,
) -> ExplainResponse:
    """Return SHAP-like feature contributions for dashboard integration."""
    features = _feature_contributions_from_parquet(date, top_k) or _static_feature_contributions()

    selected = features[:top_k]
    prediction = 0.05 + sum(item.contribution for item in selected)
    target_date = date or _latest_feature_date()
    reasoning_context = _build_reasoning_events(target_date) if target_date else []
    return ExplainResponse(
        status="fallback",
        elapsed_ms=elapsed_ms,
        timed_out=timed_out,
        date=date,
        target_date=target_date,
        base_value=0.05,
        prediction=round(prediction, 6),
        feature_contributions=_attach_feature_reasoning_context(selected, target_date),
        feature_names=[item.feature for item in selected],
        shap_values=[item.contribution for item in selected],
        reasoning_context=reasoning_context,
        message="SHAP 모듈 연결 전 feature contribution fallback입니다.",
    )


def build_explanation_response(date: str | None, top_k: int) -> ExplainResponse:
    """Return SHAP explanation from the RL module when available."""
    start = perf_counter()
    try:
        result = _shap_from_artifact(date, top_k) or _compute_ready_shap_with_timeout(date, top_k)
        target_date = result.get("target_date")
        target_date_text = str(target_date) if target_date is not None else None
        feature_contributions = [
            FeatureContribution(**item) for item in result.get("feature_contributions", [])
        ]
        top_reasoning_context = _build_reasoning_events(
            target_date_text,
            raw_context=result.get("reasoning_context"),
        )
        return ExplainResponse(
            status="ready",
            elapsed_ms=_elapsed_ms(start),
            timed_out=False,
            date=result.get("date"),
            target_date=target_date,
            base_value=result.get("base_value", 0.0),
            prediction=result.get("prediction", 0.0),
            feature_contributions=_attach_feature_reasoning_context(
                feature_contributions,
                target_date_text,
                raw_context=result.get("reasoning_context"),
            ),
            feature_names=list(result.get("feature_names", [])),
            shap_values=list(result.get("shap_values", [])),
            reasoning_context=top_reasoning_context,
            message=result.get("message", "PPO SHAP 분석 완료."),
        )
    except Exception as exc:
        return build_fallback_explanation(
            date,
            top_k,
            elapsed_ms=_elapsed_ms(start),
            timed_out=isinstance(exc, TimeoutError),
        )


def run_graph(question: str) -> dict[str, Any]:
    """Import and run LangGraph lazily so API import stays safe without an API key."""
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    from src.agent.graph import run_graph as _run_graph

    return _run_graph(question)


async def stream_graph_events(question: str) -> AsyncIterator[dict[str, Any]]:
    """Yield LangGraph fine-grained stream events for a research question."""
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    from src.agent.graph import graph

    initial_state = {
        "query": question,
        "messages": [],
        "plan": "",
        "context": "",
        "documents": [],
        "risk_tags": [],
        "distances": [],
        "retry_count": 0,
        "needs_research_retry": False,
        "response": "",
        "sources": [],
        "reasoning_trace": "",
    }
    async for event in graph.astream_events(initial_state, version="v2"):
        yield event


async def stream_research_response(question: str) -> AsyncIterator[str]:
    """Stream research progress as newline-delimited JSON for Streamlit."""
    start = perf_counter()
    yield _to_ndjson({"type": "start", "question": question})

    if not settings.OPENAI_API_KEY:
        fallback = build_fallback_research(question)
        log_e2e_event(
            "/research/stream",
            status=fallback.status,
            elapsed_ms=_elapsed_ms(start),
            timed_out=fallback.timed_out,
            sources_count=len(fallback.sources),
            risk_tags_count=len(fallback.risk_tags),
            risk_signals_count=len(fallback.risk_signals),
            report_chars=len(fallback.report),
        )
        yield _to_ndjson(
            {
                "type": "fallback",
                "name": "research",
                "data": fallback.model_dump(),
            }
        )
        return

    final_state: dict[str, Any] | None = None
    run_starts: dict[str, tuple[str, float]] = {}
    timings: dict[str, float] = {}
    try:
        async for event in stream_graph_events(question):
            elapsed_ms = _elapsed_ms(start)
            timing_key = _graph_event_timing_key(event)
            run_id = str(event.get("run_id") or timing_key)
            duration_ms = None
            if event.get("event") in {"on_chain_start", "on_tool_start"}:
                run_starts[run_id] = (timing_key, perf_counter())
            elif event.get("event") in {"on_chain_end", "on_tool_end"} and run_id in run_starts:
                started_key, started_at = run_starts.pop(run_id)
                duration_ms = round((perf_counter() - started_at) * 1000, 2)
                _record_timing(timings, started_key, duration_ms)

            event_state = _state_from_graph_event(event)
            if event_state:
                final_state = {**(final_state or {}), **event_state}
            compact = _compact_graph_event(event, elapsed_ms=elapsed_ms, duration_ms=duration_ms)
            if compact:
                yield _to_ndjson(compact)
    except Exception as exc:
        fallback = build_fallback_research(question)
        log_e2e_event(
            "/research/stream",
            status=fallback.status,
            elapsed_ms=_elapsed_ms(start),
            timed_out=fallback.timed_out,
            sources_count=len(fallback.sources),
            risk_tags_count=len(fallback.risk_tags),
            risk_signals_count=len(fallback.risk_signals),
            report_chars=len(fallback.report),
            error=exc.__class__.__name__,
        )
        yield _to_ndjson(
            {
                "type": "fallback",
                "name": "research",
                "data": {
                    **fallback.model_dump(),
                    "error": exc.__class__.__name__,
                },
            }
        )
        return

    total_ms = _elapsed_ms(start)
    timings["total_ms"] = total_ms
    response = _research_response_from_state(question, {**(final_state or {}), "timings": timings})
    log_e2e_event(
        "/research/stream",
        status=response.status,
        elapsed_ms=_elapsed_ms(start),
        timed_out=response.timed_out,
        sources_count=len(response.sources),
        risk_tags_count=len(response.risk_tags),
        risk_signals_count=len(response.risk_signals),
        report_chars=len(response.report),
    )
    yield _to_ndjson(
        {
            "type": "complete",
            "question": response.question,
            "report": response.report,
            "sources": response.sources,
            "reasoning_trace": response.reasoning_trace,
            "risk_tags": response.risk_tags,
            "risk_signals": [signal.model_dump() for signal in response.risk_signals],
            "timings": response.timings,
        }
    )


def _state_from_graph_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Extract final LangGraph state-like output from a stream event."""
    data = event.get("data") or {}
    if not isinstance(data, dict):
        return None
    output = data.get("output")
    if not isinstance(output, dict):
        return None
    useful_keys = {
        "response",
        "sources",
        "reasoning_trace",
        "risk_tags",
        "rl_risk_tags",
        "risk_signals",
        "timings",
    }
    if not any(key in output for key in useful_keys):
        return None
    return {str(key): value for key, value in output.items() if key in useful_keys}


def _graph_event_node(event: dict[str, Any]) -> str:
    metadata = event.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("langgraph_node"):
        return str(metadata["langgraph_node"])
    return str(event.get("name") or "research")


def _graph_event_timing_key(event: dict[str, Any]) -> str:
    """Return the timing bucket for a LangGraph stream event."""
    name = str(event.get("name") or "")
    if name == "route_after_grade":
        return name
    return _graph_event_node(event)


def _record_timing(timings: dict[str, float], key: str, duration_ms: float) -> None:
    """Store a timing value without overwriting repeated node executions."""
    base_key = f"{key}_ms"
    if base_key not in timings:
        timings[base_key] = duration_ms
        return

    index = 2
    while f"{key}_{index}_ms" in timings:
        index += 1
    timings[f"{key}_{index}_ms"] = duration_ms


def _compact_graph_event(
    event: dict[str, Any],
    *,
    elapsed_ms: float | None = None,
    duration_ms: float | None = None,
) -> dict[str, Any] | None:
    """Convert LangGraph events to small dashboard-oriented NDJSON payloads."""
    event_type = str(event.get("event", "event"))
    if event_type not in _STREAM_EVENT_ALLOWLIST:
        return None

    name = str(event.get("name") or "research")
    node = _graph_event_node(event)
    data = event.get("data") or {}
    text = ""
    if isinstance(data, dict):
        if event_type == "on_chat_model_stream":
            chunk = data.get("chunk")
            text = str(getattr(chunk, "content", "") or "")
        else:
            output = data.get("output") or data.get("input") or ""
            text = _compact_event_text(output)

    if event_type == "on_chat_model_stream" and not text:
        return None
    compact = {"type": event_type, "name": name, "node": node, "text": text[:500]}
    if elapsed_ms is not None:
        compact["elapsed_ms"] = elapsed_ms
    if duration_ms is not None:
        compact["duration_ms"] = duration_ms
    return compact


def _compact_event_text(value: Any) -> str:
    """Return a compact string from nested event payloads."""
    if isinstance(value, dict):
        for key in ("response", "context", "query", "plan"):
            if value.get(key):
                return str(value[key])
        return json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))[:500]
    if isinstance(value, list):
        return ", ".join(str(item) for item in value[:3])
    return str(value)


def build_research_response(question: str) -> ResearchResponse:
    """Run LangGraph when configured, otherwise return the deterministic fallback."""
    start = perf_counter()
    if not settings.OPENAI_API_KEY:
        return build_fallback_research(question, elapsed_ms=_elapsed_ms(start))

    try:
        state = _run_graph_with_timeout(question)
    except Exception as exc:
        return build_fallback_research(
            question,
            elapsed_ms=_elapsed_ms(start),
            timed_out=isinstance(exc, TimeoutError),
        )

    response = _research_response_from_state(question, state)
    response.elapsed_ms = _elapsed_ms(start)
    return response


def _run_graph_with_timeout(question: str) -> dict[str, Any]:
    """Run synchronous LangGraph research with a request-time budget."""
    future = _RESEARCH_EXECUTOR.submit(run_graph, question)
    try:
        return future.result(timeout=RESEARCH_TIMEOUT_SECONDS)
    except TimeoutError:
        future.cancel()
        raise


def _research_response_from_state(question: str, state: dict[str, Any]) -> ResearchResponse:
    """Convert LangGraph state into the public ResearchResponse schema."""
    report = str(state.get("response") or "").strip()
    if not report:
        return build_fallback_research(question)

    messages = state.get("messages") or []
    reasoning_trace = state.get("reasoning_trace") or "\n".join(str(item) for item in messages)
    raw_risk_tags = state.get("rl_risk_tags") or state.get("risk_tags") or []
    risk_tags = _normalize_rl_risk_tags(raw_risk_tags) or _infer_risk_tags(question)
    risk_signals = _normalize_risk_signals(state.get("risk_signals")) or _risk_signals_from_tags(
        risk_tags
    )
    sources = [str(source) for source in state.get("sources", []) if source]
    timings = state.get("timings") if isinstance(state.get("timings"), dict) else {}

    return ResearchResponse(
        status="ready",
        question=question,
        report=report,
        sources=sources or ["https://github.com/AI-Robo-Advisor/rl-rag-roboadvisor"],
        reasoning_trace=reasoning_trace,
        risk_tags=risk_tags,
        risk_signals=risk_signals,
        timings=timings,
    )


def build_fallback_research(
    question: str,
    *,
    elapsed_ms: float = 0.0,
    timed_out: bool = False,
) -> ResearchResponse:
    """Return a stable RAG-style payload until LangGraph is connected."""
    risk_tags = _infer_risk_tags(question)
    return ResearchResponse(
        status="fallback",
        elapsed_ms=elapsed_ms,
        timed_out=timed_out,
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
        risk_signals=_risk_signals_from_tags(risk_tags),
    )


def build_fallback_backtest(window: BacktestWindow = "final") -> BacktestResponse:
    """Return metric output from available data plus fallback ANOVA summaries."""
    metrics, dates, wf_cum, bm_cum, rewards, drawdown, sharpe_spark = _build_backtest_payload(
        window
    )
    anova = _resolve_anova_results(_load_returns()) if _can_load_data_files() else _fallback_anova()
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
        safeguard=SafeguardState(
            active=False,
            triggered_at=None,
            current_drawdown=abs(drawdown[-1]) if drawdown else current_mdd,
        ),
        mvo_cum=[],
        message=(f"Walk-Forward 백테스트 모듈 연결 전 fallback 결과입니다. " f"(window={window})"),
    )


def build_backtest_response(window: BacktestWindow = "final") -> BacktestResponse:
    """Return RL backtest/anova results when available, otherwise fallback."""
    start = perf_counter()
    try:
        response = _build_ready_backtest_response_cached(window).model_copy(deep=True)
        response.elapsed_ms = _elapsed_ms(start)
        return response
    except Exception as exc:
        response = build_fallback_backtest(window)
        response.elapsed_ms = _elapsed_ms(start)
        response.timed_out = isinstance(exc, TimeoutError)
        return response


def build_module_statuses() -> dict[str, str]:
    """Return runtime readiness using only files and modules available locally."""
    return {
        "data": "ready" if _can_load_data_files() else "fallback",
        "rl": "ready" if _is_ppo_ready() else "fallback",
        "rag": "ready" if settings.OPENAI_API_KEY and _has_local_research_corpus() else "fallback",
        "shap": "ready" if _is_shap_ready() else "fallback",
        "backtest": "ready" if _is_backtest_ready() else "fallback",
    }


def _predict_ppo_weights(
    tickers: list[str],
    risk_tags: list[str] | None = None,
    risk_signals: list[RiskSignal] | None = None,
) -> dict[str, float]:
    """Run the trained PPO policy once and return selected asset weights."""
    returns = _load_returns()
    features = _normalize_with_scaler(_load_raw_features(), "final")
    missing = [ticker for ticker in tickers if ticker not in returns.columns]
    if missing:
        raise ValueError(f"Unknown tickers for PPO model: {missing}")

    from src.rl.env import PortfolioEnv

    selected_signals = _resolve_risk_signals(risk_tags, risk_signals)
    risk_vector = _signals_to_vector(selected_signals) if selected_signals else None
    env_kwargs: dict[str, Any] = {
        "returns_df": returns,
        "features_df": features,
        "lookback": 30,
        "reward_type": "sharpe",
    }
    if risk_vector is not None:
        env_kwargs["risk_vector"] = risk_vector

    model = _load_ppo_model()
    env = PortfolioEnv(**env_kwargs)
    obs, _ = env.reset()
    env.current_step = len(env.features_df) - 1
    obs = env._get_observation()
    action, _ = model.predict(obs, deterministic=True)
    full_weights = env._normalize_action(action)
    by_asset = {asset: float(full_weights[index]) for index, asset in enumerate(env.asset_names)}
    return {ticker: by_asset[ticker] for ticker in tickers}


def _predict_ppo_weights_with_timeout(
    tickers: list[str],
    risk_tags: list[str] | None = None,
    risk_signals: list[RiskSignal] | None = None,
) -> dict[str, float]:
    """Run PPO inference with a request-time budget."""
    future = _PPO_EXECUTOR.submit(_predict_ppo_weights, tickers, risk_tags or [], risk_signals)
    try:
        return future.result(timeout=PPO_TIMEOUT_SECONDS)
    except TimeoutError:
        future.cancel()
        raise


@lru_cache(maxsize=1)
def _load_ppo_model() -> Any:
    """Load the PPO model once per API process."""
    from stable_baselines3 import PPO

    return PPO.load(str(PPO_MODEL_PATH))


def warm_runtime_caches() -> None:
    """Warm request-critical caches during API startup/import."""
    try:
        _load_returns()
        _load_features()
        _load_raw_features()
        if _is_ppo_ready():
            _load_ppo_model()
    except Exception:
        return


def _compute_ready_shap(date: str | None, top_k: int) -> dict[str, Any]:
    """Compute a bounded SHAP explanation through src.rl.shap."""
    from src.rl.shap import compute_shap_explanation

    features = _normalize_with_scaler(_load_raw_features(), "final")
    returns = _load_returns()
    return compute_shap_explanation(
        model_path=PPO_MODEL_PATH,
        features_df=features,
        returns_df=returns,
        date=date,
        top_k=top_k,
        background_size=5,
        nsamples=20,
    )


def _compute_ready_shap_with_timeout(date: str | None, top_k: int) -> dict[str, Any]:
    """Run SHAP explanation with a request-time budget."""
    future = _SHAP_EXECUTOR.submit(_compute_ready_shap_cached, date, top_k)
    try:
        return future.result(timeout=SHAP_TIMEOUT_SECONDS)
    except TimeoutError:
        future.cancel()
        raise


@lru_cache(maxsize=32)
def _compute_ready_shap_cached(date: str | None, top_k: int) -> dict[str, Any]:
    """Cache live SHAP results for repeated dashboard lookups."""
    return _compute_ready_shap(date, top_k)


@lru_cache(maxsize=1)
def _load_shap_artifact_explanations() -> list[dict[str, Any]] | None:
    """Load precomputed SHAP explanations (dict or array JSON artifact)."""
    if not SHAP_ARTIFACT_PATH.exists():
        return None
    try:
        with SHAP_ARTIFACT_PATH.open(encoding="utf-8") as fp:
            artifact = json.load(fp)
    except Exception:
        return None

    if isinstance(artifact, list):
        return [item for item in artifact if isinstance(item, dict)]
    if isinstance(artifact, dict):
        explanations = artifact.get("explanations")
        if isinstance(explanations, list):
            return [item for item in explanations if isinstance(item, dict)]
    return None


def _shap_from_artifact(date: str | None, top_k: int) -> dict[str, Any] | None:
    """Return a ready SHAP payload from a precomputed artifact."""
    explanations = _load_shap_artifact_explanations()
    if not explanations:
        return None

    target = str(date or "")
    if target:
        try:
            pd.Timestamp(target)
        except (ValueError, TypeError):
            return None
    selected: dict[str, Any] | None = None
    if target:
        exact = [
            item
            for item in explanations
            if str(item.get("target_date") or item.get("date")) == target
        ]
        if exact:
            selected = exact[0]
        else:
            candidates = [
                item
                for item in explanations
                if str(item.get("target_date") or item.get("date")) <= target
            ]
            selected = candidates[-1] if candidates else None
    if selected is None:
        selected = explanations[-1]

    contributions = list(selected.get("feature_contributions", []))[:top_k]
    if not contributions:
        return None

    target_date = selected.get("target_date") or selected.get("date")
    return {
        "date": date,
        "target_date": target_date,
        "base_value": selected.get("base_value", 0.0),
        "prediction": selected.get("prediction", 0.0),
        "feature_contributions": contributions,
        "feature_names": [item.get("feature", "") for item in contributions],
        "shap_values": [item.get("contribution", 0.0) for item in contributions],
        "reasoning_context": selected.get("reasoning_context"),
        "message": "사전 계산된 SHAP artifact 기반 해석입니다.",
    }


def _load_anova_from_artifact() -> list[dict[str, Any]] | None:
    """Load precomputed ANOVA results from data/results/anova_results.json."""
    if not ANOVA_RESULTS_PATH.exists():
        return None
    try:
        payload = json.loads(ANOVA_RESULTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, list) and payload:
        return [item for item in payload if isinstance(item, dict)]
    return None


def _anova_results_look_degenerate(results: list[dict[str, Any]]) -> bool:
    """True when live one-way ANOVA looks like equal-weight fallback (F≈0, p≈1)."""
    by_name = {str(item.get("name")): item for item in results}
    for key in ("reward_function_comparison", "strategy_comparison"):
        item = by_name.get(key)
        if not item:
            continue
        f_stat = float(item.get("f_statistic") or 0.0)
        p_val = float(item.get("p_value") or 1.0)
        if f_stat <= 0.0 and p_val >= 0.99:
            return True
    return False


def _resolve_anova_results(returns: pd.DataFrame) -> list[AnovaResult]:
    """Prefer live ANOVA; fall back to artifact JSON when live output is degenerate."""
    from src.rl.anova import run_all_anova

    live = run_all_anova(returns)
    if not _anova_results_look_degenerate(live):
        return [AnovaResult(**item) for item in live]

    artifact = _load_anova_from_artifact()
    if artifact:
        return [AnovaResult(**item) for item in artifact]

    return _fallback_anova()


def _build_ready_backtest_response(window: BacktestWindow) -> BacktestResponse:
    """Build BacktestResponse from implemented RL backtest and ANOVA modules."""
    from src.rl.backtest import WINDOWS, run_window_backtest

    returns = _load_returns()
    features = _load_raw_features()
    window_config = next(item for item in WINDOWS if item["name"] == window)
    metrics_raw, portfolio_returns, _ = run_window_backtest(
        window_config,
        returns,
        features,
        reward="sharpe",
    )
    benchmark_returns = (
        returns.loc[portfolio_returns.index, "SPY"]
        if "SPY" in returns.columns
        else returns.loc[portfolio_returns.index].iloc[:, 0]
    )
    equal_weight_returns = returns.loc[portfolio_returns.index].mean(axis=1)
    metrics = _finite_metrics(
        {
            key: value
            for key, value in metrics_raw.items()
            if isinstance(value, (int, float, np.floating))
        }
    )
    wf_cum_array = np.exp(portfolio_returns.cumsum())
    bm_cum_array = np.exp(benchmark_returns.cumsum())
    ew_cum_array = np.exp(equal_weight_returns.cumsum())
    drawdown_array = (wf_cum_array - np.maximum.accumulate(wf_cum_array)) / np.maximum.accumulate(
        wf_cum_array
    )
    current_mdd = abs(float(drawdown_array.min())) if len(drawdown_array) else 0.0
    triggered = drawdown_array[drawdown_array <= -0.15]
    anova = _resolve_anova_results(returns)

    try:
        from src.rl.mvo import run_mvo
        mvo_returns = run_mvo(
            returns,
            window_config["train_start"],
            window_config["train_end"],
            window_config["test_start"],
            window_config["test_end"],
        )
        mvo_cum_array = np.exp(mvo_returns.cumsum()) if not mvo_returns.empty else pd.Series(dtype=float)
    except Exception:
        mvo_cum_array = pd.Series(dtype=float)

    return BacktestResponse(
        status="ready",
        metrics=metrics,
        anova=anova,
        benchmark="SPY",
        dates=[index.strftime("%Y-%m-%d") for index in portfolio_returns.index],
        rewards=_finite_float_list(portfolio_returns.tail(200).cumsum()),
        wf_cum=_finite_float_list(wf_cum_array),
        bm_cum=_finite_float_list(bm_cum_array),
        ew_cum=_finite_float_list(ew_cum_array),
        mvo_cum=_finite_float_list(mvo_cum_array),
        wf_spark=_finite_float_list(wf_cum_array.tail(50)),
        sharpe_spark=_rolling_sharpe_spark(portfolio_returns),
        drawdown=_finite_float_list(drawdown_array),
        var_95=float(metrics.get("var_95", 0.0)),
        cvar_95=float(metrics.get("cvar_95", 0.0)),
        mdd=current_mdd,
        safeguard=SafeguardState(
            active=not triggered.empty,
            triggered_at=triggered.index[0].strftime("%Y-%m-%d") if not triggered.empty else None,
            current_drawdown=abs(float(drawdown_array.iloc[-1])) if len(drawdown_array) else 0.0,
        ),
        message=f"실제 Walk-Forward 백테스트 결과입니다. (window={window}, reward=sharpe)",
    )


@lru_cache(maxsize=4)
def _build_ready_backtest_response_cached(window: BacktestWindow) -> BacktestResponse:
    """Cache backtest results because they are immutable for a running API process."""
    return _build_ready_backtest_response(window)


def _is_ppo_ready() -> bool:
    """Return whether PPO inference can be attempted."""
    if not PPO_MODEL_PATH.exists():
        return False
    return find_spec("stable_baselines3") is not None and _can_load_data_files()


def _is_shap_ready() -> bool:
    """Return whether SHAP explanation can be attempted."""
    return (
        PPO_MODEL_PATH.exists()
        and _can_load_data_files()
        and find_spec("shap") is not None
        and find_spec("stable_baselines3") is not None
        and find_spec("torch") is not None
    )


def _is_backtest_ready() -> bool:
    """Return whether the full backtest + ANOVA stack is available."""
    return (
        _can_load_data_files()
        and find_spec("scipy") is not None
        and find_spec("statsmodels") is not None
    )


def _rag_has_documents() -> bool:
    """Return whether the configured Chroma collection has retrievable documents."""
    try:
        from src.agent.vectorstore import collection_document_count

        return collection_document_count(settings.CHROMA_PERSIST_DIR) > 0
    except Exception:
        return False


def _has_local_research_corpus() -> bool:
    """Return whether /research has local documents available for ready responses."""
    return _rag_has_documents()


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
        returns = _load_returns()
        usable = [ticker for ticker in weights if ticker in returns.columns]
        if not usable:
            return _static_return_series(), None, None

        usable_weights = pd.Series({ticker: weights[ticker] for ticker in usable}, dtype=float)
        usable_weights = usable_weights / usable_weights.sum()
        portfolio_returns = returns[usable].mul(usable_weights, axis=1).sum(axis=1)
        benchmark_returns = returns["SPY"] if "SPY" in returns.columns else returns[usable[0]]
        equal_weight_returns = returns[usable].mean(axis=1)

        portfolio_cum = np.exp(portfolio_returns.cumsum())
        benchmark_cum = np.exp(benchmark_returns.cumsum())
        equal_weight_cum = np.exp(equal_weight_returns.cumsum())
        years = len(portfolio_returns) / TRADING_DAYS
        expected_return = float(portfolio_cum.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0
        expected_volatility = float(portfolio_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
        return (
            ReturnSeries(
                date=[index.strftime("%Y-%m-%d") for index in returns.index],
                portfolio=_finite_float_list(portfolio_cum),
                benchmark=_finite_float_list(benchmark_cum),
                equal_weight=_finite_float_list(equal_weight_cum),
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
    equal_weight = np.cumprod(np.full(len(dates), 1.00025))
    return ReturnSeries(
        date=[item.strftime("%Y-%m-%d") for item in dates],
        portfolio=_finite_float_list(portfolio),
        benchmark=_finite_float_list(benchmark),
        equal_weight=_finite_float_list(equal_weight),
    )


def _feature_contributions_from_parquet(
    requested_date: str | None,
    top_k: int,
) -> list[FeatureContribution] | None:
    """Build deterministic SHAP-like contributions from the nearest feature row."""
    try:
        features = _load_features()
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
        features = _load_features()
        if features.empty:
            return None
        return features.index[-1].strftime("%Y-%m-%d")
    except (OSError, ValueError, ImportError):
        return None


def _build_backtest_payload(window: BacktestWindow) -> tuple[
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
        returns = _load_returns()
        if returns.empty:
            raise ValueError("returns.parquet is empty")

        windowed_returns = _slice_backtest_window(returns, window)
        if windowed_returns.empty:
            raise ValueError(f"returns.parquet has no rows for window={window}")

        portfolio_returns = windowed_returns.mean(axis=1)
        benchmark_returns = (
            windowed_returns["SPY"]
            if "SPY" in windowed_returns.columns
            else windowed_returns.iloc[:, 0]
        )
        metrics = _metrics_from_returns(portfolio_returns, benchmark_returns)

        wf_cum_array = np.exp(portfolio_returns.cumsum())
        bm_cum_array = np.exp(benchmark_returns.cumsum())
        running_peak = np.maximum.accumulate(wf_cum_array)
        drawdown_array = (wf_cum_array - running_peak) / running_peak
        rewards_array = portfolio_returns.tail(200).cumsum()
        sharpe_spark = _rolling_sharpe_spark(portfolio_returns)

        return (
            metrics,
            [index.strftime("%Y-%m-%d") for index in windowed_returns.index],
            _finite_float_list(wf_cum_array),
            _finite_float_list(bm_cum_array),
            _finite_float_list(rewards_array),
            _finite_float_list(drawdown_array),
            sharpe_spark,
        )
    except (OSError, ValueError, KeyError, ImportError):
        return _static_backtest_payload(window)


def _static_backtest_payload(window: BacktestWindow) -> tuple[
    dict[str, float],
    list[str],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Return deterministic backtest data for the requested walk-forward test window."""
    start, end = WINDOW_PERIODS[window]
    dates = pd.date_range(start, end, freq="B")
    drifts = {
        "w1": (-0.00045, -0.0007),
        "w2": (0.0005, 0.00035),
        "w3": (0.0007, 0.00048),
        "final": (0.0004, 0.0003),
    }
    portfolio_drift, benchmark_drift = drifts[window]
    seasonal = np.sin(np.linspace(0.0, 8.0 * np.pi, len(dates))) * 0.0012
    benchmark_seasonal = np.cos(np.linspace(0.0, 7.0 * np.pi, len(dates))) * 0.0009
    portfolio_returns = pd.Series(portfolio_drift + seasonal, index=dates)
    benchmark_returns = pd.Series(benchmark_drift + benchmark_seasonal, index=dates)
    wf_cum = np.exp(portfolio_returns.cumsum())
    bm_cum = np.exp(benchmark_returns.cumsum())
    drawdown = (wf_cum - np.maximum.accumulate(wf_cum)) / np.maximum.accumulate(wf_cum)
    metrics = _metrics_from_returns(portfolio_returns, benchmark_returns)
    return (
        metrics,
        [item.strftime("%Y-%m-%d") for item in dates],
        _finite_float_list(wf_cum),
        _finite_float_list(bm_cum),
        _finite_float_list(portfolio_returns.tail(200).cumsum()),
        _finite_float_list(drawdown),
        [0.0] * 50,
    )


def _slice_backtest_window(returns: pd.DataFrame, window: BacktestWindow) -> pd.DataFrame:
    """Slice raw return rows to the documented walk-forward test period."""
    start, end = WINDOW_PERIODS[window]
    return returns.loc[start:end]


def _metrics_from_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict[str, float]:
    """Compute JSON-safe metrics from a portfolio and benchmark return series."""
    from src.rl.metrics import calculate_all_metrics

    return _finite_metrics(calculate_all_metrics(portfolio_returns, benchmark_returns))


def _fallback_anova() -> list[AnovaResult]:
    """Return hardcoded ANOVA results matching anova_results.json (static fallback)."""
    return [
        AnovaResult(
            name="reward_function_comparison",
            f_statistic=18.040523,
            p_value=0.0,
            eta_squared=0.014817,
            post_hoc=[
                TukeyRow(
                    group1="PPO-mdd",
                    group2="PPO-return",
                    meandiff=0.0018,
                    p_adj=0.0002,
                    reject=True,
                ),
                TukeyRow(
                    group1="PPO-mdd",
                    group2="PPO-sharpe",
                    meandiff=0.0026,
                    p_adj=0.0,
                    reject=True,
                ),
                TukeyRow(
                    group1="PPO-return",
                    group2="PPO-sharpe",
                    meandiff=0.0008,
                    p_adj=0.1322,
                    reject=False,
                ),
            ],
        ),
        AnovaResult(
            name="strategy_comparison",
            f_statistic=57.141498,
            p_value=0.0,
            eta_squared=0.040522,
            post_hoc=[
                TukeyRow(
                    group1="MVO",
                    group2="PPO",
                    meandiff=0.0032,
                    p_adj=0.0,
                    reject=True,
                ),
                TukeyRow(
                    group1="MVO",
                    group2="동일비중",
                    meandiff=0.0001,
                    p_adj=0.9741,
                    reject=False,
                ),
                TukeyRow(
                    group1="PPO",
                    group2="동일비중",
                    meandiff=-0.0031,
                    p_adj=0.0,
                    reject=True,
                ),
            ],
        ),
        AnovaResult(
            name="market_regime_comparison",
            f_statistic=7.329351,
            p_value=0.000673,
            eta_squared=0.006902,
            post_hoc=[
                TukeyRow(
                    group1="bull",
                    group2="rate_hike",
                    meandiff=-0.0012,
                    p_adj=0.0077,
                    reject=True,
                ),
                TukeyRow(
                    group1="bull",
                    group2="recovery",
                    meandiff=0.0002,
                    p_adj=0.8879,
                    reject=False,
                ),
                TukeyRow(
                    group1="rate_hike",
                    group2="recovery",
                    meandiff=0.0014,
                    p_adj=0.0016,
                    reject=True,
                ),
            ],
            interaction=InteractionStats(
                f_statistic=0.139752,
                p_value=0.967487,
                significant=False,
            ),
            strategy_effect=StrategyEffectStats(
                f_statistic=38.78218,
                p_value=0.0,
            ),
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
        returns = _load_returns()
        features = _load_features()
    except (OSError, ValueError, ImportError):
        return False
    return not returns.empty and not features.empty


def _infer_risk_tags(question: str) -> list[str]:
    """Infer RL risk tags from a Korean or English question."""
    from src.agent.risk_tags import RL_RISK_TAGS, extract_rl_risk_tags
    return extract_rl_risk_tags(question) or [RL_RISK_TAGS[1]]  # fallback: equity_market_risk


def _default_rl_risk_tags() -> list[str]:
    """Return the default RL risk vector tag used when requests omit tags."""
    from src.agent.risk_tags import RL_RISK_TAGS

    return [RL_RISK_TAGS[1]]  # equity_market_risk


def _normalize_rl_risk_tags(tags: Any) -> list[str]:
    """Keep only tags that belong to the fixed RL observation vector."""
    try:
        from src.agent.risk_tags import RL_RISK_TAGS
        allowed = set(RL_RISK_TAGS)
    except Exception:
        return []
    return [str(tag) for tag in tags or [] if str(tag) in allowed]


def _risk_signals_from_tags(tags: list[str]) -> list[RiskSignal]:
    """Convert risk tags into default-severity signals for backward compatibility."""
    return [RiskSignal(tag=tag, severity=1.0) for tag in _normalize_rl_risk_tags(tags)]


def _resolve_risk_signals(
    risk_tags: list[str] | None,
    risk_signals: Any,
) -> list[RiskSignal]:
    """Resolve request risk_signals, falling back to legacy risk_tags only once."""
    return _normalize_risk_signals(risk_signals) or _risk_signals_from_tags(risk_tags or [])


def _normalize_risk_signals(raw_signals: Any) -> list[RiskSignal]:
    """Normalize raw risk signal payload into validated RiskSignal entries."""
    if not isinstance(raw_signals, list):
        return []
    normalized: list[RiskSignal] = []
    for item in raw_signals:
        if isinstance(item, RiskSignal):
            normalized.append(item)
            continue
        if not isinstance(item, dict):
            continue
        tag = item.get("tag")
        severity = item.get("severity")
        if tag is None or severity is None:
            continue
        try:
            normalized.append(RiskSignal(tag=str(tag), severity=float(severity)))
        except (TypeError, ValueError, ValidationError):
            continue
    return normalized


def _signals_to_vector(signals: list[RiskSignal]) -> np.ndarray:
    """Build RL observation risk vector from RiskSignal entries."""
    from src.agent.risk_tags import RL_RISK_TAGS, apply_decay

    severity_by_tag = {signal.tag: signal.severity for signal in signals}
    return np.array(
        [apply_decay(float(severity_by_tag.get(tag, 0.0)), 0, tag) for tag in RL_RISK_TAGS],
        dtype=np.float32,
    )


def _load_unified_events() -> pd.DataFrame:
    """Load unified reasoning events, reloading when the parquet asset changes."""
    global _UNIFIED_EVENTS_CACHE
    try:
        mtime_ns = UNIFIED_EVENTS_PATH.stat().st_mtime_ns
    except OSError:
        return pd.DataFrame()

    if (
        _UNIFIED_EVENTS_CACHE is not None
        and _UNIFIED_EVENTS_CACHE[0] == UNIFIED_EVENTS_PATH
        and _UNIFIED_EVENTS_CACHE[1] == mtime_ns
    ):
        return _UNIFIED_EVENTS_CACHE[2]

    try:
        events = pd.read_parquet(UNIFIED_EVENTS_PATH)
    except Exception:
        events = pd.DataFrame()
    if events.empty or "date" not in events.columns:
        loaded = pd.DataFrame()
    else:
        loaded = events.copy()
        loaded["date"] = pd.to_datetime(loaded["date"], errors="coerce")
        loaded = loaded.dropna(subset=["date"])

    _UNIFIED_EVENTS_CACHE = (UNIFIED_EVENTS_PATH, mtime_ns, loaded)
    return loaded


def _clear_unified_events_cache() -> None:
    """Clear unified event cache for tests and manual invalidation hooks."""
    global _UNIFIED_EVENTS_CACHE
    _UNIFIED_EVENTS_CACHE = None


_load_unified_events.cache_clear = _clear_unified_events_cache  # type: ignore[attr-defined]


def _normalize_reasoning_event(
    raw_event: dict[str, Any],
    target_date: pd.Timestamp,
) -> ReasoningEvent | None:
    """Convert SHAP/unified raw event payload into ReasoningEvent schema."""
    event_date_raw = raw_event.get("event_date") or raw_event.get("date")
    if event_date_raw is None:
        return None
    event_ts = pd.to_datetime(event_date_raw, errors="coerce")
    if pd.isna(event_ts):
        return None

    tag_raw = raw_event.get("tag") or raw_event.get("primary_tag")
    tag = str(tag_raw).strip() if tag_raw is not None else ""
    if tag not in _TAG_TO_SEVERITY_COL:
        return None

    severity_raw = raw_event.get("severity")
    if severity_raw is None:
        severity_raw = raw_event.get(_TAG_TO_SEVERITY_COL[tag], 0.0)
    try:
        severity = float(severity_raw)
    except (TypeError, ValueError):
        severity = 0.0
    severity = min(max(severity, 0.0), 1.0)

    days_elapsed = max(int((target_date.normalize() - event_ts.normalize()).days), 0)
    reasoning = str(raw_event.get("reasoning") or "").strip()
    if not reasoning:
        return None
    source = str(raw_event.get("source") or "")

    return ReasoningEvent(
        event_date=event_ts.strftime("%Y-%m-%d"),
        days_elapsed=days_elapsed,
        tag=tag,
        severity=round(severity, 6),
        decayed_score=float(apply_decay(severity, days_elapsed, tag)),
        reasoning=reasoning,
        source=source,
    )


def _build_reasoning_events(
    target_date: str | None,
    *,
    tag_filter: str | None = None,
    window_days: int = REASONING_WINDOW_DAYS,
    raw_context: Any = None,
) -> list[ReasoningEvent]:
    """Build normalized reasoning events from SHAP output or unified events parquet."""
    if not target_date:
        return []
    target_ts = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(target_ts):
        return []

    raw_events: list[dict[str, Any]] = []
    if isinstance(raw_context, list):
        raw_events = [item for item in raw_context if isinstance(item, dict)]
    else:
        events = _load_unified_events()
        if events.empty:
            return []
        events = events.copy()
        events["date"] = pd.to_datetime(events["date"], errors="coerce")
        events = events.dropna(subset=["date"])
        lower = target_ts - pd.Timedelta(days=window_days)
        upper = target_ts
        windowed = events[(events["date"] >= lower) & (events["date"] <= upper)].copy()
        if windowed.empty:
            return []
        raw_events = [
            {
                "event_date": row["date"],
                "source": row.get("source"),
                "tag": row.get("primary_tag"),
                "severity": row.get(row.get("primary_tag", ""), row.get("severity", 0.0)),
                "reasoning": row.get("reasoning"),
            }
            for _, row in windowed.iterrows()
        ]

    normalized = [
        event
        for item in raw_events
        for event in [_normalize_reasoning_event(item, target_ts)]
        if event is not None
    ]
    if tag_filter:
        normalized = [item for item in normalized if item.tag == tag_filter]
    normalized.sort(key=lambda item: (item.event_date, item.tag))
    return normalized


def _risk_tag_from_feature(feature_name: str) -> str | None:
    """Resolve RL risk tag from a SHAP feature name."""
    lowered = feature_name.strip().lower()
    for key, tag in _RISK_FEATURE_TO_TAG.items():
        if key in lowered:
            return tag
    return None


def _attach_feature_reasoning_context(
    contributions: list[FeatureContribution],
    target_date: str | None,
    *,
    raw_context: Any = None,
) -> list[FeatureContribution]:
    """Attach reasoning context to risk_* feature contributions only."""
    attached: list[FeatureContribution] = []
    for contribution in contributions:
        tag = _risk_tag_from_feature(contribution.feature)
        context = (
            _build_reasoning_events(target_date, tag_filter=tag, raw_context=raw_context)
            if tag and target_date
            else []
        )
        attached.append(
            FeatureContribution(
                feature=contribution.feature,
                value=contribution.value,
                contribution=contribution.contribution,
                reasoning_context=context,
            )
        )
    return attached


def _to_ndjson(payload: dict[str, Any]) -> str:
    """Serialize one compact UTF-8 NDJSON event."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def _jsonable(value: Any) -> Any:
    """Convert LangGraph event data into JSON-serializable primitives."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        pass

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    return str(value)


TENSORBOARD_LOG_DIR = Path("logs/tensorboard")
_TB_SCALAR_TAG = "rollout/ep_rew_mean"


def build_train_curve_response(
    wf_window: BacktestWindow | None = None,
    smooth_window: int | None = None,
) -> TrainCurveResponse:
    """Return per-episode reward curve from TensorBoard logs, or fallback on error."""
    start = perf_counter()
    try:
        return _read_train_curve(wf_window, smooth_window, start)
    except Exception:
        return TrainCurveResponse(
            status="fallback",
            elapsed_ms=_elapsed_ms(start),
            episode_steps=[],
            rewards=[],
            run_name="",
            message="TensorBoard 로그를 읽을 수 없습니다.",
        )


def _read_train_curve(
    wf_window: BacktestWindow | None,
    smooth_window: int | None,
    start: float,
) -> TrainCurveResponse:
    """Read rollout/ep_rew_mean from a TensorBoard run filtered by walk-forward window."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    all_run_dirs = sorted(
        TENSORBOARD_LOG_DIR.glob("ppo_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not all_run_dirs:
        return TrainCurveResponse(
            status="fallback",
            elapsed_ms=_elapsed_ms(start),
            episode_steps=[],
            rewards=[],
            run_name="",
            message="logs/tensorboard/ 에 ppo_* 디렉터리가 없습니다.",
        )

    # wf_window 지정 시 해당 패턴(ppo_*_<window>_*)을 먼저 탐색하고, 없으면 최신 run 사용
    if wf_window:
        filtered = [p for p in all_run_dirs if f"_{wf_window}_" in p.name or p.name.endswith(f"_{wf_window}")]
        run_dirs = filtered if filtered else all_run_dirs
    else:
        run_dirs = all_run_dirs

    run_dir = run_dirs[0]
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    if _TB_SCALAR_TAG not in available_tags:
        return TrainCurveResponse(
            status="fallback",
            elapsed_ms=_elapsed_ms(start),
            episode_steps=[],
            rewards=[],
            run_name=run_dir.name,
            message=f"'{_TB_SCALAR_TAG}' 태그가 없습니다 (run: {run_dir.name}).",
        )

    raw_values = [float(s.value) for s in ea.Scalars(_TB_SCALAR_TAG)]
    if smooth_window and smooth_window > 1:
        smoothed = (
            pd.Series(raw_values)
            .rolling(smooth_window, min_periods=1)
            .mean()
            .round(6)
            .tolist()
        )
    else:
        smoothed = [round(v, 6) for v in raw_values]

    return TrainCurveResponse(
        status="ready",
        elapsed_ms=_elapsed_ms(start),
        episode_steps=list(range(1, len(smoothed) + 1)),
        rewards=smoothed,
        run_name=run_dir.name,
        message=f"{run_dir.name} 학습 곡선 ({len(smoothed)}개 데이터 포인트)",
    )


def build_explain_dates(window: BacktestWindow) -> dict[str, Any]:
    """SHAP 날짜 선택 UI용 거래일 + 이벤트일 목록 반환.

    Args:
        window: 백테스트 윈도우 키.

    Returns:
        window, all_trading_dates, eventful_dates 담긴 dict.
    """
    start, end = WINDOW_PERIODS[window]
    try:
        returns = _load_returns()
        trading: list[str] = [
            d.strftime("%Y-%m-%d")
            for d in returns.loc[start:end].index
        ]
    except Exception:
        trading = []

    events = _load_unified_events()
    if not events.empty and "date" in events.columns:
        mask = (events["date"] >= pd.Timestamp(start)) & (events["date"] <= pd.Timestamp(end))
        eventful: list[str] = sorted(
            events.loc[mask, "date"].dt.strftime("%Y-%m-%d").dropna().unique().tolist()
        )
    else:
        eventful = []

    return {"window": window, "all_trading_dates": trading, "eventful_dates": eventful}
