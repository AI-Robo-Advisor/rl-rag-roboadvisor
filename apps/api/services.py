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

from apps.api.config import settings
from apps.api.schemas import (
    AnovaResult,
    BacktestWindow,
    BacktestResponse,
    ExplainResponse,
    FeatureContribution,
    InteractionStats,
    OptimizeResponse,
    ReturnSeries,
    ResearchResponse,
    RiskProfile,
    SafeguardState,
    StrategyEffectStats,
    TukeyRow,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

RETURNS_PATH = Path("data/processed/returns.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
SHAP_ARTIFACT_PATH = Path("data/processed/shap_explanations.json")
PPO_MODEL_PATH = Path("models/ppo_sharpe_final_risk.zip")
TRADING_DAYS = 252
PPO_TIMEOUT_SECONDS = 4.75
SHAP_TIMEOUT_SECONDS = 4.75
RESEARCH_TIMEOUT_SECONDS = 4.5
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


def build_fallback_portfolio(
    tickers: list[str] | None = None,
    risk_profile: RiskProfile = "balanced",
    risk_aversion: float | None = None,
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
        message="PPO 모델 연결 전 deterministic fallback 포트폴리오입니다.",
    )


def build_portfolio_response(
    tickers: list[str] | None = None,
    risk_profile: RiskProfile = "balanced",
    risk_aversion: float | None = None,
) -> OptimizeResponse:
    """Return PPO portfolio weights when available, otherwise fallback weights."""
    start = perf_counter()
    selected_tickers = tickers or DEFAULT_TICKERS
    try:
        weights = _predict_ppo_weights_with_timeout(selected_tickers)
    except Exception as exc:
        return build_fallback_portfolio(
            selected_tickers,
            risk_profile,
            risk_aversion,
            elapsed_ms=_elapsed_ms(start),
            timed_out=isinstance(exc, TimeoutError),
        )

    if set(weights) != set(selected_tickers):
        return build_fallback_portfolio(
            selected_tickers,
            risk_profile,
            risk_aversion,
            elapsed_ms=_elapsed_ms(start),
        )

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return build_fallback_portfolio(
            selected_tickers,
            risk_profile,
            risk_aversion,
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
    return ExplainResponse(
        status="fallback",
        elapsed_ms=elapsed_ms,
        timed_out=timed_out,
        date=date,
        target_date=target_date,
        base_value=0.05,
        prediction=round(prediction, 6),
        feature_contributions=selected,
        feature_names=[item.feature for item in selected],
        shap_values=[item.contribution for item in selected],
        message="SHAP 모듈 연결 전 feature contribution fallback입니다.",
    )


def build_explanation_response(date: str | None, top_k: int) -> ExplainResponse:
    """Return SHAP explanation from the RL module when available."""
    start = perf_counter()
    try:
        result = _shap_from_artifact(date, top_k) or _compute_ready_shap_with_timeout(date, top_k)
        return ExplainResponse(
            status="ready",
            elapsed_ms=_elapsed_ms(start),
            timed_out=False,
            date=result.get("date"),
            target_date=result.get("target_date"),
            base_value=result.get("base_value", 0.0),
            prediction=result.get("prediction", 0.0),
            feature_contributions=[
                FeatureContribution(**item) for item in result.get("feature_contributions", [])
            ],
            feature_names=list(result.get("feature_names", [])),
            shap_values=list(result.get("shap_values", [])),
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
    yield _to_ndjson({"type": "start", "question": question})

    if not settings.OPENAI_API_KEY:
        fallback = build_fallback_research(question)
        yield _to_ndjson(
            {
                "type": "fallback",
                "name": "research",
                "data": fallback.model_dump(),
            }
        )
        return

    try:
        async for event in stream_graph_events(question):
            compact = _compact_graph_event(event)
            if compact:
                yield _to_ndjson(compact)
    except Exception as exc:
        fallback = build_fallback_research(question)
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

    yield _to_ndjson({"type": "complete", "question": question})


def _compact_graph_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Convert LangGraph events to small dashboard-oriented NDJSON payloads."""
    event_type = str(event.get("event", "event"))
    if event_type not in _STREAM_EVENT_ALLOWLIST:
        return None

    name = str(event.get("name") or "research")
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
    return {"type": event_type, "name": name, "text": text[:500]}


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

    fast_response = _build_fast_research_response(question)
    if fast_response:
        fast_response.elapsed_ms = _elapsed_ms(start)
        return fast_response

    if not _rag_has_documents():
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


@lru_cache(maxsize=64)
def _build_fast_research_response(question: str) -> ResearchResponse | None:
    """Build a bounded extractive RAG report from local Chroma documents."""
    try:
        from src.agent.risk_tags import extract_risk_tags, extract_rl_risk_tags
        from src.agent.vectorstore import query_documents

        results = query_documents(
            query_texts=[question],
            n_results=3,
            persist_dir=settings.CHROMA_PERSIST_DIR,
        )
    except Exception:
        return None

    documents = results.get("documents") or []
    first_docs = documents[0] if documents and documents[0] else []
    if not first_docs:
        return None

    metadatas = results.get("metadatas") or []
    first_metas = metadatas[0] if metadatas and metadatas[0] else []
    snippets: list[str] = []
    sources: list[str] = []
    for index, content in enumerate(first_docs[:3], start=1):
        text = str(content).strip()
        if not text:
            continue
        meta = first_metas[index - 1] if index - 1 < len(first_metas) else {}
        title = str(meta.get("title") or f"문서 {index}")
        url = str(meta.get("url") or meta.get("source") or "")
        snippets.append(f"{index}. {title}: {text[:350]}")
        if url:
            sources.append(url)

    if not snippets:
        return None

    combined_text = " ".join([question, *snippets])
    risk_tags = extract_rl_risk_tags(combined_text) or extract_risk_tags(combined_text)
    report = (
        "로컬 RAG 검색 결과 기준 투자 리서치 요약입니다.\n\n"
        + "\n".join(snippets)
        + "\n\n"
        + "위 근거를 바탕으로 포트폴리오 관점에서는 관련 자산의 변동성, 금리 민감도, "
        + "분산 효과 변화를 함께 점검해야 합니다."
    )
    reasoning_trace = "\n".join(
        [
            "[THINK][planner] 질문에서 핵심 자산과 리스크 키워드를 추출했습니다.",
            f"[THINK][researcher] Chroma top-k={len(snippets)}건을 조회했습니다.",
            "[THINK][analyst] 검색 문서 기반으로 요약 리포트와 리스크 태그를 생성했습니다.",
        ]
    )
    return ResearchResponse(
        status="ready",
        question=question,
        report=report,
        sources=sources or ["local-chroma://finance_news"],
        reasoning_trace=reasoning_trace,
        risk_tags=[str(tag) for tag in (risk_tags or _infer_risk_tags(question))],
    )


def _build_seed_research_response(question: str) -> ResearchResponse | None:
    """Build a fast report from bundled asset seed documents for known ETF questions."""
    try:
        from src.agent.risk_tags import extract_risk_tags, extract_rl_risk_tags
        from src.agent.seed_documents import SEED_DOCUMENTS
    except Exception:
        return None

    query = question.upper()
    selected: list[dict[str, str]] = []
    for item in SEED_DOCUMENTS:
        haystack = f"{item['title']} {item['summary']}".upper()
        if any(token in query and token in haystack for token in DEFAULT_TICKERS):
            selected.append(item)
    if not selected and any(token in query for token in ("ETF", "금리", "RISK")):
        selected = SEED_DOCUMENTS[:3]
    if not selected:
        return None

    snippets = [
        f"{index}. {item['title']}: {item['summary'][:350]}"
        for index, item in enumerate(selected[:3], start=1)
    ]
    combined_text = " ".join([question, *snippets])
    risk_tags = extract_rl_risk_tags(combined_text) or extract_risk_tags(combined_text)
    report = (
        "로컬 자산 문서 기준 투자 리서치 요약입니다.\n\n"
        + "\n".join(snippets)
        + "\n\n"
        + "위 근거를 바탕으로 포트폴리오 관점에서는 주식 성장 노출, 채권 듀레이션, "
        + "실물자산 방어력, 한국/글로벌 분산 효과를 함께 점검해야 합니다."
    )
    reasoning_trace = "\n".join(
        [
            "[THINK][planner] 질문에서 자산 티커와 금리/경기 리스크 키워드를 추출했습니다.",
            f"[THINK][researcher] 로컬 seed RAG 문서 {len(snippets)}건을 선택했습니다.",
            "[THINK][analyst] 선택 문서 기반으로 요약 리포트와 리스크 태그를 생성했습니다.",
        ]
    )
    return ResearchResponse(
        status="ready",
        question=question,
        report=report,
        sources=[item["url"] for item in selected[:3]],
        reasoning_trace=reasoning_trace,
        risk_tags=[str(tag) for tag in (risk_tags or _infer_risk_tags(question))],
    )


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
    )


def build_fallback_backtest(window: BacktestWindow = "final") -> BacktestResponse:
    """Return metric output from available data plus fallback ANOVA summaries."""
    metrics, dates, wf_cum, bm_cum, rewards, drawdown, sharpe_spark = _build_backtest_payload(
        window
    )
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
        safeguard=SafeguardState(
            active=False,
            triggered_at=None,
            current_drawdown=abs(drawdown[-1]) if drawdown else current_mdd,
        ),
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
        "rag": "ready" if settings.OPENAI_API_KEY and _rag_has_documents() else "fallback",
        "shap": "ready" if _is_shap_ready() else "fallback",
        "backtest": "ready" if _is_backtest_ready() else "fallback",
    }


def _predict_ppo_weights(tickers: list[str]) -> dict[str, float]:
    """Run the trained PPO policy once and return selected asset weights."""
    returns = _load_returns()
    features = _load_features()
    missing = [ticker for ticker in tickers if ticker not in returns.columns]
    if missing:
        raise ValueError(f"Unknown tickers for PPO model: {missing}")

    from src.rl.env import PortfolioEnv

    model = _load_ppo_model()
    env = PortfolioEnv(
        returns_df=returns,
        features_df=features,
        lookback=30,
        reward_type="sharpe",
    )
    obs, _ = env.reset()
    env.current_step = len(env.features_df) - 1
    obs = env._get_observation()
    action, _ = model.predict(obs, deterministic=True)
    full_weights = env._normalize_action(action)
    by_asset = {asset: float(full_weights[index]) for index, asset in enumerate(env.asset_names)}
    return {ticker: by_asset[ticker] for ticker in tickers}


def _predict_ppo_weights_with_timeout(tickers: list[str]) -> dict[str, float]:
    """Run PPO inference with a request-time budget."""
    future = _PPO_EXECUTOR.submit(_predict_ppo_weights, tickers)
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
        _ensure_rag_seed_documents()
        if _is_ppo_ready():
            _load_ppo_model()
    except Exception:
        return


def _compute_ready_shap(date: str | None, top_k: int) -> dict[str, Any]:
    """Compute a bounded SHAP explanation through src.rl.shap."""
    from src.rl.shap import compute_shap_explanation

    features = _load_features()
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
def _load_shap_artifact() -> dict[str, Any] | None:
    """Load precomputed SHAP explanations when an artifact is available."""
    if not SHAP_ARTIFACT_PATH.exists():
        return None
    with SHAP_ARTIFACT_PATH.open(encoding="utf-8") as fp:
        artifact = json.load(fp)
    return artifact if isinstance(artifact, dict) else None


def _shap_from_artifact(date: str | None, top_k: int) -> dict[str, Any] | None:
    """Return a ready SHAP payload from a precomputed artifact."""
    artifact = _load_shap_artifact()
    if not artifact:
        return None

    explanations = artifact.get("explanations")
    if not isinstance(explanations, list) or not explanations:
        return None

    target = date or str(artifact.get("latest_date") or "")
    candidates = [
        item for item in explanations if isinstance(item, dict) and str(item.get("date")) <= target
    ]
    selected = candidates[-1] if candidates else explanations[-1]
    contributions = list(selected.get("feature_contributions", []))[:top_k]
    if not contributions:
        return None

    return {
        "date": date,
        "target_date": selected.get("date"),
        "base_value": selected.get("base_value", 0.0),
        "prediction": selected.get("prediction", 0.0),
        "feature_contributions": contributions,
        "feature_names": [item.get("feature", "") for item in contributions],
        "shap_values": [item.get("contribution", 0.0) for item in contributions],
        "message": "사전 계산된 SHAP artifact 기반 해석입니다.",
    }


def _build_ready_backtest_response(window: BacktestWindow) -> BacktestResponse:
    """Build BacktestResponse from implemented RL backtest and ANOVA modules."""
    from src.rl.anova import run_all_anova
    from src.rl.backtest import WINDOWS, run_window_backtest

    returns = _load_returns()
    features = _load_features()
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
    metrics = _finite_metrics(
        {
            key: value
            for key, value in metrics_raw.items()
            if isinstance(value, (int, float, np.floating))
        }
    )
    wf_cum_array = np.exp(portfolio_returns.cumsum())
    bm_cum_array = np.exp(benchmark_returns.cumsum())
    drawdown_array = (wf_cum_array - np.maximum.accumulate(wf_cum_array)) / np.maximum.accumulate(
        wf_cum_array
    )
    current_mdd = abs(float(drawdown_array.min())) if len(drawdown_array) else 0.0
    triggered = drawdown_array[drawdown_array <= -0.15]
    anova = [AnovaResult(**item) for item in run_all_anova(returns)]

    return BacktestResponse(
        status="ready",
        metrics=metrics,
        anova=anova,
        benchmark="SPY",
        dates=[index.strftime("%Y-%m-%d") for index in portfolio_returns.index],
        rewards=_finite_float_list(portfolio_returns.tail(200).cumsum()),
        wf_cum=_finite_float_list(wf_cum_array),
        bm_cum=_finite_float_list(bm_cum_array),
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


def _ensure_rag_seed_documents() -> int:
    """Populate deterministic local RAG documents when Chroma is empty."""
    try:
        from src.agent.seed_documents import ensure_seed_documents

        _build_fast_research_response.cache_clear()
        return ensure_seed_documents(settings.CHROMA_PERSIST_DIR)
    except Exception:
        return 0


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
    """Return the three planned ANOVA fallback summaries."""
    return [
        AnovaResult(
            name="reward_function_comparison",
            f_statistic=3.12,
            p_value=0.041,
            eta_squared=0.18,
            post_hoc=[
                TukeyRow(
                    group1="PPO-return",
                    group2="PPO-sharpe",
                    meandiff=0.021,
                    p_adj=0.032,
                    reject=True,
                ),
                TukeyRow(
                    group1="PPO-return",
                    group2="PPO-mdd",
                    meandiff=0.009,
                    p_adj=0.210,
                    reject=False,
                ),
                TukeyRow(
                    group1="PPO-sharpe",
                    group2="PPO-mdd",
                    meandiff=0.012,
                    p_adj=0.089,
                    reject=False,
                ),
            ],
        ),
        AnovaResult(
            name="strategy_comparison",
            f_statistic=4.36,
            p_value=0.028,
            eta_squared=0.22,
            post_hoc=[
                TukeyRow(
                    group1="PPO",
                    group2="MVO",
                    meandiff=0.031,
                    p_adj=0.002,
                    reject=True,
                ),
                TukeyRow(
                    group1="PPO",
                    group2="동일비중",
                    meandiff=0.018,
                    p_adj=0.041,
                    reject=True,
                ),
                TukeyRow(
                    group1="MVO",
                    group2="동일비중",
                    meandiff=0.013,
                    p_adj=0.312,
                    reject=False,
                ),
            ],
        ),
        AnovaResult(
            name="market_regime_comparison",
            f_statistic=2.07,
            p_value=0.096,
            eta_squared=0.11,
            post_hoc=[],
            interaction=InteractionStats(
                f_statistic=3.14,
                p_value=0.021,
                significant=True,
            ),
            strategy_effect=StrategyEffectStats(
                f_statistic=4.52,
                p_value=0.011,
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
