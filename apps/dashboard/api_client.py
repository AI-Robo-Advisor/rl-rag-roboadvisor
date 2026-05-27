"""HTTP helpers for the Streamlit dashboard."""

from __future__ import annotations

import json
from typing import Any, Callable

import requests

try:
    from src.agent.risk_tags import RL_RISK_TAGS
except ImportError:
    RL_RISK_TAGS = ["macro_rate_risk", "equity_market_risk", "geopolitical_fx_risk"]
_RESEARCH_NODE_LABELS = {
    "planner": "질문 분석",
    "researcher": "문서 검색",
    "grade_documents": "근거 평가",
    "analyst": "리포트 작성",
}


def calculate_period_return_metrics(
    portfolio_cumulative: list[float],
    benchmark_cumulative: list[float],
) -> tuple[float, float]:
    """Return selected-period cumulative and excess returns from cumulative wealth indexes."""
    if not portfolio_cumulative or not benchmark_cumulative:
        return 0.0, 0.0

    portfolio_start = float(portfolio_cumulative[0])
    benchmark_start = float(benchmark_cumulative[0])
    if portfolio_start <= 0 or benchmark_start <= 0:
        return 0.0, 0.0

    portfolio_return = float(portfolio_cumulative[-1]) / portfolio_start - 1.0
    benchmark_return = float(benchmark_cumulative[-1]) / benchmark_start - 1.0
    return portfolio_return, portfolio_return - benchmark_return


def explain_reasoning_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten SHAP response reasoning context for dashboard display."""
    rows: list[dict[str, Any]] = []

    def append_rows(scope: str, feature: str, events: Any) -> None:
        if not isinstance(events, list):
            return
        for event in events:
            if not isinstance(event, dict):
                continue
            rows.append(
                {
                    "구분": scope,
                    "피처": feature,
                    "날짜": str(event.get("event_date", "")),
                    "태그": str(event.get("tag", "")),
                    "강도": event.get("severity", 0.0),
                    "감쇠점수": event.get("decayed_score", 0.0),
                    "근거": str(event.get("reasoning", "")),
                    "출처": str(event.get("source", "")),
                }
            )

    append_rows("전체", "", payload.get("reasoning_context"))
    for contribution in payload.get("feature_contributions", []):
        if not isinstance(contribution, dict):
            continue
        append_rows(
            "피처",
            str(contribution.get("feature", "")),
            contribution.get("reasoning_context"),
        )
    return rows


def _mock_warning_message(endpoint: str, exc: Exception) -> str:
    """Return a user-facing warning when the dashboard falls back to mock data."""
    return f"API 연결 실패 ({endpoint}): {exc} — 이는 mock 응답입니다."


def extract_risk_tags_from_research_event(event: dict[str, Any]) -> list[str]:
    """Return RL risk tags from a research stream complete/fallback event."""
    return extract_risk_context_from_research_event(event)[0]


def extract_risk_signals_from_research_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Return validated risk_signals from research stream events."""
    return extract_risk_context_from_research_event(event)[1]


def extract_risk_context_from_research_event(
    event: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return validated risk_tags and risk_signals from a final research event."""
    event_type = event.get("type")
    if event_type == "complete":
        raw_tags = event.get("risk_tags", [])
        raw_signals = event.get("risk_signals", [])
    elif event_type == "fallback":
        data = event.get("data") or {}
        raw_tags = data.get("risk_tags", []) if isinstance(data, dict) else []
        raw_signals = data.get("risk_signals", []) if isinstance(data, dict) else []
    else:
        raw_tags = []
        raw_signals = []

    tags = [str(tag) for tag in raw_tags if str(tag) in RL_RISK_TAGS]
    normalized: list[dict[str, Any]] = []
    for item in raw_signals:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag") or "")
        if tag not in RL_RISK_TAGS:
            continue
        try:
            severity = float(item.get("severity", 0.0))
        except (TypeError, ValueError):
            continue
        if not 0.0 <= severity <= 1.0:
            continue
        normalized.append({"tag": tag, "severity": severity})
    return tags, normalized


def research_result_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a research stream final event into dashboard session data."""
    event_type = event.get("type")
    if event_type == "complete":
        tags, signals = extract_risk_context_from_research_event(event)
        return {
            "status": "ready",
            "question": str(event.get("question", "")),
            "report": str(event.get("report", "")),
            "sources": [str(source) for source in event.get("sources", []) if source],
            "reasoning_trace": str(event.get("reasoning_trace", "")),
            "risk_tags": tags,
            "risk_signals": signals,
        }
    if event_type == "fallback":
        data = event.get("data") or {}
        if not isinstance(data, dict):
            return None
        tags, signals = extract_risk_context_from_research_event(event)
        return {
            "status": str(data.get("status", "fallback")),
            "question": str(data.get("question", "")),
            "report": str(data.get("report", "")),
            "sources": [str(source) for source in data.get("sources", []) if source],
            "reasoning_trace": str(data.get("reasoning_trace", "")),
            "risk_tags": tags,
            "risk_signals": signals,
        }
    return None


def format_research_log_event(event: dict[str, Any]) -> str:
    """Format only useful research milestones for the dashboard log."""
    event_type = event.get("type", "event")
    name = str(event.get("name") or "research")

    if event_type == "start":
        return f"시작: {event.get('question', '')}\n"
    if event_type == "complete":
        tags = ", ".join(extract_risk_tags_from_research_event(event)) or "없음"
        return f"완료: 리스크 태그 {tags}\n"
    if event_type == "fallback":
        tags = ", ".join(extract_risk_tags_from_research_event(event)) or "없음"
        return f"fallback: 리스크 태그 {tags}\n"
    if event_type == "on_chat_model_stream":
        return ""
    if event_type not in {"on_chain_start", "on_chain_end", "on_tool_start", "on_tool_end"}:
        return ""

    label = _RESEARCH_NODE_LABELS.get(name, name)
    action = "시작" if event_type.endswith("_start") else "완료"
    text = str(event.get("text") or "").strip()
    if len(text) > 180:
        text = f"{text[:180]}..."
    suffix = f" - {text}" if text else ""
    return f"{label} {action}{suffix}\n"


def risk_vector_from_tags(risk_tags: list[str] | None) -> list[float]:
    """Build the dashboard display vector matching the RL observation order."""
    selected = set(risk_tags or [])
    return [1.0 if tag in selected else 0.0 for tag in RL_RISK_TAGS]


def build_optimize_payload(
    risk_aversion: float,
    risk_tags: list[str] | None = None,
    risk_signals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the /optimize request body sent by the dashboard."""
    return {
        "risk_aversion": risk_aversion,
        "risk_tags": list(risk_tags or []),
        "risk_signals": list(risk_signals or []),
    }


def get_json(
    base_url: str,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 10,
    warn: Callable[[str], None] | None = None,
    log: Callable[[str], None] | None = print,
) -> dict[str, Any] | None:
    """Perform a GET request and return JSON, or fall back to None."""
    try:
        resp = requests.get(f"{base_url}{endpoint}", params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        message = _mock_warning_message(endpoint, exc)
        if warn:
            warn(message)
        if log:
            log(f"[MOCK][{endpoint}] {message}")
        return None


def post_json(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: int = 10,
    warn: Callable[[str], None] | None = None,
    log: Callable[[str], None] | None = print,
) -> dict[str, Any] | None:
    """Perform a POST request and return JSON, or fall back to None."""
    try:
        resp = requests.post(f"{base_url}{endpoint}", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        message = _mock_warning_message(endpoint, exc)
        if warn:
            warn(message)
        if log:
            log(f"[MOCK][{endpoint}] {message}")
        return None


def stream_ndjson(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    *,
    formatter: Callable[[dict[str, Any]], str],
    timeout: int = 60,
    warn: Callable[[str], None] | None = None,
    log: Callable[[str], None] | None = print,
):
    """Stream newline-delimited JSON events as formatted strings."""
    try:
        with requests.post(
            f"{base_url}{endpoint}",
            json=payload,
            stream=True,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                event = json.loads(line.decode("utf-8"))
                yield formatter(event)
    except (requests.RequestException, json.JSONDecodeError) as exc:
        message = _mock_warning_message(endpoint, exc)
        if warn:
            warn(message)
        if log:
            log(f"[MOCK][{endpoint}] {message}")
        yield message
