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


def _mock_warning_message(endpoint: str, exc: Exception) -> str:
    """Return a user-facing warning when the dashboard falls back to mock data."""
    return f"API 연결 실패 ({endpoint}): {exc} — 이는 mock 응답입니다."


def extract_risk_tags_from_research_event(event: dict[str, Any]) -> list[str]:
    """Return RL risk tags from a research stream complete/fallback event."""
    event_type = event.get("type")
    if event_type == "complete":
        raw_tags = event.get("risk_tags", [])
    elif event_type == "fallback":
        data = event.get("data") or {}
        raw_tags = data.get("risk_tags", []) if isinstance(data, dict) else []
    else:
        raw_tags = []
    return [str(tag) for tag in raw_tags if str(tag) in RL_RISK_TAGS]


def research_result_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a research stream final event into dashboard session data."""
    event_type = event.get("type")
    if event_type == "complete":
        return {
            "status": "ready",
            "question": str(event.get("question", "")),
            "report": str(event.get("report", "")),
            "sources": [str(source) for source in event.get("sources", []) if source],
            "reasoning_trace": str(event.get("reasoning_trace", "")),
            "risk_tags": extract_risk_tags_from_research_event(event),
        }
    if event_type == "fallback":
        data = event.get("data") or {}
        if not isinstance(data, dict):
            return None
        return {
            "status": str(data.get("status", "fallback")),
            "question": str(data.get("question", "")),
            "report": str(data.get("report", "")),
            "sources": [str(source) for source in data.get("sources", []) if source],
            "reasoning_trace": str(data.get("reasoning_trace", "")),
            "risk_tags": extract_risk_tags_from_research_event(event),
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
) -> dict[str, Any]:
    """Build the /optimize request body sent by the dashboard."""
    return {"risk_aversion": risk_aversion, "risk_tags": list(risk_tags or [])}


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
