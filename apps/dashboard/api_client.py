"""HTTP helpers for the Streamlit dashboard."""

from __future__ import annotations

import json
from typing import Any, Callable

import requests

RL_RISK_TAGS = ["규제변경", "실적쇼크", "급등락"]


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
