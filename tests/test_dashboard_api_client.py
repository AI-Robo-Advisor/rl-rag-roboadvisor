"""Dashboard API helper contract tests."""

from __future__ import annotations

from typing import Any

import requests

from apps.dashboard.api_client import (
    build_optimize_payload,
    extract_risk_tags_from_research_event,
    format_research_log_event,
    get_json,
    post_json,
    research_result_from_event,
    risk_vector_from_tags,
    stream_ndjson,
)


class _DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _DummyStreamResponse:
    def __enter__(self) -> "_DummyStreamResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self) -> list[bytes]:
        return [
            b'{"type":"start","question":"q"}',
            b'{"type":"on_chain_start","name":"planner","data":{}}',
            b"",
            b'{"type":"complete","question":"q"}',
        ]


def test_get_json_returns_payload_on_success(monkeypatch) -> None:
    """GET helper should return parsed JSON when the request succeeds."""

    def fake_get(*args: Any, **kwargs: Any) -> _DummyResponse:
        return _DummyResponse({"status": "ok"})

    monkeypatch.setattr(requests, "get", fake_get)

    result = get_json("http://localhost:8000", "/health", timeout=3)

    assert result == {"status": "ok"}


def test_post_json_warns_and_logs_when_falling_back(monkeypatch) -> None:
    """POST helper should warn and print that the UI is using a mock response."""
    warnings: list[str] = []
    logs: list[str] = []

    def fake_post(*args: Any, **kwargs: Any) -> _DummyResponse:
        raise requests.RequestException("connection refused")

    monkeypatch.setattr(requests, "post", fake_post)

    result = post_json(
        "http://localhost:8000",
        "/optimize",
        {"risk_aversion": 1.5},
        timeout=3,
        warn=warnings.append,
        log=logs.append,
    )

    assert result is None
    assert warnings
    assert "이는 mock 응답입니다." in warnings[0]
    assert logs
    assert "[MOCK][/optimize]" in logs[0]


def test_stream_ndjson_yields_formatted_events(monkeypatch) -> None:
    """Streaming helper should parse NDJSON lines for st.write_stream."""

    def fake_post(*args: Any, **kwargs: Any) -> _DummyStreamResponse:
        assert kwargs["stream"] is True
        return _DummyStreamResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    chunks = list(
        stream_ndjson(
            "http://localhost:8000",
            "/research/stream",
            {"question": "q"},
            formatter=lambda event: f"{event['type']}:{event.get('name', '')}\n",
        )
    )

    assert chunks == ["start:\n", "on_chain_start:planner\n", "complete:\n"]


def test_research_complete_event_extracts_rl_risk_tags() -> None:
    """Dashboard should capture stream risk tags for session storage."""
    event = {
        "type": "complete",
        "question": "q",
        "report": "r",
        "sources": [],
        "reasoning_trace": "",
        "risk_tags": ["실적쇼크", "급등락", "금리_리스크"],
    }

    assert extract_risk_tags_from_research_event(event) == ["실적쇼크", "급등락"]
    assert risk_vector_from_tags(["실적쇼크", "급등락"]) == [0.0, 1.0, 1.0]


def test_research_result_from_complete_event() -> None:
    """Dashboard should keep final report and sources separate from logs."""
    event = {
        "type": "complete",
        "question": "q",
        "report": "최종 리포트",
        "sources": ["https://example.com"],
        "reasoning_trace": "trace",
        "risk_tags": ["실적쇼크", "금리_리스크"],
    }

    result = research_result_from_event(event)

    assert result == {
        "status": "ready",
        "question": "q",
        "report": "최종 리포트",
        "sources": ["https://example.com"],
        "reasoning_trace": "trace",
        "risk_tags": ["실적쇼크"],
    }


def test_research_log_formatter_filters_noisy_chat_chunks() -> None:
    """Dashboard log should hide raw chat token chunks and keep milestones."""
    assert format_research_log_event({"type": "on_chat_model_stream", "text": "토큰"}) == ""
    assert (
        format_research_log_event({"type": "on_chain_start", "name": "planner"})
        == "질문 분석 시작\n"
    )


def test_build_optimize_payload_includes_session_risk_tags() -> None:
    """Dashboard /optimize payload should carry session risk tags."""
    payload = build_optimize_payload(1.5, ["실적쇼크", "급등락"])

    assert payload == {"risk_aversion": 1.5, "risk_tags": ["실적쇼크", "급등락"]}
