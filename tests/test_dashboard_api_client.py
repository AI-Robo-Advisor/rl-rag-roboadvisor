"""Dashboard API helper contract tests."""

from __future__ import annotations

from typing import Any

import requests

from apps.dashboard.api_client import get_json, post_json, stream_ndjson


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
