"""Dashboard API helper contract tests."""

from __future__ import annotations

from typing import Any

import requests

from apps.dashboard.api_client import get_json, post_json


class _DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


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
