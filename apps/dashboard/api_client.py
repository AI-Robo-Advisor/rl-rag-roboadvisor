"""HTTP helpers for the Streamlit dashboard."""

from __future__ import annotations

import json
from typing import Any, Callable

import requests


def _mock_warning_message(endpoint: str, exc: Exception) -> str:
    """Return a user-facing warning when the dashboard falls back to mock data."""
    return f"API 연결 실패 ({endpoint}): {exc} — 이는 mock 응답입니다."


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
