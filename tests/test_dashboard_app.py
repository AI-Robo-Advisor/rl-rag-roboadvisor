"""Dashboard page behavior tests."""

from __future__ import annotations

import sys
from types import ModuleType
from datetime import datetime

if "streamlit_echarts" not in sys.modules:
    stub = ModuleType("streamlit_echarts")
    stub.JsCode = lambda value: value
    stub.st_echarts = lambda *args, **kwargs: None
    sys.modules["streamlit_echarts"] = stub

import apps.dashboard.app as dashboard_app


def test_mock_optimize_matches_api_shape_and_is_stable() -> None:
    """Fallback optimize data should look like a real API response and be deterministic."""
    first = dashboard_app._mock_optimize(1.5)
    second = dashboard_app._mock_optimize(1.5)

    assert first == second
    assert first["status"] == "mock"
    assert isinstance(first["message"], str) and first["message"]
    assert isinstance(first["elapsed_ms"], float)
    assert first["elapsed_ms"] == 0.0
    assert first["timed_out"] is False
    assert isinstance(first["tickers"], list) and first["tickers"]
    assert set(first["weights"]) == set(first["tickers"])
    assert isinstance(first["expected_return"], float)
    assert isinstance(first["expected_volatility"], float)
    assert set(first["returns"]) == {"date", "portfolio", "benchmark"}


def test_portfolio_data_cache_survives_rerun(monkeypatch) -> None:
    """Portfolio page should reuse the last successful optimize response across reruns."""
    state: dict[str, object] = {}
    monkeypatch.setattr(dashboard_app.st, "session_state", state)

    calls: list[tuple[str, dict[str, object]]] = []

    def fake_post(endpoint: str, payload: dict, timeout: int = 10) -> dict:
        calls.append((endpoint, payload))
        return {
            "status": "ready",
            "message": "real response",
            "elapsed_ms": 12.3,
            "timed_out": False,
            "tickers": ["SPY", "QQQ"],
            "weights": {"SPY": 0.6, "QQQ": 0.4},
            "risk_profile": "balanced",
            "expected_return": 0.084,
            "expected_volatility": 0.132,
            "returns": {
                "date": ["2024-01-01", "2024-01-02"],
                "portfolio": [1.0, 1.02],
                "benchmark": [1.0, 1.01],
            },
        }

    monkeypatch.setattr(dashboard_app, "_post", fake_post)

    first = dashboard_app._ensure_portfolio_data(1.5, ["equity_market_risk"], [], refresh=True)
    second = dashboard_app._ensure_portfolio_data(2.0, [], [], refresh=False)

    assert first == second
    assert calls == [
        (
            "/optimize",
            {
                "risk_aversion": 1.5,
                "risk_tags": ["equity_market_risk"],
                "risk_signals": [],
            },
        )
    ]
    assert state["portfolio_data"] == first
    assert state["portfolio_payload"] == {
        "risk_aversion": 1.5,
        "risk_tags": ["equity_market_risk"],
        "risk_signals": [],
    }
    assert isinstance(state["portfolio_updated_at"], str)
    datetime.strptime(state["portfolio_updated_at"], "%Y-%m-%d %H:%M:%S")
