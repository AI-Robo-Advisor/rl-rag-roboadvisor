"""Endpoint timing smoke tests for the API response budget."""

from time import perf_counter

from fastapi.testclient import TestClient

import apps.api.services as api_services
from apps.api.main import app

client = TestClient(app)


def _assert_under_5s(method: str, path: str, **kwargs):
    start = perf_counter()
    response = getattr(client, method)(path, **kwargs)
    elapsed = perf_counter() - start

    assert response.status_code == 200
    assert elapsed < 5.0
    return response


def _module_status(module: str) -> str:
    response = client.get("/health")
    assert response.status_code == 200
    return response.json()["modules"][module]


def test_processed_data_loaders_are_cached() -> None:
    """Repeated parquet loads should reuse the same in-process objects."""
    first_returns = api_services._load_returns()
    second_returns = api_services._load_returns()
    first_features = api_services._load_features()
    second_features = api_services._load_features()

    assert first_returns is second_returns
    assert first_features is second_features


def test_health_under_5s() -> None:
    _assert_under_5s("get", "/health")


def test_optimize_under_5s() -> None:
    response = _assert_under_5s(
        "post",
        "/optimize",
        json={"tickers": ["SPY", "QQQ", "TLT", "GLD"], "risk_profile": "balanced"},
    )
    if _module_status("rl") == "ready":
        assert response.json()["status"] == "ready"


def test_explain_under_5s() -> None:
    response = _assert_under_5s("post", "/explain", json={"top_k": 5})
    if _module_status("shap") == "ready":
        assert response.json()["status"] == "ready"


def test_research_under_5s() -> None:
    response = _assert_under_5s(
        "post",
        "/research",
        json={"question": "SPY와 TLT 배분 리스크는?"},
    )
    if _module_status("rag") == "ready":
        assert response.json()["status"] == "ready"
    assert response.json()["report"]


def test_backtest_under_5s() -> None:
    response = _assert_under_5s("get", "/backtest", params={"window": "final"})
    if _module_status("backtest") == "ready":
        assert response.json()["status"] == "ready"
