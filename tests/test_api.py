"""FastAPI Sprint 2 endpoint contract tests."""

import math

from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)


EXPECTED_TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ", "069500", "114260"]


def test_health_returns_module_status() -> None:
    """GET /health should expose API and integration readiness."""
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["api"]["host"]
    assert payload["api"]["port"] == 8000
    assert payload["modules"] == {
        "data": "ready",
        "rl": "fallback",
        "rag": "fallback",
        "shap": "fallback",
    }


def test_optimize_returns_normalized_weights_for_default_assets() -> None:
    """POST /optimize should return a complete normalized portfolio."""
    response = client.post("/optimize", json={"risk_profile": "balanced"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "fallback"
    assert payload["tickers"] == EXPECTED_TICKERS
    assert set(payload["weights"]) == set(EXPECTED_TICKERS)
    assert math.isclose(sum(payload["weights"].values()), 1.0, abs_tol=1e-9)
    assert all(weight > 0 for weight in payload["weights"].values())
    assert payload["risk_profile"] == "balanced"


def test_explain_returns_feature_contributions() -> None:
    """POST /explain should provide SHAP-like contribution records."""
    response = client.post("/explain", json={"date": "2024-12-31", "top_k": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "fallback"
    assert payload["date"] == "2024-12-31"
    assert len(payload["feature_contributions"]) == 5
    assert {"feature", "value", "contribution"} <= set(payload["feature_contributions"][0])
    assert isinstance(payload["base_value"], float)
    assert isinstance(payload["prediction"], float)


def test_research_returns_report_sources_trace_and_risk_tags() -> None:
    """POST /research should expose the RAG response contract for Streamlit."""
    response = client.post(
        "/research",
        json={"question": "금리 인하가 ETF 포트폴리오에 미치는 영향은?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "fallback"
    assert payload["report"]
    assert payload["sources"]
    assert payload["reasoning_trace"]
    assert payload["risk_tags"]
    assert payload["question"].startswith("금리 인하")


def test_backtest_returns_metrics_and_anova_results() -> None:
    """GET /backtest should return 12 metrics and the three ANOVA checks."""
    response = client.get("/backtest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "fallback"
    assert len(payload["metrics"]) == 12
    assert {"cumulative_return", "cagr", "sharpe_ratio", "max_drawdown"} <= set(payload["metrics"])
    assert len(payload["anova"]) == 3
    assert {item["name"] for item in payload["anova"]} == {
        "reward_function_comparison",
        "strategy_comparison",
        "market_regime_comparison",
    }


def test_openapi_schema_contains_sprint2_endpoints() -> None:
    """Swagger/OpenAPI should include the Sprint 2 API surface."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    for path in ["/health", "/optimize", "/explain", "/research", "/backtest"]:
        assert path in paths
