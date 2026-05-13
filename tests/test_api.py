"""FastAPI Sprint 2 endpoint contract tests."""

import math

from fastapi.testclient import TestClient

import apps.api.services as api_services
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
    assert payload["modules"]["data"] in {"ready", "fallback"}
    assert payload["modules"]["rl"] == "fallback"
    assert payload["modules"]["rag"] in {"ready", "fallback"}
    assert payload["modules"]["shap"] == "fallback"
    assert payload["modules"]["backtest"] == "fallback"


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


def test_optimize_accepts_dashboard_risk_aversion_and_returns_series() -> None:
    """POST /optimize should support dashboard payloads and chart-ready returns."""
    response = client.post("/optimize", json={"risk_aversion": 1.5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "fallback"
    assert math.isclose(sum(payload["weights"].values()), 1.0, abs_tol=1e-9)

    returns = payload["returns"]
    assert set(returns) == {"date", "portfolio", "benchmark"}
    assert len(returns["date"]) > 0
    assert len(returns["date"]) == len(returns["portfolio"]) == len(returns["benchmark"])
    assert all(value > 0 for value in returns["portfolio"])
    assert all(value > 0 for value in returns["benchmark"])


def test_explain_returns_feature_contributions() -> None:
    """POST /explain should provide SHAP-like contribution records."""
    response = client.post("/explain", json={"date": "2024-12-31", "top_k": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "fallback"
    assert payload["date"] == "2024-12-31"
    assert len(payload["feature_contributions"]) == 5
    assert {"feature", "value", "contribution"} <= set(payload["feature_contributions"][0])
    assert payload["target_date"] == "2024-12-31"
    assert len(payload["feature_names"]) == 5
    assert len(payload["shap_values"]) == 5
    assert payload["feature_names"] == [
        item["feature"] for item in payload["feature_contributions"]
    ]
    assert payload["shap_values"] == [
        item["contribution"] for item in payload["feature_contributions"]
    ]
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
    assert all(isinstance(source, str) for source in payload["sources"])
    assert payload["reasoning_trace"]
    assert isinstance(payload["reasoning_trace"], str)
    assert payload["risk_tags"]
    assert payload["question"].startswith("금리 인하")


def test_research_falls_back_when_graph_raises(monkeypatch) -> None:
    """POST /research should keep returning 200 when LangGraph cannot run."""

    def raise_graph_error(question: str) -> dict:
        raise RuntimeError(f"graph unavailable for {question}")

    monkeypatch.setattr(api_services.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(api_services, "run_graph", raise_graph_error)

    response = client.post("/research", json={"question": "금리 급등 리스크는?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "fallback"
    assert payload["report"]
    assert payload["risk_tags"] == ["급등락"]


def test_backtest_returns_metrics_and_anova_results() -> None:
    """GET /backtest should return 12 metrics and the three ANOVA checks."""
    response = client.get("/backtest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "fallback"
    assert len(payload["metrics"]) == 12
    assert {"cumulative_return", "cagr", "sharpe_ratio", "mdd"} <= set(payload["metrics"])
    assert len(payload["anova"]) == 3
    assert {item["name"] for item in payload["anova"]} == {
        "reward_function_comparison",
        "strategy_comparison",
        "market_regime_comparison",
    }

    assert len(payload["dates"]) > 0
    assert len(payload["dates"]) == len(payload["wf_cum"]) == len(payload["bm_cum"])
    assert len(payload["dates"]) == len(payload["drawdown"])
    assert len(payload["rewards"]) > 0
    assert payload["safeguard"]["active"] is False
    assert payload["safeguard"]["triggered_at"] is None
    assert all(isinstance(item["post_hoc"], list) for item in payload["anova"])


def test_backtest_accepts_window_query_parameter() -> None:
    """GET /backtest should accept the documented walk-forward window query."""
    response = client.get("/backtest", params={"window": "w1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "fallback"


def test_openapi_schema_does_not_expose_removed_stress_endpoint() -> None:
    """Swagger/OpenAPI should not include a separate /backtest/stress endpoint."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/backtest/stress" not in paths


def test_openapi_schema_contains_sprint2_endpoints() -> None:
    """Swagger/OpenAPI should include the Sprint 2 API surface."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    for path in ["/health", "/optimize", "/explain", "/research", "/backtest"]:
        assert path in paths
