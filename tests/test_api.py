"""FastAPI Sprint 2 endpoint contract tests."""

import math
import time
import json
from concurrent.futures import TimeoutError

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
    assert payload["modules"]["rl"] in {"ready", "fallback"}
    assert payload["modules"]["rag"] in {"ready", "fallback"}
    assert payload["modules"]["shap"] in {"ready", "fallback"}
    assert payload["modules"]["backtest"] in {"ready", "fallback"}


def test_optimize_returns_normalized_weights_for_default_assets() -> None:
    """POST /optimize should return a complete normalized portfolio."""
    response = client.post("/optimize", json={"risk_profile": "balanced"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ready", "fallback"}
    assert payload["tickers"] == EXPECTED_TICKERS
    assert set(payload["weights"]) == set(EXPECTED_TICKERS)
    assert math.isclose(sum(payload["weights"].values()), 1.0, abs_tol=1e-9)
    assert all(weight >= 0 for weight in payload["weights"].values())
    assert payload["risk_profile"] == "balanced"
    assert isinstance(payload["elapsed_ms"], float)
    assert payload["elapsed_ms"] >= 0
    assert isinstance(payload["timed_out"], bool)


def test_optimize_accepts_dashboard_risk_aversion_and_returns_series() -> None:
    """POST /optimize should support dashboard payloads and chart-ready returns."""
    response = client.post("/optimize", json={"risk_aversion": 1.5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ready", "fallback"}
    assert math.isclose(sum(payload["weights"].values()), 1.0, abs_tol=1e-9)

    returns = payload["returns"]
    assert set(returns) == {"date", "portfolio", "benchmark"}
    assert len(returns["date"]) > 0
    assert len(returns["date"]) == len(returns["portfolio"]) == len(returns["benchmark"])
    assert all(value > 0 for value in returns["portfolio"])
    assert all(value > 0 for value in returns["benchmark"])


def test_optimize_uses_ready_ppo_weights_when_available(monkeypatch) -> None:
    """POST /optimize should prefer real PPO inference over fallback weights."""

    def fake_predict_ppo_weights(tickers: list[str]) -> dict[str, float]:
        return {tickers[0]: 0.7, tickers[1]: 0.3}

    monkeypatch.setattr(
        api_services, "_predict_ppo_weights", fake_predict_ppo_weights, raising=False
    )

    response = client.post("/optimize", json={"tickers": ["SPY", "QQQ"]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["weights"] == {"SPY": 0.7, "QQQ": 0.3}
    assert "PPO" in payload["message"]


def test_explain_returns_feature_contributions() -> None:
    """POST /explain should provide SHAP-like contribution records."""
    api_services._load_shap_artifact.cache_clear()
    response = client.post("/explain", json={"date": "2024-12-31", "top_k": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ready", "fallback"}
    assert payload["date"] == "2024-12-31"
    assert len(payload["feature_contributions"]) == 5
    assert {"feature", "value", "contribution"} <= set(payload["feature_contributions"][0])
    assert payload["target_date"] <= "2024-12-31"
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
    assert isinstance(payload["elapsed_ms"], float)
    assert payload["elapsed_ms"] >= 0
    assert isinstance(payload["timed_out"], bool)


def test_explain_uses_ready_shap_module_when_available(monkeypatch) -> None:
    """POST /explain should use src.rl.shap when the module can compute a result."""

    def fake_compute_ready_shap(date: str | None, top_k: int) -> dict:
        return {
            "status": "ready",
            "date": date,
            "target_date": "2024-12-30",
            "base_value": 0.1,
            "prediction": 0.13,
            "feature_contributions": [{"feature": "SPY_RSI", "value": 0.5, "contribution": 0.03}],
            "feature_names": ["SPY_RSI"],
            "shap_values": [0.03],
            "message": f"PPO SHAP 분석 완료 top_k={top_k}",
        }

    monkeypatch.setattr(
        api_services, "_compute_ready_shap_with_timeout", fake_compute_ready_shap, raising=False
    )

    response = client.post("/explain", json={"date": "2024-12-31", "top_k": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["target_date"] == "2024-12-30"
    assert payload["feature_names"] == ["SPY_RSI"]


def test_explain_prefers_precomputed_shap_artifact(monkeypatch, tmp_path) -> None:
    """POST /explain should serve precomputed SHAP artifacts before live SHAP."""
    artifact_path = tmp_path / "shap_explanations.json"
    artifact_path.write_text(
        json.dumps(
            {
                "latest_date": "2025-12-30",
                "explanations": [
                    {
                        "date": "2025-12-30",
                        "base_value": 0.2,
                        "prediction": 0.25,
                        "feature_contributions": [
                            {"feature": "SPY_RSI", "value": 61.2, "contribution": 0.03},
                            {"feature": "TLT_volatility", "value": 0.12, "contribution": -0.01},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fail_live_shap(date: str | None, top_k: int) -> dict:
        raise AssertionError("live SHAP should not run when artifact exists")

    api_services._load_shap_artifact.cache_clear()
    monkeypatch.setattr(api_services, "SHAP_ARTIFACT_PATH", artifact_path)
    monkeypatch.setattr(api_services, "_compute_ready_shap_with_timeout", fail_live_shap)

    response = client.post("/explain", json={"date": "2025-12-30", "top_k": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["target_date"] == "2025-12-30"
    assert payload["feature_names"] == ["SPY_RSI"]
    assert "artifact" in payload["message"]
    api_services._load_shap_artifact.cache_clear()


def test_research_returns_report_sources_trace_and_risk_tags() -> None:
    """POST /research should expose the RAG response contract for Streamlit."""
    response = client.post(
        "/research",
        json={"question": "금리 인하가 ETF 포트폴리오에 미치는 영향은?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ready", "fallback"}
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
    assert payload["status"] in {"ready", "fallback"}
    assert payload["report"]
    assert payload["risk_tags"] == ["급등락"]


def test_research_uses_fast_local_rag_when_documents_exist(monkeypatch) -> None:
    """POST /research should return a ready local-RAG report without waiting for LangGraph."""
    import src.agent.vectorstore as vectorstore

    def fake_query_documents(*args, **kwargs) -> dict:
        return {
            "documents": [
                [
                    "SPY는 미국 대형주 경기 민감도를 반영하고 TLT는 장기 금리 변화에 민감합니다.",
                    "금리 인하 국면에서는 장기채 가격과 성장주 밸류에이션이 함께 움직일 수 있습니다.",
                ]
            ],
            "metadatas": [
                [
                    {"title": "SPY TLT 리스크", "url": "https://example.com/spy-tlt"},
                    {"title": "금리 인하와 ETF", "url": "https://example.com/rates-etf"},
                ]
            ],
        }

    def fail_graph(question: str) -> dict:
        raise AssertionError("LangGraph should not run for fast local RAG")

    api_services._build_fast_research_response.cache_clear()
    monkeypatch.setattr(api_services.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(vectorstore, "query_documents", fake_query_documents)
    monkeypatch.setattr(api_services, "run_graph", fail_graph)

    response = client.post("/research", json={"question": "SPY와 TLT 배분 리스크는?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert "SPY" in payload["report"]
    assert payload["sources"] == ["https://example.com/spy-tlt", "https://example.com/rates-etf"]
    assert payload["reasoning_trace"]
    assert payload["elapsed_ms"] < 5000
    api_services._build_fast_research_response.cache_clear()


def test_rag_seed_documents_enable_ready_research(monkeypatch, tmp_path) -> None:
    """Local seed documents should make an empty Chroma store usable for /research."""
    persist_dir = str(tmp_path / "chroma_seed")
    monkeypatch.setattr(api_services.settings, "CHROMA_PERSIST_DIR", persist_dir)
    monkeypatch.setattr(api_services.settings, "OPENAI_API_KEY", "test-key")
    api_services._build_fast_research_response.cache_clear()

    inserted = api_services._ensure_rag_seed_documents()
    response = client.post("/research", json={"question": "SPY와 TLT 금리 리스크는?"})

    assert inserted >= 1
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["sources"]
    assert "SPY" in payload["report"] or "TLT" in payload["report"]
    api_services._build_fast_research_response.cache_clear()


def test_research_sync_falls_back_when_graph_times_out(monkeypatch) -> None:
    """POST /research should keep the synchronous API under the request budget."""

    def raise_timeout(question: str) -> dict:
        raise TimeoutError(f"research timed out for {question}")

    monkeypatch.setattr(api_services.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(api_services, "_rag_has_documents", lambda: True)
    monkeypatch.setattr(api_services, "_build_fast_research_response", lambda question: None)
    monkeypatch.setattr(api_services, "_run_graph_with_timeout", raise_timeout)

    response = client.post("/research", json={"question": "삼성전자 실적 리스크는?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "fallback"
    assert payload["risk_tags"] == ["실적쇼크"]
    assert isinstance(payload["elapsed_ms"], float)
    assert payload["timed_out"] is True


def test_research_stream_returns_ndjson_events(monkeypatch) -> None:
    """POST /research/stream should expose graph progress as NDJSON lines."""

    async def fake_stream_graph_events(question: str):
        yield {
            "event": "on_chain_start",
            "name": "planner",
            "data": {"input": {"query": question}},
        }
        yield {
            "event": "on_chain_end",
            "name": "analyst",
            "data": {"output": {"response": "분석 완료", "risk_tags": ["급등락"]}},
        }

    monkeypatch.setattr(api_services.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(api_services, "stream_graph_events", fake_stream_graph_events)

    with client.stream(
        "POST",
        "/research/stream",
        json={"question": "금리 급등 리스크는?"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        assert response.headers["x-accel-buffering"] == "no"
        lines = [line for line in response.iter_lines() if line]

    assert len(lines) == 4
    assert '"type":"start"' in lines[0]
    assert '"type":"on_chain_start"' in lines[1]
    assert '"name":"planner"' in lines[1]
    assert '"type":"on_chain_end"' in lines[2]
    assert '"type":"complete"' in lines[-1]


def test_research_stream_falls_back_quickly_without_api_key(monkeypatch) -> None:
    """POST /research/stream should not block when LangGraph cannot run."""
    monkeypatch.setattr(api_services.settings, "OPENAI_API_KEY", "")
    started_at = time.perf_counter()

    with client.stream(
        "POST",
        "/research/stream",
        json={"question": "삼성전자 실적 쇼크 가능성은?"},
    ) as response:
        lines = [line for line in response.iter_lines() if line]

    elapsed = time.perf_counter() - started_at
    assert response.status_code == 200
    assert elapsed < 5.0
    assert lines
    assert '"type":"fallback"' in lines[-1]
    assert "실적쇼크" in lines[-1]


def test_backtest_returns_metrics_and_anova_results() -> None:
    """GET /backtest should return 12 metrics and the three ANOVA checks."""
    response = client.get("/backtest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ready", "fallback"}
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
    assert isinstance(payload["elapsed_ms"], float)
    assert payload["elapsed_ms"] >= 0
    assert payload["timed_out"] is False


def test_backtest_accepts_window_query_parameter() -> None:
    """GET /backtest should accept the documented walk-forward window query."""
    response = client.get("/backtest", params={"window": "w1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ready", "fallback"}
    assert payload["dates"][0].startswith("2022-")
    assert payload["dates"][-1].startswith("2022-")


def test_backtest_uses_ready_rl_module_when_available(monkeypatch) -> None:
    """GET /backtest should prefer the implemented RL backtest module."""

    def fake_build_ready_backtest(window: str):
        fallback = api_services.build_fallback_backtest(window)
        return fallback.model_copy(
            update={
                "status": "ready",
                "message": f"실제 Walk-Forward 백테스트 결과입니다. (window={window})",
            }
        )

    monkeypatch.setattr(
        api_services,
        "_build_ready_backtest_response",
        fake_build_ready_backtest,
        raising=False,
    )

    response = client.get("/backtest", params={"window": "w2"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert "실제 Walk-Forward" in payload["message"]


def test_backtest_window_changes_returned_period() -> None:
    """GET /backtest should return different test-period data for each window."""
    w1 = client.get("/backtest", params={"window": "w1"}).json()
    final = client.get("/backtest", params={"window": "final"}).json()

    assert w1["dates"][0] != final["dates"][0]
    assert w1["dates"][0].startswith("2022-")
    assert final["dates"][0].startswith("2025-")
    assert w1["metrics"] != final["metrics"]


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
