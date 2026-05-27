"""FastAPI Sprint 2 endpoint contract tests."""

import json
import logging
import math
import time
from concurrent.futures import TimeoutError

import pandas as pd
from fastapi.testclient import TestClient

import apps.api.services as api_services
from apps.api.main import app
from src.agent.risk_tags import RL_RISK_TAGS

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


def test_optimize_emits_e2e_log(caplog) -> None:
    """POST /optimize should emit one structured E2E log for Docker inspection."""
    caplog.set_level(logging.INFO, logger="apps.api.e2e")
    caplog.set_level(logging.INFO, logger="uvicorn.error")

    response = client.post("/optimize", json={"risk_aversion": 1.5})

    assert response.status_code == 200
    events = [
        json.loads(record.message.removeprefix("E2E "))
        for record in caplog.records
        if record.name == "apps.api.e2e" and record.message.startswith("E2E ")
    ]
    assert any(
        event["endpoint"] == "/optimize"
        and event["status"] in {"ready", "fallback"}
        and event["tickers_count"] == len(EXPECTED_TICKERS)
        for event in events
    )
    assert any(
        record.name == "uvicorn.error" and record.message.startswith("E2E ")
        for record in caplog.records
    )


def test_optimize_uses_ready_ppo_weights_when_available(monkeypatch) -> None:
    """POST /optimize should prefer real PPO inference over fallback weights."""

    def fake_predict_ppo_weights(
        tickers: list[str],
        risk_tags: list[str] | None = None,
        risk_signals: list | None = None,
    ) -> dict[str, float]:
        assert risk_tags == ["equity_market_risk"]
        assert risk_signals is None
        return {tickers[0]: 0.7, tickers[1]: 0.3}

    monkeypatch.setattr(
        api_services, "_predict_ppo_weights", fake_predict_ppo_weights, raising=False
    )

    response = client.post(
        "/optimize",
        json={"tickers": ["SPY", "QQQ"], "risk_tags": ["equity_market_risk"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["weights"] == {"SPY": 0.7, "QQQ": 0.3}
    assert "PPO" in payload["message"]


def test_predict_ppo_weights_sets_risk_vector_from_signals(monkeypatch) -> None:
    """PPO inference should pass request risk_vector during env construction."""
    captured: dict[str, object] = {}
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    returns = pd.DataFrame({"SPY": [0.001] * 40, "QQQ": [0.002] * 40}, index=dates)
    features = pd.DataFrame(
        {
            "SPY_return": [0.001] * 40,
            "QQQ_return": [0.002] * 40,
            "SPY_RSI": [50.0] * 40,
            "QQQ_RSI": [55.0] * 40,
            "SPY_MACD_signal": [0.0] * 40,
            "QQQ_MACD_signal": [0.0] * 40,
        },
        index=dates,
    )

    class FakeModel:
        def predict(self, obs, deterministic: bool = True):
            return [0.8, 0.2], None

    class FakeEnv:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs
            self.features_df = features
            self.asset_names = ["SPY", "QQQ"]

        def reset(self):
            return [0.0], {}

        def _get_observation(self):
            return [0.0]

        def _normalize_action(self, action):
            return action

        def set_risk_vector(self, risk_vector):
            captured["set_called"] = True

    import src.rl.env as rl_env

    monkeypatch.setattr(api_services, "_load_returns", lambda: returns)
    monkeypatch.setattr(api_services, "_load_features", lambda: features)
    monkeypatch.setattr(api_services, "_load_ppo_model", lambda: FakeModel())
    monkeypatch.setattr(rl_env, "PortfolioEnv", FakeEnv)

    weights = api_services._predict_ppo_weights(
        ["SPY", "QQQ"],
        ["equity_market_risk", "geopolitical_fx_risk"],
        [
            {"tag": "equity_market_risk", "severity": 0.66},
            {"tag": "geopolitical_fx_risk", "severity": 1.0},
        ],
    )

    assert weights == {"SPY": 0.8, "QQQ": 0.2}
    risk_vector = captured["init_kwargs"]["risk_vector"]
    assert risk_vector[0] == 0.0
    assert math.isclose(float(risk_vector[1]), 0.66, rel_tol=0, abs_tol=1e-6)
    assert risk_vector[2] == 1.0
    assert "set_called" not in captured


def test_predict_ppo_weights_uses_env_default_when_no_signals(monkeypatch) -> None:
    """PPO inference should rely on env defaults when no signals are supplied."""
    captured: dict[str, object] = {}
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    returns = pd.DataFrame({"SPY": [0.001] * 40, "QQQ": [0.002] * 40}, index=dates)
    features = pd.DataFrame(
        {
            "SPY_return": [0.001] * 40,
            "QQQ_return": [0.002] * 40,
            "SPY_RSI": [50.0] * 40,
            "QQQ_RSI": [55.0] * 40,
            "SPY_MACD_signal": [0.0] * 40,
            "QQQ_MACD_signal": [0.0] * 40,
        },
        index=dates,
    )

    class FakeModel:
        def predict(self, obs, deterministic: bool = True):
            return [0.8, 0.2], None

    class FakeEnv:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs
            self.features_df = features
            self.asset_names = ["SPY", "QQQ"]

        def reset(self):
            return [0.0], {}

        def _get_observation(self):
            return [0.0]

        def _normalize_action(self, action):
            return action

        def set_risk_vector(self, risk_vector):
            captured["set_called"] = True

    import src.rl.env as rl_env

    monkeypatch.setattr(api_services, "_load_returns", lambda: returns)
    monkeypatch.setattr(api_services, "_load_features", lambda: features)
    monkeypatch.setattr(api_services, "_load_ppo_model", lambda: FakeModel())
    monkeypatch.setattr(rl_env, "PortfolioEnv", FakeEnv)

    weights = api_services._predict_ppo_weights(["SPY", "QQQ"], [], [])
    assert weights == {"SPY": 0.8, "QQQ": 0.2}
    assert "risk_vector" not in captured["init_kwargs"]
    assert "set_called" not in captured


def test_normalize_risk_signals_rejects_invalid_rows() -> None:
    """Risk signal normalization should skip malformed, unknown, or out-of-range rows."""
    signals = api_services._normalize_risk_signals(
        [
            {"tag": "equity_market_risk", "severity": 0.66},
            {"tag": "unknown", "severity": 1.0},
            {"tag": "macro_rate_risk", "severity": 1.0001},
            {"tag": "geopolitical_fx_risk", "severity": "bad"},
        ]
    )

    assert [signal.model_dump() for signal in signals] == [
        {"tag": "equity_market_risk", "severity": 0.66}
    ]


def test_explain_returns_feature_contributions() -> None:
    """POST /explain should provide SHAP-like contribution records."""
    api_services._load_shap_artifact.cache_clear()
    response = client.post("/explain", json={"date": "2024-12-31", "top_k": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ready", "fallback"}
    assert payload["date"] == "2024-12-31"
    assert len(payload["feature_contributions"]) == 5
    assert {"feature", "value", "contribution", "reasoning_context"} <= set(
        payload["feature_contributions"][0]
    )
    assert payload["target_date"] <= "2024-12-31"
    assert len(payload["feature_names"]) == 5
    assert len(payload["shap_values"]) == 5
    assert "reasoning_context" in payload
    assert isinstance(payload["reasoning_context"], list)
    assert isinstance(payload["feature_contributions"][0]["reasoning_context"], list)
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


def test_explain_invalid_date_falls_back_without_reasoning_context() -> None:
    """Invalid explain dates should not fail while building reasoning context."""
    response = client.post("/explain", json={"date": "not-a-date", "top_k": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["date"] == "not-a-date"
    assert payload["reasoning_context"] == []
    assert all(item["reasoning_context"] == [] for item in payload["feature_contributions"])


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


def test_explain_attaches_reasoning_context_for_risk_features_only(
    monkeypatch,
    tmp_path,
) -> None:
    """POST /explain should attach per-feature reasoning only to risk_* features."""
    events_path = tmp_path / "unified_events.parquet"
    pd.DataFrame(
        [
            {
                "date": "2024-12-29",
                "source": "gdelt",
                "primary_tag": "equity_market_risk",
                "reasoning": "테스트 reasoning",
                "equity_market_risk": 0.66,
                "macro_rate_risk": 0.0,
                "geopolitical_fx_risk": 0.0,
            }
        ]
    ).to_parquet(events_path)

    def fake_compute_ready_shap(date: str | None, top_k: int) -> dict:
        return {
            "status": "ready",
            "date": date,
            "target_date": "2024-12-30",
            "base_value": 0.1,
            "prediction": 0.13,
            "feature_contributions": [
                {
                    "feature": "risk_equity_market_risk",
                    "value": 0.9,
                    "contribution": 0.03,
                },
                {
                    "feature": "SPY_RSI",
                    "value": 0.5,
                    "contribution": 0.01,
                },
            ],
            "feature_names": ["risk_equity_market_risk", "SPY_RSI"],
            "shap_values": [0.03, 0.01],
            "message": f"PPO SHAP 분석 완료 top_k={top_k}",
        }

    monkeypatch.setattr(api_services, "UNIFIED_EVENTS_PATH", events_path)
    api_services._load_unified_events.cache_clear()
    monkeypatch.setattr(api_services, "_shap_from_artifact", lambda date, top_k: None)
    monkeypatch.setattr(
        api_services, "_compute_ready_shap_with_timeout", fake_compute_ready_shap, raising=False
    )

    response = client.post("/explain", json={"date": "2024-12-31", "top_k": 2})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["reasoning_context"]) == 1
    assert payload["reasoning_context"][0]["reasoning"] == "테스트 reasoning"
    assert len(payload["feature_contributions"][0]["reasoning_context"]) == 1
    assert (
        payload["feature_contributions"][0]["reasoning_context"][0]["tag"]
        == "equity_market_risk"
    )
    assert payload["feature_contributions"][1]["reasoning_context"] == []


def test_build_reasoning_events_normalizes_raw_context() -> None:
    """Reasoning mapper should normalize SHAP raw events into API schema."""
    events = api_services._build_reasoning_events(
        "2024-12-31",
        raw_context=[
            {
                "date": "2024-12-30",
                "source": "manual_seed",
                "primary_tag": "macro_rate_risk",
                "reasoning": "Fed rate change +0.25%p",
            }
        ],
    )

    assert len(events) == 1
    event = events[0].model_dump()
    assert {
        "event_date",
        "days_elapsed",
        "tag",
        "severity",
        "decayed_score",
        "reasoning",
        "source",
    } <= set(event)
    assert event["event_date"] == "2024-12-30"
    assert event["days_elapsed"] == 1
    assert event["tag"] == "macro_rate_risk"
    assert event["source"] == "manual_seed"


def test_build_reasoning_events_returns_reasoning_event_schema(monkeypatch) -> None:
    """Reasoning event builder should output 7-key schema records."""
    events = pd.DataFrame(
        [
            {
                "date": "2024-12-29",
                "source": "gdelt",
                "primary_tag": "equity_market_risk",
                "reasoning": "시장 급락 신호",
                "equity_market_risk": 0.66,
                "macro_rate_risk": 0.0,
                "geopolitical_fx_risk": 0.33,
            }
        ]
    )
    monkeypatch.setattr(api_services, "_load_unified_events", lambda: events)

    records = api_services._build_reasoning_events("2024-12-30")
    assert len(records) == 1
    row = records[0].model_dump()
    assert {
        "event_date",
        "days_elapsed",
        "tag",
        "severity",
        "decayed_score",
        "reasoning",
        "source",
    } <= set(row)


def test_reasoning_risk_feature_maps_are_derived_from_rl_tags() -> None:
    """Reasoning risk-feature mappings should stay aligned with RL risk tags."""
    assert api_services._TAG_TO_SEVERITY_COL == {tag: tag for tag in RL_RISK_TAGS}
    for tag in RL_RISK_TAGS:
        assert api_services._RISK_FEATURE_TO_TAG[f"risk_{tag}"] == tag
    assert api_services._RISK_FEATURE_TO_TAG["risk_macro"] == "macro_rate_risk"
    assert api_services._RISK_FEATURE_TO_TAG["risk_equity"] == "equity_market_risk"
    assert api_services._RISK_FEATURE_TO_TAG["risk_geo"] == "geopolitical_fx_risk"


def test_build_reasoning_events_uses_past_window_and_skips_empty_reasoning(monkeypatch) -> None:
    """Reasoning context should not include future events or empty reasoning rows."""
    events = pd.DataFrame(
        [
            {
                "date": "2024-12-28",
                "source": "gdelt",
                "primary_tag": "equity_market_risk",
                "reasoning": "past market risk",
                "equity_market_risk": 0.66,
            },
            {
                "date": "2024-12-29",
                "source": "gdelt",
                "primary_tag": "equity_market_risk",
                "reasoning": "",
                "equity_market_risk": 0.5,
            },
            {
                "date": "2024-12-31",
                "source": "gdelt",
                "primary_tag": "equity_market_risk",
                "reasoning": "future market risk",
                "equity_market_risk": 0.9,
            },
        ]
    )
    monkeypatch.setattr(api_services, "_load_unified_events", lambda: events)

    records = api_services._build_reasoning_events("2024-12-30")

    assert [record.reasoning for record in records] == ["past market risk"]
    assert all(record.days_elapsed >= 0 for record in records)


def test_load_unified_events_reloads_when_parquet_mtime_changes(monkeypatch, tmp_path) -> None:
    """Unified events should be reloaded when the parquet asset changes."""
    events_path = tmp_path / "unified_events.parquet"
    first = pd.DataFrame(
        [
            {
                "date": "2024-12-29",
                "source": "gdelt",
                "primary_tag": "equity_market_risk",
                "reasoning": "first event",
            }
        ]
    )
    second = first.assign(reasoning=["second event"])
    first.to_parquet(events_path)

    monkeypatch.setattr(api_services, "UNIFIED_EVENTS_PATH", events_path)
    loaded_first = api_services._load_unified_events()

    time.sleep(0.01)
    second.to_parquet(events_path)
    loaded_second = api_services._load_unified_events()

    assert loaded_first.iloc[0]["reasoning"] == "first event"
    assert loaded_second.iloc[0]["reasoning"] == "second event"


def test_load_unified_events_returns_empty_dataframe_on_read_failure(monkeypatch, tmp_path) -> None:
    """Unified events loader should not break /explain when parquet is unreadable."""
    events_path = tmp_path / "unified_events.parquet"
    events_path.write_text("not parquet", encoding="utf-8")
    monkeypatch.setattr(api_services, "UNIFIED_EVENTS_PATH", events_path)
    api_services._load_unified_events.cache_clear()

    def raise_read_error(path):
        raise RuntimeError(f"cannot read {path}")

    monkeypatch.setattr(pd, "read_parquet", raise_read_error)

    loaded = api_services._load_unified_events()

    assert loaded.empty


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
    assert isinstance(payload["risk_signals"], list)
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
    assert payload["risk_tags"] == ["macro_rate_risk"]
    assert payload["risk_signals"] == [{"tag": "macro_rate_risk", "severity": 1.0}]


def test_research_runs_langgraph_without_fast_or_seed_shortcut(monkeypatch) -> None:
    """POST /research should use LangGraph rather than local fast/seed shortcuts."""
    calls: list[str] = []

    def fake_run_graph_with_timeout(question: str) -> dict:
        calls.append(question)
        return {
            "response": "LangGraph 분석 완료",
            "sources": ["https://example.com/langgraph"],
            "reasoning_trace": "[THINK][analyst] 완료",
            "rl_risk_tags": ["equity_market_risk"],
            "risk_signals": [{"tag": "equity_market_risk", "severity": 0.66}],
        }

    monkeypatch.setattr(api_services.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(api_services, "_rag_has_documents", lambda: False)
    monkeypatch.setattr(api_services, "_run_graph_with_timeout", fake_run_graph_with_timeout)

    response = client.post("/research", json={"question": "SPY와 TLT 배분 리스크는?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["report"] == "LangGraph 분석 완료"
    assert payload["sources"] == ["https://example.com/langgraph"]
    assert payload["risk_tags"] == ["equity_market_risk"]
    assert payload["risk_signals"] == [{"tag": "equity_market_risk", "severity": 0.66}]
    assert calls == ["SPY와 TLT 배분 리스크는?"]


def test_research_sync_falls_back_when_graph_times_out(monkeypatch) -> None:
    """POST /research should keep the synchronous API under the request budget."""

    def raise_timeout(question: str) -> dict:
        raise TimeoutError(f"research timed out for {question}")

    monkeypatch.setattr(api_services.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(api_services, "_run_graph_with_timeout", raise_timeout)

    response = client.post("/research", json={"question": "삼성전자 실적 리스크는?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "fallback"
    assert payload["risk_tags"] == ["equity_market_risk"]
    assert isinstance(payload["elapsed_ms"], float)
    assert payload["timed_out"] is True


def test_research_stream_returns_ndjson_events(monkeypatch) -> None:
    """POST /research/stream should expose graph progress as NDJSON lines."""

    async def fake_stream_graph_events(question: str):
        yield {"event": "on_parser_start", "name": "ignored", "data": {"input": "skip"}}
        yield {"event": "on_chain_start", "name": "planner", "data": {"input": {"query": question}}}
        yield {
            "event": "on_chain_end",
            "name": "analyst",
            "data": {
                "output": {
                    "response": "분석 완료",
                    "risk_tags": ["equity_market_risk"],
                    "risk_signals": [{"tag": "equity_market_risk", "severity": 0.66}],
                }
            },
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
    assert '"data"' not in lines[1]
    assert len(lines[1]) < 1000
    assert '"type":"on_chain_end"' in lines[2]
    assert '"type":"complete"' in lines[-1]
    assert '"report":"분석 완료"' in lines[-1]
    assert "equity_market_risk" in lines[-1]
    assert '"risk_signals":[{"tag":"equity_market_risk","severity":0.66}]' in lines[-1]


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
    assert "equity_market_risk" in lines[-1]


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
