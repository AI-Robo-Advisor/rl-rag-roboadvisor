"""Dashboard API helper contract tests."""

from __future__ import annotations

from typing import Any

import requests

from apps.dashboard.api_client import (
    anova_attainment_cards,
    anova_summary_rows,
    build_optimize_payload,
    calculate_period_return_metrics,
    explain_reasoning_rows,
    extract_analyst_draft_delta,
    filter_post_hoc_rows,
    extract_risk_signals_from_research_event,
    extract_risk_tags_from_research_event,
    format_research_log_event,
    get_json,
    post_json,
    post_hoc_group_options,
    research_result_from_event,
    risk_vector_from_signals,
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


def test_api_helpers_strip_trailing_slash_from_base_url(monkeypatch) -> None:
    """HTTP helpers should normalize trailing slashes in the base URL."""
    captured: list[str] = []

    def fake_get(url: str, **kwargs: Any) -> _DummyResponse:
        captured.append(url)
        return _DummyResponse({"status": "ok"})

    class _DummyStream:
        def __enter__(self) -> "_DummyStream":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self) -> list[bytes]:
            return [b'{"type":"complete"}']

    def fake_post(url: str, **kwargs: Any) -> _DummyResponse | _DummyStream:
        captured.append(url)
        if kwargs.get("stream"):
            return _DummyStream()
        return _DummyResponse({"status": "ok"})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)

    assert get_json("http://localhost:8000/", "/health", timeout=3) == {"status": "ok"}
    assert post_json(
        "http://localhost:8000/",
        "/optimize",
        {"risk_aversion": 1.5},
        timeout=3,
    ) == {"status": "ok"}

    list(
        stream_ndjson(
            "http://localhost:8000/",
            "/research/stream",
            {"question": "q"},
            formatter=lambda event: "",
        )
    )

    assert captured == [
        "http://localhost:8000/health",
        "http://localhost:8000/optimize",
        "http://localhost:8000/research/stream",
    ]


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
        "risk_tags": ["equity_market_risk", "geopolitical_fx_risk", "macro_rate_risk"],
    }

    assert extract_risk_tags_from_research_event(event) == ["equity_market_risk", "geopolitical_fx_risk", "macro_rate_risk"]
    assert risk_vector_from_tags(["equity_market_risk", "geopolitical_fx_risk"]) == [0.0, 1.0, 1.0]


def test_risk_vector_from_signals_uses_severity_values() -> None:
    """Dashboard display vector should preserve risk signal severity."""
    signals = [
        {"tag": "equity_market_risk", "severity": 0.66},
        {"tag": "geopolitical_fx_risk", "severity": 0.33},
    ]

    assert risk_vector_from_signals(signals, ["macro_rate_risk"]) == [0.0, 0.66, 0.33]


def test_risk_vector_from_signals_falls_back_to_tags_when_empty() -> None:
    """Dashboard display vector should remain compatible with tag-only events."""
    assert risk_vector_from_signals([], ["macro_rate_risk"]) == [1.0, 0.0, 0.0]


def test_research_result_from_complete_event() -> None:
    """Dashboard should keep final report and sources separate from logs."""
    event = {
        "type": "complete",
        "question": "q",
        "report": "최종 리포트",
        "sources": ["https://example.com"],
        "reasoning_trace": "trace",
        "risk_tags": ["equity_market_risk", "macro_rate_risk"],
    }

    result = research_result_from_event(event)

    assert result == {
        "status": "ready",
        "question": "q",
        "report": "최종 리포트",
        "sources": ["https://example.com"],
        "reasoning_trace": "trace",
        "risk_tags": ["equity_market_risk", "macro_rate_risk"],
        "risk_signals": [],
    }


def test_research_log_formatter_filters_noisy_chat_chunks() -> None:
    """Dashboard log should hide raw chat token chunks and keep milestones."""
    analyst_token = {"type": "on_chat_model_stream", "node": "analyst", "text": "토큰"}
    assert format_research_log_event(analyst_token) == ""
    assert extract_analyst_draft_delta(analyst_token) == "토큰"
    assert (
        extract_analyst_draft_delta(
            {"type": "on_chat_model_stream", "node": "planner", "text": "무시"}
        )
        == ""
    )
    assert (
        format_research_log_event({"type": "on_chain_start", "name": "planner"})
        == "질문 분석 시작\n"
    )
    assert (
        format_research_log_event(
            {
                "type": "on_chain_end",
                "name": "analyst",
                "elapsed_ms": 8800,
                "duration_ms": 6200,
            }
        )
        == "리포트 작성 완료 (소요 6.20s, 누적 8.80s)\n"
    )


def test_build_optimize_payload_includes_session_risk_tags() -> None:
    """Dashboard /optimize payload should carry session risk tags."""
    payload = build_optimize_payload(1.5, ["equity_market_risk", "geopolitical_fx_risk"])

    assert payload == {
        "risk_aversion": 1.5,
        "risk_tags": ["equity_market_risk", "geopolitical_fx_risk"],
        "risk_signals": [],
    }


def test_calculate_period_return_metrics_rebases_cumulative_series() -> None:
    """Dashboard period metrics should compare returns within the selected slice."""
    cumulative_return, excess_return = calculate_period_return_metrics(
        [2.0, 3.0],
        [5.0, 6.0],
    )

    assert cumulative_return == 0.5
    assert round(excess_return, 10) == 0.3


def test_anova_summary_rows_flattens_computed_values_without_fixed_outcomes() -> None:
    """Dashboard should render ANOVA rows from the computed API payload."""
    rows = anova_summary_rows(
        [
            {
                "name": "reward_function_comparison",
                "f_statistic": 1.23,
                "p_value": 0.123,
                "eta_squared": 0.11,
                "post_hoc": [],
            },
            {
                "name": "strategy_comparison",
                "f_statistic": 4.56,
                "p_value": 0.004,
                "eta_squared": 0.22,
                "post_hoc": [],
            },
            {
                "name": "market_regime_comparison",
                "f_statistic": 7.89,
                "p_value": 0.001,
                "eta_squared": 0.33,
                "post_hoc": [],
                "strategy_effect": {"f_statistic": 8.76, "p_value": 0.0001},
                "interaction": {"f_statistic": 0.12, "p_value": 0.456, "significant": False},
            },
        ]
    )

    assert rows == [
        {
            "검증": "검증 1",
            "효과": "보상 함수 비교",
            "방법": "One-way",
            "F 통계량": 1.23,
            "p-value": 0.123,
            "η²": 0.11,
            "판정": "유의하지 않음",
            "해석": "효과 크기와 비유의 원인 해석 필요",
        },
        {
            "검증": "검증 2",
            "효과": "전략 비교",
            "방법": "One-way",
            "F 통계량": 4.56,
            "p-value": 0.004,
            "η²": 0.22,
            "판정": "유의함",
            "해석": "전략 간 성과 차이 확인",
        },
        {
            "검증": "검증 3",
            "효과": "국면 주효과",
            "방법": "Two-way",
            "F 통계량": 7.89,
            "p-value": 0.001,
            "η²": 0.33,
            "판정": "유의함",
            "해석": "시장 국면별 성과 차이 확인",
        },
        {
            "검증": "검증 3",
            "효과": "전략 주효과",
            "방법": "Two-way",
            "F 통계량": 8.76,
            "p-value": 0.0001,
            "η²": None,
            "판정": "유의함",
            "해석": "전략 자체의 성과 차이 확인",
        },
        {
            "검증": "검증 3",
            "효과": "국면 × 전략 교호작용",
            "방법": "Two-way",
            "F 통계량": 0.12,
            "p-value": 0.456,
            "η²": None,
            "판정": "유의하지 않음",
            "해석": "전략 우위 일관성 확인",
        },
    ]


def test_anova_attainment_cards_reflect_rubric_from_computed_values() -> None:
    """Dashboard cards should describe ANOVA rubric status from computed values."""
    cards = anova_attainment_cards(
        [
            {"name": "reward_function_comparison", "p_value": 0.001, "eta_squared": 0.1, "post_hoc": []},
            {"name": "strategy_comparison", "p_value": 0.002, "eta_squared": 0.2, "post_hoc": []},
            {
                "name": "market_regime_comparison",
                "p_value": 0.003,
                "eta_squared": 0.3,
                "post_hoc": [],
                "strategy_effect": {"p_value": 0.004},
                "interaction": {"p_value": 0.9},
            },
        ]
    )

    assert cards == [
        {
            "label": "보고 완성도",
            "status": "달성",
            "detail": "p-value, η², 사후검정/비유의 해석 항목 보고",
        },
        {
            "label": "주요 효과 유의성",
            "status": "달성",
            "detail": "보상 함수, 전략, 국면, 전략 주효과 p < 0.05 기준",
        },
        {
            "label": "교호작용 해석",
            "status": "일관성 확인",
            "detail": "비유의이면 전략 우위가 국면별로 크게 뒤집히지 않음",
        },
    ]


def test_post_hoc_filter_uses_actual_group_names() -> None:
    """Post-hoc filters should come from Tukey rows, not fixed strategy names."""
    rows = [
        {"group1": "PPO-return", "group2": "PPO-sharpe"},
        {"group1": "PPO-mdd", "group2": "PPO-return"},
    ]

    assert post_hoc_group_options(rows) == ["PPO-mdd", "PPO-return", "PPO-sharpe"]
    assert filter_post_hoc_rows(rows, []) == rows
    assert filter_post_hoc_rows(rows, ["PPO-sharpe"]) == [rows[0]]


def test_explain_reasoning_rows_includes_global_and_feature_context() -> None:
    """Dashboard should render both response-level and per-feature SHAP reasoning."""
    rows = explain_reasoning_rows(
        {
            "reasoning_context": [
                {
                    "event_date": "2024-12-30",
                    "tag": "equity_market_risk",
                    "severity": 0.66,
                    "decayed_score": 0.5,
                    "reasoning": "시장 급락",
                    "source": "gdelt",
                }
            ],
            "feature_contributions": [
                {
                    "feature": "risk_equity_market_risk",
                    "reasoning_context": [
                        {
                            "event_date": "2024-12-30",
                            "tag": "equity_market_risk",
                            "severity": 0.66,
                            "decayed_score": 0.5,
                            "reasoning": "시장 급락",
                            "source": "gdelt",
                        }
                    ],
                }
            ],
        }
    )

    assert rows == [
        {
            "구분": "전체",
            "피처": "",
            "날짜": "2024-12-30",
            "태그": "equity_market_risk",
            "강도": 0.66,
            "감쇠점수": 0.5,
            "근거": "시장 급락",
            "출처": "gdelt",
        },
        {
            "구분": "피처",
            "피처": "risk_equity_market_risk",
            "날짜": "2024-12-30",
            "태그": "equity_market_risk",
            "강도": 0.66,
            "감쇠점수": 0.5,
            "근거": "시장 급락",
            "출처": "gdelt",
        },
    ]


def test_extract_risk_signals_from_research_event_filters_schema() -> None:
    """Dashboard should keep only valid risk signal rows."""
    event = {
        "type": "complete",
        "risk_signals": [
            {"tag": "equity_market_risk", "severity": 0.66},
            {"tag": "unknown", "severity": 1.0},
            {"tag": "macro_rate_risk", "severity": "bad"},
            {"tag": "macro_rate_risk", "severity": 1.0001},
            {"tag": "geopolitical_fx_risk", "severity": -0.1},
        ],
    }

    assert extract_risk_signals_from_research_event(event) == [
        {"tag": "equity_market_risk", "severity": 0.66}
    ]
