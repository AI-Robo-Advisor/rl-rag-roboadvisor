"""HTTP helpers for the Streamlit dashboard."""

from __future__ import annotations

import json
from typing import Any, Callable

import requests

try:
    from src.agent.risk_tags import RL_RISK_TAGS
except ImportError:
    RL_RISK_TAGS = ["macro_rate_risk", "equity_market_risk", "geopolitical_fx_risk"]
_RESEARCH_NODE_LABELS = {
    "planner": "질문 분석",
    "researcher": "문서 검색",
    "grade_documents": "근거 평가",
    "analyst": "리포트 작성",
}


def _build_url(base_url: str, endpoint: str) -> str:
    """Join a dashboard base URL and endpoint without double slashes."""
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def calculate_period_return_metrics(
    portfolio_cumulative: list[float],
    benchmark_cumulative: list[float],
) -> tuple[float, float]:
    """Return selected-period cumulative and excess returns from cumulative wealth indexes."""
    if not portfolio_cumulative or not benchmark_cumulative:
        return 0.0, 0.0

    portfolio_start = float(portfolio_cumulative[0])
    benchmark_start = float(benchmark_cumulative[0])
    if portfolio_start <= 0 or benchmark_start <= 0:
        return 0.0, 0.0

    portfolio_return = float(portfolio_cumulative[-1]) / portfolio_start - 1.0
    benchmark_return = float(benchmark_cumulative[-1]) / benchmark_start - 1.0
    return portfolio_return, portfolio_return - benchmark_return


def explain_reasoning_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten SHAP response reasoning context for dashboard display."""
    rows: list[dict[str, Any]] = []

    def append_rows(scope: str, feature: str, events: Any) -> None:
        if not isinstance(events, list):
            return
        for event in events:
            if not isinstance(event, dict):
                continue
            rows.append(
                {
                    "구분": scope,
                    "피처": feature,
                    "날짜": str(event.get("event_date", "")),
                    "태그": str(event.get("tag", "")),
                    "강도": float(event.get("severity") or 0.0),
                    "감쇠점수": float(event.get("decayed_score") or 0.0),
                    "근거": str(event.get("reasoning", "")),
                    "출처": str(event.get("source", "")),
                }
            )

    append_rows("전체", "", payload.get("reasoning_context"))
    for contribution in payload.get("feature_contributions", []):
        if not isinstance(contribution, dict):
            continue
        append_rows(
            "피처",
            str(contribution.get("feature", "")),
            contribution.get("reasoning_context"),
        )
    return rows


def _mock_warning_message(endpoint: str, exc: Exception) -> str:
    """Return a user-facing warning when the dashboard falls back to mock data."""
    return f"API 연결 실패 ({endpoint}): {exc} — 이는 mock 응답입니다."


def extract_risk_tags_from_research_event(event: dict[str, Any]) -> list[str]:
    """Return RL risk tags from a research stream complete/fallback event."""
    return extract_risk_context_from_research_event(event)[0]


def extract_risk_signals_from_research_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Return validated risk_signals from research stream events."""
    return extract_risk_context_from_research_event(event)[1]


def extract_risk_context_from_research_event(
    event: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return validated risk_tags and risk_signals from a final research event."""
    event_type = event.get("type")
    if event_type == "complete":
        raw_tags = event.get("risk_tags", [])
        raw_signals = event.get("risk_signals", [])
    elif event_type == "fallback":
        data = event.get("data") or {}
        raw_tags = data.get("risk_tags", []) if isinstance(data, dict) else []
        raw_signals = data.get("risk_signals", []) if isinstance(data, dict) else []
    else:
        raw_tags = []
        raw_signals = []

    tags = [str(tag) for tag in raw_tags if str(tag) in RL_RISK_TAGS]
    normalized: list[dict[str, Any]] = []
    for item in raw_signals:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag") or "")
        if tag not in RL_RISK_TAGS:
            continue
        try:
            severity = float(item.get("severity", 0.0))
        except (TypeError, ValueError):
            continue
        if not 0.0 <= severity <= 1.0:
            continue
        normalized.append({"tag": tag, "severity": severity})
    return tags, normalized


def research_result_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a research stream final event into dashboard session data."""
    event_type = event.get("type")
    if event_type == "complete":
        tags, signals = extract_risk_context_from_research_event(event)
        return {
            "status": "ready",
            "question": str(event.get("question", "")),
            "report": str(event.get("report", "")),
            "sources": [str(source) for source in event.get("sources", []) if source],
            "reasoning_trace": str(event.get("reasoning_trace", "")),
            "risk_tags": tags,
            "risk_signals": signals,
        }
    if event_type == "fallback":
        data = event.get("data") or {}
        if not isinstance(data, dict):
            return None
        tags, signals = extract_risk_context_from_research_event(event)
        return {
            "status": str(data.get("status", "fallback")),
            "question": str(data.get("question", "")),
            "report": str(data.get("report", "")),
            "sources": [str(source) for source in data.get("sources", []) if source],
            "reasoning_trace": str(data.get("reasoning_trace", "")),
            "risk_tags": tags,
            "risk_signals": signals,
        }
    return None


def format_research_log_event(event: dict[str, Any]) -> str:
    """Format only useful research milestones for the dashboard log."""
    event_type = event.get("type", "event")
    name = str(event.get("name") or "research")

    if event_type == "start":
        return f"시작: {event.get('question', '')}\n"
    if event_type == "complete":
        tags = ", ".join(extract_risk_tags_from_research_event(event)) or "없음"
        return f"완료: 리스크 태그 {tags}\n"
    if event_type == "fallback":
        tags = ", ".join(extract_risk_tags_from_research_event(event)) or "없음"
        return f"fallback: 리스크 태그 {tags}\n"
    if event_type == "on_chat_model_stream":
        return ""
    if event_type not in {"on_chain_start", "on_chain_end", "on_tool_start", "on_tool_end"}:
        return ""

    label = _RESEARCH_NODE_LABELS.get(name, name)
    action = "시작" if event_type.endswith("_start") else "완료"
    text = str(event.get("text") or "").strip()
    if len(text) > 180:
        text = f"{text[:180]}..."
    suffix = f" - {text}" if text else ""
    timing = _format_research_event_timing(event)
    return f"{label} {action}{suffix}{timing}\n"


def _format_research_event_timing(event: dict[str, Any]) -> str:
    """Return compact latency text for streamed research milestones."""
    parts: list[str] = []
    duration_ms = event.get("duration_ms")
    elapsed_ms = event.get("elapsed_ms")
    if isinstance(duration_ms, int | float):
        parts.append(f"소요 {duration_ms / 1000:.2f}s")
    if isinstance(elapsed_ms, int | float):
        parts.append(f"누적 {elapsed_ms / 1000:.2f}s")
    return f" ({', '.join(parts)})" if parts else ""


def extract_analyst_draft_delta(event: dict[str, Any]) -> str:
    """Return analyst token-stream text for the live draft panel."""
    if event.get("type") != "on_chat_model_stream":
        return ""
    if str(event.get("node") or event.get("name") or "") != "analyst":
        return ""
    return str(event.get("text") or "")


def risk_vector_from_tags(risk_tags: list[str] | None) -> list[float]:
    """Build the dashboard display vector matching the RL observation order."""
    selected = set(risk_tags or [])
    return [1.0 if tag in selected else 0.0 for tag in RL_RISK_TAGS]


def risk_vector_from_signals(
    risk_signals: list[dict[str, Any]] | None,
    fallback_tags: list[str] | None = None,
) -> list[float]:
    """Build the dashboard display vector from severity-bearing risk signals."""
    severity_by_tag: dict[str, float] = {}
    for item in risk_signals or []:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag") or "")
        if tag not in RL_RISK_TAGS:
            continue
        try:
            severity = float(item.get("severity", 0.0))
        except (TypeError, ValueError):
            continue
        if 0.0 <= severity <= 1.0:
            severity_by_tag[tag] = severity

    if not severity_by_tag:
        return risk_vector_from_tags(fallback_tags)

    return [severity_by_tag.get(tag, 0.0) for tag in RL_RISK_TAGS]


def build_optimize_payload(
    risk_aversion: float,
    risk_tags: list[str] | None = None,
    risk_signals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the /optimize request body sent by the dashboard."""
    return {
        "risk_aversion": risk_aversion,
        "risk_tags": list(risk_tags or []),
        "risk_signals": list(risk_signals or []),
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _anova_significance_label(p_value: float | None) -> str:
    if p_value is None:
        return "계산값 없음"
    return "유의함" if p_value < 0.05 else "유의하지 않음"


def _is_significant(p_value: Any) -> bool:
    numeric = _float_or_none(p_value)
    return numeric is not None and numeric < 0.05


def _anova_item_by_name(anova_list: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next((item for item in anova_list if item.get("name") == name), {})


def _anova_row_interpretation(effect: str, p_value: float | None) -> str:
    significant = p_value is not None and p_value < 0.05
    if effect == "보상 함수 비교":
        return "보상 함수별 성과 차이 확인" if significant else "효과 크기와 비유의 원인 해석 필요"
    if effect == "전략 비교":
        return "전략 간 성과 차이 확인" if significant else "효과 크기와 비유의 원인 해석 필요"
    if effect == "국면 주효과":
        return "시장 국면별 성과 차이 확인" if significant else "효과 크기와 비유의 원인 해석 필요"
    if effect == "전략 주효과":
        return "전략 자체의 성과 차이 확인" if significant else "효과 크기와 비유의 원인 해석 필요"
    if effect == "국면 × 전략 교호작용":
        return "국면별 전략 우위 변화 확인" if significant else "전략 우위 일관성 확인"
    return "계산 결과 확인"


def anova_attainment_cards(anova_list: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Summarize whether the ANOVA reporting rubric is satisfied."""
    rows = [item for item in anova_list or [] if isinstance(item, dict)]
    reward = _anova_item_by_name(rows, "reward_function_comparison")
    strategy = _anova_item_by_name(rows, "strategy_comparison")
    regime = _anova_item_by_name(rows, "market_regime_comparison")
    strategy_effect = regime.get("strategy_effect") or {}
    interaction = regime.get("interaction") or {}

    required = [reward, strategy, regime]
    reporting_complete = all(
        item
        and _float_or_none(item.get("p_value")) is not None
        and _float_or_none(item.get("eta_squared")) is not None
        and isinstance(item.get("post_hoc"), list)
        for item in required
    )
    main_effects_significant = all(
        [
            _is_significant(reward.get("p_value")),
            _is_significant(strategy.get("p_value")),
            _is_significant(regime.get("p_value")),
            isinstance(strategy_effect, dict) and _is_significant(strategy_effect.get("p_value")),
        ]
    )
    interaction_p = _float_or_none(interaction.get("p_value") if isinstance(interaction, dict) else None)
    interaction_consistent = interaction_p is not None and interaction_p >= 0.05

    return [
        {
            "label": "보고 완성도",
            "status": "달성" if reporting_complete else "점검 필요",
            "detail": "p-value, η², 사후검정/비유의 해석 항목 보고",
        },
        {
            "label": "주요 효과 유의성",
            "status": "달성" if main_effects_significant else "해석 필요",
            "detail": "보상 함수, 전략, 국면, 전략 주효과 p < 0.05 기준",
        },
        {
            "label": "교호작용 해석",
            "status": "일관성 확인" if interaction_consistent else "국면별 차이 확인",
            "detail": "비유의이면 전략 우위가 국면별로 크게 뒤집히지 않음",
        },
    ]


def anova_summary_rows(anova_list: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Flatten computed ANOVA results into dashboard summary rows."""
    labels = {
        "reward_function_comparison": ("검증 1", "보상 함수 비교", "One-way"),
        "strategy_comparison": ("검증 2", "전략 비교", "One-way"),
        "market_regime_comparison": ("검증 3", "국면 주효과", "Two-way"),
    }
    rows: list[dict[str, Any]] = []
    for item in anova_list or []:
        if not isinstance(item, dict):
            continue
        check, effect, method = labels.get(
            str(item.get("name") or ""),
            (str(item.get("name") or "ANOVA"), "주효과", "ANOVA"),
        )
        f_statistic = _float_or_none(item.get("f_statistic"))
        p_value = _float_or_none(item.get("p_value"))
        rows.append(
            {
                "검증": check,
                "효과": effect,
                "방법": method,
                "F 통계량": f_statistic,
                "p-value": p_value,
                "η²": _float_or_none(item.get("eta_squared")),
                "판정": _anova_significance_label(p_value),
                "해석": _anova_row_interpretation(effect, p_value),
            }
        )

        if str(item.get("name") or "") != "market_regime_comparison":
            continue
        strategy_effect = item.get("strategy_effect") or {}
        if isinstance(strategy_effect, dict):
            strategy_p = _float_or_none(strategy_effect.get("p_value"))
            rows.append(
                {
                    "검증": check,
                    "효과": "전략 주효과",
                    "방법": method,
                    "F 통계량": _float_or_none(strategy_effect.get("f_statistic")),
                    "p-value": strategy_p,
                    "η²": None,
                    "판정": _anova_significance_label(strategy_p),
                    "해석": _anova_row_interpretation("전략 주효과", strategy_p),
                }
            )
        interaction = item.get("interaction") or {}
        if isinstance(interaction, dict):
            interaction_p = _float_or_none(interaction.get("p_value"))
            rows.append(
                {
                    "검증": check,
                    "효과": "국면 × 전략 교호작용",
                    "방법": method,
                    "F 통계량": _float_or_none(interaction.get("f_statistic")),
                    "p-value": interaction_p,
                    "η²": None,
                    "판정": _anova_significance_label(interaction_p),
                    "해석": _anova_row_interpretation("국면 × 전략 교호작용", interaction_p),
                }
            )
    return rows


def post_hoc_group_options(post_hoc: list[dict[str, Any]] | None) -> list[str]:
    """Return group filter options from computed Tukey HSD rows."""
    groups: set[str] = set()
    for row in post_hoc or []:
        if not isinstance(row, dict):
            continue
        for key in ("group1", "group2"):
            value = str(row.get(key) or "")
            if value:
                groups.add(value)
    return sorted(groups)


def filter_post_hoc_rows(
    post_hoc: list[dict[str, Any]] | None,
    selected_groups: list[str] | None,
) -> list[dict[str, Any]]:
    """Filter Tukey HSD rows by selected groups, preserving all rows when empty."""
    rows = [row for row in post_hoc or [] if isinstance(row, dict)]
    selected = set(selected_groups or [])
    if not selected:
        return rows
    return [
        row
        for row in rows
        if str(row.get("group1") or "") in selected or str(row.get("group2") or "") in selected
    ]


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
        resp = requests.get(_build_url(base_url, endpoint), params=params, timeout=timeout)
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
        resp = requests.post(_build_url(base_url, endpoint), json=payload, timeout=timeout)
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
            _build_url(base_url, endpoint),
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
