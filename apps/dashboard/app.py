"""
Streamlit 대시보드.

streamlit-echarts 기반 인터랙티브 차트 + st.navigation 사이드바 네비게이션.
FastAPI HTTP 통신만 사용하며, 모델을 직접 로드하지 않습니다.
API 미완성 상태에서는 mock 데이터로 UI를 렌더링합니다.
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import JsCode, st_echarts

try:
    from apps.dashboard.api_client import (
        RL_RISK_TAGS,
        build_optimize_payload,
        calculate_period_return_metrics,
        explain_reasoning_rows,
        extract_analyst_draft_delta,
        extract_risk_context_from_research_event,
        format_research_log_event,
        get_json,
        post_json,
        research_result_from_event,
        risk_vector_from_tags,
        stream_ndjson,
    )
except ModuleNotFoundError:
    # Streamlit file-entry execution inside Docker may not resolve the package root.
    from api_client import (
        RL_RISK_TAGS,
        build_optimize_payload,
        calculate_period_return_metrics,
        explain_reasoning_rows,
        extract_analyst_draft_delta,
        extract_risk_context_from_research_event,
        format_research_log_event,
        get_json,
        post_json,
        research_result_from_event,
        risk_vector_from_tags,
        stream_ndjson,
    )

try:
    from config import TICKERS_GLOBAL, TICKERS_KR
except ModuleNotFoundError:
    TICKERS_GLOBAL = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ"]
    TICKERS_KR = ["069500", "114260"]

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
_TIMEOUT_DEFAULT: int = 10
_TIMEOUT_RESEARCH: int = 60  # LangGraph 루프 최대 3회 대응

_PALETTE = [
    "#5470c6",
    "#91cc75",
    "#fac858",
    "#ee6666",
    "#73c0de",
    "#3ba272",
    "#fc8452",
    "#9a60b4",
    "#ea7ccc",
    "#48b8d0",
]

_PERIOD_MONTHS = {"1개월": 21, "3개월": 63, "6개월": 126, "12개월": 252, "전체": None}


# ─────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────


def _get(endpoint: str, params: dict | None = None) -> dict[str, Any] | None:
    return get_json(
        API_BASE_URL,
        endpoint,
        params=params,
        timeout=_TIMEOUT_DEFAULT,
        warn=st.warning,
    )


def _post(endpoint: str, payload: dict, timeout: int = _TIMEOUT_DEFAULT) -> dict[str, Any] | None:
    return post_json(
        API_BASE_URL,
        endpoint,
        payload,
        timeout=timeout,
        warn=st.warning,
    )


def _format_research_event(event: dict[str, Any]) -> str:
    """Format one /research/stream NDJSON event for st.write_stream."""
    return format_research_log_event(event)


def _remember_research_risk_tags(event: dict[str, Any]) -> None:
    """Store stream risk tags in Streamlit session state for /optimize."""
    tags, signals = extract_risk_context_from_research_event(event)
    if event.get("type") in {"complete", "fallback"}:
        st.session_state["risk_tags"] = tags
        st.session_state["risk_signals"] = signals
        st.session_state["risk_tags_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = research_result_from_event(event)
        if result:
            st.session_state["research_result"] = result


def _stream_research(question: str, on_draft: Callable[[str], None] | None = None):
    def formatter(event: dict[str, Any]) -> str:
        _remember_research_risk_tags(event)
        draft_delta = extract_analyst_draft_delta(event)
        if draft_delta and on_draft:
            on_draft(draft_delta)
        return _format_research_event(event)

    return stream_ndjson(
        API_BASE_URL,
        "/research/stream",
        {"question": question},
        formatter=formatter,
        timeout=_TIMEOUT_RESEARCH,
        warn=st.warning,
    )


# ─────────────────────────────────────────────
# ECharts 공통 빌더
# ─────────────────────────────────────────────


def _echarts_line(
    x: list,
    series: list[dict],
    title: str = "",
    height: str = "380px",
    y_formatter: str = "",
    zoom: bool = True,
    key: str = "chart",
    legend: bool = True,
) -> None:
    _legend = (
        {
            "top": 28,
            "type": "scroll",
            "itemWidth": 28,
            "itemHeight": 14,
            "textStyle": {"fontSize": 12},
            "icon": "roundRect",
        }
        if legend
        else {"show": False}
    )
    _grid_top = "22%" if legend else "12%"
    opts: dict = {
        "title": {"text": title, "left": "center", "top": 4, "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "axis"},
        "legend": _legend,
        "grid": {"bottom": "18%" if zoom else "8%", "top": _grid_top, "containLabel": True},
        "xAxis": {"type": "category", "data": x, "boundaryGap": False},
        "yAxis": {"type": "value", "axisLabel": {"formatter": y_formatter} if y_formatter else {}},
        "series": series,
    }
    if zoom:
        opts["dataZoom"] = [
            {"type": "inside", "start": 0, "end": 100},
            {"type": "slider", "start": 0, "end": 100, "height": 36, "bottom": 16},
        ]
    st_echarts(options=opts, height=height, theme="streamlit", key=key)


def _echarts_bar_h(
    names: list,
    values: list,
    title: str = "",
    colors: list | None = None,
    height: str = "360px",
    key: str = "bar_h",
) -> None:
    bar_data = (
        [{"value": v, "itemStyle": {"color": c}} for v, c in zip(values, colors)]
        if colors
        else values
    )
    opts = {
        "title": {"text": title, "left": "center", "top": 4, "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "3%", "right": "8%", "bottom": "8%", "containLabel": True},
        "xAxis": {"type": "value"},
        "yAxis": {"type": "category", "data": names},
        "series": [{"type": "bar", "data": bar_data}],
    }
    st_echarts(options=opts, height=height, theme="streamlit", key=key)


def _echarts_donut(
    labels: list,
    values: list,
    title: str = "",
    height: str = "380px",
    key: str = "donut",
) -> None:
    data = [{"name": l, "value": round(v, 4)} for l, v in zip(labels, values)]
    opts = {
        "title": {"text": title, "left": "center", "top": 4, "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "item", "formatter": "{b}: {d}%"},
        "legend": {
            "bottom": 6,
            "type": "scroll",
            "itemWidth": 12,
            "itemHeight": 12,
            "textStyle": {"fontSize": 11},
            "icon": "circle",
        },
        "series": [
            {
                "type": "pie",
                "radius": ["38%", "65%"],
                "avoidLabelOverlap": True,
                "itemStyle": {"borderRadius": 8, "borderColor": "#fff", "borderWidth": 2},
                "label": {"show": True, "formatter": "{b}\n{d}%", "fontSize": 11},
                "emphasis": {"label": {"show": True, "fontSize": 13, "fontWeight": "bold"}},
                "data": data,
            }
        ],
    }
    st_echarts(options=opts, height=height, theme="streamlit", key=key)


def _echarts_gauge(
    value: float,
    title: str,
    max_val: float = 0.1,
    height: str = "260px",
    key: str = "gauge",
) -> None:
    pct = round(value * 100, 2)
    color = "#ee6666" if pct > 3 else "#fac858" if pct > 1.5 else "#91cc75"
    opts = {
        "series": [
            {
                "type": "gauge",
                "min": 0,
                "max": round(max_val * 100, 1),
                "progress": {"show": True, "width": 14},
                "axisLine": {"lineStyle": {"width": 14}},
                "axisTick": {"show": False},
                "splitLine": {"length": 10, "lineStyle": {"width": 2, "color": "#999"}},
                "axisLabel": {
                    "distance": 20,
                    "fontSize": 11,
                    "formatter": JsCode("function(v){return v+'%'}"),
                },
                "detail": {
                    "valueAnimation": True,
                    "fontSize": 28,
                    "offsetCenter": [0, "60%"],
                    "formatter": JsCode("function(v){return v.toFixed(2)+'%'}"),
                },
                "title": {"offsetCenter": [0, "88%"], "fontSize": 12},
                "data": [{"value": pct, "name": title}],
                "itemStyle": {"color": color},
            }
        ],
    }
    st_echarts(options=opts, height=height, theme="streamlit", key=key)


# ─────────────────────────────────────────────
# Mock 데이터
# ─────────────────────────────────────────────

_ASSETS = TICKERS_GLOBAL + TICKERS_KR
_rng = np.random.default_rng(42)


def _mock_optimize(risk_aversion: float = 1.0) -> dict:
    seed = 4242 + int(round(float(risk_aversion) * 1000))
    rng = np.random.default_rng(seed)
    concentration = np.full(len(_ASSETS), 1.0 / max(float(risk_aversion), 0.1))
    weights = rng.dirichlet(concentration)
    portfolio_daily = rng.normal(0.0005, 0.012, 252)
    benchmark_daily = rng.normal(0.0003, 0.010, 252)
    equal_weight_daily = rng.normal(0.00035, 0.011, 252)
    portfolio_cum = np.cumprod(1 + portfolio_daily)
    benchmark_cum = np.cumprod(1 + benchmark_daily)
    equal_weight_cum = np.cumprod(1 + equal_weight_daily)
    dates = pd.date_range("2024-01-01", periods=252, freq="B").strftime("%Y-%m-%d").tolist()
    portfolio_return = float(portfolio_cum[-1] / portfolio_cum[0] - 1.0)
    return {
        "status": "mock",
        "message": "API 연결 실패로 mock 포트폴리오를 표시합니다.",
        "elapsed_ms": 0.0,
        "timed_out": False,
        "tickers": list(_ASSETS),
        "weights": dict(zip(_ASSETS, weights.tolist())),
        "risk_profile": "balanced",
        "expected_return": round(portfolio_return, 6),
        "expected_volatility": round(float(np.std(portfolio_daily, ddof=1) * np.sqrt(252)), 6),
        "returns": {
            "date": dates,
            "portfolio": portfolio_cum.tolist(),
            "benchmark": benchmark_cum.tolist(),
            "equal_weight": equal_weight_cum.tolist(),
        },
    }


def _ensure_portfolio_data(
    risk_aversion: float,
    current_risk_tags: list[str],
    current_risk_signals: list[dict[str, Any]],
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    """Return the latest portfolio result, refreshing only when requested."""
    payload = build_optimize_payload(risk_aversion, current_risk_tags, current_risk_signals)
    cached_data = st.session_state.get("portfolio_data")

    if cached_data and not refresh:
        return cached_data
    with st.spinner("POST /optimize 호출 중…"):
        data = _post("/optimize", payload)

    if not data:
        data = _mock_optimize(risk_aversion)
    else:
        data = dict(data)
        data.setdefault("status", "ready")
        data.setdefault("message", "포트폴리오 최적화 결과입니다.")

    st.session_state["portfolio_data"] = data
    st.session_state["portfolio_payload"] = payload
    st.session_state["portfolio_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return data


def _mock_backtest() -> dict:
    dates = pd.date_range("2024-01-01", periods=252, freq="B").strftime("%Y-%m-%d").tolist()
    wf = _rng.normal(0.001, 0.015, 252)
    bm = _rng.normal(0.0003, 0.010, 252)
    ew = _rng.normal(0.0006, 0.011, 252)
    prices = np.cumprod(1 + wf)
    drawdown = ((prices - np.maximum.accumulate(prices)) / np.maximum.accumulate(prices)).tolist()
    metrics = {
        "cumulative_return": 0.248,
        "cagr": 0.231,
        "annualized_volatility": 0.182,
        "var_95": 0.021,
        "cvar_95": 0.031,
        "mdd": 0.127,
        "sharpe_ratio": 1.27,
        "sortino_ratio": 1.85,
        "calmar_ratio": 1.82,
        "alpha": 0.043,
        "beta": 0.92,
        "information_ratio": 0.68,
    }
    return {
        "dates": dates,
        "rewards": _rng.normal(0.002, 0.05, 200).cumsum().tolist(),
        "wf_cum": np.cumprod(1 + wf).tolist(),
        "bm_cum": np.cumprod(1 + bm).tolist(),
        "ew_cum": np.cumprod(1 + ew).tolist(),
        "wf_spark": np.cumprod(1 + wf[:50]).tolist(),
        "sharpe_spark": (np.cumsum(_rng.normal(0, 0.3, 50)) + 1.27).tolist(),
        "drawdown": drawdown,
        "metrics": metrics,
        "anova": [
            {
                "name": "reward_function_comparison",
                "f_statistic": 8.71,
                "p_value": 0.0002,
                "eta_squared": 0.142,
                "post_hoc": [
                    {
                        "group1": "PPO-return",
                        "group2": "PPO-sharpe",
                        "meandiff": 0.0003,
                        "p_adj": 0.031,
                        "reject": True,
                    },
                    {
                        "group1": "PPO-return",
                        "group2": "PPO-mdd",
                        "meandiff": 0.0005,
                        "p_adj": 0.004,
                        "reject": True,
                    },
                    {
                        "group1": "PPO-sharpe",
                        "group2": "PPO-mdd",
                        "meandiff": 0.0002,
                        "p_adj": 0.218,
                        "reject": False,
                    },
                ],
            },
            {
                "name": "strategy_comparison",
                "f_statistic": 12.34,
                "p_value": 0.0003,
                "eta_squared": 0.187,
                "post_hoc": [
                    {
                        "group1": "PPO",
                        "group2": "MVO",
                        "meandiff": 0.0008,
                        "p_adj": 0.002,
                        "reject": True,
                    },
                    {
                        "group1": "PPO",
                        "group2": "동일비중",
                        "meandiff": 0.0006,
                        "p_adj": 0.041,
                        "reject": True,
                    },
                    {
                        "group1": "MVO",
                        "group2": "동일비중",
                        "meandiff": 0.0002,
                        "p_adj": 0.312,
                        "reject": False,
                    },
                ],
            },
            {
                "name": "market_regime_comparison",
                "f_statistic": 2.07,
                "p_value": 0.127,
                "eta_squared": 0.038,
                "post_hoc": [],
                "interaction": {"f_statistic": 3.14, "p_value": 0.014, "significant": True},
                "strategy_effect": {"f_statistic": 4.52, "p_value": 0.011},
            },
        ],
        "var_95": metrics["var_95"],
        "cvar_95": metrics["cvar_95"],
        "mdd": metrics["mdd"],
        "safeguard": {"active": False, "triggered_at": None, "current_drawdown": 0.043},
    }


def _mock_explain(target_date: str) -> dict:
    feat = [f"{t}_{f}" for t in _ASSETS for f in ("return", "RSI", "MACD", "MACD_signal")]
    vals = _rng.normal(0, 0.05, len(feat)).tolist()
    base = 0.002
    return {
        "target_date": target_date,
        "feature_names": feat,
        "shap_values": vals,
        "base_value": base,
        "prediction": base + sum(vals),
    }


def _mock_research(question: str) -> dict:
    return {
        "report": (
            f"**[Mock 리포트]** '{question}'에 대한 분석입니다.\n\n"
            "현재 시장은 글로벌 금리 인상 기조와 반도체 업황 회복 사이의 긴장 속에 있습니다. "
            "국내 대형주는 외국인 수급 개선으로 단기 반등 가능성이 있으나, "
            "미 연준의 금리 경로 불확실성은 여전히 상방 리스크로 작용합니다.\n\n"
            "---\n**[면책 조항]** 본 분석은 교육 목적으로만 제공됩니다."
        ),
        "sources": ["https://news.example.com/article/1", "https://news.example.com/article/2"],
        "reasoning_trace": (
            "[THINK][planner] 질의 분석 시작\n"
            "[THINK][researcher] 초기 검색: Chroma hit=5건\n"
            "[THINK][grade_documents] 판정: 충분 — analyst 진행\n"
            "[THINK][analyst] 최종 리포트 생성 착수"
        ),
        "risk_tags": ["equity_market_risk"],
        "risk_signals": [{"tag": "equity_market_risk", "severity": 1.0}],
    }


def _render_research_result(result: dict[str, Any] | None) -> None:
    """Render the latest research output as report, sources, and tags (in that order)."""
    if not result:
        return

    tags = result.get("risk_tags", [])
    updated_at = st.session_state.get("risk_tags_updated_at")

    with st.container(border=True):
        st.markdown("**리서치 결과**")
        if result.get("question"):
            st.caption(f"질문: {result['question']}")
        st.markdown(result.get("report") or "리서치 결과가 비어 있습니다.")

    with st.container(border=True):
        st.markdown("**출처**")
        sources = result.get("sources") or []
        if sources:
            for source in sources:
                st.markdown(f"- {source}")
        else:
            st.caption("표시할 출처가 없습니다.")

    with st.container(border=True):
        st.markdown("**Optimize 반영 상태**")
        st.caption(
            f"최근 리서치 태그: {', '.join(tags) if tags else '없음'}"
            + (f" | 저장 시각: {updated_at}" if updated_at else "")
        )
        st.caption(f"RL 관측 벡터 {RL_RISK_TAGS}: {risk_vector_from_tags(tags)}")


# ─────────────────────────────────────────────
# 페이지 함수
# ─────────────────────────────────────────────


def portfolio_page() -> None:
    # 페이지 전용 사이드바 컨트롤
    with st.sidebar:
        st.divider()
        st.subheader(":material/tune: 포트폴리오 설정")
        risk_aversion = st.slider(
            "위험 회피 계수",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
            help="값이 클수록 분산 투자 비중 증가",
        )
        period: str = st.selectbox("분석 기간", list(_PERIOD_MONTHS.keys()), index=3)

    st.title("포트폴리오 현황")
    current_risk_tags = st.session_state.get("risk_tags", [])
    current_risk_signals = st.session_state.get("risk_signals", [])
    latest_research = st.session_state.get("research_result", {})
    latest_question = latest_research.get("question")
    latest_updated_at = st.session_state.get("risk_tags_updated_at")

    if current_risk_tags:
        caption = f"이번 최적화에 반영 예정인 리스크 태그: {', '.join(current_risk_tags)}"
    else:
        caption = "이번 최적화에 반영 예정인 리스크 태그: 없음"
    if latest_question:
        caption += f" | 최근 리서치: {latest_question}"
    if latest_updated_at:
        caption += f" | 저장 시각: {latest_updated_at}"
    st.caption(caption)

    refresh_requested = st.button("최적화 실행", key="btn_optimize")

    current_payload = build_optimize_payload(risk_aversion, current_risk_tags, current_risk_signals)
    cached_payload = st.session_state.get("portfolio_payload")
    if (
        cached_payload is not None
        and cached_payload != current_payload
        and not refresh_requested
    ):
        st.warning("설정이 변경되었습니다. '최적화 실행'을 다시 눌러 결과를 업데이트하세요.")

    data = _ensure_portfolio_data(
        risk_aversion,
        current_risk_tags,
        current_risk_signals,
        refresh=refresh_requested or "portfolio_data" not in st.session_state,
    )

    # 기간 슬라이싱
    n = _PERIOD_MONTHS[period]
    ret = data["returns"]
    x = ret["date"][-n:] if n else ret["date"]
    port_vals = ret["portfolio"][-n:] if n else ret["portfolio"]
    bm_vals = ret["benchmark"][-n:] if n else ret["benchmark"]
    ew_full = ret.get("equal_weight") or []
    ew_vals = ew_full[-n:] if (n and ew_full) else ew_full
    data_status = data.get("status", "unknown")
    data_message = data.get("message", "")
    elapsed_ms = float(data.get("elapsed_ms", 0.0) or 0.0)
    data_start = x[0] if x else "-"
    data_end = x[-1] if x else "-"
    if data_status == "mock":
        st.error(
            "API 연결 실패로 mock 응답을 표시합니다. 실제 PPO 결과가 아니며, "
            "도커/로컬 API 기동 여부를 확인하세요."
        )
    st.caption(
        f"상태: {data_status} | 메시지: {data_message} | 소요 시간: {elapsed_ms:.1f}ms | "
        f"표시 구간: {data_start} ~ {data_end}"
    )
    st.caption(
        ":material/info: 아래 누적 수익률은 **현재 PPO 가중치 1세트를 과거 구간에 정적으로 적용한 "
        "시뮬레이션**입니다 (시점별 리밸런싱이 없는 buy-and-hold). "
        "Walk-Forward 백테스트 결과는 **강화학습 성과** 탭을 참조하세요."
    )

    cum_ret, excess = calculate_period_return_metrics(port_vals, bm_vals)
    top_asset = max(data["weights"], key=data["weights"].get)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("누적 수익률", f"{cum_ret:.1%}", border=True)
    k2.metric("초과 수익", f"{excess:.1%}", delta=f"{excess:.2%} vs SPY", border=True)
    k3.metric("최대 비중 자산", top_asset, f"{data['weights'][top_asset]:.1%}", border=True)
    k4.metric("편입 종목 수", f"{len(data['weights'])}개", border=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            _echarts_donut(
                labels=list(data["weights"].keys()),
                values=list(data["weights"].values()),
                title="자산 비중",
                key="p_donut",
            )
    with col2:
        with st.container(border=True):
            p_series: list[dict[str, Any]] = [
                {
                    "name": "포트폴리오 (PPO)",
                    "type": "line",
                    "smooth": True,
                    "areaStyle": {"opacity": 0.15},
                    "data": port_vals,
                    "itemStyle": {"color": _PALETTE[0]},
                },
                {
                    "name": "벤치마크 (SPY)",
                    "type": "line",
                    "smooth": True,
                    "data": bm_vals,
                    "itemStyle": {"color": _PALETTE[3]},
                },
            ]
            if ew_vals:
                p_series.append(
                    {
                        "name": "동일가중",
                        "type": "line",
                        "smooth": True,
                        "lineStyle": {"type": "dashed"},
                        "data": ew_vals,
                        "itemStyle": {"color": _PALETTE[2]},
                    }
                )
            _echarts_line(
                x=x,
                series=p_series,
                title=f"정적 가중치 누적 수익률 ({period})",
                key="p_line",
            )

    with st.expander("비중 상세 테이블"):
        st.dataframe(
            pd.DataFrame(
                {
                    "자산": list(data["weights"].keys()),
                    "비중": [f"{v:.2%}" for v in data["weights"].values()],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )


def rl_page() -> None:
    with st.sidebar:
        st.divider()
        st.subheader(":material/tune: 분석 설정")
        period: str = st.selectbox(
            "분석 기간", list(_PERIOD_MONTHS.keys()), index=3, key="rl_period"
        )
        strategies: list[str] = st.multiselect(
            "비교 전략",
            ["PPO", "벤치마크 (SPY)", "동일비중", "MVO"],
            default=["PPO", "벤치마크 (SPY)", "동일비중"],
            help="강화학습 성과 탭에서 비교할 전략",
        )

    st.title("강화학습 성과")
    with st.spinner("GET /backtest 호출 중…"):
        bt = _get("/backtest") or _mock_backtest()

    # 기간 슬라이싱
    n = _PERIOD_MONTHS[period]
    dates = bt["dates"][-n:] if n else bt["dates"]
    wf_cum = bt["wf_cum"][-n:] if n else bt["wf_cum"]
    bm_cum = bt["bm_cum"][-n:] if n else bt["bm_cum"]
    ew_cum_full = bt.get("ew_cum") or []
    ew_cum = ew_cum_full[-n:] if n else ew_cum_full

    m = bt["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "누적 수익률",
        f"{m['cumulative_return']:.1%}",
        border=True,
        chart_data=bt.get("wf_spark"),
        chart_type="area",
    )
    c2.metric(
        "샤프 비율",
        f"{m['sharpe_ratio']:.2f}",
        border=True,
        chart_data=bt.get("sharpe_spark"),
        chart_type="line",
    )
    c3.metric("MDD", f"{m['mdd']:.1%}", border=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            _echarts_line(
                x=list(range(1, len(bt["rewards"]) + 1)),
                series=[
                    {
                        "name": "누적 보상",
                        "type": "line",
                        "smooth": True,
                        "areaStyle": {"opacity": 0.12},
                        "data": bt["rewards"],
                        "itemStyle": {"color": _PALETTE[1]},
                    }
                ],
                title="학습 곡선 (에피소드 누적 보상)",
                key="rl_reward",
                legend=False,
            )
    with col2:
        with st.container(border=True):
            # 비교 전략 필터 적용
            series_list = []
            if "PPO" in strategies:
                series_list.append(
                    {
                        "name": "PPO",
                        "type": "line",
                        "smooth": True,
                        "areaStyle": {"opacity": 0.12},
                        "data": wf_cum,
                        "itemStyle": {"color": _PALETTE[0]},
                    }
                )
            if "벤치마크 (SPY)" in strategies:
                series_list.append(
                    {
                        "name": "벤치마크 (SPY)",
                        "type": "line",
                        "smooth": True,
                        "data": bm_cum,
                        "itemStyle": {"color": _PALETTE[3]},
                    }
                )
            if "동일비중" in strategies and ew_cum:
                series_list.append(
                    {
                        "name": "동일비중",
                        "type": "line",
                        "smooth": True,
                        "lineStyle": {"type": "dashed"},
                        "data": ew_cum,
                        "itemStyle": {"color": _PALETTE[2]},
                    }
                )
            if "MVO" in strategies:
                series_list.append(
                    {
                        "name": "MVO (mock)",
                        "type": "line",
                        "smooth": True,
                        "lineStyle": {"type": "dotted"},
                        "data": (np.array(wf_cum) * 0.92).tolist(),
                        "itemStyle": {"color": _PALETTE[4] if len(_PALETTE) > 4 else _PALETTE[2]},
                    }
                )
            if not series_list:
                st.info("비교 전략을 하나 이상 선택하세요.")
            else:
                _echarts_line(
                    x=dates,
                    series=series_list,
                    title=f"Walk-Forward 백테스트 ({period})",
                    key="rl_wf",
                )

    with st.container(border=True):
        st.markdown("**성과 지표 전체**")
        st.dataframe(
            pd.DataFrame([{"지표": k, "값": f"{v:.4f}"} for k, v in m.items()]),
            hide_index=True,
            use_container_width=True,
        )


def shap_page() -> None:
    # 페이지 전용 사이드바 컨트롤
    with st.sidebar:
        st.divider()
        st.subheader(":material/calendar_month: SHAP 설정")
        target_date = st.date_input(
            "분석 날짜",
            value=date.today() - timedelta(days=1),
            min_value=date(2020, 1, 1),
            max_value=date.today(),
        )

    st.title("SHAP 해석")

    if st.button("SHAP 분석 실행", key="btn_explain"):
        with st.spinner("POST /explain 호출 중…"):
            sd = _post("/explain", {"date": str(target_date)}) or _mock_explain(str(target_date))
    else:
        sd = _mock_explain(str(target_date))

    feat, vals, base, pred = (
        sd["feature_names"],
        sd["shap_values"],
        sd["base_value"],
        sd["prediction"],
    )
    st.markdown(
        f"분석 날짜: **{target_date}** | 기준값: **`{base:.4f}`** → 예측값: **`{pred:.4f}`**"
    )

    shap_df = pd.DataFrame({"피처": feat, "SHAP값": vals}).sort_values("SHAP값")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            summary = shap_df.copy()
            summary["절대값"] = summary["SHAP값"].abs()
            summary = summary.sort_values("절대값")
            _echarts_bar_h(
                names=summary["피처"].tolist(),
                values=summary["절대값"].round(4).tolist(),
                title="Summary Plot (|SHAP| 절대값)",
                key="shap_summary",
            )
    with col2:
        with st.container(border=True):
            colors = ["#ee6666" if v > 0 else "#5470c6" for v in shap_df["SHAP값"]]
            _echarts_bar_h(
                names=shap_df["피처"].tolist(),
                values=shap_df["SHAP값"].round(4).tolist(),
                colors=colors,
                title="Force Plot (빨강=양, 파랑=음)",
                key="shap_force",
            )

    reasoning_rows = explain_reasoning_rows(sd)
    with st.container(border=True):
        st.markdown("**Reasoning Context**")
        if reasoning_rows:
            st.dataframe(pd.DataFrame(reasoning_rows), hide_index=True, use_container_width=True)
        else:
            st.caption("해당 분석 날짜와 top SHAP 피처에 연결된 reasoning context가 없습니다.")


def research_page() -> None:
    st.title("에이전트 리서치 (RAG)")

    question = st.text_area(
        "투자 질문 입력",
        placeholder="ex. SPY와 TLT 배분 리스크는?",
        height=80,
    )
    st.caption("데이터셋 자산 예시: SPY, QQQ, IWM, EFA, EEM, TLT, GLD, VNQ, 069500, 114260")

    # Enter → 리서치 실행, Shift+Enter → 줄바꿈
    st.components.v1.html(
        """
    <script>
    (function() {
        function attachHandler() {
            const textareas = window.parent.document.querySelectorAll('textarea');
            textareas.forEach(function(ta) {
                if (ta._researchBound) return;
                ta._researchBound = true;
                ta.addEventListener('keydown', function(e) {
                    if (e.isComposing) return;
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        ta.blur();
                        setTimeout(function() {
                            const buttons = window.parent.document.querySelectorAll('button');
                            for (const btn of buttons) {
                                if (btn.innerText.trim() === '리서치 실행') {
                                    btn.click();
                                    break;
                                }
                            }
                        }, 100);
                    }
                });
            });
        }
        attachHandler();
        new MutationObserver(attachHandler).observe(
            window.parent.document.body, { childList: true, subtree: true }
        );
    })();
    </script>
    """,
        height=0,
    )

    if st.button("리서치 실행", key="btn_research"):
        if not question.strip():
            st.error("질문을 입력하세요.")
        else:
            # 1. 결과 위치 예약 (스트림 종료 후 위에 채워짐)
            result_container = st.container()

            # 2. 추론 로그 expander — default 닫힘. 사용자가 토글 열면 실시간 진행 노출
            draft_placeholder = st.empty()
            draft_text = ""

            def update_draft(delta: str) -> None:
                nonlocal draft_text
                draft_text += delta
                draft_placeholder.markdown(f"**실시간 리포트 초안**\n\n{draft_text}")

            with st.expander("추론 로그 보기 (LangGraph reasoning trace)", expanded=False):
                stream_placeholder = st.empty()
                stream_placeholder.caption("LangGraph 진행 상황 수신 대기 중…")

            # 3. spinner 동안 스트림 chunk를 expander 내부 placeholder에 누적 갱신
            with st.spinner("LangGraph 리서치 진행 중…"):
                stream_chunks: list[str] = []
                for chunk in _stream_research(question, on_draft=update_draft):
                    stream_chunks.append(str(chunk))
                    running_text = "".join(stream_chunks).strip()
                    stream_placeholder.code(
                        running_text[-4000:] or "...",
                        language="text",
                    )
                stream_text = "".join(stream_chunks)

            st.session_state["research_stream_text"] = stream_text
            stream_placeholder.code(
                stream_text.strip() or "표시할 진행 로그가 없습니다.",
                language="text",
            )

            # 4. 결과는 위에 예약된 컨테이너에 그림
            if draft_text:
                draft_placeholder.empty()
            with result_container:
                _render_research_result(st.session_state.get("research_result"))
    else:
        st.info("위에서 질문을 입력하고 '리서치 실행' 버튼을 누르세요.")
        _render_research_result(st.session_state.get("research_result"))
        cached_stream = st.session_state.get("research_stream_text")
        if cached_stream:
            with st.expander("추론 로그 보기 (LangGraph reasoning trace)", expanded=False):
                st.code(cached_stream.strip(), language="text")


def anova_page() -> None:
    with st.sidebar:
        st.divider()
        st.subheader(":material/tune: 분석 설정")
        strategies: list[str] = st.multiselect(
            "비교 전략",
            ["PPO", "MVO", "동일비중"],
            default=["PPO"],
            help="사후 검정 결과 필터",
        )

    st.title("ANOVA 검증 결과")
    with st.spinner("GET /backtest 호출 중…"):
        bt5 = _get("/backtest") or _mock_backtest()

    anova_list: list = bt5.get("anova", _mock_backtest()["anova"])

    _EXP_LABELS = {
        "reward_function_comparison": "검증 1 — 보상함수 비교",
        "strategy_comparison": "검증 2 — 전략 비교",
        "market_regime_comparison": "검증 3 — 국면 × 전략 (Two-way)",
    }
    tab_labels = [_EXP_LABELS.get(a.get("name", ""), a.get("name", "")) for a in anova_list]
    tabs = st.tabs(tab_labels) if tab_labels else []

    for tab, anova in zip(tabs, anova_list):
        with tab:
            a1, a2, a3 = st.columns(3)
            a1.metric("F 통계량", f"{anova.get('f_statistic', 0):.2f}", border=True)
            a2.metric("p-value", f"{anova.get('p_value', 1):.4f}", border=True)
            a3.metric("η² (효과 크기)", f"{anova.get('eta_squared', 0):.3f}", border=True)

            if anova.get("p_value", 1) < 0.05:
                st.success("✅ 집단 간 성과 차이가 통계적으로 유의합니다 (p < 0.05)")
            else:
                st.warning("⚠️ 통계적으로 유의한 차이 없음 (p ≥ 0.05)")

            # Two-way 교호작용 표시 (검증 3 전용)
            interaction = anova.get("interaction")
            if interaction:
                sig = interaction.get("significant", False)
                label = (
                    "✅ 교호작용 유의 (전략 효과가 국면에 따라 다름)" if sig else "교호작용 비유의"
                )
                st.info(
                    f"**교호작용** — F={interaction.get('f_statistic', 0):.2f}, "
                    f"p={interaction.get('p_value', 1):.4f}  |  {label}"
                )
                strat = anova.get("strategy_effect", {})
                st.caption(
                    f"전략 주효과 — F={strat.get('f_statistic', 0):.2f}, "
                    f"p={strat.get('p_value', 1):.4f}"
                )

            with st.container(border=True):
                st.markdown("**사후 검정 결과 (Tukey HSD)**")
                posthoc_all = anova.get("post_hoc", [])
                posthoc = (
                    [
                        row
                        for row in posthoc_all
                        if row["group1"] in strategies or row["group2"] in strategies
                    ]
                    if strategies
                    else posthoc_all
                )

                if posthoc:
                    ph = pd.DataFrame(posthoc)
                    ph["유의여부"] = ph["reject"].map({True: "✅", False: "—"})
                    ph["p_adj"] = ph["p_adj"].map("{:.4f}".format)
                    st.dataframe(
                        ph[["group1", "group2", "p_adj", "유의여부"]].rename(
                            columns={"group1": "집단 A", "group2": "집단 B", "p_adj": "p-adj"}
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.info(
                        "사후 검정 결과 없음 (p ≥ 0.05 또는 왼쪽 사이드바에서 전략을 선택하세요)."
                    )


def risk_page() -> None:
    with st.sidebar:
        st.divider()
        st.subheader(":material/tune: 분석 설정")
        period: str = st.selectbox(
            "분석 기간", list(_PERIOD_MONTHS.keys()), index=3, key="risk_period"
        )

    st.title("리스크 모니터링")
    current_risk_tags = st.session_state.get("risk_tags", [])
    risk_vector = risk_vector_from_tags(current_risk_tags)

    st.markdown("**리서치 기반 RL 리스크 관측 벡터**")
    tag_cols = st.columns(3)
    for col, tag, value in zip(tag_cols, RL_RISK_TAGS, risk_vector):
        col.metric(tag, "감지" if value else "미감지", border=True)
    st.caption(f"관측 벡터 순서 {RL_RISK_TAGS}: {risk_vector}")

    with st.spinner("GET /backtest 호출 중…"):
        bt6 = _get("/backtest") or _mock_backtest()

    # 기간 슬라이싱
    n = _PERIOD_MONTHS[period]
    dates = bt6["dates"][-n:] if n else bt6["dates"]
    drawdown = bt6["drawdown"][-n:] if n else bt6["drawdown"]

    sg = bt6.get("safeguard", {})
    if sg.get("active"):
        st.error(f"🔴 Safe-Guard 발동 중 — {sg['triggered_at']} 이후 매매 중단")
    else:
        st.success(f"🟢 Safe-Guard 정상 — 현재 낙폭 {sg.get('current_drawdown', 0):.1%}")

    g1, g2, g3 = st.columns(3)
    with g1:
        with st.container(border=True):
            _echarts_gauge(bt6["var_95"], "VaR 95%", max_val=0.08, key="r_var")
    with g2:
        with st.container(border=True):
            _echarts_gauge(bt6["cvar_95"], "CVaR 95%", max_val=0.08, key="r_cvar")
    with g3:
        with st.container(border=True):
            _echarts_gauge(bt6["mdd"], "MDD", max_val=0.4, key="r_mdd")

    with st.container(border=True):
        _echarts_line(
            x=dates,
            series=[
                {
                    "name": "낙폭",
                    "type": "line",
                    "smooth": True,
                    "areaStyle": {"opacity": 0.3, "color": "#ee6666"},
                    "lineStyle": {"color": "#ee6666"},
                    "itemStyle": {"color": "#ee6666"},
                    "data": drawdown,
                }
            ],
            title=f"MDD 추이 ({period})",
            y_formatter=JsCode("function(v){return (v*100).toFixed(1)+'%'}"),
            key="r_drawdown",
        )

    with st.container(border=True):
        st.markdown("**리스크 지표 요약**")
        m6 = bt6["metrics"]
        st.dataframe(
            pd.DataFrame(
                [
                    {"지표": "VaR 95%", "값": f"{bt6['var_95']:.2%}"},
                    {"지표": "CVaR 95%", "값": f"{bt6['cvar_95']:.2%}"},
                    {"지표": "MDD", "값": f"{bt6['mdd']:.2%}"},
                    {"지표": "연환산 변동성", "값": f"{m6['annualized_volatility']:.2%}"},
                    {"지표": "베타", "값": f"{m6['beta']:.2f}"},
                    {"지표": "샤프 비율", "값": f"{m6['sharpe_ratio']:.2f}"},
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


# ─────────────────────────────────────────────
# 앱 진입점
# ─────────────────────────────────────────────

st.set_page_config(page_title="AI Robo Advisor", layout="wide", page_icon="📈")

st.markdown(
    """
<style>
/* 기본 running 인디케이터(운동하는 사람) 숨기고 🌀 이모지로 교체
   버전 의존 CSS 패치 — streamlit 버전 고정 필요 (requirements.txt 참고) */
@keyframes spin { to { transform: rotate(360deg); } }
[data-testid="stStatusWidget"] {
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px;
}
[data-testid="stStatusWidget"] svg { display: none !important; }
[data-testid="stStatusWidget"]::before {
    content: "🌀";
    font-size: 18px;
    display: inline-block;
    animation: spin 1s linear infinite;
}
</style>
""",
    unsafe_allow_html=True,
)

# 네비게이션 (템플릿과 동일한 st.navigation + st.Page 방식)
pg = st.navigation(
    [
        st.Page(portfolio_page, title="포트폴리오 현황", icon=":material/pie_chart:", default=True),
        st.Page(rl_page, title="강화학습 성과", icon=":material/psychology:"),
        st.Page(shap_page, title="SHAP 해석", icon=":material/auto_graph:"),
        st.Page(research_page, title="에이전트 리서치", icon=":material/article:"),
        st.Page(anova_page, title="ANOVA 검증", icon=":material/science:"),
        st.Page(risk_page, title="리스크 모니터링", icon=":material/shield:"),
    ]
)

with st.sidebar:
    st.divider()
    st.caption(f"API: `{API_BASE_URL}`")
    st.caption("FastAPI 미연결 시 mock 데이터로 렌더링됩니다.")

pg.run()
