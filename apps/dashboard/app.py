"""
Streamlit 대시보드.

streamlit-echarts 기반 인터랙티브 차트 + st.navigation 사이드바 네비게이션.
FastAPI HTTP 통신만 사용하며, 모델을 직접 로드하지 않습니다.
API 미완성 상태에서는 mock 데이터로 UI를 렌더링합니다.
"""
from __future__ import annotations

import os
import re
from datetime import date, timedelta
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import JsCode, st_echarts

try:
    from apps.dashboard.api_client import get_json, post_json
except ModuleNotFoundError:
    # Streamlit file-entry execution inside Docker may not resolve the package root.
    from api_client import get_json, post_json

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
_TIMEOUT_DEFAULT: int = 10
_TIMEOUT_RESEARCH: int = 60  # LangGraph 루프 최대 3회 대응

_PALETTE = ["#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de",
            "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc", "#48b8d0"]

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


# ─────────────────────────────────────────────
# ECharts 공통 빌더
# ─────────────────────────────────────────────

def _echarts_line(
    x: list, series: list[dict], title: str = "",
    height: str = "380px", y_formatter: str = "",
    zoom: bool = True, key: str = "chart",
) -> None:
    opts: dict = {
        "title": {"text": title, "left": "center", "top": 4, "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0, "type": "scroll"},
        "grid": {"bottom": "18%" if zoom else "12%", "top": "12%", "containLabel": True},
        "xAxis": {"type": "category", "data": x, "boundaryGap": False},
        "yAxis": {"type": "value",
                  "axisLabel": {"formatter": y_formatter} if y_formatter else {}},
        "series": series,
    }
    if zoom:
        opts["dataZoom"] = [
            {"type": "inside", "start": 0, "end": 100},
            {"type": "slider", "start": 0, "end": 100, "height": 18, "bottom": 24},
        ]
    st_echarts(options=opts, height=height, theme="streamlit", key=key)


def _echarts_bar_h(
    names: list, values: list, title: str = "",
    colors: list | None = None, height: str = "360px", key: str = "bar_h",
) -> None:
    bar_data = (
        [{"value": v, "itemStyle": {"color": c}} for v, c in zip(values, colors)]
        if colors else values
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
    labels: list, values: list, title: str = "",
    height: str = "380px", key: str = "donut",
) -> None:
    data = [{"name": l, "value": round(v, 4)} for l, v in zip(labels, values)]
    opts = {
        "title": {"text": title, "left": "center", "top": 4, "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "item", "formatter": "{b}: {d}%"},
        "legend": {"bottom": 0, "type": "scroll"},
        "series": [{
            "type": "pie", "radius": ["38%", "65%"], "avoidLabelOverlap": True,
            "itemStyle": {"borderRadius": 8, "borderColor": "#fff", "borderWidth": 2},
            "label": {"show": True, "formatter": "{b}\n{d}%", "fontSize": 11},
            "emphasis": {"label": {"show": True, "fontSize": 13, "fontWeight": "bold"}},
            "data": data,
        }],
    }
    st_echarts(options=opts, height=height, theme="streamlit", key=key)


def _echarts_gauge(
    value: float, title: str, max_val: float = 0.1,
    height: str = "260px", key: str = "gauge",
) -> None:
    pct = round(value * 100, 2)
    color = "#ee6666" if pct > 3 else "#fac858" if pct > 1.5 else "#91cc75"
    opts = {
        "series": [{
            "type": "gauge", "min": 0, "max": round(max_val * 100, 1),
            "progress": {"show": True, "width": 14},
            "axisLine": {"lineStyle": {"width": 14}},
            "axisTick": {"show": False},
            "splitLine": {"length": 10, "lineStyle": {"width": 2, "color": "#999"}},
            "axisLabel": {"distance": 20, "fontSize": 11,
                          "formatter": JsCode("function(v){return v+'%'}")},
            "detail": {"valueAnimation": True, "fontSize": 28, "offsetCenter": [0, "60%"],
                       "formatter": JsCode("function(v){return v.toFixed(2)+'%'}")},
            "title": {"offsetCenter": [0, "88%"], "fontSize": 12},
            "data": [{"value": pct, "name": title}],
            "itemStyle": {"color": color},
        }],
    }
    st_echarts(options=opts, height=height, theme="streamlit", key=key)


# ─────────────────────────────────────────────
# Mock 데이터
# ─────────────────────────────────────────────

_ASSETS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ", "069500", "114260"]
_rng = np.random.default_rng(42)


def _mock_optimize(risk_aversion: float = 1.0) -> dict:
    w = _rng.dirichlet(np.ones(len(_ASSETS)) * (1 / risk_aversion))
    dates = pd.date_range("2024-01-01", periods=252, freq="B").strftime("%Y-%m-%d").tolist()
    return {
        "weights": dict(zip(_ASSETS, w.tolist())),
        "returns": {
            "date": dates,
            "portfolio": np.cumprod(1 + _rng.normal(0.0005, 0.012, 252)).tolist(),
            "benchmark": np.cumprod(1 + _rng.normal(0.0003, 0.010, 252)).tolist(),
        },
    }


def _mock_backtest() -> dict:
    dates = pd.date_range("2024-01-01", periods=252, freq="B").strftime("%Y-%m-%d").tolist()
    wf = _rng.normal(0.001, 0.015, 252)
    bm = _rng.normal(0.0003, 0.010, 252)
    prices = np.cumprod(1 + wf)
    drawdown = ((prices - np.maximum.accumulate(prices)) / np.maximum.accumulate(prices)).tolist()
    metrics = {
        "cumulative_return": 0.248, "cagr": 0.231, "annualized_volatility": 0.182,
        "var_95": 0.021, "cvar_95": 0.031, "mdd": 0.127,
        "sharpe_ratio": 1.27, "sortino_ratio": 1.85, "calmar_ratio": 1.82,
        "alpha": 0.043, "beta": 0.92, "information_ratio": 0.68,
    }
    return {
        "dates": dates,
        "rewards": _rng.normal(0.002, 0.05, 200).cumsum().tolist(),
        "wf_cum": np.cumprod(1 + wf).tolist(),
        "bm_cum": np.cumprod(1 + bm).tolist(),
        "wf_spark": np.cumprod(1 + wf[:50]).tolist(),
        "sharpe_spark": (np.cumsum(_rng.normal(0, 0.3, 50)) + 1.27).tolist(),
        "drawdown": drawdown,
        "metrics": metrics,
        "anova": [
            {
                "name": "reward_function_comparison",
                "f_statistic": 8.71, "p_value": 0.0002, "eta_squared": 0.142,
                "post_hoc": [
                    {"group1": "PPO-return", "group2": "PPO-sharpe", "meandiff": 0.0003, "p_adj": 0.031, "reject": True},
                    {"group1": "PPO-return", "group2": "PPO-mdd",    "meandiff": 0.0005, "p_adj": 0.004, "reject": True},
                    {"group1": "PPO-sharpe", "group2": "PPO-mdd",    "meandiff": 0.0002, "p_adj": 0.218, "reject": False},
                ],
            },
            {
                "name": "strategy_comparison",
                "f_statistic": 12.34, "p_value": 0.0003, "eta_squared": 0.187,
                "post_hoc": [
                    {"group1": "PPO",  "group2": "MVO",    "meandiff": 0.0008, "p_adj": 0.002, "reject": True},
                    {"group1": "PPO",  "group2": "동일비중", "meandiff": 0.0006, "p_adj": 0.041, "reject": True},
                    {"group1": "MVO",  "group2": "동일비중", "meandiff": 0.0002, "p_adj": 0.312, "reject": False},
                ],
            },
            {
                "name": "market_regime_comparison",
                "f_statistic": 2.07, "p_value": 0.127, "eta_squared": 0.038,
                "post_hoc": [],
                "interaction":     {"f_statistic": 3.14, "p_value": 0.014, "significant": True},
                "strategy_effect": {"f_statistic": 4.52, "p_value": 0.011},
            },
        ],
        "var_95": metrics["var_95"], "cvar_95": metrics["cvar_95"], "mdd": metrics["mdd"],
        "safeguard": {"active": False, "triggered_at": None, "current_drawdown": 0.043},
    }


def _mock_explain(target_date: str) -> dict:
    feat = [
        f"{t}_{f}"
        for t in _ASSETS
        for f in ("return", "RSI", "MACD", "MACD_signal")
    ]
    vals = _rng.normal(0, 0.05, len(feat)).tolist()
    base = 0.002
    return {"target_date": target_date, "feature_names": feat,
            "shap_values": vals, "base_value": base, "prediction": base + sum(vals)}


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
        "risk_tags": ["급등락"],
    }


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
            min_value=0.5, max_value=5.0, value=1.0, step=0.5,
            help="값이 클수록 분산 투자 비중 증가",
        )
        period: str = st.selectbox("분석 기간", list(_PERIOD_MONTHS.keys()), index=3)

    st.title("포트폴리오 현황")

    if st.button("최적화 실행", key="btn_optimize"):
        with st.spinner("POST /optimize 호출 중…"):
            data = _post("/optimize", {"risk_aversion": risk_aversion}) or _mock_optimize(risk_aversion)
    else:
        data = _mock_optimize(risk_aversion)

    # 기간 슬라이싱
    n = _PERIOD_MONTHS[period]
    ret = data["returns"]
    x = ret["date"][-n:] if n else ret["date"]
    port_vals = ret["portfolio"][-n:] if n else ret["portfolio"]
    bm_vals = ret["benchmark"][-n:] if n else ret["benchmark"]

    ret_arr = np.array(port_vals)
    bm_arr = np.array(bm_vals)
    cum_ret = float(ret_arr[-1] - 1)
    excess = float(ret_arr[-1] - bm_arr[-1])
    top_asset = max(data["weights"], key=data["weights"].get)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("누적 수익률", f"{cum_ret:.1%}", border=True)
    k2.metric("초과 수익", f"{excess:.1%}", delta=f"{excess:.2%} vs KOSPI", border=True)
    k3.metric("최대 비중 자산", top_asset, f"{data['weights'][top_asset]:.1%}", border=True)
    k4.metric("편입 종목 수", f"{len(data['weights'])}개", border=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            _echarts_donut(
                labels=list(data["weights"].keys()),
                values=list(data["weights"].values()),
                title="자산 비중", key="p_donut",
            )
    with col2:
        with st.container(border=True):
            _echarts_line(
                x=x,
                series=[
                    {"name": "포트폴리오", "type": "line", "smooth": True,
                     "areaStyle": {"opacity": 0.15}, "data": port_vals,
                     "itemStyle": {"color": _PALETTE[0]}},
                    {"name": "벤치마크(KOSPI)", "type": "line", "smooth": True,
                     "data": bm_vals, "itemStyle": {"color": _PALETTE[3]}},
                ],
                title=f"누적 수익률 ({period})", key="p_line",
            )

    with st.expander("비중 상세 테이블"):
        st.dataframe(
            pd.DataFrame({"자산": list(data["weights"].keys()),
                          "비중": [f"{v:.2%}" for v in data["weights"].values()]}),
            hide_index=True, use_container_width=True,
        )


def rl_page() -> None:
    with st.sidebar:
        st.divider()
        st.subheader(":material/tune: 분석 설정")
        period: str = st.selectbox("분석 기간", list(_PERIOD_MONTHS.keys()), index=3, key="rl_period")
        strategies: list[str] = st.multiselect(
            "비교 전략", ["PPO", "MVO", "동일비중"], default=["PPO"],
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

    m = bt["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("누적 수익률", f"{m['cumulative_return']:.1%}", border=True,
              chart_data=bt.get("wf_spark"), chart_type="area")
    c2.metric("샤프 비율", f"{m['sharpe_ratio']:.2f}", border=True,
              chart_data=bt.get("sharpe_spark"), chart_type="line")
    c3.metric("MDD", f"{m['mdd']:.1%}", border=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            _echarts_line(
                x=list(range(1, len(bt["rewards"]) + 1)),
                series=[{"name": "누적 보상", "type": "line", "smooth": True,
                         "areaStyle": {"opacity": 0.12}, "data": bt["rewards"],
                         "itemStyle": {"color": _PALETTE[1]}}],
                title="학습 곡선 (에피소드 누적 보상)", key="rl_reward",
            )
    with col2:
        with st.container(border=True):
            # 비교 전략 필터 적용
            series_list = []
            if "PPO" in strategies:
                series_list.append({
                    "name": "PPO", "type": "line", "smooth": True,
                    "areaStyle": {"opacity": 0.12}, "data": wf_cum,
                    "itemStyle": {"color": _PALETTE[0]},
                })
            if "MVO" in strategies:
                series_list.append({
                    "name": "MVO (mock)", "type": "line", "smooth": True,
                    "data": (np.array(wf_cum) * 0.92).tolist(),
                    "itemStyle": {"color": _PALETTE[2]},
                })
            if "동일비중" in strategies:
                series_list.append({
                    "name": "동일비중 (mock)", "type": "line", "smooth": True,
                    "lineStyle": {"type": "dashed"},
                    "data": bm_cum,
                    "itemStyle": {"color": _PALETTE[3]},
                })
            if not series_list:
                st.info("비교 전략을 하나 이상 선택하세요.")
            else:
                _echarts_line(
                    x=dates, series=series_list,
                    title=f"Walk-Forward 백테스트 ({period})", key="rl_wf",
                )

    with st.container(border=True):
        st.markdown("**성과 지표 전체**")
        st.dataframe(
            pd.DataFrame([{"지표": k, "값": f"{v:.4f}"} for k, v in m.items()]),
            hide_index=True, use_container_width=True,
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

    feat, vals, base, pred = sd["feature_names"], sd["shap_values"], sd["base_value"], sd["prediction"]
    st.markdown(f"분석 날짜: **{target_date}** | 기준값: **`{base:.4f}`** → 예측값: **`{pred:.4f}`**")

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
                title="Summary Plot (|SHAP| 절대값)", key="shap_summary",
            )
    with col2:
        with st.container(border=True):
            colors = ["#ee6666" if v > 0 else "#5470c6" for v in shap_df["SHAP값"]]
            _echarts_bar_h(
                names=shap_df["피처"].tolist(),
                values=shap_df["SHAP값"].round(4).tolist(),
                colors=colors,
                title="Force Plot (빨강=양, 파랑=음)", key="shap_force",
            )


def research_page() -> None:
    st.title("에이전트 리서치 (RAG)")

    question = st.text_area(
        "투자 질문 입력",
        placeholder="ex. 삼성전자 HBM 반도체 실적 전망은?",
        height=80,
    )

    # Enter → 리서치 실행, Shift+Enter → 줄바꿈
    st.components.v1.html("""
    <script>
    (function() {
        function attachHandler() {
            const textareas = window.parent.document.querySelectorAll('textarea');
            textareas.forEach(function(ta) {
                if (ta._researchBound) return;
                ta._researchBound = true;
                ta.addEventListener('keydown', function(e) {
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
    """, height=0)

    if st.button("리서치 실행", key="btn_research"):
        if not question.strip():
            st.error("질문을 입력하세요.")
        else:
            with st.status("에이전트 리서치 진행 중…", expanded=True) as status:
                st.write("LangGraph 파이프라인 실행 (planner → researcher → analyst)")
                res = _post("/research", {"question": question}, timeout=_TIMEOUT_RESEARCH)
                if res is None:
                    res = _mock_research(question)
                    status.update(label="API 미연결 — mock 응답", state="error", expanded=False)
                else:
                    status.update(label="리서치 완료", state="complete", expanded=False)

            col1, col2 = st.columns([2, 1])
            with col1:
                with st.container(border=True):
                    st.markdown("**분석 리포트**")
                    report_text = re.sub(
                        r'\(출처: ([^)\n]+)$', r'(출처: \1)', res["report"], flags=re.MULTILINE
                    )
                    st.markdown(report_text)
            with col2:
                with st.container(border=True):
                    st.markdown("**출처 URL**")
                    for url in res.get("sources", []):
                        domain = urlparse(url).netloc or url
                        st.markdown(f"- [{domain}]({url})")
                with st.container(border=True):
                    st.markdown("**리스크 태그**")
                    tags = res.get("risk_tags", [])
                    if tags:
                        for t in tags:
                            st.warning(f"⚠️ {t}")
                    else:
                        st.success("감지된 리스크 없음")

            with st.expander("추론 트레이스 (reasoning_trace)"):
                st.code(res.get("reasoning_trace", ""), language="text")
    else:
        st.info("위에서 질문을 입력하고 '리서치 실행' 버튼을 누르세요.")


def anova_page() -> None:
    with st.sidebar:
        st.divider()
        st.subheader(":material/tune: 분석 설정")
        strategies: list[str] = st.multiselect(
            "비교 전략", ["PPO", "MVO", "동일비중"], default=["PPO"],
            help="사후 검정 결과 필터",
        )

    st.title("ANOVA 검증 결과")
    with st.spinner("GET /backtest 호출 중…"):
        bt5 = _get("/backtest") or _mock_backtest()

    anova_list: list = bt5.get("anova", _mock_backtest()["anova"])

    _EXP_LABELS = {
        "reward_function_comparison": "검증 1 — 보상함수 비교",
        "strategy_comparison":        "검증 2 — 전략 비교",
        "market_regime_comparison":   "검증 3 — 국면 × 전략 (Two-way)",
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
                label = "✅ 교호작용 유의 (전략 효과가 국면에 따라 다름)" if sig else "교호작용 비유의"
                st.info(f"**교호작용** — F={interaction.get('f_statistic', 0):.2f}, "
                        f"p={interaction.get('p_value', 1):.4f}  |  {label}")
                strat = anova.get("strategy_effect", {})
                st.caption(f"전략 주효과 — F={strat.get('f_statistic', 0):.2f}, "
                           f"p={strat.get('p_value', 1):.4f}")

            with st.container(border=True):
                st.markdown("**사후 검정 결과 (Tukey HSD)**")
                posthoc_all = anova.get("post_hoc", [])
                posthoc = [
                    row for row in posthoc_all
                    if row["group1"] in strategies or row["group2"] in strategies
                ] if strategies else posthoc_all

                if posthoc:
                    ph = pd.DataFrame(posthoc)
                    ph["유의여부"] = ph["reject"].map({True: "✅", False: "—"})
                    ph["p_adj"] = ph["p_adj"].map("{:.4f}".format)
                    st.dataframe(
                        ph[["group1", "group2", "p_adj", "유의여부"]].rename(
                            columns={"group1": "집단 A", "group2": "집단 B", "p_adj": "p-adj"}
                        ),
                        hide_index=True, use_container_width=True,
                    )
                else:
                    st.info("사후 검정 결과 없음 (p ≥ 0.05 또는 왼쪽 사이드바에서 전략을 선택하세요).")


def risk_page() -> None:
    with st.sidebar:
        st.divider()
        st.subheader(":material/tune: 분석 설정")
        period: str = st.selectbox("분석 기간", list(_PERIOD_MONTHS.keys()), index=3, key="risk_period")

    st.title("리스크 모니터링")
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
            series=[{
                "name": "낙폭", "type": "line", "smooth": True,
                "areaStyle": {"opacity": 0.3, "color": "#ee6666"},
                "lineStyle": {"color": "#ee6666"},
                "itemStyle": {"color": "#ee6666"},
                "data": drawdown,
            }],
            title=f"MDD 추이 ({period})",
            y_formatter=JsCode("function(v){return (v*100).toFixed(1)+'%'}"),
            key="r_drawdown",
        )

    with st.container(border=True):
        st.markdown("**리스크 지표 요약**")
        m6 = bt6["metrics"]
        st.dataframe(
            pd.DataFrame([
                {"지표": "VaR 95%",       "값": f"{bt6['var_95']:.2%}"},
                {"지표": "CVaR 95%",      "값": f"{bt6['cvar_95']:.2%}"},
                {"지표": "MDD",           "값": f"{bt6['mdd']:.2%}"},
                {"지표": "연환산 변동성", "값": f"{m6['annualized_volatility']:.2%}"},
                {"지표": "베타",          "값": f"{m6['beta']:.2f}"},
                {"지표": "샤프 비율",     "값": f"{m6['sharpe_ratio']:.2f}"},
            ]),
            hide_index=True, use_container_width=True,
        )


# ─────────────────────────────────────────────
# 앱 진입점
# ─────────────────────────────────────────────

st.set_page_config(page_title="AI Robo Advisor", layout="wide", page_icon="📈")

st.markdown("""
<style>
/* 기본 running 인디케이터(운동하는 사람) 숨기고 🌀 이모지로 교체 */
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
""", unsafe_allow_html=True)

# 네비게이션 (템플릿과 동일한 st.navigation + st.Page 방식)
pg = st.navigation([
    st.Page(portfolio_page, title="포트폴리오 현황", icon=":material/pie_chart:",   default=True),
    st.Page(rl_page,        title="강화학습 성과",   icon=":material/psychology:"),
    st.Page(shap_page,      title="SHAP 해석",       icon=":material/auto_graph:"),
    st.Page(research_page,  title="에이전트 리서치", icon=":material/article:"),
    st.Page(anova_page,     title="ANOVA 검증",      icon=":material/science:"),
    st.Page(risk_page,      title="리스크 모니터링", icon=":material/shield:"),
])

with st.sidebar:
    st.divider()
    st.caption(f"API: `{API_BASE_URL}`")
    st.caption("FastAPI 미연결 시 mock 데이터로 렌더링됩니다.")

pg.run()
