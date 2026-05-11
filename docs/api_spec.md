# AI 로보어드바이저 API 명세서

> **버전**: v0.3.0 (Sprint 3 기준)  
> **작성일**: 2026-05-09  
> **대상**: 박지민(백엔드), 강유영(대시보드·분석), 이문정(RL·인프라)

---

## 목차

1. [개요](#1-개요)
2. [공통 규격](#2-공통-규격)
3. [엔드포인트 명세](#3-엔드포인트-명세)
   - [GET /health](#31-get-health)
   - [POST /optimize](#32-post-optimize)
   - [POST /explain](#33-post-explain)
   - [POST /research](#34-post-research)
   - [GET /backtest](#35-get-backtest)
   - [GET /backtest/stress](#36-get-backtessstress-sprint-3-신규) *(Sprint 3 신규)*
4. [스키마 정의](#4-스키마-정의)
5. [Sprint 3 통합 목표](#5-sprint-3-통합-목표)
6. [알려진 불일치 및 수정 계획](#6-알려진-불일치-및-수정-계획)

---

## 1. 개요

| 항목 | 값 |
|------|----|
| 프레임워크 | FastAPI |
| 로컬 Base URL | `http://localhost:8000` |
| Docker Base URL | `http://api:8000` |
| 인증 | 없음 (내부 서비스) |
| 콘텐츠 타입 | `application/json` |
| 문자 인코딩 | UTF-8 |

대시보드(Streamlit)는 FastAPI HTTP 통신만 사용하며, 모델·DB를 직접 로드하지 않는다.  
API 미연결 시 대시보드는 mock 데이터로 렌더링한다.

---

## 2. 공통 규격

### 2-1. Status 타입

```
"ready"      → 실제 모듈과 연결되어 정상 동작 중
"fallback"   → 모듈 미연결, deterministic 더미 데이터 반환
"unavailable"→ 모듈 오류 또는 의존성 없음
```

### 2-2. 공통 에러 응답

FastAPI 기본 validation 에러 응답을 그대로 사용한다.

```json
{
  "detail": [
    {
      "loc": ["body", "field_name"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

| HTTP 코드 | 상황 |
|-----------|------|
| 200 | 정상 (status가 "fallback"이어도 200) |
| 422 | 요청 바디 유효성 검사 실패 |
| 500 | 서버 내부 오류 |

### 2-3. 날짜 형식

모든 날짜는 `"YYYY-MM-DD"` 형식 문자열.

---

## 3. 엔드포인트 명세

---

### 3.1 GET /health

서비스 가용성 확인 및 하위 모듈 상태 반환.

**담당**: 박지민

#### 요청

파라미터 없음.

#### 응답 `200 OK`

```json
{
  "status": "ok",
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "log_level": "INFO"
  },
  "modules": {
    "data":     "ready | fallback",
    "rl":       "ready | fallback",
    "rag":      "ready | fallback",
    "shap":     "ready | fallback",
    "backtest": "ready | fallback"
  }
}
```

| 모듈 키 | "ready" 조건 |
|---------|------------|
| `data` | `returns.parquet`, `features.parquet` 모두 읽을 수 있음 |
| `rl` | PPO 모델 파일 로드 성공 (Sprint 3 연동 후) |
| `rag` | `OPENAI_API_KEY` 환경변수 존재 |
| `shap` | SHAP 모듈 import 성공 (Sprint 3 연동 후) |
| `backtest` | `data/results/` 결과 파일 존재 (Sprint 3 연동 후) |

#### 구현 상태

✅ Sprint 2 완료

---

### 3.2 POST /optimize

PPO 강화학습 기반 포트폴리오 최적 비중 계산.

**담당**: 박지민

#### 요청 바디

```json
{
  "tickers": ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ", "069500", "114260"],
  "risk_profile": "balanced",
  "risk_aversion": 1.0
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `tickers` | `list[str]` | 아니오 | 10개 전체 자산 | 최적화할 자산 티커 목록 (1개 이상) |
| `risk_profile` | `"conservative" \| "balanced" \| "aggressive"` | 아니오 | `"balanced"` | 위험 성향 프리셋 |
| `risk_aversion` | `float` (> 0) | 아니오 | `null` | 수치형 위험 회피 계수. 설정 시 `risk_profile` 보다 우선 |

> **대시보드 호출 예시**: `POST /optimize` `{"risk_aversion": 1.5}`

#### 응답 `200 OK`

```json
{
  "status": "fallback",
  "tickers": ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ", "069500", "114260"],
  "weights": {
    "SPY": 0.1234,
    "QQQ": 0.0987,
    "...": "..."
  },
  "risk_profile": "balanced",
  "expected_return": 0.084,
  "expected_volatility": 0.132,
  "returns": {
    "date":      ["2024-01-02", "2024-01-03", "..."],
    "portfolio": [1.0003, 1.0007, "..."],
    "benchmark": [1.0001, 1.0004, "..."]
  },
  "message": "PPO 모델 연결 전 deterministic fallback 포트폴리오입니다."
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `status` | `EndpointStatus` | PPO 연결 전 `"fallback"`, 연결 후 `"ready"` |
| `tickers` | `list[str]` | 요청된 자산 목록 |
| `weights` | `dict[str, float]` | 합이 1.0인 자산별 비중 |
| `risk_profile` | `RiskProfile` | 적용된 위험 성향 |
| `expected_return` | `float` | 연환산 기대수익률 |
| `expected_volatility` | `float` | 연환산 기대변동성 |
| `returns.date` | `list[str]` | 날짜 배열 |
| `returns.portfolio` | `list[float]` | 포트폴리오 누적 수익률 시계열 (기준: 1.0) |
| `returns.benchmark` | `list[float]` | 벤치마크(SPY) 누적 수익률 시계열 (기준: 1.0) |
| `message` | `str` | 상태 설명 메시지 |

#### Sprint 3 통합 목표

- PPO 모델(`models/ppo_*.zip`)을 로드해 실제 비중 반환
- `status`가 `"ready"`로 변경됨
- PPO 로드 실패 시 현재 fallback 유지

---

### 3.3 POST /explain

SHAP 기반 피처 기여도 설명. PPO 모델이 특정 날짜에 내린 결정의 근거를 해석한다.

**담당**: 박지민 (API), 이문정 (SHAP 모듈)

#### 요청 바디

```json
{
  "date": "2024-06-15",
  "top_k": 8
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `date` | `str` (YYYY-MM-DD) | 아니오 | `null` (최신 날짜) | SHAP 분석 기준 날짜. 없으면 features.parquet 마지막 날짜 |
| `top_k` | `int` (1 ~ 20) | 아니오 | `8` | 반환할 상위 기여 피처 수 |

#### 응답 `200 OK`

```json
{
  "status": "fallback",
  "date": "2024-06-15",
  "target_date": "2024-06-14",
  "base_value": 0.05,
  "prediction": 0.063,
  "feature_contributions": [
    {"feature": "SPY_return", "value": 0.018, "contribution": 0.031},
    {"feature": "QQQ_RSI",   "value": 61.2,  "contribution": 0.019},
    "..."
  ],
  "feature_names": ["SPY_return", "QQQ_RSI", "..."],
  "shap_values":   [0.031, 0.019, "..."],
  "message": "SHAP 모듈 연결 전 feature contribution fallback입니다."
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `status` | `EndpointStatus` | SHAP 연결 전 `"fallback"` |
| `date` | `str \| null` | 요청한 날짜 (없으면 null) |
| `target_date` | `str \| null` | 실제 사용된 데이터 날짜 (요청일 이전 마지막 거래일) |
| `base_value` | `float` | SHAP base value (모델의 평균 예측값) |
| `prediction` | `float` | `base_value + sum(shap_values)` |
| `feature_contributions` | `list[FeatureContribution]` | 피처별 기여도 (top_k 개, \|SHAP\| 내림차순) |
| `feature_names` | `list[str]` | 피처명 배열 (feature_contributions와 순서 동일) |
| `shap_values` | `list[float]` | SHAP 값 배열 (feature_contributions와 순서 동일) |
| `message` | `str` | 상태 설명 메시지 |

#### Sprint 3 통합 목표

- `src/rl/shap.py` (이문정)에서 SHAP 값 계산 후 반환
- 피처명은 `features.parquet` 컬럼명과 동일: `{ticker}_{return|RSI|MACD|MACD_signal}`

---

### 3.4 POST /research

LangGraph RAG 에이전트를 통한 투자 리서치 리포트 생성.

**담당**: 박지민 (API), 강유영 (LangGraph 에이전트)

#### 요청 바디

```json
{
  "question": "삼성전자 HBM 반도체 실적 전망은?"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `question` | `str` (1자 이상) | **예** | 투자 관련 질문 |

> **처리 시간 안내**: LangGraph Self-Correction 루프(최대 3회)로 인해 10~30초 소요 가능.  
> 대시보드에서 `st.spinner()`로 처리하며, 타임아웃은 대시보드 측 `REQUEST_TIMEOUT = 10`초.  
> Sprint 3에서는 **동기 방식 유지** (폴링·SSE 미적용). 타임아웃 값은 30초로 늘리는 것을 권장.

#### 응답 `200 OK`

```json
{
  "status": "ready",
  "question": "삼성전자 HBM 반도체 실적 전망은?",
  "report": "**분석 리포트**\n현재 HBM 시장은...",
  "sources": [
    "https://news.google.com/rss/...",
    "https://..."
  ],
  "reasoning_trace": "[THINK][planner] 질의 분석 시작\n[THINK][researcher] Chroma hit=5건\n...",
  "risk_tags": ["급등락", "실적쇼크"]
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `status` | `EndpointStatus` | `OPENAI_API_KEY` 있고 LangGraph 성공 시 `"ready"`, 아니면 `"fallback"` |
| `question` | `str` | 요청한 질문 (echo) |
| `report` | `str` | Markdown 형식 투자 분석 리포트 |
| `sources` | `list[str]` | 참고한 뉴스 URL 목록 (없으면 GitHub 레포 URL) |
| `reasoning_trace` | `str` | LangGraph 내부 추론 과정 (`[THINK][노드명]` 접두사) |
| `risk_tags` | `list[str]` | 감지된 리스크 태그 (`"규제변경"`, `"실적쇼크"`, `"급등락"` 중 해당 항목) |

#### LangGraph 연동 시그니처 (강유영 제공)

```python
from src.agent.graph import run_graph

state = run_graph("삼성전자 최근 투자 리스크 분석해줘")
# 반환 dict 키:
#   state["response"]         → str  최종 리포트 (Markdown)
#   state["sources"]          → list[str]  참고 URL
#   state["reasoning_trace"]  → str  추론 과정
#   state["rl_risk_tags"]     → list[str]  리스크 태그
#   state["messages"]         → list  LangGraph 메시지 (reasoning_trace fallback)
```

---

### 3.5 GET /backtest

Walk-Forward 백테스트 성과 지표 및 ANOVA 검증 결과 반환.

**담당**: 박지민 (API), 강유영 (metrics·anova·backtest), 이문정 (RL 학습 결과 파일)

#### 요청 쿼리 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| `window` | `str` | 아니오 | `"final"` | Walk-Forward 윈도우 선택 (`"w1"` \| `"w2"` \| `"w3"` \| `"final"`) |

**윈도우 정의**:

| window | 학습 기간 | 검증 기간 | 특징 |
|--------|----------|---------|------|
| `w1` | 2018-01-01 ~ 2021-12-31 | 2022-01-01 ~ 2022-12-31 | 금리 충격 |
| `w2` | 2019-01-01 ~ 2022-12-31 | 2023-01-01 ~ 2023-12-31 | 회복장 |
| `w3` | 2020-01-01 ~ 2023-12-31 | 2024-01-01 ~ 2024-12-31 | AI 랠리 |
| `final` | 2021-01-01 ~ 2024-12-31 | 2025-01-01 ~ 2025-12-31 | 최신 구간 |

> `window` 파라미터는 Sprint 3에서 추가. Sprint 2 현재는 파라미터 무시하고 전체 기간 데이터 반환.  
> 모델 파일 네이밍: `models/ppo_{reward}_{window}.zip` (예: `ppo_sharpe_w3.zip`), 총 12개 (3보상 × 4윈도우).

#### 응답 `200 OK`

```json
{
  "status": "fallback",
  "benchmark": "SPY",
  "metrics": {
    "cumulative_return":     0.248,
    "cagr":                  0.231,
    "annualized_volatility": 0.182,
    "var_95":                0.021,
    "cvar_95":               0.031,
    "mdd":                   0.127,
    "sharpe_ratio":          1.27,
    "sortino_ratio":         1.85,
    "calmar_ratio":          1.82,
    "alpha":                 0.043,
    "beta":                  0.92,
    "information_ratio":     0.68
  },
  "anova": [
    {
      "name": "reward_function_comparison",
      "f_statistic": 3.12,
      "p_value": 0.041,
      "eta_squared": 0.18,
      "post_hoc": [
        {"group1": "PPO-return", "group2": "PPO-sharpe", "meandiff": 0.021, "p_adj": 0.032, "reject": true},
        {"group1": "PPO-return", "group2": "PPO-mdd",    "meandiff": 0.009, "p_adj": 0.210, "reject": false},
        {"group1": "PPO-sharpe", "group2": "PPO-mdd",    "meandiff": 0.012, "p_adj": 0.089, "reject": false}
      ]
    },
    {
      "name": "strategy_comparison",
      "f_statistic": 4.36,
      "p_value": 0.028,
      "eta_squared": 0.22,
      "post_hoc": [
        {"group1": "PPO",  "group2": "MVO",   "meandiff": 0.031, "p_adj": 0.002, "reject": true},
        {"group1": "PPO",  "group2": "동일비중", "meandiff": 0.018, "p_adj": 0.041, "reject": true},
        {"group1": "MVO",  "group2": "동일비중", "meandiff": 0.013, "p_adj": 0.312, "reject": false}
      ]
    },
    {
      "name": "market_regime_comparison",
      "f_statistic": 2.07,
      "p_value": 0.096,
      "eta_squared": 0.11,
      "post_hoc": [
        {"group1": "bull", "group2": "bear",  "meandiff": 0.042, "p_adj": 0.071, "reject": false},
        {"group1": "bull", "group2": "crisis","meandiff": 0.087, "p_adj": 0.013, "reject": true},
        {"group1": "bear", "group2": "crisis","meandiff": 0.045, "p_adj": 0.102, "reject": false}
      ]
    }
  ],
  "dates":       ["2024-01-02", "2024-01-03", "..."],
  "rewards":     [0.001, -0.002, "..."],
  "wf_cum":      [1.001, 0.999, "..."],
  "bm_cum":      [1.000, 0.998, "..."],
  "wf_spark":    [1.001, 1.003, "..."],
  "sharpe_spark":[1.24, 1.26, "..."],
  "drawdown":    [-0.001, -0.003, "..."],
  "var_95":      0.021,
  "cvar_95":     0.031,
  "mdd":         0.127,
  "safeguard": {
    "active":           false,
    "triggered_at":     null,
    "current_drawdown": 0.043
  },
  "message": "Walk-Forward 백테스트 모듈 연결 전 fallback 결과입니다."
}
```

#### 필드 상세

**metrics** (12개 고정, `src/rl/metrics.py` 출력):

| 키 | 단위 | 설명 |
|----|------|------|
| `cumulative_return` | 소수 | 전체 기간 누적 수익률 |
| `cagr` | 소수/년 | 연평균 복합 성장률 |
| `annualized_volatility` | 소수/년 | 연환산 표준편차 |
| `var_95` | 소수 | 95% VaR (양수 = 손실) |
| `cvar_95` | 소수 | 95% CVaR / Expected Shortfall (양수 = 손실) |
| `mdd` | 소수 | 최대 낙폭 (양수) |
| `sharpe_ratio` | 비율 | 연환산 샤프 비율 |
| `sortino_ratio` | 비율 | 연환산 소르티노 비율 |
| `calmar_ratio` | 비율 | CAGR / MDD |
| `alpha` | 소수/년 | 젠센 알파 (연환산) |
| `beta` | 비율 | 벤치마크 대비 베타 |
| `information_ratio` | 비율 | 연환산 정보 비율 |

**anova** (`list[AnovaResult]`, 3개 고정):

| `name` | 설명 | 담당 |
|--------|------|------|
| `reward_function_comparison` | 보상함수 3종(return·sharpe·mdd) 성과 차이 검증 | 강유영 (`anova.py`) |
| `strategy_comparison` | PPO vs MVO vs 동일비중 성과 차이 검증 | 강유영 (`anova.py`) |
| `market_regime_comparison` | bull·bear·crisis 국면별 성과 차이 검증 | 강유영 (`anova.py`) |

각 `AnovaResult` 필드:

| 필드 | 타입 | 설명 |
|------|------|------|
| `name` | `str` | 실험 식별자 |
| `f_statistic` | `float` | 일원분산분석 F 통계량 |
| `p_value` | `float` | p-value (< 0.05 = 통계적 유의) |
| `eta_squared` | `float` | 효과 크기 η² |
| `post_hoc` | `list[TukeyRow]` | Tukey HSD 사후 검정 결과 |

각 `TukeyRow` 필드:

| 필드 | 타입 | 설명 |
|------|------|------|
| `group1` | `str` | 비교 그룹 A |
| `group2` | `str` | 비교 그룹 B |
| `meandiff` | `float` | 그룹 간 평균 차이 |
| `p_adj` | `float` | 보정된 p-value |
| `reject` | `bool` | `true` = 귀무가설 기각 (유의미한 차이 존재) |

**시계열 배열** (모두 동일 길이, `dates`와 인덱스 대응):

| 필드 | 설명 |
|------|------|
| `dates` | 날짜 배열 |
| `rewards` | 최근 200 에피소드 누적 보상 (학습 곡선) |
| `wf_cum` | Walk-Forward 포트폴리오 누적 수익률 |
| `bm_cum` | 벤치마크(SPY) 누적 수익률 |
| `wf_spark` | `wf_cum` 마지막 50개 (스파크라인) |
| `sharpe_spark` | 30일 롤링 샤프 비율 마지막 50개 |
| `drawdown` | 일별 낙폭 (음수, 0 이하) |

**safeguard**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `active` | `bool` | Safe-Guard 발동 중 여부 |
| `triggered_at` | `str \| null` | 발동 날짜 (YYYY-MM-DD), 미발동 시 null |
| `current_drawdown` | `float` | 현재 낙폭 (양수) |

> Safe-Guard 임계값: MDD ≥ 15%. 코로나 KOSPI MDD 38%의 절반 이하(조기 경보 기준).

#### ANOVA 입력 파일 (이문정 → 강유영)

강유영이 `anova.py`를 구현하려면 아래 파일이 필요하다.

```
data/results/backtest_return.csv   # 보상함수: 수익률 최대화
data/results/backtest_sharpe.csv   # 보상함수: 샤프 비율 최대화
data/results/backtest_mdd.csv      # 보상함수: MDD 최소화
```

CSV 포맷 (헤더 포함):
```
date,episode_return
2024-01-02,0.0023
2024-01-03,-0.0011
```

#### Sprint 3 통합 목표

- `data/results/` 파일 생성 후 `anova.py` 실제 계산 연동
- `window` 쿼리 파라미터 지원 추가
- `status`가 `"ready"`로 변경됨

---

### 3.6 GET /backtest/stress *(Sprint 3 신규)*

코로나 구간(2020-02-01 ~ 2020-05-31) 스트레스 테스트.  
코로나 데이터는 전체 학습 데이터에 포함되나, **Final Holdout 모델(학습 2021~2024)은 코로나를 학습하지 않아** 해당 모델 기준 OOS 검증이 유효하다.  
극단적 하락장에서의 Safe-Guard 작동 여부를 검증한다.

**담당**: 강유영 (`src/rl/backtest.py`), 박지민 (API 라우터 추가)

#### 요청

파라미터 없음.

#### 응답 `200 OK`

```json
{
  "status": "ready",
  "period": {
    "start": "2020-02-01",
    "end":   "2020-05-31"
  },
  "metrics": {
    "cumulative_return":     -0.183,
    "mdd":                    0.312,
    "var_95":                 0.041,
    "cvar_95":                0.058,
    "sharpe_ratio":          -1.24
  },
  "benchmark_metrics": {
    "cumulative_return":     -0.342,
    "mdd":                    0.381,
    "var_95":                 0.052,
    "cvar_95":                0.073,
    "sharpe_ratio":          -2.01
  },
  "safeguard_events": [
    {
      "triggered_at":    "2020-03-12",
      "drawdown_at_trigger": 0.152,
      "resumed_at":      "2020-04-08"
    }
  ],
  "dates":     ["2020-02-03", "2020-02-04", "..."],
  "ppo_cum":   [0.998, 0.991, "..."],
  "bm_cum":    [0.997, 0.987, "..."],
  "drawdown":  [-0.002, -0.009, "..."],
  "message":   "코로나 구간 스트레스 테스트 결과입니다."
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `period` | `{start, end}` | 스트레스 테스트 기간 |
| `metrics` | `dict[str, float]` | PPO 포트폴리오 지표 (5개) |
| `benchmark_metrics` | `dict[str, float]` | SPY 벤치마크 동일 기간 지표 |
| `safeguard_events` | `list[SafeguardEvent]` | Safe-Guard 발동·재개 이벤트 목록 |
| `dates` / `ppo_cum` / `bm_cum` / `drawdown` | `list[float]` | 시계열 (동일 길이) |

각 `SafeguardEvent`:

| 필드 | 타입 | 설명 |
|------|------|------|
| `triggered_at` | `str` | Safe-Guard 발동 날짜 |
| `drawdown_at_trigger` | `float` | 발동 시점 낙폭 (임계값 15% 근접) |
| `resumed_at` | `str \| null` | 재개 날짜 (기간 내 재개 없으면 null) |

> **fallback 처리**: `data/results/` 파일 없으면 `status: "fallback"`, 모든 수치 0, 빈 배열 반환.

---

## 4. 스키마 정의

```python
# apps/api/schemas.py 기준

RiskProfile = Literal["conservative", "balanced", "aggressive"]
EndpointStatus = Literal["ready", "fallback", "unavailable"]

class TukeyRow(BaseModel):
    group1:    str
    group2:    str
    meandiff:  float
    p_adj:     float
    reject:    bool

class AnovaResult(BaseModel):
    name:        str
    f_statistic: float
    p_value:     float
    eta_squared: float
    post_hoc:    list[TukeyRow]   # ← Sprint 3에서 str → list[TukeyRow] 변경

class SafeguardEvent(BaseModel):    # Sprint 3 신규
    triggered_at:        str
    drawdown_at_trigger: float
    resumed_at:          str | None

class StressTestResponse(BaseModel):  # Sprint 3 신규
    status:            EndpointStatus
    period:            dict[str, str]
    metrics:           dict[str, float]
    benchmark_metrics: dict[str, float]
    safeguard_events:  list[SafeguardEvent]
    dates:             list[str]
    ppo_cum:           list[float]
    bm_cum:            list[float]
    drawdown:          list[float]
    message:           str
```

> 나머지 스키마(`OptimizeRequest`, `OptimizeResponse`, `ExplainRequest`, `ExplainResponse`,  
> `ResearchRequest`, `ResearchResponse`, `BacktestResponse`)는 `apps/api/schemas.py` 참고.

---

## 5. Sprint 3 통합 목표

| 엔드포인트 | 담당 | 목표 | 연결 모듈 |
|-----------|------|------|---------|
| `POST /optimize` | 박지민 | PPO 비중 실제 반환 | `models/ppo_*.zip` |
| `POST /explain` | 박지민 + 이문정 | SHAP 값 실제 반환 | `src/rl/shap.py` |
| `GET /backtest` | 박지민 + 강유영 | walk-forward 실측 결과 반환 | `src/rl/metrics.py`, `src/rl/anova.py`, `data/results/` |
| `GET /backtest/stress` | 박지민 + 강유영 | 스트레스 테스트 신규 추가 | `src/rl/backtest.py` |
| `GET /backtest` | 박지민 | `window` 쿼리 파라미터 지원 | — |
| `schemas.py` | 박지민 | `AnovaResult.post_hoc: list[TukeyRow]` 변경 | — |
| `apps/dashboard/app.py` | 강유영 | ANOVA 탭 `list[AnovaResult]` 처리로 업데이트 | — |

---

## 6. 알려진 불일치 및 수정 계획

### 6-1. [중요] ANOVA 스키마 불일치

**문제**: `BacktestResponse.anova`는 `list[AnovaResult]`이지만, 대시보드 `anova_page()`는 `anova`를 단일 dict로 처리한다.

```python
# 현재 대시보드 코드 (버그)
anova = bt5.get("anova", ...)
a1.metric("F 통계량", f"{anova['f_statistic']:.2f}")  # list는 dict 키 접근 불가 → 오류

# 현재 AnovaResult.post_hoc = str (구조화 불가)
```

**Sprint 3 수정 계획**:
1. `schemas.py`: `AnovaResult.post_hoc: str` → `post_hoc: list[TukeyRow]` 변경 (박지민)
2. `schemas.py`: `TukeyRow` 신규 스키마 추가 (박지민)
3. `dashboard/app.py` `anova_page()`: `list[AnovaResult]` 처리로 수정 (강유영)
   - `strategy_comparison` 항목을 대표 지표(F, p, η²) 표시에 사용
   - 실험 선택 탭 또는 selectbox 추가

```python
# Sprint 3 수정 후 대시보드 예시
anova_list: list = bt5.get("anova", [])
strategy_anova = next(
    (a for a in anova_list if a["name"] == "strategy_comparison"), {}
)
a1.metric("F 통계량", f"{strategy_anova.get('f_statistic', 0):.2f}")
```

### 6-2. 대시보드 REQUEST_TIMEOUT

`apps/dashboard/app.py` `REQUEST_TIMEOUT = 10`초는 `/research` 호출 시 부족할 수 있다.

**수정 계획**: `REQUEST_TIMEOUT`을 엔드포인트별로 분리 (강유영)
```python
_TIMEOUT_DEFAULT: int = 10
_TIMEOUT_RESEARCH: int = 60   # LangGraph 루프 최대 3회
```

### 6-3. /backtest window 파라미터 (Sprint 3 추가)

현재 `GET /backtest`는 파라미터 없음. Sprint 3에서 `?window=w1|w2|w3|final` 추가 예정.  
대시보드도 사이드바 selectbox로 윈도우 선택 UI를 추가해야 한다.  
Walk-Forward 윈도우는 총 4개(`src/rl/backtest.py`의 `WINDOWS` 상수 참고).

---

*최종 업데이트: 2026-05-11 / 작성: 강유영*
