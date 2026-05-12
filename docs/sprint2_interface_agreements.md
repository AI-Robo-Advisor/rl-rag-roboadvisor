# Sprint 2 인터페이스 협의 정리

> 작성자: 강유영 | 작성일: 2026-04-30 | 대상: 박지민, 이문정

---

## 1. 박지민 → 강유영 질문 답변

### `/research` — LangGraph 에이전트 호출

#### Q1. agent 호출 함수 시그니처가 뭐야?

**함수**: `run_graph(query: str) -> dict`  
위치: `src/agent/graph.py`

```python
# 호출 예시
from src.agent.graph import run_graph

result = run_graph("삼성전자 최근 투자 리스크 분석해줘")
```

**input**: 질문 string 하나만 넘기면 됨  
**output**:

```python
{
    "report": str,          # 최종 투자 분석 리포트 (markdown 형식)
    "sources": List[str],   # 참고 뉴스 URL 목록
    "reasoning_trace": str, # 내부 추론 과정 로그 (THINK: 접두사)
    "risk_tags": List[str], # ["급등락", "실적쇼크"] 등 감지된 리스크 태그
}
```

#### Q2. sources 배열 안의 구조가 뭐야?

`sources`는 **`List[str]`** (URL 문자열 목록)

```python
result["sources"]
# → ["https://news.google.com/...", "https://..."]
```

title, date 같은 추가 key는 현재 없음. API 응답 포맷 구성 시 URL만 사용하거나,
필요하면 ChromaDB 메타데이터에서 `title`, `date` 추가 추출 가능 (강유영에게 요청).

#### Q3. `/research` 비동기 처리 방식

**권장: BackgroundTasks + 폴링**

```
POST /research  →  { task_id: "abc123" }
GET  /research/abc123  →  { status: "running"|"done", result: {...} }
```

이유:
- LangGraph Self-Correction 루프가 최대 3회 반복 → 응답 시간 10~30초 가능
- `st.write_stream()` 스트리밍은 FastAPI SSE 구현 필요 → Sprint 2에서는 복잡도 높음
- 폴링 방식이면 Streamlit에서 `st.spinner()` + `time.sleep(2)` 루프로 구현 가능

박지민이 FastAPI에서 어떤 방식으로 구현할지 결정한 뒤 강유영에게 알려줄 것.

---

### `/backtest` — metrics.py + anova.py 호출

#### Q1. backtest.py / metrics.py 실행 함수 시그니처가 뭐야?

**함수**: `calculate_all_metrics(returns, benchmark) -> dict`  
위치: `src/rl/metrics.py`

```python
import pandas as pd
from src.rl.metrics import calculate_all_metrics

returns   = pd.Series(...)   # 포트폴리오 일별 로그수익률
benchmark = pd.Series(...)   # 벤치마크(KOSPI 등) 일별 로그수익률

result = calculate_all_metrics(returns, benchmark)
```

**output** (12개 지표):

```python
{
    "cumulative_return":     float,  # 누적 수익률
    "cagr":                  float,  # 연평균 성장률
    "annualized_volatility": float,  # 연간 변동성
    "var_95":                float,  # 95% VaR
    "cvar_95":               float,  # 95% CVaR
    "mdd":                   float,  # 최대 낙폭
    "sharpe_ratio":          float,  # 샤프 비율
    "sortino_ratio":         float,  # 소르티노 비율
    "calmar_ratio":          float,  # 칼마 비율
    "alpha":                 float,  # 알파
    "beta":                  float,  # 베타
    "information_ratio":     float,  # 정보 비율
}
```

`years` 파라미터는 내부에서 `len(returns) / 252`로 자동 계산.

#### Q2. ANOVA 결과 구조가 뭐야?

**제안하는 출력 포맷** (`src/rl/anova.py`에서 반환):

```python
{
    "f_stat":     float,   # F-통계량
    "p_value":    float,   # p-value (< 0.05이면 유의미한 차이)
    "eta_squared": float,  # 효과 크기 (η²)
    "tukey_hsd":  [        # 사후검정 쌍별 비교
        {
            "group1": str,   # e.g. "PPO-return"
            "group2": str,   # e.g. "PPO-sharpe"
            "meandiff": float,
            "p_adj": float,
            "reject": bool,
        },
        ...
    ],
    "groups": {             # 그룹별 기술통계
        "PPO-return":  {"mean": float, "std": float, "n": int},
        "PPO-sharpe":  {"mean": float, "std": float, "n": int},
        "PPO-mdd":     {"mean": float, "std": float, "n": int},
    }
}
```

> anova.py 구현 담당이 강유영이므로 위 포맷대로 구현 예정.  
> 박지민은 FastAPI `/backtest` 응답에 `metrics`와 `anova` 두 key를 포함시켜 줄 것.

---

## 2. 강유영 ↔ 이문정 인터페이스 협의

### 2-1. 리스크 태그 → RL 관측공간 연동 (핵심)

#### 강유영 쪽 구현 현황

`src/agent/risk_tags.py`에 RL 전용 3-태그 시스템 이미 구현됨.  
태그 종류·순서·벡터 변환 예시 → `docs/labels_and_interfaces.md` 2-2 참고.

#### 이문정에게 물어볼 것

1. **관측공간 shape 변경 수락 여부**  
   - 현재: `(330,)` = `(30+3) × 10`  
   - 변경 후: `(333,)` = 기존 330 + 리스크 벡터 3  
   - 리스크 벡터는 관측벡터 **맨 뒤**에 붙이는 것을 제안  
   - shape 변경 시 모델 재학습 필요하므로 이문정 판단 필요

2. **학습 시 날짜별 리스크 태그 매핑**  
   - 강유영이 `data/processed/risk_tags.parquet` (날짜-태그 매핑 파일) 생성 가능  
   - 이문정이 `env.py`에서 현재 step 날짜에 해당하는 리스크 벡터를 불러와 관측에 추가  
   - 이 방식이 괜찮은지 확인 필요

3. **Sprint 2에서는 연동 생략 가능**  
   - `get_risk_vector()` API는 완성된 상태  
   - Sprint 3에서 연동해도 무방하나, 포맷만 지금 합의해두면 충돌 방지

#### 합의 필요 사항 요약

| 항목 | 강유영 제안 | 이문정 확인 필요 |
|------|------------|----------------|
| 리스크 벡터 포맷 | `np.ndarray shape=(3,) float32` | OK 여부 |
| 관측공간 위치 | 기존 벡터 맨 뒤 append | OK 여부 |
| 학습 데이터 매핑 | `risk_tags.parquet` (날짜→벡터) | env.py 수정 가능한지 |
| Sprint 적용 시기 | Sprint 3 이후로 미뤄도 OK | 동의 여부 |

---

### 2-2. ANOVA 입력 데이터 — 이문정이 강유영에게 줘야 할 것

#### 강유영이 필요한 파일

**ANOVA 검증 1**: 보상함수 3종 비교

```
data/results/backtest_return.csv
data/results/backtest_sharpe.csv
data/results/backtest_mdd.csv
```

각 CSV 포맷:
```
date,       episode_return
2024-01-02, 0.0023
2024-01-03, -0.0011
...
```

**ANOVA 검증 2**: Walk-Forward 백테스트 포트폴리오 비중

```
data/results/weights_return.parquet
```

포맷 (행: 날짜, 열: 자산 ticker):
```
           005930.KS  035720.KS  ...
2024-01-02  0.12       0.08      ...
2024-01-03  0.11       0.09      ...
```

#### 이문정에게 물어볼 것

1. **결과 파일을 위 포맷으로 저장해줄 수 있어?**  
   - 파일 경로: `data/results/` 디렉터리  
   - 보상함수 3종 각각에 대해 에피소드별 수익률 시계열 필요  

2. **Walk-Forward 결과가 현재 어디에 저장돼?**  
   - 강유영이 `backtest.py`를 구현하려면 포트폴리오 비중 시계열 필요  
   - 현재 `models/ppo_return_50k.zip`만 있는 상태  

3. **ANOVA 1번 기준 지표가 뭐야?**  
   - 에피소드별 최종 수익률? 아니면 샤프 비율?  
   - 강유영은 수익률 시계열을 받아서 `calculate_all_metrics()`로 12개 지표 모두 계산할 계획  

---

## 3. 전체 액션 아이템

| 협의 주제 | 강유영 할 일 | 이문정 할 일 | 박지민 할 일 |
|-----------|------------|------------|------------|
| `/research` 시그니처 | `run_graph()` 문서화 완료 ✅ | — | FastAPI `/research` 엔드포인트 구현 |
| `/backtest` 시그니처 | `calculate_all_metrics()` 문서화 완료 ✅ | `data/results/` 파일 생성 | FastAPI `/backtest` 엔드포인트 구현 |
| 비동기 처리 방식 | 폴링 방식 선호 의견 전달 | — | FastAPI 방식 결정 후 강유영에게 통보 |
| 리스크 태그 포맷 | `get_risk_vector()` 구현 완료 ✅ | 관측공간 수락 여부 + 재학습 계획 | — |
| ANOVA 입력 데이터 | `data/results/` 파일 포맷 명세 전달 | `data/results/*.csv,*.parquet` 생성 | — |
| 백테스트 연동 | `backtest.py` input 포맷 명세 전달 | 포트폴리오 비중 시계열 저장 방식 확정 | — |

---

## 4. 우선순위

1. **즉시 협의**: 리스크 태그 관측공간 연동 포맷 합의 (Sprint 3에서 구현해도 포맷은 지금 확정)
2. **이번 주**: 이문정이 `data/results/` 파일 경로·포맷 확정 → 강유영이 `anova.py` 구현 시작
3. **다음 주**: 박지민이 비동기 방식 결정 → 강유영이 대시보드 `/research` 탭 연동

---

> **주의**: 리스크 태그 관측공간 연동은 shape 변경을 수반하므로 **이문정이 모델 재학습 타이밍을 고려해야 함**.  
> Sprint 2에서 포맷만 문서화하고, 실제 연동은 Sprint 3에서 진행하는 것을 권장.
