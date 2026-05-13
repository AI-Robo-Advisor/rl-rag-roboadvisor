# API Contract Design

> Branch: `feature-j-api-contract`  
> Date: `2026-05-13`  
> Scope: `apps/api` contract design only. No implementation changes are assumed by this document.

## 1. Goal

`docs/과제명세서.md`를 최우선 기준으로 두고, `apps/api`의 HTTP 계약을 `schemas.py`, `routers`, `services` 관점에서 정리한다. 이번 문서는 구현 전 검토용 설계 문서이며, 이후 `docs/api_spec.md`와 코드 반영의 기준으로 사용한다.

## 2. Contract Authority Order

1. `docs/과제명세서.md`
2. `docs/labels_and_interfaces.md`
3. `docs/api_spec.md`
4. `apps/dashboard/app.py`의 실제 API 호출 및 mock payload
5. `apps/api/*.py` 현재 구현

이 순서를 적용하는 이유는 과제 요구사항이 팀 내부 임시 구현보다 우선이고, RL/RAG 공유 인터페이스 문서가 API 세부 설계보다 상위 의도를 담고 있기 때문이다.

## 3. Current-State Summary

현재 `apps/api`는 모든 핵심 엔드포인트를 노출하고 있지만, 계약 수준에서는 아직 Sprint 2 fallback 구조에 머물러 있다.

- `/health`
  - 동작 중.
  - readiness semantics는 단순 파일/환경변수 기반이다.
- `/optimize`
  - deterministic fallback 반환 구조는 이미 존재한다.
  - 실제 PPO 어댑터 경계는 아직 문서화만 되어 있다.
- `/explain`
  - fallback 설명 응답 존재.
  - SHAP 모듈 연결 시 반환 shape는 추가 정규화가 필요하다.
- `/research`
  - fallback + LangGraph lazy run 구조 존재.
  - `reasoning_trace`, `risk_tags`, `sources` key 정합성은 비교적 안정적이다.
- `/backtest`
  - metrics 시계열은 fallback/partial-real 혼합 구조다.
  - `anova` shape가 문서와 코드 사이에서 가장 크게 어긋난다.

## 4. Confirmed Mismatches

### 4-1. Backtest ANOVA Schema

현재 코드:
- `AnovaResult.post_hoc: str`

문서와 dashboard mock:
- `post_hoc: list[TukeyRow]`
- optional `interaction`
- optional `strategy_effect`

결론:
- `post_hoc` 문자열 계약은 폐기한다.
- `/backtest` ANOVA는 구조화된 nested schema로 고정한다.

### 4-2. Window Query Parameter

현재 코드:
- `/backtest` 파라미터 없음

문서:
- `window=w1|w2|w3|final`

결론:
- `window`는 `final` 기본값의 typed query parameter로 설계한다.
- fallback에서도 파라미터를 무시하지 않고 echo 또는 적용 가능한 범위에서 반영하는 구조가 바람직하다.

### 4-3. Dashboard Label Drift

현재 API 문서와 코드:
- benchmark는 `SPY` 중심

현재 dashboard 표시:
- 일부 문구는 `KOSPI`를 전제

결론:
- API 계약의 benchmark literal은 `SPY`를 유지한다.
- dashboard 표현은 별도 표시 로직에서 해결한다.

## 5. Endpoint Design Decisions

### 5-1. GET `/health`

유지 원칙:
- 응답 shape는 현행 유지.
- `modules`는 `data`, `rl`, `rag`, `shap`, `backtest` 5개 고정.

설계 결정:
- `ready`는 단순 import 가능 여부가 아니라 "API가 해당 엔드포인트에서 실제 값을 반환할 준비가 되었는가" 기준으로 해석한다.
- 따라서 `shap`, `rl`, `backtest`는 실제 파일/모듈/결과물 연결 조건이 명시되어야 한다.

### 5-2. POST `/optimize`

유지 원칙:
- 현재 request/response shape는 크게 변경하지 않는다.

설계 결정:
- `risk_aversion`이 있으면 `risk_profile`보다 우선한다.
- `weights`는 항상 합이 1.0인 normalized dict다.
- benchmark series는 optimize 단계에서도 `SPY` 기준을 유지한다.

### 5-3. POST `/explain`

유지 원칙:
- 현재 `feature_names`, `shap_values`, `feature_contributions` 3중 표현은 dashboard 호환성을 위해 유지한다.

설계 결정:
- `target_date`는 실제 사용된 거래일이다.
- SHAP upstream이 더 풍부한 구조를 주더라도 API는 현재 단순화된 shape로 정규화한다.

### 5-4. POST `/research`

유지 원칙:
- fallback과 ready shape를 동일하게 유지한다.

설계 결정:
- `risk_tags`는 최종적으로 `docs/labels_and_interfaces.md`의 RL 연동 3종 태그와 합치되, 일반 리포트 태그 확장은 별도 논의 대상으로 남긴다.
- `sources`는 비어 있지 않도록 보장한다.
- `reasoning_trace`는 plain string contract를 유지한다.

### 5-5. GET `/backtest`

설계 결정:
- request surface에 `window` 추가
- response는 아래 nested schema를 사용
  - `metrics`
  - `anova: list[AnovaResult]`
  - `safeguard`
  - 시계열 배열
- `w1`이 2022 금리 충격 핵심 OOS 구간을 대표하므로 별도 `stress` 엔드포인트는 두지 않는다.
- `metrics` key set은 현재 문서의 12개를 표준으로 본다.
- `anova`는 정확히 3개 실험명을 표준 식별자로 사용한다.

## 6. Schema Design

### 6-1. Keep As-Is

- `RiskProfile`
- `EndpointStatus`
- `ApiStatus`
- `HealthResponse`
- `OptimizeRequest`
- `OptimizeResponse`
- `ExplainRequest`
- `ExplainResponse`
- `ResearchRequest`
- `ResearchResponse`

### 6-2. Add Or Refine

- `TukeyRow`
- `InteractionStats`
- `StrategyEffectStats`
- `AnovaResult`
- `SafeguardState`
- optional `BacktestWindow` literal

### 6-3. Target Backtest Schema Shape

```python
BacktestWindow = Literal["w1", "w2", "w3", "final"]

class TukeyRow(BaseModel):
    group1: str
    group2: str
    meandiff: float
    p_adj: float
    reject: bool

class InteractionStats(BaseModel):
    f_statistic: float
    p_value: float
    significant: bool

class StrategyEffectStats(BaseModel):
    f_statistic: float
    p_value: float

class AnovaResult(BaseModel):
    name: str
    f_statistic: float
    p_value: float
    eta_squared: float
    post_hoc: list[TukeyRow]
    interaction: InteractionStats | None = None
    strategy_effect: StrategyEffectStats | None = None
```

이 구조를 쓰는 이유:
- dashboard mock과 `docs/api_spec.md`가 이미 nested shape를 기대한다.
- `dict | None`보다 typed model이 더 안전하고 추후 회귀를 줄인다.

## 7. Router Responsibilities

router는 아래 책임만 가진다.

- path/query/body surface 선언
- response model 선언
- service 호출
- FastAPI validation 위임

router가 가지면 안 되는 책임:
- fallback payload 조립
- parquet/모델 파일 직접 로드
- RL/RAG/SHAP 모듈 import 분기

정리 기준:
- `/backtest` router는 `window` query를 받는다.

## 8. Service Responsibilities

service는 논리적으로 아래 세 층으로 정리한다.

### 8-1. Readiness Layer

- health용 모듈 상태 판정
- 파일 존재/모듈 import/환경변수 체크

### 8-2. Fallback Builder Layer

- deterministic fallback payload 생성
- schema shape를 실제 구현과 동일하게 유지

### 8-3. Runtime Adapter Layer

- `src.agent.graph`
- `src.rl.metrics`
- `src.rl.anova`
- `src.rl.backtest`
- 향후 `src.rl.shap`

upstream 모듈이 어떤 내부 shape를 주더라도, service에서 API schema로 normalize한다.

## 9. Out Of Scope For This Round

- `docs/api_spec.md` 직접 수정
- `apps/api/schemas.py` 실제 코드 변경
- `apps/api/routers/*` 실제 경로 추가
- `apps/api/services.py` 리팩토링
- dashboard 구현 수정

이번 라운드는 설계 확정이 목적이다.

## 10. Proposed Next Execution Order

1. 이 문서 검토 및 승인
2. `docs/api_spec.md`와 `docs/labels_and_interfaces.md`에 설계 반영
3. `apps/api/schemas.py` 반영
4. `routers` surface 반영
5. `services.py` 책임 분리 및 fallback shape 정렬
6. `CLAUDE.md` 상태 갱신
