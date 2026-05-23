# 설계 기준서 (Design Baseline) — 초안

> **상태**: 초안 / 승인 대기
> **작성일**: 2026-05-23
> **목적**: "이슈 문서가 잘못되었을 수 있다"는 전제 하에, 코드·문서·이슈를 1:1로 비교해
> (1) 무엇이 사실인지, (2) 어디서 충돌하는지, (3) 어떤 선택지가 있는지를 정리한다.
> **이 문서가 승인되기 전까지 코드 변경 금지**.

## 0. 원칙

- **사실(Fact)**: 코드와 산출물에서 직접 확인된 동작만 적는다.
- **충돌(Conflict)**: "이슈가 기대한 동작" vs "코드가 실제 하는 동작" vs "문서가 적은 약속"의 차이를 명시한다.
- **결정(Decision)**: 사람(=프로젝트 오너)이 선택해야 하며, 본 문서는 옵션 A/B만 제시한다.
- **검증(Verification)**: 결정이 적용된 뒤 어떤 테스트로 확증할지 함께 적는다.

---

## 1. SHAP 호출 인터페이스 — 이슈 vs 코드

### Fact

- `src/rl/shap.py`의 `generate_summary_plot()` 시그니처에는 `scaler_path` **없음**.
  외부에서 미리 정규화된 `features_df`를 받는다.
- `src/rl/train_walkforward.py`/`backtest.py`는 `raw_features.parquet` + window별 `scalers/{w}_feature_stats.json`로 정규화한 뒤 환경/모델에 넘긴다.
- `docs/임시_유영_분석보고서.md` §5는 "각 window scaler로 정규화한 features_df를 SHAP에 사용"이라고 명시.

### Conflict

- `prompt.md` 70–118행의 "이슈 1"은 `generate_summary_plot(..., scaler_path=...)` 호출을 예시로 제시 → **실제 함수에 그 파라미터가 없다**.
- 즉 이슈 자체가 함수 인터페이스를 잘못 기술. 코드/보고서는 일관됨.

### Decision options

- **A. 이슈 문서를 코드 기준으로 정정** (권장)
  - `prompt.md`의 SHAP 예제를 "외부 정규화 → features 전달"로 수정하고, `scaler_path` 언급 삭제.
  - 코드 변경 0.
- **B. 코드에 `scaler_path` 인자를 추가**해서 이슈 예제대로 동작하게 변경
  - 함수 시그니처 확장, 내부에서 scaler 로드/정규화.
  - 호출처 전부 갱신 필요. 임시 보고서 §5 호환 점검 필요.

### Verification

- A: 문서 diff만. 별도 테스트 불필요.
- B: 새 인자 사용/미사용 두 경로 모두 `tests/test_shap.py`에 케이스 추가, 기존 결과와 수치 일치 확인.

---

## 2. reasoning context → SHAP 연결

### Fact

- `src/rl/shap.py`에 `_load_reasoning_context()`가 이미 존재(임시 보고서 §5-3, §1 체크리스트).
- `compute_shap_explanation()` 반환에 `reasoning_context`가 들어있고, `data/results/shap_force_*.reasoning.json` 사이드카가 9 시점 생성됨.
- 데이터 소스는 `data/processed/unified_events.parquet`의 `reasoning` 컬럼.

### Conflict

- `prompt.md` 120–130행의 "이슈 2"는 이 작업을 "추가 필요"라고 표기 → **이미 구현·산출물 생성 완료 상태와 불일치**.

### Decision options

- **A. 이슈 문서에 "완료"로 표기 + 산출물 경로 명시** (권장)
- **B. 이슈 자체 삭제**(완료된 항목은 prompt.md에 남기지 않음)

### Verification

- 어느 쪽이든 코드 변경 없음. `tests/test_shap.py`가 `reasoning_context` 키를 검증하는지만 확인 후, 누락 시 케이스 1개 추가.

---

## 3. backtest CSV의 윈도우 분리

### Fact

- `src/rl/backtest.py`는 4 윈도우 결과를 단일 CSV로 concat 저장(`backtest_{reward}.csv`, `date` 인덱스).
- `data/results/backtest_metrics.csv`는 12행(3 reward × 4 window) 별도 저장. 임시 보고서 §3-1 표 A가 이를 사용.
- 보고서/대시보드는 날짜 슬라이싱(`loc['2022-01-01':'2022-12-31']`)으로 W1을 분리.

### Conflict

- `prompt.md` 131–141행의 "이슈 3"은 "W1 행만 유효"라는 전제 + CSV 분리 코드 누락을 지적 → **현재 보고서는 4 윈도우 모두 유효 분석 완료** 상태.
- 즉 "W1만 유효"라는 전제 자체가 더 이상 맞지 않음(설계 진척으로 폐기됨).

### Decision options

- **A. 이슈 폐기 + 보고서 §2(윈도우 정의)와 §3-1(표 A) 링크로 대체** (권장)
- **B. backtest.py에 윈도우별 분리 저장 기능 추가** (`backtest_{reward}_{window}.csv`)
  - 다운스트림(보고서/대시보드/테스트) 영향 점검 필요.

### Verification

- A: 문서 diff만.
- B: `tests/test_backtest.py`에 분리 파일 생성 검증 추가, 임시 보고서 표 A 재현 가능 여부 확인.

---

## 4. 리스크 태그 이름 — Full form vs Short form (TODO 1의 핵심)

### Fact (현재 코드)

| 위치 | 이름 형식 | 예시 |
|------|----------|------|
| In-memory RL 태그 (`RL_RISK_TAGS`) | **Full** (`_risk` 접미사) | `macro_rate_risk`, `equity_market_risk`, `geopolitical_fx_risk` |
| Raw event parquet (`unified_events.parquet`, `manual_seed_events.parquet` 등) | **Full** | `macro_rate_risk`, `equity_market_risk`, `geopolitical_fx_risk` (severity 0~1) |
| Daily decay parquet (`risk_vectors_daily.parquet`) | **Short** | `risk_macro`, `risk_equity`, `risk_geo` |
| `PortfolioEnv` obs 순서 (3차원) | **Full** 기준 | `[macro_rate_risk, equity_market_risk, geopolitical_fx_risk]` |
| SHAP 피처명(임시 보고서 §5-2) | **Short(`risk_*`)** | `risk_equity_market_risk` 등 (env 내부 명명 다름) |
| 문서 (`spec_tag_mapping.md`) | 이중 명시 | §1(raw)=Full, §3(daily)=Short |

### Conflict (= "이슈 자체가 잘못되었을 가능성")

- 직전 TODO("RL_RISK_TAGS *_risk, keywords, normalize alias")는 이 **이중 명명을 한쪽으로 통일**해야 한다고 전제.
- 그러나 코드와 문서를 같이 보면 **이중 명명은 의도된 분리**일 수 있음:
  - Full(`macro_rate_risk`): 이벤트 강도(severity), 0/0.33/0.66/1.0 값.
  - Short(`risk_macro`): 일별 누적 decay 스코어, 0~1 연속값.
- 둘은 의미·단위·생애주기가 다르므로 같은 이름을 쓰면 오히려 혼동.

### Decision options

- **A. 이중 명명 유지(의도된 설계로 확정)** — **권장 후보**
  - `risk_tags.py`의 `RL_RISK_TAGS`는 Full 그대로.
  - 문서 `labels_and_interfaces.md`·`spec_tag_mapping.md`에 "Full=이벤트 severity / Short=일별 decay"라는 의미 분리를 1문단으로 추가.
  - 코드 변경 없음. 직전 TODO는 "불필요한 통일 작업"으로 취소.
- **B. Full로 완전 통일**
  - `risk_vectors_daily.parquet` 컬럼명을 `risk_macro` → `macro_rate_risk` 등으로 변경.
  - 영향 범위(grep 결과 기준): `build_risk_parquet.py`, `train_walkforward.py`, `env.py`, `shap.py`, `services.py`, `dashboard/*`, `tests/test_build_risk_parquet.py`, `tests/test_risk.py`, 문서 4종.
  - 기존 parquet 산출물 재생성 필요(이미 커밋된 `risk_vectors_daily.parquet`도 재빌드).
- **C. Short로 완전 통일**
  - `RL_RISK_TAGS = ["risk_macro", ...]`로 변경.
  - 영향 범위 B와 거의 동일하지만 RL/PortfolioEnv 인터페이스 변경이 추가됨(가장 큰 영향).

### Verification

- A: 의미 분리 문단 추가 후 `pytest tests/test_risk.py tests/test_build_risk_parquet.py tests/test_shap.py`가 그대로 통과하는지만 확인.
- B/C: 이름 변경 PR 단위로 진행 후 동일 테스트 + `python scripts/build_risk_parquet.py --sanity` 통과 확인.

---

## 5. 워크트리 상태(부수 이슈)

### Fact

- `feature/u-analysis-v2`에 미추적 파일 다수: `data/processed/*.jsonl/json`, `data/raw/ecos/*`, `data/raw/manual_seed/*`, `data/results/anova_results.json`, `backtest_*.csv`, `shap_explanations.json`, `shap_force_*.html/.reasoning.json`, `data/results_backup_pre_analysis_20260523/`, `docs/캡디과제설명서.md`.
- 수정/삭제: `docs/rag_eval_results.md`(M), `docs/임시_유영_분석보고서.md`(M), `src/rl/shap.py`(M), `docs/과제명세서.md`(D).

### Conflict

- 이슈/보고서가 참조하는 산출물 일부는 아직 git에 올라오지 않음 → 다른 팀원이 재현 불가.

### Decision options

- **A. 산출물별로 분리 커밋**(권장): (1) `data/results/*` 분석 산출물, (2) `data/raw/{ecos,manual_seed}`, (3) `data/processed/llm_batch_*`, (4) docs 변경.
- **B. 일괄 한 커밋**: 추적 단순화 우선.
- **C. 일부 산출물은 `.gitignore` 처리** (예: 백업 디렉토리 `data/results_backup_pre_analysis_20260523/`).

### Verification

- 어느 쪽이든 `git status`가 의도한 상태(추적/무시)로 떨어지는지 확인.

---

## 6. 종합 — 결정 요약 폼

아래 표에 사용자(=프로젝트 오너)가 직접 X 표시 후 회신.

| 항목 | A | B | C | 비고 |
|------|---|---|---|------|
| 1. SHAP 인터페이스 | [ ] | [ ] | — | A=문서 정정 / B=함수 확장 |
| 2. reasoning 연결 표기 | [ ] | [ ] | — | A=완료 표기 / B=이슈 삭제 |
| 3. backtest 윈도우 분리 | [ ] | [ ] | — | A=문서 정정 / B=파일 분리 추가 |
| 4. 리스크 태그 명명 | [ ] | [ ] | [ ] | A=이중 유지 / B=Full 통일 / C=Short 통일 |
| 5. 워크트리 정리 | [ ] | [ ] | [ ] | A=분리 커밋 / B=일괄 / C=일부 ignore |

> **승인 시 다음 작업 순서**
> 1. 본 문서 커밋(설계 기준 동결).
> 2. 선택된 옵션만 코드/문서에 반영(옵션별 분리 커밋).
> 3. 영향 테스트(`pytest` 해당 파일) 통과 확인.
> 4. PR 본문에 본 문서 링크.

