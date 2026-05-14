# 레이블·태그·RL 공유 인터페이스 정의

> **버전**: v1.0 (Sprint 3 기준)  
> **작성일**: 2026-05-12  
> **대상**: 박지민(백엔드), 강유영(분석·대시보드), 이문정(RL)

팀 전체가 공유하는 레이블, 태그, RL 인터페이스를 한 곳에 정리한다.  
변경 시 **팀 합의 후 이 문서를 먼저 수정**하고, 코드에 반영한다.

---

## 1. 시장 국면(Market Regime) 레이블

**정의 위치**: `src/rl/anova.py` — `_REGIME_SLICES`  
**데이터 파일**: `data/processed/regime_labels.csv` (별도 파일, features.parquet에 컬럼 추가 안 함)  
**활용 범위**: ANOVA 분석 + SHAP 분석 전용  
**ANOVA 검증 3**: regime × strategy **Two-way ANOVA** — Factor 1로 사용 (3개 국면, 균형 3×3 설계)  
**RL 관측공간 투입 금지**: 사후 라벨링(사후 정답지)이므로 미래 정보 누수에 해당

| 레이블 | 기간 | 설명 | 활용 |
|--------|------|------|------|
| `rate_hike` | 2022-01-01 ~ 2022-12-31 | 금리 급등, 주식·채권 동반 하락 (하락장) | **ANOVA** — OOS W1 테스트 구간 |
| `recovery` | 2023-01-01 ~ 2023-12-31 | 금리 인상 후 회복장 | **ANOVA** — OOS W2 테스트 구간 |
| `bull` | 2024-01-01 ~ 2024-12-31 | AI 랠리 강세장 | **ANOVA** — OOS W3 테스트 구간 |
| `crisis` | 2020-02-01 ~ 2020-05-31 | 코로나 폭락 | **SHAP 전용** — in-sample, ANOVA 제외 |

> **`rate_hike`가 이 프로젝트의 핵심 OOS 스트레스 구간**  
> W1 모델(학습 2018~2021)이 금리 인상 패턴을 본 적 없는 상태에서 2022년을 테스트.  
> 3개 국면(rate_hike / recovery / bull)은 각각 W1·W2·W3 테스트 구간에 1:1 대응하며,  
> PPO/MVO/EW 모두 해당 구간 백테스트 결과가 존재 → 결측 셀 없는 균형 3×3 설계.  
> `crisis`(코로나)는 in-sample 구간이라 PPO/MVO 결과 없음 → SHAP 전용으로 분리  
> (`shap.py` — crisis 날짜 필터링).

### regime_labels.csv 포맷

```
date,regime
2018-01-03,normal
...
2020-02-20,crisis
2020-02-21,crisis
...
2020-05-31,crisis
2022-01-03,rate_hike
...
2022-12-30,rate_hike
2023-01-02,recovery
...
2023-12-29,recovery
2024-01-02,bull
...
2024-12-31,bull
```

### 데이터 파이프라인 내 위치

```
[returns/features.parquet 생성]  ← 박지민
        ↓
[regime_labels.csv 생성]          ← 날짜 기반 생성 (강유영, src/rl/anova.py)
        ↓
[RL 학습] features.parquet만 사용  ← regime 레이블 투입 안 함
        ↓
[백테스트] PPO 결정 + 수익률 기록
        ↓
[ANOVA]  backtest 결과에 regime JOIN → Two-way 분석
[SHAP]   crisis 날짜만 필터링 → 위기 구간 피처 중요도 분석
```

---

## 2. 리스크 태그(Risk Tags)

### 2-1. 일반 리스크 태그 (RAG 리포트용)

**정의 위치**: `src/agent/risk_tags.py` — `RISK_KEYWORD_MAP`  
**활용**: `POST /research` 응답의 `risk_tags` 필드

| 태그 | 심각도 | 트리거 키워드 (예시) |
|------|--------|---------------------|
| `지정학_리스크` | HIGH | 지정학, 전쟁, 분쟁, 제재 |
| `경기침체_리스크` | HIGH | 침체, 리세션, 경기둔화 |
| `신용_리스크` | HIGH | 부도, 파산, 디폴트 |
| `금리_리스크` | MEDIUM | 금리, 기준금리 |
| `인플레이션_리스크` | MEDIUM | 인플레이션, 물가 |
| `유동성_리스크` | MEDIUM | 유동성 |
| `시장_리스크` | MEDIUM | 하락, 폭락, 급락 |
| `환율_리스크` | LOW | 환율, 달러 |
| `규제_리스크` | LOW | 규제, 법안 |
| `불확실성_리스크` | LOW | 불확실성 |
| `변동성_리스크` | LOW | 변동성 |

### 2-2. RL 관측공간 연동 태그 (3종 고정)

**정의 위치**: `src/agent/risk_tags.py` — `RL_RISK_TAGS`  
**활용**: `PortfolioEnv` 관측공간 벡터 (이문정 담당)  
**변경 시**: 이문정과 반드시 사전 협의 필요 (env.py 관측공간 차원 변경 연동)

| 순서 | 태그 | 트리거 키워드 |
|------|------|--------------|
| 0 | `규제변경` | 규제변경, 규정 변경, 규제 강화, 법 개정, 금융 규제, 규제 개편 |
| 1 | `실적쇼크` | 실적쇼크, 어닝쇼크, 실적 충격, 실적 부진, 영업손실 |
| 2 | `급등락` | 급등락, 급등, 급락, 폭등, 폭락, 급변동 |

**벡터 변환 예시**:
```python
from src.agent.risk_tags import get_risk_vector, extract_rl_risk_tags

tags = extract_rl_risk_tags("금융당국이 규제를 강화했다.")  # → ["규제변경"]
vec  = get_risk_vector(tags)   # → array([1., 0., 0.], dtype=float32)
#                                         규제변경 실적쇼크 급등락
```

---

## 3. RL 공유 인터페이스

### 3-1. PortfolioEnv 관측공간 구조

**정의 위치**: `src/rl/env.py`  
**담당**: 이문정

| 구성 요소 | 차원 | 내용 |
|----------|------|------|
| 과거 수익률 윈도우 | `lookback × n_assets` | `features_df`의 `{ticker}_return` (lookback=30 기본) |
| 현재 포트폴리오 비중 | `n_assets` | 이전 step의 `weights` (env 내부 상태) |
| RSI | `n_assets` | `features_df`의 `{ticker}_RSI` |
| MACD signal | `n_assets` | `features_df`의 `{ticker}_MACD_signal` |
| **risk_vector** | `3` | RL_RISK_TAGS 3종 (규제변경 / 실적쇼크 / 급등락), `set_risk_vector()`로 갱신 |
| **합계** | `(lookback + 3) × n_assets + 3` | → `(30 + 3) × 10 + 3 = 333` 차원 |

> `obs_dim = (lookback + 3) * n_assets + 3` (`env.py`)  
> features_df 컬럼 네이밍: `{ticker}_return`, `{ticker}_RSI`, `{ticker}_MACD_signal`

**액션 공간**: `Box(low=0.0, high=1.0, shape=(n_assets,))` — softmax 정규화 후 포트폴리오 비중으로 사용

### 3-2. 보상함수 종류

**정의 위치**: `src/rl/env.py` — `reward_type`

| 값 | 설명 |
|----|------|
| `"return"` | 일별 포트폴리오 로그수익률 |
| `"sharpe"` | 롤링 샤프 비율 (volatility_window=20 기본) |
| `"mdd"` | 로그수익률 − lambda_mdd × MDD 페널티 (lambda_mdd=1.0 기본) |

### 3-3. 모델 파일 네이밍 규칙

**저장 경로**: `models/`  
**형식**: `ppo_{reward}_{window}_risk.zip` (`_risk` = obs_dim 333 환경으로 학습된 모델)  
**총 12개** (3 보상함수 × 4 윈도우)

| reward \ window | w1 | w2 | w3 | final |
|----------------|----|----|-----|-------|
| `return` | ppo_return_w1_risk.zip | ppo_return_w2_risk.zip | ppo_return_w3_risk.zip | ppo_return_final_risk.zip |
| `sharpe` | ppo_sharpe_w1_risk.zip | ppo_sharpe_w2_risk.zip | ppo_sharpe_w3_risk.zip | ppo_sharpe_final_risk.zip |
| `mdd` | ppo_mdd_w1_risk.zip | ppo_mdd_w2_risk.zip | ppo_mdd_w3_risk.zip | ppo_mdd_final_risk.zip |

### 3-4. Walk-Forward 윈도우 정의

**정의 위치**: `src/rl/backtest.py` — `WINDOWS`

| window | 학습 기간 | 테스트 기간 | 특징 |
|--------|----------|------------|------|
| `w1` | 2018-01-01 ~ 2021-12-31 | 2022-01-01 ~ 2022-12-31 | 금리 충격 (핵심 OOS) |
| `w2` | 2019-01-01 ~ 2022-12-31 | 2023-01-01 ~ 2023-12-31 | 회복장 |
| `w3` | 2020-01-01 ~ 2023-12-31 | 2024-01-01 ~ 2024-12-31 | AI 랠리 |
| `final` | 2021-01-01 ~ 2024-12-31 | 2025-01-01 ~ 2025-12-31 | 최신 구간 |

### 3-5. Safe-Guard 임계값

**정의 위치**: `src/rl/env.py` (`safe_guard_mdd = 0.15`), `src/rl/backtest.py` (`SAFEGUARD_THRESHOLD = 0.15`)

| 항목 | 값 | 근거 |
|------|-----|------|
| MDD 임계값 | **15%** | 코로나 KOSPI MDD 38%의 절반 이하 (조기 경보 수준) |
| 발동 조건 | MDD ≥ 15% | backtest.py `_detect_safeguard_events()` |
| 해제 조건 | MDD < 7.5% (임계값 × 0.5) | 회복 확인 후 재개 |

### 3-6. 거래 비용 설정

**정의 위치**: `src/rl/env.py`

| 항목 | 값 |
|------|----|
| 거래비용 (fee_rate) | 0.015% |
| 슬리피지 (slippage_rate) | 0.05% |
| 합계 (total_cost_rate) | 0.065% |

---

## 4. 데이터 파일 인터페이스

**상세 내용**: `docs/data_interface.md` 참고

| 파일 | 컬럼 구성 | 담당 |
|------|----------|------|
| `data/processed/returns.parquet` | 날짜 index + 10개 자산 티커 컬럼 (로그수익률) | 박지민 |
| `data/processed/features.parquet` | 날짜 index + `{ticker}_return`, `{ticker}_RSI`, `{ticker}_MACD_signal` (40컬럼) | 박지민 |
| `data/results/backtest_{reward}.csv` | `date`, `episode_return` | 강유영 (backtest.py 생성) |
| `data/results/weights_{reward}.parquet` | 날짜 index + 10개 자산 비중 컬럼 | 강유영 (backtest.py 생성) |

---

*최종 업데이트: 2026-05-12 / 작성: 강유영*
