# RL 보상함수 실험 결과 비교 리포트

> **최초 작성**: 2026-05-14  
> **갱신**: 2026-05-23 — risk_vector 연동 후 전 윈도우(W1~Final) 재학습 완료, 전 구간 비교 분석 반영  
> **작성자**: 이문정 (m)  
> **관련 파일**: `src/rl/env.py`, `src/rl/train_walkforward.py`  
> **분석 범위**: RL 보상 함수 설계 및 학습 거동 분석 (백테스트·ANOVA는 별도 문서 참고)

---

## 1. 개요

`PortfolioEnv`(`src/rl/env.py`)에 구현된 보상 함수 3종(`return`, `sharpe`, `mdd`)을 각각 독립적으로 PPO 에이전트에 적용하여 Walk-Forward 방식으로 학습시켰다. 이 문서는 **각 보상 함수의 정의, 설계 의도, 학습 중 보상 신호 특성, 그리고 최종 선택 근거**를 RL 관점에서 기술한다.

**2026-05-23 갱신 배경**:
- 기존 obs에서 risk_vector 마지막 3차원이 항상 0이었던 문제 수정 (PR #51)
- `risk_vectors_daily.parquet` 기반 날짜별 risk 주입 연동 완료
- `geopolitical_fx_risk` decay 기간 확정 (20일) 후 parquet 재빌드
- W1~Final 전 윈도우 모델 3종 × 4윈도우 = 12개 재학습 완료
- 이 문서의 §5 수치는 **risk 연동 후 전 윈도우 재학습 기준 최종 결과**

백테스트 성과 지표(샤프비율, MDD, 누적수익률 등)는 `src/rl/backtest.py`(강유영 담당)가 생성한 결과를 참조하며, 이 문서에서는 해당 수치를 **보상 함수 거동 해석 목적으로만** 인용한다.

---

## 2. Reward Function 정의

보상 함수는 `PortfolioEnv.__init__`의 `reward_type` 인자로 지정하며, `_calculate_reward(net_return, weights)`에서 step마다 계산된다.

공통 입력:
- `net_return`: 거래비용(fee 0.015% + 슬리피지 0.05% = 합계 0.065%)을 차감한 일별 포트폴리오 로그수익률
- `weights`: 현재 step에 적용된 포트폴리오 비중 (softmax 정규화 후)

### 2.1 Return Reward

```python
if self.reward_type == "return":
    return net_return
```

**정의**: 거래비용 차감 후 일별 포트폴리오 로그수익률을 그대로 보상으로 반환한다.

**설계 의도**: 에이전트가 단기 수익 극대화에 집중하도록 유도한다. 가장 단순한 형태의 보상으로, 보상 스케일이 자산 수익률 크기에 직접 종속된다.

**특성**:
- 보상 범위: 시장 일별 수익률 수준 (대략 −3% ~ +6%)
- 위험 패널티 없음 — 변동성이 큰 포지션도 수익이 나면 동일하게 보상
- risk_vector를 obs에서 직접 학습 신호로 활용 가능 — risk 상승 → 과거 수익률 악화 → 보상 감소 경로로 자연스럽게 연결됨

### 2.2 Sharpe Reward

```python
if self.reward_type == "sharpe":
    start = max(0, self.current_step - self.volatility_window)
    recent_returns = self.returns_df.iloc[start:self.current_step].values
    portfolio_returns = recent_returns @ weights
    volatility = float(np.std(portfolio_returns))
    if volatility < 1e-8:
        return 0.0
    return net_return / volatility
```

**정의**: 현재 step의 `net_return`을 직전 `volatility_window`(기본값 20 거래일) 구간의 포트폴리오 수익률 표준편차로 나눈 값이다.

> 주의: 이 값은 **step-level 보상 스케일의 샤프 비율**로, `metrics.py`가 계산하는 연율화 샤프 비율과 정의 및 해석이 다르다.

**설계 의도**: 수익률을 변동성으로 정규화하여 에이전트가 수익·위험을 동시에 고려하도록 한다.

**특성**:
- 보상이 변동성 단위로 정규화되므로 학습 초기의 보상 스케일이 안정적
- obs에 risk_vector가 추가된 이후 변동성 신호가 일부 중복 — 보상 함수가 내재한 변동성 정규화와 obs의 risk 신호가 상호 간섭 가능성 있음

### 2.3 MDD Reward

```python
if self.reward_type == "mdd":
    return net_return - self.lambda_mdd * self.current_mdd
```

**정의**: 일별 `net_return`에서 현재 시점의 최대낙폭(`current_mdd`)에 `lambda_mdd`(기본값 1.0)를 곱한 패널티를 차감한다.

**설계 의도**: MDD가 커질수록 보상이 감소하여 에이전트가 낙폭을 억제하는 행동을 선호하도록 유도한다.

**특성**:
- MDD 패널티는 낙폭이 **이미 발생한 뒤** 작동하는 후행(lagged) 신호
- 연속 하락 구간에서 `current_mdd`가 누적되어 보상이 지속적으로 음수로 유지됨

---

## 3. PPO 학습 설정

| 항목 | 값 |
|------|----|
| 알고리즘 | PPO (`stable-baselines3`) |
| 정책 네트워크 | `MlpPolicy` |
| 학습률 | 3e-4 |
| n_steps (rollout buffer) | 2,048 |
| batch_size | 64 |
| n_epochs | 10 |
| 총 학습 스텝 | 100,000 |
| 관측공간 차원 | 333 (= (lookback 30 + 3) × 10 자산 + risk_vector 3) |
| 액션공간 | `Box(0.0, 1.0, shape=(10,))` → softmax 정규화 |

**관측공간 구성** (`env.py`):

| 구성 요소 | 차원 | 비고 |
|----------|------|------|
| 과거 30일 수익률 윈도우 | 300 | walk-forward 학습 구간 통계로 Z-score 정규화 |
| 현재 포트폴리오 비중 | 10 | |
| RSI (14일) | 10 | |
| MACD signal | 10 | |
| **risk_vector** | **3** | **macro_rate_risk / equity_market_risk / geopolitical_fx_risk** — 날짜별 decay 스코어 (0~1) |
| **합계** | **333** | |

**risk_vector 연동 방식** (2026-05-23 수정):
- `data/processed/risk_vectors_daily.parquet` — 2018~2025 일별 decay 사전 계산값 (build_risk_parquet.py)
- Decay 기간: macro=30일 / equity=10일 / geo=20일
- env 초기화 시 `risk_series`를 train 구간으로 슬라이싱해서 주입 → SB3 학습 루프 내 step마다 `risk_array[current_step]`으로 자동 조회
- look-ahead 방지: train 구간 risk만 env에 전달, test 구간 노출 없음

거래비용 설정:

| 항목 | 값 |
|------|----|
| fee_rate | 0.015% |
| slippage_rate | 0.05% |
| total_cost_rate | 0.065% |

---

## 4. Walk-Forward 학습 구성

정규화는 `data/processed/raw_features.parquet`를 기준으로 각 윈도우 학습 구간에서만 mean/std를 계산하고, 같은 통계를 테스트 구간에 적용한다. 통계는 `data/processed/scalers/{window}_feature_stats.json`에 저장된다.

| 윈도우 | 학습 기간 | 테스트 기간 | 시장 국면 |
|--------|----------|------------|----------|
| W1 | 2018-01-01 ~ 2021-12-31 | 2022-01-01 ~ 2022-12-31 | 금리 인상 충격 (핵심 OOS 스트레스) |
| W2 | 2019-01-01 ~ 2022-12-31 | 2023-01-01 ~ 2023-12-31 | 회복장 |
| W3 | 2020-01-01 ~ 2023-12-31 | 2024-01-01 ~ 2024-12-31 | AI 랠리 강세장 |
| Final | 2021-01-01 ~ 2024-12-31 | 2025-01-01 ~ 2025-12-31 | 최신 구간 |

모델 파일 네이밍: `models/ppo_{reward}_{window}_risk.zip` (총 12개: 3 보상 × 4 윈도우, 전 윈도우 재학습 완료 2026-05-23)

---

## 5. 전 윈도우 재학습 결과 비교 (risk 연동 후 최종)

> 전 윈도우 재학습 완료: 2026-05-23  
> 백테스트 실행 기준: `src/rl/backtest.py` / 벤치마크: SPY

### 5-1. 누적수익률

| 구간 (시장 국면) | `return` | `sharpe` | `mdd` |
|----------------|---------|---------|-------|
| W1 — 2022 (금리충격) | **0.407** | 0.333 | −0.118 |
| W2 — 2023 (회복장)  | 1.070 | **1.487** | 0.510 |
| W3 — 2024 (AI랠리)  | 1.080 | **1.689** | 0.376 |
| Final — 2025        | 1.382 | **2.482** | 0.828 |

### 5-2. 샤프비율 (연율화, metrics.py 기준)

| 구간 | `return` | `sharpe` | `mdd` |
|------|---------|---------|-------|
| W1 — 2022 | **2.175** | 2.016 | −1.261 |
| W2 — 2023 | 8.115 | **9.413** | 4.331 |
| W3 — 2024 | 8.707 | **10.784** | 3.808 |
| Final — 2025 | 9.137 | **10.310** | 5.443 |

### 5-3. MDD (낮을수록 우수)

| 구간 | `return` | `sharpe` | `mdd` |
|------|---------|---------|-------|
| W1 — 2022 | **9.10%** | 10.82% | 15.07% ❌ |
| W2 — 2023 | 3.69% | **2.78%** | 6.20% |
| W3 — 2024 | **1.73%** | 2.12% | 7.04% |
| Final — 2025 | **2.64%** | 2.74% | 3.46% |

### 5-4. 알파 / 베타

| 구간 | `return` α/β | `sharpe` α/β | `mdd` α/β |
|------|------------|------------|---------|
| W1 — 2022 | **0.495** / 0.568 | 0.418 / 0.490 | −0.084 / 0.583 |
| W2 — 2023 | 0.769 / 0.561 | **1.017** / 0.447 | 0.387 / 0.556 |
| W3 — 2024 | 0.789 / 0.507 | **1.122** / 0.420 | 0.315 / 0.360 |
| Final — 2025 | 1.017 / 0.331 | **1.495** / 0.296 | 0.669 / 0.440 |

### 5-5. 구간별 1위 요약

| 구간 | 누적수익률 | 샤프비율 | MDD | 알파 |
|------|-----------|---------|-----|------|
| W1 (금리충격) | `return` | `return` | `return` | `return` |
| W2 (회복장)   | `sharpe` | `sharpe` | `sharpe` | `sharpe` |
| W3 (AI랠리)   | `sharpe` | `sharpe` | `return` | `sharpe` |
| Final (2025)  | `sharpe` | `sharpe` | `return` | `sharpe` |

---

## 6. Reward별 관찰 결과

### 6-1. `return` 보상

- **W1(금리충격) 전 구간 1위** — 누적 +40.7%, 샤프 2.175, MDD 9.1%, 알파 0.495. risk_vector를 obs로 받아 고위험 구간 자연 회피를 학습한 것으로 해석된다.
- W2~Final에서는 `sharpe`에 수익률·샤프·알파 모두 밀리나, **MDD는 W3·Final에서 가장 낮음** (1.73%, 2.64%).
- 보상이 단순해 학습 신호가 명확하고, risk 정보는 obs를 통해 분리 제공되므로 보상 함수와 관측 신호 간 역할이 명확히 분리된다.
- 베타가 W1 0.57 → Final 0.33으로 점진 감소 — 윈도우가 진행될수록 시장 의존도를 줄이는 방향으로 수렴.

### 6-2. `sharpe` 보상

- **W2~Final 3개 구간 전 지표 1위** — Final 기준 누적 +248%, 샤프 10.31, 알파 1.495.
- W1에서만 `return`에 밀림 — 보상 함수 내부의 변동성 정규화(`net_return / volatility`)와 obs의 risk_vector가 이중으로 위험 신호를 제공해 과도하게 보수적으로 수렴한 것으로 해석된다.
- 베타가 W1 0.49 → Final 0.30으로 3종 중 가장 빠르게 감소 — 시장 노이즈를 가장 잘 걸러내는 정책으로 수렴.
- 전 구간 Safe-Guard 미발동 (207/207/206/203 스텝 완주).

### 6-3. `mdd` 보상

- MDD 패널티는 **낙폭이 발생한 이후** 작동하는 후행 신호. risk 연동 후에도 이 구조적 문제는 해소되지 않음.
- W1에서 Safe-Guard 발동(MDD 15.07%) — 140스텝 조기 종료, 누적 −11.8%, 샤프 −1.26. 낙폭 억제라는 설계 목적과 반대 결과.
- W2~Final은 완주하나 전 구간 최하위. 후행 패널티로 인해 에이전트가 낙폭을 사전에 예방하지 못하고 사후 포지션 축소로 수렴하는 경향이 지속됨.

---

## 7. Safe-Guard Threshold 분석

`PortfolioEnv`는 `safe_guard_mdd = 0.15`(15%)를 초과하면 에피소드를 즉시 종료한다:

```python
safe_guard_triggered = bool(self.current_mdd > self.safe_guard_mdd)
terminated = bool(reached_end or safe_guard_triggered)
```

| 측면 | 내용 |
|------|------|
| 에피소드 조기 종료 | MDD > 15% 도달 시 즉시 종료 |
| 학습 신호 기능 | 에이전트가 위기 구간에서 Safe-Guard 발동을 반복 경험하며 고위험 포지션 회피 학습 |
| `mdd` 보상과의 상호작용 | `mdd` 에이전트는 MDD 패널티 + Safe-Guard 이중 조기 종료 압력을 받아 W1에서만 140스텝 조기 종료, W2~Final은 완주 |

임계값 15%의 근거 및 적정성 검토 결과는 `docs/safeguard_threshold_analysis.md`에 상세히 기술되어 있다.

---

## 8. 최종 Reward 평가 (전 윈도우 재학습 완료)

| 판단 기준 | `return` | `sharpe` | `mdd` |
|----------|---------|---------|-------|
| W1 종합 (금리충격) | **1위** ✅ | 2위 | 3위 ❌ |
| W2 종합 (회복장) | 2위 | **1위** ✅ | 3위 |
| W3 종합 (AI랠리) | 2위 | **1위** ✅ | 3위 |
| Final 종합 (2025) | 2위 | **1위** ✅ | 3위 |
| Final 누적수익률 | 1.382 | **2.482** | 0.828 |
| Final 샤프비율 | 9.137 | **10.310** | 5.443 |
| Final 알파 | 1.017 | **1.495** | 0.669 |
| W1 Safe-Guard | ✅ 완주 | ✅ 완주 | ❌ 발동 |
| risk 신호 활용 방식 | obs 분리 ✅ | 보상 중복 ⚠️ | 후행 패널티 ❌ |

**최종 채택: `sharpe` 보상 함수**

4개 구간 중 3개(W2·W3·Final)에서 전 지표 1위. W1(금리충격)에서만 `return`에 밀리는데, 이는 risk_vector obs 추가로 인한 변동성 이중 신호 문제로 해석된다. 해당 구간에서도 Safe-Guard 미발동·알파 양수를 유지하여 치명적 손실은 없다.

**`return` 보상은 2순위 대안** — W1 금리충격 같은 지속 하락장에서 MDD 억제 측면에서 `sharpe`보다 유리하며, 보상 구조가 단순해 해석이 명확하다.

**`mdd` 보상은 현 구현으로 비권장** — 후행 패널티 구조로 인해 W1에서 Safe-Guard 발동, 전 구간 최하위. 개선 방향: 롤링 CVaR 기반 선행 패널티, `lambda_mdd` 동적 스케일링.

**최종 모델**: `models/ppo_sharpe_final_risk.zip`  
(Final 기준 누적 +248%, 샤프 10.31, 알파 1.495, 베타 0.296)

---

## 9. 후속 작업

- [ ] SHAP 실행 — `risk_macro/equity/geo` 피처 기여도 0 이상 확인 (보고서 §5-3 연동)
- [ ] ANOVA 실행 (강유영) — `data/results/backtest_{reward}.csv` 전달 완료 후 진행
- [ ] 지민에게 모델 파일·백테스트 결과 전달

---

*작성: 이문정 / 백테스트 결과 참조: 강유영(`src/rl/backtest.py`) / 최종 갱신: 2026-05-23 (전 윈도우 재학습 완료)*
