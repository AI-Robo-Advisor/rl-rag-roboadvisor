# risk_vector 핸드오프 — 이문정 담당

> **작성자**: 강유영  
> **작성일**: 2026-05-21  
> **대상**: 이문정 (`src/rl/train_walkforward.py`, `src/rl/backtest.py`)

---

## 1. 파일 위치 및 스키마

```
data/processed/risk_vectors_daily.parquet
```

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `date` | datetime64[ns] | 2018-01-01 ~ 2025-12-31 (매일, 비거래일 포함) |
| `risk_macro` | float64 | macro_rate_risk Decay 스코어 (0.0~1.0) |
| `risk_equity` | float64 | equity_market_risk Decay 스코어 (0.0~1.0) |
| `risk_geo` | float64 | geopolitical_fx_risk Decay 스코어 (0.0~1.0) |

---

## 2. train_walkforward.py / backtest.py 에 붙일 코드

```python
import numpy as np
import pandas as pd

# ── 학습 시작 전 1회 로드 ──────────────────────────────────────
risk_df = pd.read_parquet("data/processed/risk_vectors_daily.parquet")
risk_df["date"] = pd.to_datetime(risk_df["date"])
risk_df = risk_df.set_index("date")

# ── 매 스텝(거래일)마다 호출 ──────────────────────────────────
def get_risk_vector(date: pd.Timestamp) -> np.ndarray:
    """해당 날짜의 risk vector를 반환합니다. 데이터 없으면 zeros."""
    if date in risk_df.index:
        row = risk_df.loc[date]
        return np.array([row["risk_macro"], row["risk_equity"], row["risk_geo"]],
                        dtype=np.float32)
    return np.zeros(3, dtype=np.float32)

# ── env 주입 (이미 env.py에 set_risk_vector 있음) ──────────────
# 에피소드 루프 내에서:
current_date = returns_df.index[current_step]   # 현재 거래일
env.set_risk_vector(get_risk_vector(current_date))
obs, reward, done, truncated, info = env.step(action)
```

**obs 순서**: `[..., risk_macro, risk_equity, risk_geo]` (마지막 3개)  
→ `env.py` 코드에서 `self.risk_vector` 그대로 이어 붙임.

---

## 3. returns.parquet 거래일 align 예시

```python
returns_df = pd.read_parquet("data/processed/returns.parquet")  # 거래일 index

# risk_df는 매일(비거래일 포함) → 거래일만 필터링하려면:
trading_days = returns_df.index
risk_aligned = risk_df.reindex(trading_days, method="ffill")  # 비거래일 → 직전 거래일 값 사용
# risk_aligned 컬럼: risk_macro, risk_equity, risk_geo
```

---

## 4. Walk-Forward 윈도우별 날짜 범위

| 윈도우 | 학습 기간 | 테스트 기간 |
|--------|----------|------------|
| w1 | 2018-01-01 ~ 2021-12-31 | 2022-01-01 ~ 2022-12-31 |
| w2 | 2019-01-01 ~ 2022-12-31 | 2023-01-01 ~ 2023-12-31 |
| w3 | 2020-01-01 ~ 2023-12-31 | 2024-01-01 ~ 2024-12-31 |
| final (w4) | 2021-01-01 ~ 2024-12-31 | 2025-01-01 ~ 2025-12-31 |

`risk_vectors_daily.parquet`은 2018-01-01부터 커버하므로 모든 윈도우 학습 구간 포함.

---

## 5. A/B 데이터 도착 후 재빌드

박지민(GDELT) 또는 이문정(FRED) parquet이 `data/raw/` 에 추가되면:

```bash
python scripts/build_risk_parquet.py
```

한 줄로 재빌드 완료. `data/processed/risk_vectors_daily.parquet` 덮어씀.

---

## 6. 확인 사항

- [ ] `get_risk_vector()` 함수를 에피소드 스텝 루프 내부에서 매 거래일마다 호출하는지
- [ ] `env.set_risk_vector(vec)` 호출 후 `env.step()` 순서인지 (관측에 반영되려면 step 전에 set해야 함)
- [ ] W1 재학습 시 학습 구간(2018~2021) risk_vector가 0이 아닌지 확인:  
  ```python
  # 2020-03-16 (코로나): risk_equity 높아야 함
  print(get_risk_vector(pd.Timestamp("2020-03-16")))  # [~0.x, ~1.0, ~0.x]
  ```
