# 과제 명세 ↔ 팀 리스크 태그 매핑

> **버전**: v1.0  
> **작성일**: 2026-05-21  
> **목적**: 과제 명세서의 자연어 예시와 팀 구현 3축의 대응 관계를 명확히 기록.  
> 리포트 작성, RL 관측공간 설명, ChromaDB 메타데이터 필드 설명에 동일하게 사용.

---

## 1. 명세 예시 ↔ 팀 3축 대응표

| 과제 명세 예시 (자연어) | 팀 RL 축 | parquet 컬럼 | obs 순서 |
|------------------------|----------|-------------|---------|
| 금리 인상·인하, FOMC, Fed, CPI·PPI, 국채금리, 달러 강세, 한국은행 기준금리, 채권 가격 변동 | **macro_rate_risk** | `macro_rate_risk` | 0 |
| 증시 급락·폭락, 경기침체 우려, 어닝쇼크, VIX 급등, 패닉셀, KOSPI 급락 | **equity_market_risk** | `equity_market_risk` | 1 |
| 전쟁·지정학 갈등, 관세·무역 전쟁, 반도체 수출 규제, 공급망 충격, 환율 급변 | **geopolitical_fx_risk** | `geopolitical_fx_risk` | 2 |

> **구 태그(레거시) 매핑**
> | 구 태그 | 신 태그 | 비고 |
> |--------|--------|------|
> | 규제변경 | `macro_rate_risk` + `geopolitical_fx_risk` (내용에 따라) | 정책·법률 규제 → macro, 수출규제·제재 → geo |
> | 실적쇼크 | `equity_market_risk` | |
> | 급등락 | `equity_market_risk` 주도 (+ 원인에 따라 macro/geo 보완) | |

---

## 2. RL 관측공간 내 위치

```
obs = [
  과거 lookback일 수익률 (n_assets × lookback),   # features_df {ticker}_return
  현재 포트폴리오 비중 (n_assets),                 # env 내부 weights
  RSI (n_assets),                                  # features_df {ticker}_RSI
  MACD signal (n_assets),                          # features_df {ticker}_MACD_signal
  risk_macro,    # obs[-3] — macro_rate_risk decay 스코어 (0~1)
  risk_equity,   # obs[-2] — equity_market_risk decay 스코어 (0~1)
  risk_geo,      # obs[-1] — geopolitical_fx_risk decay 스코어 (0~1)
]
obs_dim = (lookback + 3) * n_assets + 3  →  (30+3)*10+3 = 333
```

`env.set_risk_vector(np.array([risk_macro, risk_equity, risk_geo], dtype=np.float32))`

---

## 3. 일별 Parquet 스키마

**파일**: `data/processed/risk_vectors_daily.parquet`

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `date` | datetime64[ns] | 캘린더 일별(비거래일 포함), 2018-01-01 ~ 2025-12-31. RL 학습/백테스트 시 `returns.parquet` index(거래일)로 align 필요 (`ffill` 또는 inner join) |
| `risk_macro` | float32 | macro_rate_risk 이벤트 Exponential Decay 누적 (0~1) |
| `risk_equity` | float32 | equity_market_risk 이벤트 Exponential Decay 누적 (0~1) |
| `risk_geo` | float32 | geopolitical_fx_risk 이벤트 Exponential Decay 누적 (0~1) |

**Decay 기간**: macro=30일, equity=10일, geo=60일  
**집계 방식**: 동일 축의 여러 이벤트 → max pooling (포화 방지)

---

## 4. 이벤트 원본 Parquet 스키마 (raw)

모든 소스(GDELT·FRED·ECOS·manual_seed)의 공통 필드:

| 컬럼 | 값 예시 |
|------|--------|
| `macro_rate_risk` | 0.0 \| 0.33 \| 0.66 \| 1.0 |
| `equity_market_risk` | 0.0 \| 0.33 \| 0.66 \| 1.0 |
| `geopolitical_fx_risk` | 0.0 \| 0.33 \| 0.66 \| 1.0 |
| `primary_tag` | `"macro_rate_risk"` \| `"equity_market_risk"` \| `"geopolitical_fx_risk"` \| `"none"` |
| `label_method` | `"rule_based"` \| `"manual"` \| `"llm"` |
