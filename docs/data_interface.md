# 데이터 인터페이스 명세

> **대상 독자**: 이문정 (`src/rl/env.py`) · 강유영 (`src/rl/backtest.py`)
>
> 이 문서는 `src/data/` 파이프라인이 생성하는 파일의 스키마와 사용 시 주의사항을 정리한다.
> 컬럼명·파일 경로·정규화 방식 변경 시 박지민에게 반드시 사전 공지.

---

## 파일 목록

| 파일 | 경로 | 용도 |
|------|------|------|
| 원본 가격 | `data/raw/prices.parquet` | EDA·검증용 (직접 사용 불필요) |
| 로그수익률 | `data/processed/returns.parquet` | RL 환경 학습·추론 |
| 피처 | `data/processed/features.parquet` | RL 관측 공간·백테스트 |

---

## returns.parquet

### 스키마

| 항목 | 값 |
|------|----|
| index | `Date (datetime64[ns])`, 중복 없음 |
| columns | `SPY, QQQ, IWM, EFA, EEM, TLT, GLD, VNQ, 069500, 114260` (10열) |
| 값 | **raw 로그수익률** — 정규화 전 |
| 결측치 | 0개 |
| 실제 shape | `(1423, 10)` |
| 날짜 범위 | 2020-01-03 ~ 2025-03-28 (US–KR inner join 기준) |

### 생성 로직

```
yfinance Adj Close + pykrx 종가
→ inner join (공통 거래일)
→ ffill → dropna
→ np.log(prices / prices.shift(1)).dropna()
```

### ⚠️ env.py 사용 시 주의

`returns.parquet`은 **정규화 전 값**이다.
`trading_env.py`에서 자체 정규화(Z-score 또는 rolling)를 수행해야 한다.
이 파일을 그대로 관측 공간에 넣으면 스케일 불일치가 발생한다.

---

## features.parquet

### 스키마

| 항목 | 값 |
|------|----|
| index | `Date (datetime64[ns])`, 중복 없음 |
| columns | 아래 40개 |
| 값 | **Z-score 정규화 후** (전체 기간 기준) |
| 결측치 | 0개 |
| 실제 shape | `(1390, 40)` |
| 날짜 범위 | 2020-02-25 ~ 2025-03-28 (MACD 초기 NaN 제거 후) |

### 컬럼명 규칙 (티커 × 4 = 40열)

```
{ticker}_return        # Z-score 정규화된 로그수익률
{ticker}_RSI           # RSI(14), Z-score 정규화
{ticker}_MACD          # MACD(12/26/9) 값, Z-score 정규화
{ticker}_MACD_signal   # MACD 시그널선(9), Z-score 정규화
```

티커 순서: `SPY, QQQ, IWM, EFA, EEM, TLT, GLD, VNQ, 069500, 114260`

### 왜 returns보다 33행 적은가?

MACD 계산 시 slow=26, signal=9 → 초기 `26 + 9 - 2 = 33`행이 NaN으로 생성된다.
`build_features()` 내에서 `dropna()`로 이 행을 제거하므로
`features` 시작일이 `returns` 시작일보다 약 33 거래일(~7주) 늦다. **이는 정상이다.**

### 정규화 방식

전체 기간 기준 Z-score: `(x - mean) / std`

Look-ahead bias는 `trading_env.py`의 Walk-Forward 분리에서 처리한다.
collector 단계의 전체 기간 정규화는 이문정과 합의된 방식이다.

### backtest.py 사용 시 주의

- 컬럼 슬라이싱 예시:
  ```python
  return_cols = [f"{t}_return" for t in tickers]
  features["SPY_RSI"]         # SPY RSI
  features["069500_MACD"]     # 069500 MACD 값
  ```
- features 시작일(2020-02-25)과 returns 시작일(2020-01-03)이 다르므로
  두 파일을 날짜로 join할 때 기준일 정렬 필요.

---

## 데이터 재생성

```bash
# 전체 파이프라인 재실행 (수분 소요)
python -m src.data.collector

# 검증
pytest tests/test_data.py -v
```

---

## 자산 목록

| 티커 | 설명 | 시장 |
|------|------|------|
| SPY | S&P 500 ETF | 미국 |
| QQQ | 나스닥 100 ETF | 미국 |
| IWM | 러셀 2000 ETF | 미국 |
| EFA | 선진국 ETF | 글로벌 |
| EEM | 신흥국 ETF | 글로벌 |
| TLT | 미국 장기채 ETF | 미국 |
| GLD | 금 ETF | 글로벌 |
| VNQ | 리츠 ETF | 미국 |
| 069500 | KODEX 200 | 한국 |
| 114260 | KODEX 국고채10년 | 한국 |

---

*최종 업데이트: 2026-04-08 / 작성: 박지민*
