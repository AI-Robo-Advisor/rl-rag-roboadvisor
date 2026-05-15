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
| 로그수익률 | `data/processed/returns.parquet` | 실제 포트폴리오 수익률 계산 |
| 정규화 전 피처 | `data/processed/raw_features.parquet` | Walk-Forward 정규화 입력 |
| legacy 피처 | `data/processed/features.parquet` | EDA·기존 코드 호환용, Walk-Forward 학습 직접 사용 금지 |

---

## returns.parquet

### 스키마

| 항목 | 값 |
|------|----|
| index | `Date (datetime64[ns])`, 중복 없음 |
| columns | `SPY, QQQ, IWM, EFA, EEM, TLT, GLD, VNQ, 069500, 114260` (10열) |
| 값 | **raw 로그수익률** — 정규화 전 |
| 결측치 | 0개 |
| 실제 shape | `(1898, 10)` |
| 날짜 범위 | 2018-01-03 ~ 2025-12-30 (US–KR inner join 기준) |

### 생성 로직

```
yfinance Adj Close + pykrx 종가
→ inner join (공통 거래일)
→ ffill → dropna
→ np.log(prices / prices.shift(1)).dropna()
```

### 사용 시 주의

`returns.parquet`은 **정규화 전 값**이다.
`src/rl/env.py`의 실제 수익률 계산에는 이 파일을 사용한다.
관측 공간의 `{ticker}_return`은 `raw_features.parquet`에서 가져온 뒤
Walk-Forward 학습 구간 통계로 정규화한 값을 사용해야 한다.

---

## raw_features.parquet

### 스키마

| 항목 | 값 |
|------|----|
| index | `Date (datetime64[ns])`, 중복 없음 |
| columns | 아래 40개 |
| 값 | **정규화 전 raw feature** |
| 결측치 | 0개 |
| 실제 shape | `(1865, 40)` |
| 날짜 범위 | 2018-02-23 ~ 2025-12-30 (MACD 초기 NaN 제거 후) |

### 컬럼명 규칙 (티커 × 4 = 40열)

```
{ticker}_return        # raw 로그수익률
{ticker}_RSI           # RSI(14), 0~100 스케일
{ticker}_MACD          # MACD(12/26/9) 값, 정규화 전
{ticker}_MACD_signal   # MACD 시그널선(9), 정규화 전
```

티커 순서: `SPY, QQQ, IWM, EFA, EEM, TLT, GLD, VNQ, 069500, 114260`

### 왜 returns보다 33행 적은가?

MACD 계산 시 slow=26, signal=9 → 초기 `26 + 9 - 2 = 33`행이 NaN으로 생성된다.
`build_raw_features()` 내에서 `dropna()`로 이 행을 제거하므로
`raw_features` 시작일이 `returns` 시작일보다 약 33 거래일(~7주) 늦다. **이는 정상이다.**

### 정규화 방식

`raw_features.parquet`에는 정규화를 적용하지 않는다.

Walk-Forward 학습/백테스트에서 각 윈도우별로 다음 절차를 수행한다.

1. 학습 기간의 `raw_features`만으로 mean/std 계산
2. 학습 기간과 테스트 기간 모두 같은 train mean/std로 Z-score 변환
3. 변환된 feature를 `src/rl/env.py`의 `features_df`로 전달

테스트 기간의 mean/std를 사용하면 look-ahead bias가 발생한다.

### TODO — RL 학습/백테스트 담당

- [ ] `src/rl/train_walkforward.py`에서 `raw_features.parquet` 로드
- [ ] 윈도우별 train mean/std 계산 및 저장
- [ ] `src/rl/backtest.py`에서 학습 때 저장한 mean/std로 test feature 변환
- [ ] 기존 `models/ppo_*_risk.zip`, `data/results/backtest_*.csv`, `weights_*.parquet` 재생성

---

## features.parquet

`features.parquet`은 기존 코드 호환과 EDA 검증을 위해 당분간 유지하는
**legacy 산출물**이다. 값은 전체 기간 기준 Z-score이므로 Walk-Forward 학습/백테스트에
직접 사용하면 look-ahead bias가 생길 수 있다.

### 스키마

| 항목 | 값 |
|------|----|
| index | `Date (datetime64[ns])`, 중복 없음 |
| columns | `raw_features.parquet`과 동일한 40개 |
| 값 | **전체 기간 기준 Z-score** |
| 용도 | EDA·기존 테스트 호환용 |

### backtest.py 사용 시 주의

- 컬럼 슬라이싱 예시:
  ```python
  return_cols = [f"{t}_return" for t in tickers]
  raw_features["SPY_RSI"]         # SPY RSI
  raw_features["069500_MACD"]     # 069500 MACD 값
  ```
- raw_features 시작일(2018-02-23)과 returns 시작일(2018-01-03)이 다르므로
  두 파일을 날짜로 join할 때 기준일 정렬 필요.
- Walk-Forward 백테스트에는 `features.parquet`이 아니라 `raw_features.parquet`에서
  윈도우별로 정규화한 DataFrame을 전달해야 한다.

---

## 데이터 재생성

```bash
# 전체 파이프라인 재실행 (수분 소요)
.venv/bin/python -m src.data.collector

# 검증
.venv/bin/pytest tests/test_data.py -v
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
| 114260 | KODEX 국고채3년 | 한국 |

---

*최종 업데이트: 2026-05-15 / 작성: 박지민*
