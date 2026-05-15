# Walk-Forward 분할 계획

> 관련 모듈: `src/data/collector.py` (박지민), `src/rl/train_walkforward.py` (이문정), `src/rl/backtest.py` (강유영), `src/rl/env.py` (이문정)
>
> 데이터 기간: 2018-01-03 ~ 2025-12-30 (returns.parquet 기준, 총 1898 거래일)

---

## 왜 Walk-Forward인가

전체 기간을 단순 train/test로 분리하면 학습 시점보다 이전 정규화 파라미터(mean, std)가
미래 데이터에 누출되는 Look-ahead bias가 발생한다.
Walk-Forward는 학습 윈도우를 앞에서 뒤로 밀면서 각 윈도우에서 독립적으로 정규화하므로
bias를 방지하고, 시장 국면 변화(COVID 급락, 금리 인상기 등)에 대한 일반화 성능을 측정할 수 있다.

---

## 윈도우 정의

| Window | 학습(Train) | 검증(Test) | 거래일(추정) |
|--------|------------|-----------|-------------|
| W1 | 2018-01-01 ~ 2021-12-31 | 2022-01-01 ~ 2022-12-31 | 학습 955일 / 검증 237일 |
| W2 | 2019-01-01 ~ 2022-12-31 | 2023-01-01 ~ 2023-12-31 | 학습 957일 / 검증 237일 |
| W3 | 2020-01-01 ~ 2023-12-31 | 2024-01-01 ~ 2024-12-31 | 학습 955일 / 검증 236일 |
| Final Holdout | 2021-01-01 ~ 2024-12-31 | 2025-01-01 ~ 2025-12-31 | 학습 950일 / 검증 233일 |

---

## 윈도우 설계 근거

- **학습 기간 4년**: PPO 에이전트가 충분한 시장 국면을 학습하려면 bull/bear 사이클이 최소 1회 이상 포함되어야 한다. 2018~2024 구간은 2020 COVID 급락, 2021 회복·성장기, 2022 금리 인상 하락기, 2023~2024 반등·AI 랠리를 포함한다.
- **검증 기간 1년**: 단일 연도 검증으로 과적합 여부를 명확히 측정할 수 있다. 각 검증 연도는 2022 금리 충격, 2023 회복장, 2024 AI 랠리, 2025 최신 시장 구간을 분리한다.
- **슬라이딩 스텝 1년**: 윈도우를 1년씩 밀어 시장 구조 변화에 따른 일반화 성능을 점검한다.

---

## 구현 가이드라인

```python
WINDOWS = [
    {"train_start": "2018-01-01", "train_end": "2021-12-31",
     "test_start":  "2022-01-01", "test_end":  "2022-12-31"},
    {"train_start": "2019-01-01", "train_end": "2022-12-31",
     "test_start":  "2023-01-01", "test_end":  "2023-12-31"},
    {"train_start": "2020-01-01", "train_end": "2023-12-31",
     "test_start":  "2024-01-01", "test_end":  "2024-12-31"},
    {"train_start": "2021-01-01", "train_end": "2024-12-31",
     "test_start":  "2025-01-01", "test_end":  "2025-12-31"},
]
```

> 모델 파일 네이밍 → `docs/labels_and_interfaces.md` 3-3 참고

각 윈도우에서:
1. `data/processed/raw_features.parquet`를 날짜로 분리
2. 학습 기간 데이터만으로 Z-score 파라미터(mean, std) 계산
3. 계산된 파라미터로 학습 기간과 검증 기간을 모두 정규화 (학습 기간 통계 재사용)
4. 정규화된 학습 feature를 `PortfolioEnv(features_df=...)`에 전달
5. 정규화된 검증 feature로 성과 측정

`data/processed/features.parquet`은 전체 기간 Z-score가 적용된 legacy 산출물이므로
Walk-Forward 학습/백테스트에 직접 사용하지 않는다.

## TODO — RL 학습/백테스트 구현

- [ ] `train_walkforward.py`에서 `raw_features.parquet` 기반 per-window 정규화 구현
- [ ] window별 mean/std 저장 및 재사용 경로 확정
- [x] `backtest.py`에서 test feature를 train mean/std로 변환
- [ ] look-ahead bias 제거 후 PPO 모델과 백테스트 결과 재생성

---

## 보고서 연결

- **섹션 5 (RL 환경 설계)**: Walk-Forward 분리가 look-ahead bias를 방지하는 이유 서술
- **섹션 7 (실험 결과)**: W1·W2·W3·Final Holdout 각각의 샤프비율·MDD·누적수익률 비교 테이블

---

*최종 업데이트: 2026-05-12 / 작성: 박지민*
