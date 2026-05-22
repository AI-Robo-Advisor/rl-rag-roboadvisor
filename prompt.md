# 강유영 남은 작업 현황

> 업데이트: 2026-05-22

---

## 과제 1. End-to-end 파이프라인 (리스크태그 문제)

### 완료
- [x] 코드 전체 canonical naming (`_risk` suffix) 적용 — PR #48 merged
- [x] `build_risk_parquet.py` / `label_events_llm.py` RAW_PATHS `2018_2025`로 수정 — PR #50 merged
- [x] ECOS 수집 (`collect_ecos.py`, look-ahead bias 수정 포함)
- [x] manual_seed 수집
- [x] GDELT+FRED dev 반영 후 `build_risk_parquet.py` 재실행 완료
  - ECOS+FRED+manual_seed 포함, sanity check 통과 (자이언트 스텝 macro=1.0, 러-우 전쟁 geo=1.0)
- [x] `unified_events.parquet` FRED+ECOS+manual_seed 통합 (1227건)
- [x] `scripts/build_chroma_from_parquet.py` 작성 및 실행 — ChromaDB 1316건

### 남은 것
- [ ] `train.py` risk_vector 연동 — **이문정 담당**, 완료 후 재학습 진행
- [ ] `apps/api/` `/research` `rl_risk_tags` 확인 — **박지민 담당**

---

## 과제 2. RAG 성능 향상 및 검증

### 완료
- [x] English 키워드 casefold 지원 추가 — PR #44
- [x] `docs/rag_eval_questions.json` 평가 질문 12개 작성
- [x] ChromaDB 과거 데이터 bulk upsert — `scripts/build_chroma_from_parquet.py`
- [x] RAG 품질 평가 실행 — `scripts/rag_eval.py`
  - 검색 히트율: 12/12 = **100%** (dist < 0.4)
  - 태그 정확도: 10/12 = **83.3%**
  - 실패 2건: Q08(인플레이션), Q11(한국은행 기준금리) — ECOS 데이터 매칭 부족

---

## 과제 3. 데이터 수집 C·D 파트

### 완료
- [x] C: `scripts/collect_ecos.py`
- [x] D: `scripts/collect_manual_seed.py`

---

## 과제 4. 신규 데이터 라벨링 + 금융 데이터 연결 (3인 공동)

> **브랜치**: dev 기반 새 브랜치 별도 생성 (3명 공동 작업)

- [ ] 새로 모은 데이터 3개 리스크 라벨링
- [ ] 금융 데이터(parquet)에 연결 통합
- [ ] `build_risk_parquet.py` 재실행

---

## 과제 5. 분석 보고서 (`docs/임시_유영_분석보고서.md`)

> **선행 조건**: 이문정 모델 재학습 완료 후 진행

- [ ] 백테스트 재실행 → 표 A·B·C 기입
- [ ] ANOVA 재실행 → 표 D·E 기입
- [ ] SHAP force plot 재생성 → 표 F 기입
- [ ] §5-3 RAG-RL 연동 스토리 작성 (risk vector 연동 완료 후)

---

## 블로커 요약

| 블로커 | 담당 | 내 액션 |
|--------|------|---------|
| `train.py` risk_vector 연동 + 재학습 | 이문정 | 완료 연락 기다리는 중 |
| `apps/api/` `/research` rl_risk_tags 확인 | 박지민 | 별도 요청 필요 |


# 🔧 공통 가이드라인 (모든 담당자 필독)

## 최종 산출물 통일 스키마

모든 소스의 수집 결과는 **동일한 Parquet 스키마**로 통합돼. 각자 자기 소스에서 이 스키마에 맞춰 저장:

```python
# 공통 스키마 (모든 담당자가 동일하게 맞춤)
COLUMNS = {
    "event_id":        "string",     # 고유 ID, 소스별 prefix 사용
    "date":            "datetime",   # YYYY-MM-DD (KST 기준, market close 정렬 후)
    "source":          "string",     # 'gdelt' | 'fred' | 'ecos' | 'manual_seed'
    "title":           "string",     # 뉴스 제목 또는 이벤트 명
    "summary":         "string",     # 본문 요약 (300자 이내)
    "url":             "string",     # 원문 URL (수동 시드는 위키피디아 등)
    "language":        "string",     # 'en' | 'ko'
    "raw_data":        "string",     # 원본 데이터 JSON 문자열 (디버깅용)
    "label_method":    "string",     # 'llm' | 'rule_based' | 'manual'
    # 라벨링은 별도 단계에서 추가되므로 여기서는 비워둠
}

# event_id 규칙
# gdelt-{YYYYMMDD}-{eventcode}-{idx}
# fred-{YYYYMMDD}-{series_name}
# ecos-{YYYYMMDD}-{series_name}
# seed-{YYYYMMDD}-{slug}
```

## 저장 위치 (Git 레포 구조)

```
data/
├── raw/
│   ├── gdelt/
│   │   └── gdelt_events_2018_2025.parquet      # 담당자 A
│   ├── fred/
│   │   └── fred_events_2018_2025.parquet       # 담당자 B
│   ├── ecos/
│   │   └── ecos_events_2018_2025.parquet       # 담당자 C
│   └── manual_seed/
│       └── manual_seed_events.parquet          # 담당자 D
└── processed/
    └── unified_events.parquet                   # 통합 (마지막에 합침)
```

## 수집 기간

**2018-01-01 ~ 2025-12-31** (8년치, 윈도우 4개 커버)

⚠️  Walk-Forward 윈도우 4개(W1=2022, W2=2023, W3=2024, W4=2025)를 모두 커버하려면 학습 데이터까지 포함해서 **2018년부터** 수집하는 게 안전해.

## 미래 정보 누수 방지 — 모든 담당자 공통 규칙

```python
def adjust_date_for_market_close(news_datetime_utc, asset_region):
    """
    뉴스가 시장 마감 후 발표되면 다음 거래일로 라벨 날짜 이동
    """
    if asset_region == "US":
        # 미국 장 마감: 16:00 ET = 21:00 UTC (DST 무시 단순화)
        market_close_utc = news_datetime_utc.replace(hour=21, minute=0)
        if news_datetime_utc > market_close_utc:
            return next_us_trading_day(news_datetime_utc.date())
    elif asset_region == "KR":
        # 한국 장 마감: 15:30 KST = 06:30 UTC
        market_close_utc = news_datetime_utc.replace(hour=6, minute=30)
        if news_datetime_utc > market_close_utc:
            return next_kr_trading_day(news_datetime_utc.date())
    return news_datetime_utc.date()
```

각자 수집 코드 마지막에 이 함수 적용. 거래일 계산은 `pandas_market_calendars` 라이브러리 추천.

## 통합 환경 설정

```bash
# requirements.txt (모든 담당자 동일)
google-cloud-bigquery==3.25.0
fredapi==0.5.2
requests==2.32.3
pandas==2.2.3
pyarrow==17.0.0
pandas_market_calendars==4.4.1
python-dotenv==1.0.1
```

```bash
# .env (각자 본인 키만 채워서 사용, GitHub 업로드 금지!)
GOOGLE_APPLICATION_CREDENTIALS=path/to/gcp-key.json
FRED_API_KEY=your_fred_key
ECOS_API_KEY=your_ecos_key
```

---

# 👤 담당자 A: GDELT 2.0 (BigQuery)

## 목표

2018-01-01 ~ 2025-12-31 기간 동안 **미국/글로벌 매크로·지정학 이벤트** 약 2,000~3,000건 수집.

## 사전 준비

### 1. Google Cloud 계정 셋업

1. https://console.cloud.google.com 가입 (Gmail 계정으로 가능)
2. 신용카드 등록 (무료 티어 사용, 알림만 설정하면 과금 위험 거의 없음)
3. **새 프로젝트 생성**: 이름 `robo-advisor-gdelt`
4. **BigQuery API 활성화**: API 라이브러리 → BigQuery API → Enable
5. **서비스 계정 키 생성**:
    - IAM & Admin → Service Accounts → Create
    - 권한: `BigQuery Data Viewer` + `BigQuery Job User`
    - JSON 키 다운로드 → `.env`에 경로 저장

### 2. 비용 알림 설정 ⚠️ 필수

- Billing → Budgets & alerts → Create Budget
- 한도: **$5** (무료 티어는 월 1TB 쿼리 무료, 우리는 안 넘을 거지만 안전망)
- 50% 도달 시 이메일 알림

## ⚠️ BigQuery 안전 수칙

**절대 지켜야 할 3가지:**

```sql
-- ❌ 절대 금지: 파티션 필터 없는 SELECT *
SELECT * FROM `gdelt-bq.gdeltv2.events`
-- → 페타바이트 스캔, 무료 크레딧 즉시 소진

-- ✅ 반드시: _PARTITIONTIME 또는 SQLDATE 필터 먼저
WHERE _PARTITIONTIME BETWEEN TIMESTAMP('2018-01-01') AND TIMESTAMP('2025-12-31')

-- ✅ 처음엔 LIMIT으로 형태 확인
LIMIT 100
```

작업 흐름:

1. **1일치 + LIMIT 100**으로 먼저 쿼리 → 결과 형태 확인
2. **1개월치**로 확장 → 노이즈 수준 점검 (GoldsteinScale 분포 확인)
3. 전체 5년치는 **마지막 한 번만** 실행

## 단계 1: 탐색 쿼리 (먼저 이거 돌려보기)

```sql
-- 1일치 샘플 확인용
SELECT
    SQLDATE,
    EventCode,
    EventBaseCode,
    Actor1CountryCode,
    Actor1Name,
    Actor2CountryCode,
    Actor2Name,
    GoldsteinScale,
    NumMentions,
    AvgTone,
    SOURCEURL
FROM `gdelt-bq.gdeltv2.events`
WHERE _PARTITIONTIME = TIMESTAMP('2022-02-24')  -- 우크라전 개전일
  AND (Actor1CountryCode IN ('USA','RUS','UKR','CHN','KOR')
       OR Actor2CountryCode IN ('USA','RUS','UKR','CHN','KOR'))
  AND GoldsteinScale < -3.0
LIMIT 100;
```

이 쿼리로 GDELT 데이터 구조 익히기. **이게 작동하면 다음 단계 진행.**

## 단계 2: 본 수집 쿼리

```sql
SELECT
    CONCAT('gdelt-',
           FORMAT_DATE('%Y%m%d', PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING))),
           '-', CAST(EventCode AS STRING),
           '-', GENERATE_UUID()) AS event_id,
    PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) AS date,
    'gdelt' AS source,
    CONCAT(IFNULL(Actor1Name, ''), ' - ', IFNULL(Actor2Name, ''),
           ' (Event ', CAST(EventCode AS STRING), ')') AS title,
    CONCAT('Goldstein: ', CAST(GoldsteinScale AS STRING),
           ', Tone: ', CAST(AvgTone AS STRING),
           ', Mentions: ', CAST(NumMentions AS STRING)) AS summary,
    SOURCEURL AS url,
    'en' AS language,
    TO_JSON_STRING(STRUCT(EventCode, EventBaseCode, GoldsteinScale,
                          NumMentions, AvgTone, Actor1CountryCode,
                          Actor2CountryCode)) AS raw_data,
    'llm' AS label_method
FROM `gdelt-bq.gdeltv2.events`
WHERE _PARTITIONTIME BETWEEN TIMESTAMP('2018-01-01') AND TIMESTAMP('2025-12-31')
  -- 1. 매크로 관련 국가 필터
  AND (Actor1CountryCode IN ('USA','CHN','KOR','RUS','UKR','EUR','JPN','TWN')
       OR Actor2CountryCode IN ('USA','CHN','KOR','RUS','UKR','EUR','JPN','TWN'))
  -- 2. 이벤트 코드: 외교 갈등, 군사 충돌, 강압/제재, 경제 정책
  AND (EventCode LIKE '03%'   -- 외교 협력/갈등
       OR EventCode LIKE '13%'   -- 군사 충돌
       OR EventCode LIKE '17%'   -- 강압/제재
       OR EventCode LIKE '04%'   -- 정책 발표
       OR EventCode LIKE '14%'   -- 무력 위협
       OR EventCode LIKE '19%')  -- 전쟁/충돌
  -- 3. 노이즈 필터: 충분히 보도되고 부정적 톤
  AND NumMentions >= 50
  AND GoldsteinScale < -3.0
ORDER BY date, NumMentions DESC;
```

**필터 근거:**

- `NumMentions >= 50`: 최소 50개 이상 언론사가 보도한 사건만 (단발성 노이즈 제거)
- `GoldsteinScale < -3.0`: 부정적 톤 이벤트만 (Goldstein 척도는 -10 ~ +10)
- `EventCode` 필터: 매크로 영향 있는 CAMEO 코드만

**예상 결과량:** 2,000~3,000건 (조정 가능)

## 단계 3: BigQuery → Parquet 변환

```python
# scripts/collect_gdelt.py
from google.cloud import bigquery
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

client = bigquery.Client()

QUERY = """
-- 위의 본 수집 쿼리 그대로
"""

# 비용 미리 확인 (dry run)
job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
dry_job = client.query(QUERY, job_config=job_config)
print(f"예상 처리 데이터량: {dry_job.total_bytes_processed / 1e9:.2f} GB")
# 1TB(1000GB) 이하면 무료 티어 내 안전

# 실제 실행
input("총 데이터량 확인 후 Enter를 눌러 실행: ")
df = client.query(QUERY).to_dataframe()
print(f"수집 완료: {len(df)}건")

# market close 보정 (위 공통 가이드 함수 적용)
# GDELT는 미국 중심이므로 US 마켓 기준
df['date'] = df['date'].apply(lambda d: adjust_date_for_market_close(d, 'US'))

# 저장
os.makedirs("data/raw/gdelt", exist_ok=True)
df.to_parquet("data/raw/gdelt/gdelt_events_2018_2025.parquet", index=False)
print("저장 완료")
```

## 체크리스트

- [ ]  GCP 프로젝트 생성, BigQuery API 활성화
- [ ]  서비스 계정 키 발급, `.env` 등록
- [ ]  비용 알림 $5 설정
- [ ]  탐색 쿼리(1일치, LIMIT 100) 성공
- [ ]  dry_run으로 예상 데이터량 확인 (< 1TB)
- [ ]  본 수집 실행, 결과 2,000~3,000건 확인
- [ ]  market close 보정 적용
- [ ]  Parquet 저장 (`data/raw/gdelt/`)
- [ ]  결과 샘플 10건 팀에 공유 (Slack/Notion)

---

# 👤 담당자 B: FRED API (미국 매크로 지표)

## 목표

2018-01-01 ~ 2025-12-31 기간 **미국 정량 매크로 이벤트** 약 200~300건 수집. **룰 기반 라벨링**까지 한 번에.

## 사전 준비

### FRED API 키 발급 (1분)

1. https://fred.stlouisfed.org/docs/api/api_key.html
2. 회원가입 + API 키 신청 → **즉시 발급**
3. `.env`에 `FRED_API_KEY=xxx` 저장

## 수집 대상 시계열

| FRED Series ID | 의미 | 이벤트 변환 규칙 | 영향 태그 |
| --- | --- | --- | --- |
| `DFEDTARU` | Fed Funds Target Rate (상단) | 변경일 → 이벤트 | macro_rate |
| `CPIAUCSL` | CPI (전체) | YoY 4% 초과 발표일 | macro_rate |
| `UNRATE` | 실업률 | 0.3%p 이상 변동 | equity_market |
| `DGS10` | 10년 국채금리 | 일변동 15bp 초과 | macro_rate |
| `DEXKOUS` | 원/달러 환율 | 일변동 1.5% 초과 | geopolitical_fx |
| `VIXCLS` | VIX | 30 이상 진입/유지 | equity_market |
| `DCOILWTICO` | WTI 원유 | 일변동 5% 초과 | geopolitical_fx |

## 룰 기반 라벨링 함수

```python
# scripts/collect_fred.py
from fredapi import Fred
import pandas as pd
import os
from dotenv import load_dotenv
import json
import uuid

load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

def label_fed_rate_change(diff_pct):
    """Fed 금리 변경 → macro_rate_risk 강도"""
    abs_change = abs(diff_pct)
    if abs_change >= 0.75: return 1.0    # 자이언트 스텝
    elif abs_change >= 0.50: return 1.0  # 빅 스텝
    elif abs_change >= 0.25: return 0.66 # 일반 스텝
    elif abs_change > 0:     return 0.33 # 미세 조정
    return 0.0

def label_cpi(yoy_pct):
    """CPI YoY → macro_rate_risk"""
    if yoy_pct >= 6.0:   return 1.0   # 인플레이션 위기
    elif yoy_pct >= 4.0: return 0.66  # 높은 인플레
    elif yoy_pct >= 3.0: return 0.33  # 약간 높음
    return 0.0

def label_treasury_yield(diff_bp):
    """10년물 일변동 → macro_rate_risk"""
    abs_change = abs(diff_bp)
    if abs_change >= 25:   return 1.0
    elif abs_change >= 15: return 0.66
    elif abs_change >= 10: return 0.33
    return 0.0

def label_usdkrw(daily_change_pct):
    """원/달러 일변동 → geopolitical_fx_risk"""
    abs_change = abs(daily_change_pct)
    if abs_change >= 2.0:   return 1.0
    elif abs_change >= 1.5: return 0.66
    elif abs_change >= 1.0: return 0.33
    return 0.0

def label_vix(vix_value):
    """VIX 수준 → equity_market_risk"""
    if vix_value >= 40:    return 1.0   # 패닉
    elif vix_value >= 30:  return 0.66  # 공포
    elif vix_value >= 25:  return 0.33  # 불안
    return 0.0

def label_oil(daily_change_pct):
    """WTI 일변동 → geopolitical_fx_risk"""
    abs_change = abs(daily_change_pct)
    if abs_change >= 8.0:  return 1.0
    elif abs_change >= 5.0: return 0.66
    elif abs_change >= 3.0: return 0.33
    return 0.0
```

## 수집 + 라벨링 통합 코드

```python
def collect_fred_events():
    events = []

    # 1. Fed 금리 변경
    fed_rate = fred.get_series('DFEDTARU', '2018-01-01', '2025-12-31')
    fed_changes = fed_rate.diff().dropna()
    for date, diff in fed_changes[fed_changes != 0].items():
        severity = label_fed_rate_change(diff)
        if severity > 0:
            events.append({
                "event_id": f"fred-{date.strftime('%Y%m%d')}-fedrate",
                "date": date.date(),
                "source": "fred",
                "title": f"Fed Funds Rate {'+' if diff > 0 else ''}{diff:.2f}%p",
                "summary": f"Fed target rate changed by {diff:.2f}%p to {fed_rate[date]:.2f}%",
                "url": "https://fred.stlouisfed.org/series/DFEDTARU",
                "language": "en",
                "raw_data": json.dumps({
                    "series": "DFEDTARU",
                    "diff": float(diff),
                    "new_rate": float(fed_rate[date])
                }),
                "label_method": "rule_based",
                # 룰 기반 라벨링 결과 (LLM 안 거치고 바로)
                "macro_rate_risk": severity,
                "equity_market_risk": severity * 0.5,  # 간접 영향
                "geopolitical_fx_risk": severity * 0.3,
                "primary_tag": "macro_rate",
                "reasoning": f"Fed rate change {diff:+.2f}%p",
                "confidence": 1.0
            })

    # 2. CPI 발표 (월별)
    cpi = fred.get_series('CPIAUCSL', '2017-01-01', '2025-12-31')
    cpi_yoy = cpi.pct_change(12) * 100  # YoY
    for date, yoy in cpi_yoy.dropna().items():
        if date < pd.Timestamp('2018-01-01'): continue
        severity = label_cpi(yoy)
        if severity > 0:
            events.append({
                "event_id": f"fred-{date.strftime('%Y%m%d')}-cpi",
                "date": date.date(),
                "source": "fred",
                "title": f"US CPI YoY {yoy:.1f}%",
                "summary": f"Consumer Price Index YoY change: {yoy:.2f}%",
                "url": "https://fred.stlouisfed.org/series/CPIAUCSL",
                "language": "en",
                "raw_data": json.dumps({"series": "CPIAUCSL", "yoy": float(yoy)}),
                "label_method": "rule_based",
                "macro_rate_risk": severity,
                "equity_market_risk": severity * 0.3,
                "geopolitical_fx_risk": 0.0,
                "primary_tag": "macro_rate",
                "reasoning": f"CPI YoY {yoy:.1f}%",
                "confidence": 1.0
            })

    # 3. 10년물 국채금리 (일변동 큰 날만)
    treasury = fred.get_series('DGS10', '2018-01-01', '2025-12-31')
    treasury_diff = (treasury.diff() * 100).dropna()  # bp 단위
    for date, diff in treasury_diff.items():
        severity = label_treasury_yield(diff)
        if severity > 0:
            events.append({
                "event_id": f"fred-{date.strftime('%Y%m%d')}-treasury",
                "date": date.date(),
                "source": "fred",
                "title": f"US 10Y Treasury {'+' if diff > 0 else ''}{diff:.0f}bp",
                "summary": f"10Y Treasury yield changed by {diff:.1f}bp",
                "url": "https://fred.stlouisfed.org/series/DGS10",
                "language": "en",
                "raw_data": json.dumps({"series": "DGS10", "diff_bp": float(diff)}),
                "label_method": "rule_based",
                "macro_rate_risk": severity,
                "equity_market_risk": severity * 0.4,
                "geopolitical_fx_risk": 0.0,
                "primary_tag": "macro_rate",
                "reasoning": f"10Y yield {diff:+.0f}bp",
                "confidence": 1.0
            })

    # 4. VIX (30 이상 진입일)
    vix = fred.get_series('VIXCLS', '2018-01-01', '2025-12-31').dropna()
    # 25 이하에서 25 이상으로 올라간 첫 날만 이벤트로
    vix_above_25 = vix >= 25
    vix_entries = vix_above_25 & ~vix_above_25.shift(1, fill_value=False)
    for date in vix[vix_entries].index:
        severity = label_vix(vix[date])
        if severity > 0:
            events.append({
                "event_id": f"fred-{date.strftime('%Y%m%d')}-vix",
                "date": date.date(),
                "source": "fred",
                "title": f"VIX spiked to {vix[date]:.1f}",
                "summary": f"VIX entered {vix[date]:.1f} (fear zone)",
                "url": "https://fred.stlouisfed.org/series/VIXCLS",
                "language": "en",
                "raw_data": json.dumps({"series": "VIXCLS", "value": float(vix[date])}),
                "label_method": "rule_based",
                "macro_rate_risk": 0.0,
                "equity_market_risk": severity,
                "geopolitical_fx_risk": 0.0,
                "primary_tag": "equity_market",
                "reasoning": f"VIX spike to {vix[date]:.1f}",
                "confidence": 1.0
            })

    # 5. WTI 원유 (일변동 큰 날)
    oil = fred.get_series('DCOILWTICO', '2018-01-01', '2025-12-31').dropna()
    oil_pct = oil.pct_change() * 100
    for date, pct in oil_pct.dropna().items():
        severity = label_oil(pct)
        if severity > 0:
            events.append({
                "event_id": f"fred-{date.strftime('%Y%m%d')}-oil",
                "date": date.date(),
                "source": "fred",
                "title": f"WTI Oil {'+' if pct > 0 else ''}{pct:.1f}%",
                "summary": f"WTI Crude Oil price changed {pct:.2f}% in one day",
                "url": "https://fred.stlouisfed.org/series/DCOILWTICO",
                "language": "en",
                "raw_data": json.dumps({"series": "DCOILWTICO", "daily_change_pct": float(pct)}),
                "label_method": "rule_based",
                "macro_rate_risk": 0.0,
                "equity_market_risk": severity * 0.3,
                "geopolitical_fx_risk": severity,
                "primary_tag": "geopolitical_fx",
                "reasoning": f"Oil daily change {pct:+.1f}%",
                "confidence": 1.0
            })

    df = pd.DataFrame(events)

    # market close 보정 (FRED 발표는 미국 시간 기준)
    df['date'] = df['date'].apply(lambda d: adjust_date_for_market_close(
        pd.Timestamp(d).tz_localize('UTC'), 'US'
    ))

    os.makedirs("data/raw/fred", exist_ok=True)
    df.to_parquet("data/raw/fred/fred_events_2018_2025.parquet", index=False)
    print(f"FRED 이벤트 수집 완료: {len(df)}건")
    return df

if __name__ == "__main__":
    collect_fred_events()
```

## 체크리스트

- [ ]  FRED API 키 발급, `.env` 등록
- [ ]  `fredapi` 라이브러리 설치
- [ ]  각 시계열 개별로 1년치 먼저 테스트 (`fred.get_series('DFEDTARU', '2024-01-01', '2024-12-31')`)
- [ ]  룰 기반 라벨링 함수 단위 테스트 (Fed 0.75%p → 1.0 나오는지)
- [ ]  전체 수집 실행, 200~300건 확인
- [ ]  **2022년 6월 FOMC 0.75%p 인상이 1.0으로 라벨링됐는지 sanity check**
- [ ]  **2022년 6월 CPI 9.1% (40년 만의 최고치) 잡혔는지 확인**
- [ ]  Parquet 저장
- [ ]  결과 분포 (태그별 강도 히스토그램) 팀 공유

---

# 👤 담당자 C: ECOS API (한국은행 매크로 지표)

## 목표

2018-01-01 ~ 2025-12-31 기간 **한국 정량 매크로 이벤트** 약 150~250건 수집. **룰 기반 라벨링** 포함.

## 사전 준비

### ECOS API 키 발급 (10분)

1. https://ecos.bok.or.kr/api
2. 회원가입 + 인증키 신청 → 즉시 발급
3. `.env`에 `ECOS_API_KEY=xxx` 저장
4. 일일 호출 한도: **10,000건** (우리는 안 넘김)

## 수집 대상 시계열

| ECOS 통계코드 | 항목코드 | 의미 | 라벨링 규칙 | 영향 태그 |
| --- | --- | --- | --- | --- |
| 722Y001 | 0101000 | 한국은행 기준금리 | 변경일 → 이벤트 | macro_rate |
| 901Y009 | 0 | 소비자물가지수 (총지수) | YoY 4% 초과 | macro_rate |
| 731Y001 | 0000003 | 국고채 3년 수익률 | 일변동 10bp 초과 | macro_rate (114260 직접 영향) |
| 731Y001 | 0000005 | 국고채 10년 수익률 | 일변동 15bp 초과 | macro_rate |
| 731Y001 | 0000001 | 원/달러 환율 | 일변동 1.5% 초과 | geopolitical_fx |
| 802Y001 | 0000001 | KOSPI | 일변동 -2% 이하 | equity_market |

⚠️ 통계코드는 ECOS 사이트에서 직접 확인 권장 (가끔 변경됨). 통계 검색 페이지에서 "기준금리" 등으로 검색하면 코드 확인 가능.

## ECOS API 사용법

ECOS는 REST API라 `requests`로 직접 호출.

```python
# URL 형식
# http://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/1000/{통계코드}/{주기}/{시작일}/{종료일}/{항목코드1}

# 예: 한국은행 기준금리 (722Y001), 일별(D), 2020~2025
url = f"http://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/10000/722Y001/D/20180101/20251231/0101000"
```

## 수집 + 라벨링 통합 코드

```python
# scripts/collect_ecos.py
import requests
import pandas as pd
import os
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()
API_KEY = os.getenv("ECOS_API_KEY")
BASE_URL = "http://ecos.bok.or.kr/api/StatisticSearch"

def fetch_ecos(stat_code, item_code, cycle, start, end):
    """ECOS API 호출 헬퍼"""
    url = f"{BASE_URL}/{API_KEY}/json/kr/1/10000/{stat_code}/{cycle}/{start}/{end}/{item_code}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if 'StatisticSearch' not in data:
        print(f"⚠️ 데이터 없음: {stat_code}/{item_code}")
        print(data)
        return pd.DataFrame()

    rows = data['StatisticSearch']['row']
    df = pd.DataFrame(rows)
    df['TIME'] = pd.to_datetime(df['TIME'], format='%Y%m%d' if cycle == 'D' else '%Y%m')
    df['DATA_VALUE'] = pd.to_numeric(df['DATA_VALUE'])
    return df[['TIME', 'DATA_VALUE']].sort_values('TIME')

# 라벨링 함수 (FRED 담당자와 동일한 기준)
def label_bok_rate_change(diff_pct):
    abs_change = abs(diff_pct)
    if abs_change >= 0.50:   return 1.0
    elif abs_change >= 0.25: return 0.66
    elif abs_change > 0:     return 0.33
    return 0.0

def label_kr_cpi(yoy_pct):
    if yoy_pct >= 5.0:   return 1.0
    elif yoy_pct >= 4.0: return 0.66
    elif yoy_pct >= 3.0: return 0.33
    return 0.0

def label_kr_bond_yield(diff_bp, maturity='3y'):
    threshold = {'3y': 8, '10y': 12}[maturity]
    abs_change = abs(diff_bp)
    if abs_change >= threshold * 2:   return 1.0
    elif abs_change >= threshold * 1.5: return 0.66
    elif abs_change >= threshold:       return 0.33
    return 0.0

def label_usdkrw(daily_change_pct):
    abs_change = abs(daily_change_pct)
    if abs_change >= 2.0:   return 1.0
    elif abs_change >= 1.5: return 0.66
    elif abs_change >= 1.0: return 0.33
    return 0.0

def label_kospi_drop(daily_change_pct):
    """KOSPI 하락폭 → equity_market_risk"""
    if daily_change_pct <= -3.0:   return 1.0
    elif daily_change_pct <= -2.0: return 0.66
    elif daily_change_pct <= -1.5: return 0.33
    return 0.0

def collect_ecos_events():
    events = []

    # 1. 한국은행 기준금리
    rate = fetch_ecos('722Y001', '0101000', 'D', '20180101', '20251231')
    rate_changes = rate.set_index('TIME')['DATA_VALUE'].diff().dropna()
    for date, diff in rate_changes[rate_changes != 0].items():
        severity = label_bok_rate_change(diff)
        if severity > 0:
            events.append({
                "event_id": f"ecos-{date.strftime('%Y%m%d')}-bokrate",
                "date": date.date(),
                "source": "ecos",
                "title": f"한국은행 기준금리 {'+' if diff > 0 else ''}{diff:.2f}%p",
                "summary": f"한국은행 기준금리 {diff:+.2f}%p 조정",
                "url": "https://ecos.bok.or.kr",
                "language": "ko",
                "raw_data": json.dumps({"series": "722Y001/0101000", "diff": float(diff)}),
                "label_method": "rule_based",
                "macro_rate_risk": severity,
                "equity_market_risk": severity * 0.5,
                "geopolitical_fx_risk": severity * 0.3,
                "primary_tag": "macro_rate",
                "reasoning": f"한은 기준금리 {diff:+.2f}%p",
                "confidence": 1.0
            })

    # 2. 한국 CPI (월별)
    cpi = fetch_ecos('901Y009', '0', 'M', '201701', '202512')
    cpi = cpi.set_index('TIME')['DATA_VALUE']
    cpi_yoy = cpi.pct_change(12) * 100
    for date, yoy in cpi_yoy.dropna().items():
        if date < pd.Timestamp('2018-01-01'): continue
        severity = label_kr_cpi(yoy)
        if severity > 0:
            events.append({
                "event_id": f"ecos-{date.strftime('%Y%m%d')}-cpi",
                "date": date.date(),
                "source": "ecos",
                "title": f"한국 CPI YoY {yoy:.1f}%",
                "summary": f"소비자물가지수 전년 동월 대비 {yoy:.2f}%",
                "url": "https://ecos.bok.or.kr",
                "language": "ko",
                "raw_data": json.dumps({"series": "901Y009", "yoy": float(yoy)}),
                "label_method": "rule_based",
                "macro_rate_risk": severity,
                "equity_market_risk": severity * 0.3,
                "geopolitical_fx_risk": 0.0,
                "primary_tag": "macro_rate",
                "reasoning": f"한국 CPI YoY {yoy:.1f}%",
                "confidence": 1.0
            })

    # 3. 국고채 3년 (114260 직접 영향, 매우 중요!)
    bond3y = fetch_ecos('731Y001', '0000003', 'D', '20180101', '20251231')
    bond3y_diff = (bond3y.set_index('TIME')['DATA_VALUE'].diff() * 100).dropna()
    for date, diff in bond3y_diff.items():
        severity = label_kr_bond_yield(diff, '3y')
        if severity > 0:
            events.append({
                "event_id": f"ecos-{date.strftime('%Y%m%d')}-bond3y",
                "date": date.date(),
                "source": "ecos",
                "title": f"국고채 3년 {'+' if diff > 0 else ''}{diff:.0f}bp",
                "summary": f"국고채 3년물 수익률 {diff:+.1f}bp 변동 (114260 직접 영향)",
                "url": "https://ecos.bok.or.kr",
                "language": "ko",
                "raw_data": json.dumps({"series": "731Y001/0000003", "diff_bp": float(diff)}),
                "label_method": "rule_based",
                "macro_rate_risk": severity,
                "equity_market_risk": 0.0,
                "geopolitical_fx_risk": 0.0,
                "primary_tag": "macro_rate",
                "reasoning": f"국고채 3년 {diff:+.0f}bp",
                "confidence": 1.0
            })

    # 4. 원/달러 환율
    usdkrw = fetch_ecos('731Y001', '0000001', 'D', '20180101', '20251231')
    usdkrw_pct = usdkrw.set_index('TIME')['DATA_VALUE'].pct_change() * 100
    for date, pct in usdkrw_pct.dropna().items():
        severity = label_usdkrw(pct)
        if severity > 0:
            events.append({
                "event_id": f"ecos-{date.strftime('%Y%m%d')}-usdkrw",
                "date": date.date(),
                "source": "ecos",
                "title": f"원/달러 {'+' if pct > 0 else ''}{pct:.2f}%",
                "summary": f"원/달러 환율 일변동 {pct:.2f}%",
                "url": "https://ecos.bok.or.kr",
                "language": "ko",
                "raw_data": json.dumps({"series": "731Y001/0000001", "daily_change_pct": float(pct)}),
                "label_method": "rule_based",
                "macro_rate_risk": 0.0,
                "equity_market_risk": severity * 0.3,
                "geopolitical_fx_risk": severity,
                "primary_tag": "geopolitical_fx",
                "reasoning": f"원/달러 {pct:+.2f}%",
                "confidence": 1.0
            })

    # 5. KOSPI 하락일
    kospi = fetch_ecos('802Y001', '0000001', 'D', '20180101', '20251231')
    kospi_pct = kospi.set_index('TIME')['DATA_VALUE'].pct_change() * 100
    for date, pct in kospi_pct.dropna().items():
        severity = label_kospi_drop(pct)
        if severity > 0:
            events.append({
                "event_id": f"ecos-{date.strftime('%Y%m%d')}-kospi",
                "date": date.date(),
                "source": "ecos",
                "title": f"KOSPI {pct:.2f}%",
                "summary": f"KOSPI 일간 변동 {pct:.2f}%",
                "url": "https://ecos.bok.or.kr",
                "language": "ko",
                "raw_data": json.dumps({"series": "802Y001/0000001", "daily_change_pct": float(pct)}),
                "label_method": "rule_based",
                "macro_rate_risk": 0.0,
                "equity_market_risk": severity,
                "geopolitical_fx_risk": severity * 0.3,
                "primary_tag": "equity_market",
                "reasoning": f"KOSPI {pct:+.2f}%",
                "confidence": 1.0
            })

    df = pd.DataFrame(events)

    # market close 보정 (한국 시장 기준)
    df['date'] = df['date'].apply(lambda d: adjust_date_for_market_close(
        pd.Timestamp(d).tz_localize('Asia/Seoul'), 'KR'
    ))

    os.makedirs("data/raw/ecos", exist_ok=True)
    df.to_parquet("data/raw/ecos/ecos_events_2018_2025.parquet", index=False)
    print(f"ECOS 이벤트 수집 완료: {len(df)}건")
    return df

if __name__ == "__main__":
    collect_ecos_events()
```

## 체크리스트

- [ ]  ECOS 회원가입, API 키 발급
- [ ]  통계코드 사전 확인 (사이트에서 직접 검색 → 코드가 맞는지)
- [ ]  단일 시계열로 먼저 테스트 (기준금리 1년치)
- [ ]  라벨링 함수 단위 테스트
- [ ]  전체 수집 실행, 150~250건 확인
- [ ]  **2022년 한국은행 빅스텝(0.5%p) 인상이 1.0으로 잡혔는지 sanity check**
- [ ]  **2022년 9월 원/달러 1,400원 돌파 시점 잡혔는지 확인**
- [ ]  Parquet 저장
- [ ]  결과 분포 팀 공유

---

# 👤 담당자 D: 수동 시드 큐레이션

## 목표

2018-01-01 ~ 2025-12-31 기간 **시장에 큰 충격을 준 30~50개 빅 이벤트**를 직접 정리. 자동 수집의 안전망 + LLM 라벨링 검증셋 역할.

## 수집 방법

### 참고 소스 (이 순서로 훑기)

1. **위키피디아 영문**
    - "2020 in economics", "2021 in economics", ... "2025 in economics"
    - "2020 stock market crash", "2022 stock market decline"
    - "Russo-Ukrainian War" → 주요 분기점 날짜
2. **위키피디아 한글**
    - "2020년 코로나19 범유행으로 인한 경제적 영향"
    - "2022년 한국 경제"
3. **한국은행 통화정책방향 결정문 목록**
    - https://www.bok.or.kr/portal/bbs/B0000216/list.do?menuNo=200755
    - 빅스텝/자이언트 스텝 결정일 정리
4. **블룸버그/로이터 "Year in Review" 기사** (Google 검색)
    - "biggest market events 2022", "key financial events 2023" 등
5. **나무위키 (보조)**
    - "코로나19/경제적 영향", "2022년 인플레이션" 등

## 큐레이션 가이드라인

### 포함 기준

- 시장에 **5% 이상 충격을 준 단일 이벤트**
- 또는 **장기적 추세 전환점** (Fed 정책 전환, 전쟁 발발 등)
- 또는 **자산군 가격 패턴이 명확히 바뀐 날**

### 제외 기준

- 점진적 변화 (자동 수집에서 잡힘)
- 단발성 종목 이슈 (ETF에 영향 작음)

### 라벨링 가이드 (4단계)

- **1.0**: 단일 이벤트로 시장 -3% 이상 또는 패러다임 전환
- **0.66**: 시장 -1.5%~-3% 또는 명확한 정책 변경
- **0.33**: 시장 -1%~-1.5% 또는 중요한 발표/신호
- **0.0**: 사용 안 함 (시드는 모두 의미 있는 이벤트만)

## 시드 데이터 CSV 템플릿

`data/raw/manual_seed/manual_seed_events.csv` 파일을 직접 작성:

```
event_id,date,source,title,summary,url,language,macro_rate_risk,equity_market_risk,geopolitical_fx_risk,primary_tag,reasoning,confidence
seed-20200316-covid-crash,2020-03-16,manual_seed,코로나 패닉 - 미 증시 서킷브레이커 발동,"S&P 500 -12%, 다우 -2997p, 두 번째 서킷브레이커. WHO 팬데믹 선언 이후 글로벌 자산 폭락",https://en.wikipedia.org/wiki/2020_stock_market_crash,ko,0.66,1.0,0.66,equity_market,코로나 패닉 서킷브레이커 12% 폭락,1.0
seed-20200323-fed-qe-infinity,2020-03-23,manual_seed,Fed 무제한 QE 발표,Fed가 무제한 양적완화와 회사채 매입 발표. 시장 바닥 형성,https://en.wikipedia.org/wiki/Federal_Reserve_responses_to_the_COVID-19_pandemic,en,1.0,0.66,0.33,macro_rate,Fed 무제한 QE 정책 전환,1.0
seed-20220224-russia-ukraine,2022-02-24,manual_seed,러시아 우크라이나 침공 개시,러시아 우크라이나 전면 침공. 유가 100달러 돌파 안전자산 급등,https://en.wikipedia.org/wiki/Russian_invasion_of_Ukraine,en,0.33,0.66,1.0,geopolitical_fx,전쟁 발발 유가 100달러 돌파,1.0
seed-20220615-fomc-75bp,2022-06-15,manual_seed,FOMC 0.75%p 자이언트 스텝,22년 만의 0.75%p 인상. 파월 추가 인상 시사. 성장주 급락,https://en.wikipedia.org/wiki/2022_United_States_inflation,en,1.0,0.66,0.33,macro_rate,자이언트 스텝 22년만의 충격,1.0
seed-20221007-chip-export,2022-10-07,manual_seed,미국 대중 반도체 수출 규제 강화,미국 반도체 장비 수출 통제 발표. 한국 반도체주 영향,https://en.wikipedia.org/wiki/2022_United_States_export_controls_on_China,ko,0.0,0.66,1.0,geopolitical_fx,반도체 수출 규제 한국 충격,1.0
seed-20230310-svb-collapse,2023-03-10,manual_seed,실리콘밸리은행(SVB) 파산,SVB 파산으로 은행주 폭락 VIX 28 돌파,https://en.wikipedia.org/wiki/Collapse_of_Silicon_Valley_Bank,en,0.66,1.0,0.0,equity_market,SVB 파산 패닉,1.0
```

## 빅 이벤트 후보 리스트 (출발점)

이 리스트를 시작점으로 채워넣어. 30개 정도 골라서 정리하면 돼.

### 2020 (코로나 충격)

- 2020-03-09: 사우디-러시아 유가 전쟁, WTI 25% 폭락
- 2020-03-12: 미국 유럽발 입국 금지, 시장 -10%
- 2020-03-16: 두 번째 서킷브레이커, S&P -12%
- 2020-03-23: Fed 무제한 QE 발표 (시장 바닥)
- 2020-08-27: Fed 평균물가목표제(AIT) 도입

### 2021 (회복과 인플레 시작)

- 2021-01-27: 게임스톱 숏스퀴즈 (변동성)
- 2021-11-30: 파월 "transitory" 단어 폐기, 매파적 전환
- 2021-12-15: FOMC 테이퍼링 가속화 결정

### 2022 (금리 충격기)

- 2022-01-05: FOMC 의사록 매파적, 성장주 조정 시작
- 2022-02-24: 러시아 우크라이나 침공
- 2022-03-16: Fed 첫 금리인상 (0.25%p)
- 2022-05-04: Fed 0.5%p 인상 (빅스텝)
- 2022-06-13: 미국 CPI 8.6%, 시장 -3.9%
- 2022-06-15: FOMC 0.75%p 인상 (자이언트 스텝)
- 2022-07-13: 한국은행 빅스텝(0.5%p) 첫 인상
- 2022-09-21: FOMC 0.75%p 3연속
- 2022-09-23: 영국 미니예산 위기, 파운드 폭락
- 2022-09-28: 영국 BOE 긴급 개입, 국채 매입
- 2022-10-07: 미국 대중 반도체 수출 규제
- 2022-10-25: 원/달러 1,444원 (13년 만의 최고)
- 2022-11-10: 미국 CPI 7.7% 둔화, 시장 +5.5%

### 2023 (은행 위기와 안정화)

- 2023-03-10: SVB 파산
- 2023-03-19: UBS의 크레디트 스위스 인수
- 2023-08-01: Fitch 미국 신용등급 강등
- 2023-10-07: 하마스 이스라엘 공격, 중동 분쟁
- 2023-12-13: FOMC 비둘기 전환

### 2024 (AI 랠리와 후반 변동성)

- 2024-02-21: 엔비디아 어닝 서프라이즈
- 2024-03-04: 비트코인 ATH (자산 가격 거품 논의)
- 2024-08-05: 일본 닛케이 -12% (역대 최대 하락), 글로벌 변동성
- 2024-09-18: FOMC 50bp 인하 (피벗 확인)
- 2024-11-06: 트럼프 대선 승리 (관세 우려)

### 2025 (관세/지정학)

- 2025-02-01: 트럼프 캐나다/멕시코 관세 발표
- 2025-04-02: 트럼프 "Liberation Day" 상호 관세 (대규모 시장 충격)
- 2025-04-09: 관세 90일 유예 발표
- (그 외 2025년 본인 기억 + 검색으로 보완)

## 변환 스크립트

CSV로 정리한 다음 Parquet으로 변환:

```python
# scripts/collect_manual_seed.py
import pandas as pd
import os

df = pd.read_csv("data/raw/manual_seed/manual_seed_events.csv")
df['date'] = pd.to_datetime(df['date']).dt.date
df['label_method'] = 'manual'
df['raw_data'] = '{}'

# 컬럼 순서 통일
columns = ['event_id', 'date', 'source', 'title', 'summary', 'url', 'language',
           'raw_data', 'label_method',
           'macro_rate_risk', 'equity_market_risk', 'geopolitical_fx_risk',
           'primary_tag', 'reasoning', 'confidence']
df = df[columns]

os.makedirs("data/raw/manual_seed", exist_ok=True)
df.to_parquet("data/raw/manual_seed/manual_seed_events.parquet", index=False)
print(f"수동 시드 변환 완료: {len(df)}건")
```

## 체크리스트

- [ ]  위키피디아 "2018~2025 in economics" 페이지 훑기
- [ ]  후보 리스트 50개 정도 추려내기
- [ ]  CSV 템플릿에 30~50개 입력
- [ ]  각 이벤트마다 4단계 강도 신중하게 부여 (애매하면 0.66)
- [ ]  **각 연도 윈도우(W1~W4)마다 최소 5개 이상 이벤트 확보**
- [ ]  Parquet 변환
- [ ]  팀 리뷰: 다른 팀원들이 빠진 이벤트 추가 제안 받기 (특히 본인이 잘 모르는 분야)

---

# 📊 통합 단계 (모두 끝난 후)

각자 수집 완료하면 PM이 통합:

```python
# scripts/merge_all_sources.py
import pandas as pd
import os

dfs = []
for source in ['gdelt', 'fred', 'ecos', 'manual_seed']:
    if source == "manual_seed":
        path = f"data/raw/{source}/manual_seed_events.parquet"
    else:
        path = f"data/raw/{source}/{source}_events_2018_2025.parquet"
    df = pd.read_parquet(path)
    df['_source_file'] = source
    dfs.append(df)
    print(f"{source}: {len(df)}건")

merged = pd.concat(dfs, ignore_index=True)
merged = merged.sort_values('date').reset_index(drop=True)

# 중복 제거 (동일 날짜 + 동일 source + 유사 제목)
merged = merged.drop_duplicates(subset=['date', 'source', 'title'], keep='first')

os.makedirs("data/processed", exist_ok=True)
merged.to_parquet("data/processed/unified_events.parquet", index=False)
print(f"\n총 통합: {len(merged)}건")
print(merged.groupby('source').size())
```

---

# 📋 작업 분담 요약표

| 담당자 | 소스 | 예상 작업 시간 | 산출물 |
| --- | --- | --- | --- |
| A | GDELT 2.0 (BigQuery) | 8~12h | 2,000~3,000건 (LLM 라벨링 대기) |
| B | FRED API | 4~6h | 200~300건 (룰 기반 라벨링 완료) |
| C | ECOS API | 4~6h | 150~250건 (룰 기반 라벨링 완료) |
| D | 수동 시드 큐레이션 | 3~5h | 30~50건 (수동 라벨링 완료) |

**전체 병렬 작업 가능. 의존성 없음.**

라벨링 단계는 GDELT만 LLM 처리하면 되니까 담당자 A 끝난 후에 별도로 진행하면 돼. B/C/D는 룰 기반/수동이라 수집과 동시에 라벨링까지 완료.

---

# 🚨 모든 담당자에게 강조

1. **API 키 절대 GitHub에 올리지 마.** `.env`는 `.gitignore`에 반드시 추가.
2. **결과 sanity check 필수.** 본인이 정한 빅 이벤트(예: 2022 자이언트 스텝)가 1.0으로 잡혔는지 직접 확인.
3. **수집 코드도 같이 커밋.** 누가 봐도 재현 가능하게.
4. **작업 중 막히면 즉시 공유.** 4명 병렬이라 한 명이 막히면 통합 단계가 지연됨.