# ChromaDB 파이프라인 설계 문서

> 작성: 강유영 | 최종 수정: 2026-05-23

---

## 1. 전체 구조

ChromaDB `finance_news` 컬렉션에는 **두 경로**로 데이터가 유입된다.

```
[경로 A] 과거 이벤트 (1회성 배치)
  unified_events.parquet
  (FRED + ECOS + manual_seed, 1,226건)
        │
        ▼
  build_chroma_from_parquet.py --clear
        │
        ▼
  finance_news 컬렉션

[경로 B] 실시간 뉴스 (주기적 실행)
  Google News RSS (3개 피드)
  + ECOS StatisticTableList (보조)
        │
        ▼
  news_collector.py
  collect_google_news_and_store()
        │
        ▼
  finance_news 컬렉션 (upsert, URL MD5 중복 방지)

[사용]
  /research API 호출
        │
        ▼
  researcher_node() → query_documents()
        │
        ▼
  RAG 검색 결과 → analyst_node() → 리포트 생성
```

GDELT는 ChromaDB에서 **영구 제외**. title/summary가 GKG 기계코드 나열 형식이라 자연어 임베딩 품질이 극히 낮음. GDELT LLM 라벨링 결과는 `risk_vectors_daily.parquet` 생성(RL 학습용)에만 사용.

---

## 2. 데이터 소스별 크롤링 기준

### 2-1. 과거 이벤트 (경로 A)

| 소스 | 수집 기간 | 수집 기준 | 라벨링 방식 | 건수 |
|------|-----------|-----------|-------------|------|
| FRED | 2018-01-01 ~ 2025-12-31 | 금리 변경, CPI YoY ≥4%, VIX ≥25 진입, WTI 일변동 ≥5%, 10년물 ≥15bp | 룰 기반 | ~732건 |
| ECOS | 2018-01-01 ~ 2025-12-31 | 한은 기준금리 변경, 국고채 3Y ≥8bp, 원/달러 일변동 ≥1.5%, KOSPI 일변동 ≤-1.5% | 룰 기반 | ~455건 |
| manual_seed | 2018-01-01 ~ 2025-12-31 | 시장 충격 ≥5% 또는 패러다임 전환점 — 연구자가 직접 선정 | 수동 | 39건 |
| GDELT | 2018-01-01 ~ 2025-12-31 | NumMentions ≥50, GoldsteinScale < -3.0, 주요 국가 관련 이벤트 | LLM | 1,197건 (ChromaDB 제외) |

**수집 기간 근거**: Walk-Forward 윈도우 4개 (W1 테스트=2022, W2=2023, W3=2024, W4=2025)를 모두 커버하려면 학습 데이터 기준 2018년부터 필요.

**FRED 수집 지표 선정 근거**:
- `DFEDTARU` (Fed 금리): RL 포트폴리오의 채권 ETF(114260) 직접 영향
- `CPIAUCSL` (CPI): 금리 방향성 선행 지표
- `VIXCLS` (VIX): 시장 공포 지수, 리스크 온/오프 판단
- `DCOILWTICO` (WTI): 에너지 ETF 및 인플레이션 연동
- `DGS10` (10년물): 장기 금리 환경
- `DEXKOUS` (원/달러): 국내 자산에 직접 영향

**ECOS 수집 지표 선정 근거**:
- 한은 기준금리: 국내 채권 ETF 직접 영향
- 국고채 3Y: 114260(KODEX 국고채 3년) 기초 자산
- 원/달러: 외화 자산 수익률 환산
- KOSPI 하락일: 국내 주식형 ETF 리스크 신호

### 2-2. 실시간 뉴스 (경로 B)

| 소스 | 피드/엔드포인트 | 수집 분야 | 수집 주기 |
|------|----------------|-----------|-----------|
| Google News RSS | `news.google.com/rss/search` | 미국주식, 글로벌시장, 채권금리, 금대체자산, 한국시장 | 수동 or 일 1회 cron |
| ECOS StatisticTableList | `ecos.bok.or.kr/api/StatisticTableList` | 거시 통계표 메타데이터 (보조) | 동일 |

**Google News 카테고리 및 대상 자산**:

| 카테고리 | 검색 키워드 | 대상 자산 |
|---------|-----------|---------|
| `미국주식` | 미국증시·S&P500·나스닥·뉴욕증시 | SPY(S&P500), QQQ(Nasdaq100), IWM(Russell2000) |
| `글로벌시장` | 신흥국증시·선진국주식·MSCI·글로벌시장 | EFA(선진국), EEM(신흥국) |
| `채권금리` | 미국국채·장기금리·금리·연준 | TLT(20yr Treasury), 114260(국고채 3Y) |
| `금대체자산` | 금시세·리츠·부동산·원자재 | GLD(금), VNQ(리츠) |
| `한국시장` | 코스피·한국증시·한국국채·코스닥 | 069500(KODEX 200), 114260(국고채 3Y) |

**카테고리 선정 근거**: 초기 설계(급등락·실적쇼크·규제변경)는 3종 리스크 태그를 그대로 피드로 옮긴 것이었음. 포트폴리오 자산 10종이 확정된 이후, 편입 자산에 직접 영향을 주는 뉴스를 우선 수집하도록 재설계.

**수집 주기 근거**: 요청 시 실시간 수집은 RSS 지연(1~3초)으로 응답 시간 저하. 사전 수집 후 ChromaDB 검색이 응답 안정성 우선인 경진대회 환경에 적합.

---

## 3. ChromaDB 컬렉션 구조

### 3-1. 컬렉션 설정

| 항목 | 값 |
|------|----|
| 컬렉션명 | `finance_news` |
| 거리 함수 | cosine (`hnsw:space=cosine`) |
| 저장 경로 | `./chroma_db` (환경변수 `CHROMA_PERSIST_DIR`로 변경 가능) |
| 임베딩 모델 | ChromaDB 기본 (`all-MiniLM-L6-v2`) |

### 3-2. 문서 ID 규칙

| 경로 | ID 형식 | 예시 |
|------|---------|------|
| 과거 parquet (경로 A) | `parquet_{event_id}` | `parquet_fred-20220615-fedrate` |
| 실시간 RSS (경로 B) | URL MD5 (32자 hex) | `a3f2b1c4d5e6...` |

두 경로가 ID 공간이 분리되어 있어 upsert 시 충돌 없음.

### 3-3. 메타데이터 스키마

| 필드 | 타입 | 설명 | 경로 A | 경로 B |
|------|------|------|--------|--------|
| `title` | str | 이벤트 제목 (≤200자) | ✅ | ✅ |
| `summary` | str | 본문 요약 (≤300자) | ✅ | ✅ |
| `url` | str | 원문 URL | ✅ | ✅ |
| `date` | str | 기준일 `YYYY-MM-DD` | ✅ | ✅ |
| `category` | str | 리스크 카테고리 | `macro_rate_risk` 등 | `급등락` 등 |
| `source` | str | 데이터 출처 | `fred` / `ecos` / `manual_seed` | `google_news` |
| `risk_label` | str | RL 3축 태그 (쉼표 구분) | ✅ | ✅ |

> **주의**: `category` 필드 형식이 경로 A(영문 _risk suffix)와 경로 B(한국어)가 다름. RAG 검색 자체는 텍스트 유사도 기반이라 category 필터링을 쓰지 않으면 영향 없음. 향후 통일 필요.

### 3-4. 리스크 태그 (`risk_label`) 값 정의

| 태그 | 의미 | 대표 이벤트 |
|------|------|------------|
| `macro_rate_risk` | 금리·통화정책·CPI | Fed 금리 인상, 한은 빅스텝 |
| `equity_market_risk` | 주가·변동성 | 코로나 서킷브레이커, SVB 파산 |
| `geopolitical_fx_risk` | 지정학·환율·무역 | 러-우 전쟁, 반도체 수출 규제 |

쉼표 구분으로 복수 태그 가능. 예: `macro_rate_risk,equity_market_risk`

---

## 4. 데이터 갱신 방법

### 4-1. 서버 최초 배포 시 (필수)

`chroma_db/`는 gitignore 대상이라 서버에 자동 배포되지 않는다.
컨테이너 기동 후 아래 명령을 **서버에서 1회 실행**해야 RAG가 동작한다.

```bash
# api 컨테이너 기동 후
docker compose exec api PYTHONPATH=. python scripts/build_chroma_from_parquet.py --clear
```

### 4-2. 로컬 개발 환경

```bash
# 과거 데이터 재빌드 (unified_events 변경 시)
PYTHONPATH=. python scripts/build_chroma_from_parquet.py --clear

# RAG 품질 검증
PYTHONPATH=. python scripts/rag_eval.py
```

### 4-3. 실시간 뉴스 수집 자동화

**로컬 cron은 적합하지 않다.** 로컬 PC는 상시 켜져 있지 않아 실행 시점을 보장할 수 없다.
자동화가 필요하다면 배포 서버에서 아래 방식 중 하나를 사용해야 한다.

```bash
# 서버에서 수동 실행 (현재 방식)
docker compose exec api PYTHONPATH=. python -m src.agent.news_collector
```

> **경진대회 데모 환경에서는 자동화 불필요.**
> 과거 8년치 데이터(1,226건)가 ChromaDB에 적재된 상태면 RAG는 정상 동작한다.
> 실시간 최신성이 필수인 서비스가 아니므로 배포 시 1회 실행으로 충분하다.

---

## 5. 알려진 한계 및 향후 개선

| 항목 | 현재 상태 | 개선 방향 |
|------|-----------|-----------|
| GDELT ChromaDB 미포함 | GKG 기계코드 형식 | title/summary 자연어 재생성 후 포함 |
| category 필드 불일치 | 경로 A/B 형식 다름 | new style 태그로 통일 |
| 실시간 수집 자동화 없음 | 서버 배포 후 수동 실행 | 배포 서버에 cron worker 컨테이너 추가 |
| Q08·Q11 태그 정확도 낮음 | ECOS 통계 자연어 매칭 약함 | 요약 텍스트 보강 |
