# RAG 평가 결과

> **측정일**: 2026-05-24
> **ChromaDB 상태**: FRED(732) + ECOS(455) + manual_seed(39) = **1,226건**.
> GDELT(1,197건)는 GKG 기계코드 형식이라 ChromaDB 영구 제외, `risk_vectors_daily.parquet` 집계에만 반영.
> **실시간 RSS는 평가 풀에서 제외**(재현성 확보). 운영(`/research`) 시에는 별도 수집 스케줄로 추가 적재됨.
> **재빌드 명령**: `PYTHONPATH=. python scripts/build_chroma_from_parquet.py --clear`
> **평가 스크립트**: `PYTHONPATH=. python scripts/rag_eval.py --json`
> **평가 질문**: `docs/rag_eval_questions.json` (12문항)
> **검색 파라미터**: `n_results=3`, `hit_threshold=0.4` (cosine distance, ChromaDB 기본 임베딩 `all-MiniLM-L6-v2`)
> **리스크 태그 체계**: 신버전 3축 — `macro_rate_risk` / `equity_market_risk` / `geopolitical_fx_risk` (`_risk` suffix 통일)

---

## 요약

| 지표 | 결과 |
|------|------|
| 검색 히트율 (top-1 cosine dist < 0.4) | **12 / 12 = 100%** |
| 태그 매칭률 `tag_ok` (`expect ∩ found ≠ ∅`) | **12 / 12 = 100%** |
| top_dist 평균 | 0.248 |
| top_dist 범위 | 0.203 ~ 0.367 |
| 메타데이터 회수 (`date`, `risk_label` 필드) | 12 / 12 |

> ⚠️ **`tag_ok` 지표의 정확한 정의**: `scripts/rag_eval.py` line 69 `tag_ok = bool(expect & found)` — **기대 태그 집합과 검색된 태그 집합의 교집합이 비어 있지 않은지** 확인하는 조건이다(부분집합 조건이 아님). 기대 태그 중 하나라도 회수되면 통과한다. `n_results=3` 검색에서 3축 태그가 거의 모든 질문에서 동시에 회수되므로 사실상 자동 통과에 가깝다. 실제 1위 문서의 적절성은 `top_dist`(전 구간 0.4 미만)와 `top_doc` 정성 검토로 보완 판단한다.

---

## 질문별 상세

| id | 질문 (요약) | hit | top_dist | expect | found_tags | tag_ok |
|----|-------------|-----|----------|--------|------------|--------|
| Q01 | 금리 인상이 ETF 포트폴리오에 미치는 영향 | OK | 0.2079 | macro | macro/equity/geo | OK |
| Q02 | 한국 금융당국 규제 강화·정책 변경 시 투자 리스크 | OK | 0.2362 | geo | macro/equity/geo | OK |
| Q03 | 코스피 급락·변동성 확대 시 포트폴리오 리스크 | OK | 0.2030 | equity | macro/equity/geo | OK |
| Q04 | 어닝쇼크·실적 부진 관련 최근 증시 리스크 | OK | 0.2090 | equity | macro/equity/geo | OK |
| Q05 | 미국 연준 기준금리 결정이 한국 채권 ETF에 미치는 영향 | OK | 0.2586 | macro | macro/equity/geo | OK |
| Q06 | 원달러 환율 급등 시 포트폴리오 리스크 관리 | OK | 0.2070 | geo | macro/equity/geo | OK |
| Q07 | 지정학적 리스크(전쟁·무역분쟁)가 국내 증시에 미치는 영향 | OK | 0.2755 | equity, geo | macro/equity/geo | OK |
| Q08 | 인플레이션 상승이 자산 배분 전략에 미치는 영향 | OK | 0.2975 | macro | macro/equity/geo | OK |
| Q09 | 시장 변동성 VIX 지수 급등 시 방어적 포트폴리오 구성 | OK | 0.3674 | equity | macro/equity/geo | OK |
| Q10 | 공급망 차질·원자재 가격 급등이 ETF 수익률에 미치는 영향 | OK | 0.2052 | macro, geo | macro/equity/geo | OK |
| Q11 | 한국은행 기준금리 인상이 부동산·채권 시장에 미치는 영향 | OK | 0.2626 | macro | macro/equity/geo | OK |
| Q12 | 글로벌 경기침체 우려가 국내 주식형 ETF에 미치는 리스크 | OK | 0.2519 | macro, equity | macro/equity/geo | OK |

---

## 유효한 검증 항목

- **검색 파이프라인 작동**: 12/12 (`top_dist < 0.4` 통과)
- **메타데이터 회수**: 12/12 (`date`·`risk_label` 모두 회수, `top_doc`/`top_date` 출력 확인). URL·source 등 추가 메타데이터 회수 여부는 별도 컬렉션 dump로 확인 예정.
- **신규 태그 체계 적용**: 모든 `found_tags`가 `_risk` suffix 신버전으로 회수됨(구버전 `equity_market`·`macro_rate` 잔재 없음).
- **태그 분포 균형**: 12 질문 전반에서 macro / equity / geo 3 태그가 골고루 회수됨.

## 변별력이 약한 항목

- **`tag_ok` 100%의 의미 제한**: 위 요약 표 주석 참조. `n_results=3` + 3축 태그 풀이라는 구조적 이유로 `tag_ok`가 사실상 자동 통과.
- **개선 metric 후보**:
  - 1위 문서의 `primary_tag` 메타데이터가 expect와 정확 일치하는지 확인.
  - `precision = |found ∩ expect| / |found|`, `recall = |found ∩ expect| / |expect|`.
  - MRR(Mean Reciprocal Rank), Recall@k.

## 평가 범위의 한계

- **검색 단계 단일 호출만 평가**: 본 평가는 ChromaDB 검색 1회로만 수행된다. LangGraph 기반 Self-Correction 재검색 루프, 리포트 생성 단계(`/research/stream`), risk_tags 후처리는 본 평가 범위 밖이며, Self-Correction 효과는 별도 시나리오 테스트로 후속 검증한다.
- **평가 풀 도메인 편향**: ChromaDB 적재 풀이 FRED(매크로·금리 중심) + ECOS(매크로·환율) + manual_seed(소량 큐레이션)로 구성되어 equity·geo 도메인 비중이 macro 대비 얕다. 이는 모든 질문에서 3 태그가 동시에 회수되는 현상의 한 원인.
- **운영 풀과의 분리**: 본 평가 수치는 과거 데이터 풀(1,226건)이며 운영 시 실시간 RSS가 추가된 풀(시점에 따라 변동)에서는 hit·top_dist가 다소 달라질 수 있다.

---

## 수동 정성 검수 (상위 5개) — RAG 담당자 작성

> RAG 파이프라인 실행 후 `top_doc` 텍스트 직접 검토하여 작성.

| id | 관련성 (Y/N) | 출처 (Y/N) | 태그 (Y/N) | 메모 |
| --- | --- | --- | --- | --- |
| Q01 | — | — | — |  |
| Q02 | — | — | — |  |
| Q03 | — | — | — |  |
| Q04 | — | — | — |  |
| Q05 | — | — | — |  |
