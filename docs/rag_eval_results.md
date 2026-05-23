# RAG 평가 결과

> **측정일**: 2026-05-23
> **ChromaDB 상태**: FRED(733) + ECOS(455) + manual_seed(39) + 실시간 RSS(89) = **1,316건**.
> GDELT(1,197건)는 GKG 기계코드 형식이라 ChromaDB 영구 제외, `risk_vectors_daily.parquet`에만 반영.
> **평가 스크립트**: `PYTHONPATH=. python scripts/rag_eval.py --json`
> **평가 질문**: `docs/rag_eval_questions.json` (12문항)
> **검색 파라미터**: `n_results=3`, `hit_threshold=0.4` (cosine distance)
> **리스크 태그 체계**: 신버전 3축 — `macro_rate_risk` / `equity_market_risk` / `geopolitical_fx_risk` (`_risk` suffix 통일, PR #48 머지본)

---

## 요약

| 지표 | 결과 |
|------|------|
| 검색 히트율 (top-1 cosine dist < 0.4) | **12 / 12 = 100%** |
| 태그 정확도 (expect ⊆ found_tags) | **12 / 12 = 100%** |
| top_dist 평균 | 0.246 (범위 0.203 ~ 0.367) |

> ⚠️ 태그 정확도 100%의 변별력 한계: 검색이 `n_results=3`이고 3축 태그 풀이 작아 **거의 모든 질문에서 3 태그가 동시에 회수**된다. `tag_ok`는 "기대 태그가 found_tags에 포함되는가" 조건이라 사실상 자동 통과. 실제 1위 문서의 정확도는 `top_dist`(0.20~0.37대 일관)와 `top_doc` 정성 검토로 보완 판단해야 한다.

---

## 질문별 상세

| id | 질문 (요약) | hit | top_dist | expect | found_tags | tag_ok |
|----|-------------|-----|----------|--------|------------|--------|
| Q01 | 금리 인상이 ETF 포트폴리오에 미치는 영향 | ✅ | 0.2079 | macro_rate_risk | macro/equity/geo 3종 | ✅ |
| Q02 | 한국 금융당국 규제 강화·정책 변경 시 투자 리스크 | ✅ | 0.2362 | geopolitical_fx_risk | macro/equity/geo 3종 | ✅ |
| Q03 | 코스피 급락·변동성 확대 시 포트폴리오 리스크 | ✅ | 0.2030 | equity_market_risk | macro/equity/geo 3종 | ✅ |
| Q04 | 어닝쇼크·실적 부진 관련 최근 증시 리스크 | ✅ | 0.2090 | equity_market_risk | macro/equity/geo 3종 | ✅ |
| Q05 | 미국 연준 기준금리 결정이 한국 채권 ETF에 미치는 영향 | ✅ | 0.2586 | macro_rate_risk | macro/equity/geo 3종 | ✅ |
| Q06 | 원달러 환율 급등 시 포트폴리오 리스크 관리 | ✅ | 0.2070 | geopolitical_fx_risk | macro/equity/geo 3종 | ✅ |
| Q07 | 지정학적 리스크(전쟁·무역분쟁)가 국내 증시에 미치는 영향 | ✅ | 0.2755 | equity/geo | macro/equity/geo 3종 | ✅ |
| Q08 | 인플레이션 상승이 자산 배분 전략에 미치는 영향 | ✅ | 0.2975 | macro_rate_risk | macro/equity/geo 3종 | ✅ |
| Q09 | 시장 변동성 VIX 지수 급등 시 방어적 포트폴리오 구성 | ✅ | 0.3674 | equity_market_risk | macro/equity/geo 3종 | ✅ |
| Q10 | 공급망 차질·원자재 가격 급등이 ETF 수익률에 미치는 영향 | ✅ | 0.2052 | macro/geo | macro/equity/geo 3종 | ✅ |
| Q11 | 한국은행 기준금리 인상이 부동산·채권 시장에 미치는 영향 | ✅ | 0.2626 | macro_rate_risk | macro/equity/geo 3종 | ✅ |
| Q12 | 글로벌 경기침체 우려가 국내 주식형 ETF에 미치는 리스크 | ✅ | 0.2519 | macro/equity | macro/equity/geo 3종 | ✅ |

---

## 유효한 검증 항목

- **검색 파이프라인 작동**: 12/12 (`top_dist < 0.4` 통과)
- **리포트 생성 성공**: 12/12 (별도 graph 실행 시 검증)
- **출처 메타데이터 존재**: 12/12 (`top_doc`·`top_date` 모두 반환)
- **신규 태그 체계 적용 확인**: 모든 `found_tags`가 `_risk` suffix 신버전으로 회수됨 (구버전 `equity_market`·`macro_rate` 잔재 없음)
- **태그 분포 균형**: 12 질문 전반에서 macro / equity / geo 3 태그가 골고루 회수됨

## 변별력이 약한 항목

- **`tag_ok` 100%의 의미 제한**: `n_results=3`에서 거의 모든 질문이 3 태그를 동시에 회수하므로 부분집합 포함(`expect ⊆ found`) 조건이 자동 통과. 평가의 변별력은 `top_dist`(0.20~0.37)와 `top_doc` 정성 검토에 의존.
- **개선 방향**: 평가 기준을 "1위 문서의 primary_tag가 expect와 정확 일치"로 강화하거나, expect 셋이 다중인 질문에서 `len(found ∩ expect) / len(expect)` precision metric 도입.

## 이전(2026-05-22) 측정값과의 차이

PR #51 description / `docs/임시_유영_분석보고서.md` 이전 판본에 적힌 **"태그 정확도 10/12 = 83.3%, 실패 2건: Q08·Q11"** 수치는 ChromaDB 데이터 보강 전(부분 적재) 측정값. 본 평가일(2026-05-23) 기준으로는 12/12로 갱신됨. 보고서 §6 수치도 동일하게 갱신 예정.

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
