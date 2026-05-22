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
