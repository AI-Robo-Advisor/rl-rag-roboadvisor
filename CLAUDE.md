# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 프로젝트 개요 (rl-rag-roboadvisor)

## 프로젝트 개요

PPO 강화학습 기반 자산배분 엔진과 LangGraph RAG 에이전트를 결합한 로보어드바이저.
핵심은 두 시스템의 **연동**: 리서치 에이전트가 뉴스에서 탐지한 위험 이벤트를 리스크 태그로 변환하여 RL 환경의 관측 공간에 반영한다.

- **팀**: 강유영(u) · 박지민(j) · 이문정(m)
- **GitHub**: https://github.com/AI-Robo-Advisor/rl-rag-roboadvisor

---

## 시스템 아키텍처

```
[뉴스 수집]               [RL 엔진]
feedparser (RSS)           Gymnasium 환경 (env.py)
ECOS API    ──────────→   관측 공간: 수익률/비중/기술지표 + 리스크태그
    ↓                      PPO 학습 (train.py, SB3)
ChromaDB                       ↓
    ↓                   포트폴리오 비중 + SHAP 해석
[LangGraph 에이전트]                ↓
planner → researcher           [FastAPI]  ←── [Streamlit 대시보드]
    ↑         ↓ grade          5개 엔드포인트    6개 탭
    └── (재검색, 최대 3회)                       HTTP 통신만 허용
          ↓ analyst
    투자 리포트 + 리스크 태그
```

**핵심 연동**: `src/agent/`의 리스크 태그 → `src/rl/env.py`의 관측 공간 주입.
이 인터페이스 변경 시 강유영·이문정 합의 필요.

---

## 모듈 구조 및 담당

| 경로 | 역할 | 담당 |
|------|------|------|
| `src/agent/` | LangGraph 워크플로우, ChromaDB 검색, 뉴스 수집 | 강유영 |
| `apps/dashboard/` | Streamlit 6탭 (FastAPI HTTP 통신만 사용) | 강유영 |
| `src/rl/env.py` | Gymnasium 환경, 관측·행동 공간, Safe-Guard | 이문정 |
| `src/rl/train.py` | PPO 학습 루프, 학습 곡선 저장 | 이문정 |
| `src/rl/shap.py` | SHAP 값 계산, Force/Summary Plot | 이문정 |
| `src/rl/backtest.py` | Walk-Forward 백테스트, 12개 지표 계산 | 강유영 |
| `src/rl/metrics.py` | 샤프비율·MDD 등 지표 함수 | 강유영 |
| `src/rl/anova.py` | 3가지 ANOVA 검증 | 강유영 |
| `apps/api/` | FastAPI 5개 엔드포인트, Pydantic 스키마 | 박지민 |
| `src/data/collector.py` | yfinance/pykrx 수집, 로그수익률 계산 | 박지민 |
| `src/data/indicators.py` | RSI, MACD 등 기술적 지표 | 박지민 |
| `docker-compose.yml`, `Dockerfile.*` | 컨테이너 설정 | 이문정 |

**담당 외 파일 수정 금지.** 수정이 필요하면 PR로 요청.

---

## FastAPI 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 서버 상태, 모델 로드 여부 |
| POST | `/optimize` | 현재 시장 데이터 → 포트폴리오 비중 |
| POST | `/explain` | 특정 결정 → SHAP 해석 |
| POST | `/research` | 투자 질문 → 에이전틱 리서치 결과 |
| GET | `/backtest` | 백테스트 성과 지표 |

Streamlit은 위 엔드포인트만 호출. 모델·DB 직접 로드 금지.

---

## RL 환경 핵심 스펙

- **관측 공간**: 과거 N일 수익률(N=20~60) + 현재 포트폴리오 비중 + 기술지표(RSI·MACD 이상 2개) + 리스크 태그
- **행동 공간**: 연속(비중 직접 조절) 또는 이산(리밸런싱) — `env.py` 확인
- **거래비용**: 수수료 0.015% + 슬리피지 0.05%
- **Safe-Guard**: MDD 15% 초과 시 에피소드 조기 종료
- **보상 함수 3종**: 단순수익률 / 샤프비율 기반 / 수익률+MDD페널티(`λ` 범위 0.5~5.0)

---

## ChromaDB

- **위치**: `./chroma_db` (커밋 금지)
- **유사도**: 코사인
- 컬렉션 구조·메타데이터 스키마는 `src/agent/` 코드 참고

---

## 코딩 규칙

- **Python 3.10+**, 타입 힌트 필수
- **네이밍**: `snake_case` (함수·변수) · `PascalCase` (클래스) · `UPPER_SNAKE_CASE` (상수)
- **Docstring**: Google 스타일 (Args / Returns / Raises)
- **예외 처리**: 구체적 타입 (`except requests.Timeout`), `except Exception` 최소화
- **임포트 순서**: stdlib → third-party → local
- **함수 길이**: 50줄 이내 권장

---

## Git 규칙

**브랜치**: `feature/이니셜-작업내용`
- 강유영: `feature/u-langgraph` · `feature/u-dashboard` · `feature/u-analysis`
- 박지민: `feature/j-fastapi` · `feature/j-data`
- 이문정: `feature/m-rl-env` · `feature/m-ppo-train` · `feature/m-docker` · `feature/m-shap`

**커밋**: `feat` · `fix` · `docs` · `refactor` · `test` · `chore`

PR 대상: `dev` | 셀프 merge 금지 | `CLAUDE.md` 변경도 리뷰 필요

---

## 환경변수 (.env)

```
OPENAI_API_KEY=
BOK_API_KEY=           # 없으면 ECOS 수집 스킵
CHROMA_PERSIST_DIR=./chroma_db
API_HOST=0.0.0.0
API_PORT=8000
DASHBOARD_PORT=8501
API_BASE_URL=http://api:8000
LOG_LEVEL=INFO
```

`.env` 커밋 절대 금지.

---

## 주요 명령어

```bash
# 테스트
pytest tests/ -v -m "not integration"
pytest tests/ -v
pytest tests/test_data.py -v   # 데이터 파이프라인 단독 검증

# 데이터 파이프라인 (박지민 담당)
python -m src.data.collector          # 전체 실행: 수집 → 병합 → returns/features/prices parquet 저장
python -m src.data.indicators         # indicators만 재생성 (returns.parquet 필요)

# 로컬 실행
uvicorn apps.api.main:app --reload
streamlit run apps/dashboard/app.py
python -m src.agent.graph

# Docker
docker compose up
docker compose up api

# 뉴스 수집 smoke test
COLLECTOR_SMOKE_TMP=1 python scripts/collector_smoke_test.py
```

> **현재 구현 상태** (2026-04-09 기준):
>
> | 모듈 | 상태 |
> |------|------|
> | `apps/api/` | `/health` 엔드포인트만 구현. `/optimize`, `/explain`, `/research`, `/backtest` 미구현 |
> | `apps/dashboard/` | 플레이스홀더만 존재 |
> | `src/data/collector.py` | ✅ 완료 — yfinance·pykrx 수집, 병합, 로그수익률 계산, parquet 저장 |
> | `src/data/indicators.py` | ✅ 완료 — RSI(14), MACD(12/26/9) 계산, Z-score 정규화, features.parquet 저장 |
> | `data/raw/prices.parquet` | ✅ 생성됨 (10자산, 2020-01-01~2025-12-31) |
> | `data/processed/returns.parquet` | ✅ 생성됨 (~1423행, 10열, raw 로그수익률) |
> | `data/processed/features.parquet` | ✅ 생성됨 (~1390행, 40열, Z-score 정규화) |
> | `src/agent/`, `src/rl/` | 미구현 (개발 예정) |
