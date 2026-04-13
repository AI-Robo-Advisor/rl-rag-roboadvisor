# rl-rag-roboadvisor

## 프로젝트 개요
RL + RAG 기반 AI 로보어드바이저.
금융 뉴스를 ChromaDB에 저장하고, LangGraph 에이전트가 RAG로 검색하여 투자 분석 리포트를 생성합니다.

## 팀 정보
- **팀명**: 혈스방지모임 | **팀장**: 강유영 | **팀원**: 박지민, 이문정
- **GitHub**: https://github.com/AI-Robo-Advisor/rl-rag-roboadvisor
- **개발 기간**: 8주 (스프린트 5개) | **완료 보고**: 2026-06-15

## 역할 분담
| 담당자 | 이니셜 | 담당 영역 | 담당 파일/디렉터리 |
|--------|--------|-----------|------------------|
| 강유영 | u | RAG/프론트 | `src/agent/`, `apps/dashboard/`, `src/rl/backtest.py`, `src/rl/metrics.py`, `src/rl/anova.py` |
| 박지민 | j | 데이터/백엔드 | `apps/api/`, `src/data/collector.py`, `src/data/indicators.py` |
| 이문정 | m | RL/인프라 | `src/rl/env.py`, `src/rl/train.py`, `src/rl/shap.py`, `docker-compose.yml`, `Dockerfile.*` |

## 기술 스택
- **LLM**: OpenAI `gpt-4o-mini` (LangChain) | **에이전트**: LangGraph `StateGraph`
- **벡터스토어**: ChromaDB (로컬, `./chroma_db`, 코사인 유사도)
- **뉴스 수집**: feedparser (구글 뉴스 RSS) + ECOS API (한국은행)
- **RL**: Gymnasium + Stable-Baselines3 (PPO) | **API**: FastAPI | **대시보드**: Streamlit
- **컨테이너**: Docker Compose

## 디렉터리 구조
```
src/agent/       # LangGraph + ChromaDB + 뉴스 수집 (강유영)
src/rl/          # RL 환경·학습·분석 (이문정/강유영)
apps/dashboard/  # Streamlit 6탭 (강유영)
apps/api/        # FastAPI 백엔드 (박지민)
scripts/         # 수동 실행 스크립트
tests/           # pytest 테스트
```

## LangGraph 워크플로우
```
START → planner → researcher → grade_documents → analyst → END
                      ↑              │ (부족, 최대 3회)
                      └──────────────┘
```

## 코딩 컨벤션
- **언어**: Python 3.10+, 타입 힌트 필수 (`def fn(x: str) -> int:`)
- **네이밍**: 함수·변수 `snake_case`, 클래스 `PascalCase`, 상수 `UPPER_SNAKE_CASE`
- **Docstring**: Google 스타일 (Args / Returns / Raises)
- **에러 처리**: 구체적 예외 타입 사용 (`except requests.Timeout`), `except Exception` 최소화
- **주석**: 한국어 허용. 로직이 자명하지 않은 곳에만 작성
- **임포트 순서**: stdlib → third-party → local (isort 기준)
- **함수 길이**: 50줄 이내 권장. 초과 시 분리 검토

## Git 규칙
**브랜치 형식**: `feature/이니셜-작업내용`
- 강유영(u): `feature/u-langgraph` / `feature/u-dashboard` / `feature/u-analysis`
- 박지민(j): `feature/j-fastapi` / `feature/j-data`
- 이문정(m): `feature/m-rl-env` / `feature/m-ppo-train` / `feature/m-docker` / `feature/m-shap`

**커밋 메시지 형식**: `<type>: <내용>`
| type | 사용 상황 |
|------|----------|
| `feat` | 새 기능 추가 |
| `fix` | 버그 수정 |
| `docs` | 문서 수정 |
| `refactor` | 리팩토링 |
| `test` | 테스트 코드 |
| `chore` | 빌드·설정 변경 |

- PR은 반드시 `dev` 브랜치로, **셀프 merge 금지**
- `CLAUDE.md` 변경도 PR 리뷰 대상 (컨벤션 변경 = 팀 합의 필요)

## 금지 사항
- `.env` 파일 절대 커밋 금지 — API 키 유출 위험
- **타 팀원 담당 파일 무단 수정 금지** (역할 분담표 참고)
- `apps/api/` 직접 수정 금지 — 박지민 담당, PR로 요청
- `src/rl/env.py`, `src/rl/train.py` 직접 수정 금지 — 이문정 담당
- Streamlit(`apps/dashboard/`)에서 모델·DB 직접 로드 금지 — FastAPI HTTP 통신만 사용
- `chroma_db/` 디렉터리 커밋 금지 — 로컬 생성 데이터
- `--no-verify` 플래그로 hook 우회 금지

## 환경변수 (.env)
```
OPENAI_API_KEY=          # LLM 호출
BOK_API_KEY=             # 한국은행 ECOS (없으면 ECOS 수집 스킵)
CHROMA_PERSIST_DIR=./chroma_db
API_HOST=0.0.0.0
API_PORT=8000
DASHBOARD_HOST=0.0.0.0
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
