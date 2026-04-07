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
| 강유영 | u | RAG/프론트 | `src/agent/`, `src/dashboard/`, `src/rl/backtest.py`, `src/rl/metrics.py`, `src/rl/anova.py` |
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
src/dashboard/   # Streamlit 6탭 (강유영)
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
- Streamlit(`src/dashboard/`)에서 모델·DB 직접 로드 금지 — FastAPI HTTP 통신만 사용
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

## 자주 쓰는 명령어
```bash
# 테스트
pytest tests/ -v -m "not integration"   # 단위 테스트
pytest tests/ -v                         # 전체 테스트

# 로컬 직접 실행
uvicorn apps.api.main:app --reload
streamlit run src/dashboard/app.py

# 뉴스 수집 (smoke test)
COLLECTOR_SMOKE_TMP=1 python scripts/collector_smoke_test.py

# LangGraph 실행
python -m src.agent.graph

# Docker
docker compose up          # 전체 서비스 실행
docker compose up api      # API만 실행
docker compose build       # 이미지 재빌드
```

## 스프린트 일정
| 스프린트 | 기간 | 주요 목표 |
|---------|------|---------|
| Sprint 1 | 3/31~4/13 | 환경 세팅 + 기초 구현 |
| Sprint 2 | 4/14~4/27 | 핵심 기능 구현 |
| Sprint 3 | 4/28~5/11 | 분석 + 연동 |
| Sprint 4 | 5/12~5/25 | 통합 테스트 + 버그픽스 |
| Sprint 5 | 5/26~6/15 | 문서 + 최종 마무리 |

**주요 마감**: 4/13 제안 발표 | 5/11 중간 발표 | 6/15 완료 보고

## 성능 기준
- RL 샤프비율: 동일가중 대비 +0.2 이상
- 백테스트 누적 수익률: 벤치마크(SPY) 대비 +10%p 이상
- API 응답 시간: 요청당 5초 이내
- pytest: 10개 이상 통과
