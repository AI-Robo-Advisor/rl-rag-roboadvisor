# 배포 가이드

서버: AWS Lightsail (Amazon Linux 2023) — `3.35.148.87`
서비스: FastAPI `8000`, Streamlit Dashboard `8501`

---

## CI/CD 파이프라인 개요

```
PR to dev   →  GitHub Actions: pytest (non-integration)
                    ↓ 통과 시 수동 merge
dev → main  →  GitHub Actions: Docker 이미지 빌드 & 푸시
                    ↓
              서버: docker compose pull → ChromaDB 갱신 → compose up
```

- **PR 생성**: 자동으로 `pytest` 실행. 실패하면 merge 불가
- **main merge**: 자동으로 서버 배포까지 완료
- **셀프 merge 금지** (팀 컨벤션)

---

## 일반 개발 워크플로우 (팀원 전체)

### 1. 기능 개발 후 PR 생성

```bash
# feature 브랜치에서 작업
git checkout -b feature/j-새기능
# ... 작업 ...
git push origin feature/j-새기능
```

GitHub에서 **dev 브랜치**로 PR 생성 → 팀원 리뷰 요청

### 2. CI 확인

PR 페이지 하단에서 `CI / test` 체크가 ✅ 통과인지 확인.
실패 시 로컬에서 확인:

```bash
pytest tests/ -v -m "not integration" --tb=short
```

### 3. dev → main 배포 PR

`dev`에 기능이 충분히 쌓이면 팀장이 `dev → main` PR 생성 → 팀원 리뷰 → merge.
merge 즉시 자동 배포 시작.

### 4. 배포 상태 확인

GitHub → Actions 탭 → `Deploy` 워크플로 확인.
완료 후:

```
http://3.35.148.87:8000/health  → {"status": "ok"}
http://3.35.148.87:8501         → 대시보드
```

---

## 서버 직접 접속 (디버깅 필요 시)

### SSH 키 발급 요청

서버 접속 키는 팀장에게 요청. 받은 후:

```bash
chmod 600 ~/Downloads/lightsail_roboadvisor
ssh -i ~/Downloads/lightsail_roboadvisor ec2-user@3.35.148.87
```

### 서버에서 자주 쓰는 명령어

```bash
# 실행 중인 컨테이너 상태
docker compose -f /opt/roboadvisor/docker-compose.prod.yml ps

# API 로그 실시간 확인
docker compose -f /opt/roboadvisor/docker-compose.prod.yml logs -f api

# 대시보드 로그
docker compose -f /opt/roboadvisor/docker-compose.prod.yml logs -f dashboard

# ChromaDB 문서 수 확인
docker compose -f /opt/roboadvisor/docker-compose.prod.yml run --rm collector \
  python -c "from src.agent.vectorstore import collection_document_count; print(collection_document_count())"
```

---

## 수동 배포 (긴급 시)

GitHub Actions를 기다리지 않고 바로 배포해야 할 때:

```bash
ssh -i ~/Downloads/lightsail_roboadvisor ec2-user@3.35.148.87

cd /opt/roboadvisor

# Docker Hub 로그인
echo "<DOCKERHUB_TOKEN>" | docker login -u "<DOCKERHUB_USERNAME>" --password-stdin

# 이미지 업데이트 & 서비스 재시작
docker compose -f docker-compose.prod.yml pull api dashboard
docker compose -f docker-compose.prod.yml run --rm collector || true
docker compose -f docker-compose.prod.yml up -d api dashboard
```

---

## 롤백

이전 버전으로 돌아가야 할 때 (Docker Hub에 `sha-<커밋해시>` 태그로 저장됨):

```bash
ssh -i ~/Downloads/lightsail_roboadvisor ec2-user@3.35.148.87
cd /opt/roboadvisor

# docker-compose.prod.yml에서 :latest → :sha-되돌릴커밋해시 로 임시 수정
# 예시:
# image: myuser/rl-rag-api:sha-abc1234

docker compose -f docker-compose.prod.yml pull api dashboard
docker compose -f docker-compose.prod.yml up -d api dashboard
```

커밋 해시는 GitHub → Actions → 해당 Deploy 실행 → 상단 커밋 링크에서 확인.

---

## 트러블슈팅

### 서비스가 안 뜰 때

```bash
# 컨테이너 상태 + 종료 코드 확인
docker compose -f /opt/roboadvisor/docker-compose.prod.yml ps -a

# 최근 에러 로그
docker compose -f /opt/roboadvisor/docker-compose.prod.yml logs --tail=50 api
```

### ChromaDB 데이터가 비었을 때

```bash
# 뉴스 수동 수집
docker compose -f /opt/roboadvisor/docker-compose.prod.yml run --rm collector
```

### 디스크 공간 부족

```bash
docker system prune -f
docker image prune -a -f
```

### .env 수정이 필요할 때

```bash
vim /opt/roboadvisor/.env
# 저장 후 서비스 재시작
docker compose -f /opt/roboadvisor/docker-compose.prod.yml up -d api dashboard
```

> **.env 파일은 절대 git에 올리지 말 것.** API 키가 포함되어 있음.
