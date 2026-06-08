# 배포 가이드

서버: AWS Lightsail (Amazon Linux 2023) — `<서버주소>`
서비스: FastAPI `8000`, Streamlit Dashboard `8501`

> 서버 주소와 SSH 키 파일 경로는 팀장에게 문의.

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
git checkout -b feature/이니셜-새기능
# ... 작업 ...
git push origin feature/이니셜-새기능
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
http://<서버주소>:8000/health  → {"status": "ok"}
http://<서버주소>:8501         → 대시보드
```

---

## 서버 직접 접속 (디버깅 필요 시)

### SSH 접속

팀장에게 키 파일 발급 요청 후:

```bash
chmod 600 <키파일경로>
ssh -i <키파일경로> ec2-user@<서버주소>
```

### 서버에서 자주 쓰는 명령어

서버 접속 후 `/opt/roboadvisor/` 에서 실행. 이하 명령어의 `-f docker-compose.prod.yml` 은 서버 내 경로 기준.

```bash
cd /opt/roboadvisor

# 실행 중인 컨테이너 상태
docker compose -f docker-compose.prod.yml ps

# API 로그 실시간 확인
docker compose -f docker-compose.prod.yml logs -f api

# 대시보드 로그
docker compose -f docker-compose.prod.yml logs -f dashboard

# ChromaDB 저장된 문서 수 확인
docker compose -f docker-compose.prod.yml run --rm collector \
  python -c "from src.agent.vectorstore import collection_document_count; print(collection_document_count())"
```

---

## 수동 배포 (긴급 시)

GitHub Actions 없이 바로 배포:

```bash
ssh -i <키파일경로> ec2-user@<서버주소>
cd /opt/roboadvisor

echo "<DOCKERHUB_TOKEN>" | docker login -u "<DOCKERHUB_USERNAME>" --password-stdin

docker compose -f docker-compose.prod.yml pull api dashboard
docker compose -f docker-compose.prod.yml run --rm collector || true
docker compose -f docker-compose.prod.yml up -d api dashboard
```

---

## 롤백

Docker Hub에 `sha-<커밋해시>` 태그로 이전 버전이 저장됨.
커밋 해시는 GitHub → Actions → 해당 Deploy 실행 → 상단 커밋 링크에서 확인.

```bash
ssh -i <키파일경로> ec2-user@<서버주소>
cd /opt/roboadvisor

# docker-compose.prod.yml에서 :latest → :sha-되돌릴커밋해시 로 임시 수정
# 예: image: myuser/rl-rag-api:sha-abc1234
vim docker-compose.prod.yml

docker compose -f docker-compose.prod.yml pull api dashboard
docker compose -f docker-compose.prod.yml up -d api dashboard
```

---

## 트러블슈팅

### 서비스가 안 뜰 때

```bash
cd /opt/roboadvisor
docker compose -f docker-compose.prod.yml ps -a
docker compose -f docker-compose.prod.yml logs --tail=50 api
```

### ChromaDB 데이터가 비었을 때

```bash
cd /opt/roboadvisor
docker compose -f docker-compose.prod.yml run --rm collector
```

### 디스크 공간 부족

```bash
docker system prune -f
docker image prune -a -f
```

### .env 수정이 필요할 때

```bash
vim /opt/roboadvisor/.env
cd /opt/roboadvisor
docker compose -f docker-compose.prod.yml up -d api dashboard
```

> **.env 파일은 절대 git에 올리지 말 것.** API 키가 포함되어 있음.
