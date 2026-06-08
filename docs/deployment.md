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

## 최초 배포 전 필수 데이터 셋업

자동 배포(GitHub Actions)는 코드·이미지만 처리합니다. 아래 두 항목은 **서버에 직접 올려야** 합니다.

### 1. TensorBoard 학습곡선 데이터 (`/train_curve` API 용)

`logs/tensorboard/`는 `.dockerignore`로 이미지에서 제외되어 있으므로, 서버 디렉터리에 직접 업로드하고 bind mount로 컨테이너에 연결합니다.

**이문정에게 `tensorboard_logs.tar.gz` 공유 요청 후:**

```bash
# 로컬에서 서버로 업로드 (한 번만)
scp -i <키파일경로> tensorboard_logs.tar.gz ec2-user@<서버주소>:/opt/roboadvisor/

# 서버 SSH 접속 후 압축 해제
ssh -i <키파일경로> ec2-user@<서버주소>
cd /opt/roboadvisor
tar -xzf tensorboard_logs.tar.gz
# logs/tensorboard/ppo_*/ 구조로 압축되어 있어 그대로 해제하면 됩니다.
```

> `docker-compose.prod.yml`에 `/opt/roboadvisor/logs/tensorboard:/app/logs/tensorboard:ro` bind mount가 이미 설정되어 있습니다. 서버에 디렉터리가 존재하면 자동으로 마운트됩니다.

업로드 후 API 재시작:

```bash
cd /opt/roboadvisor
docker compose -f docker-compose.prod.yml up -d api
```

확인:

```bash
curl http://<서버주소>:8000/train_curve
# {"windows": [...]} 형태면 정상
```

---

### 2. ChromaDB (RAG `/research` 탭 용)

ChromaDB가 비어 있으면 `/research`가 뉴스 링크 대신 GitHub 저장소 링크를 출처로 반환합니다.

**배포 시 `collector` 서비스가 자동으로 실행**되지만, 실패하면 기존 데이터를 유지한 채 배포가 계속됩니다. 아래로 확인하세요.

현재 저장된 문서 수 확인:

```bash
cd /opt/roboadvisor
docker compose -f docker-compose.prod.yml run --rm collector \
  python -c "
import chromadb
client = chromadb.PersistentClient(path='/app/chroma_db')
col = client.get_or_create_collection('news')
print('문서 수:', col.count())
"
```

문서 수가 0이면 수동으로 수집:

```bash
cd /opt/roboadvisor
docker compose -f docker-compose.prod.yml run --rm collector
```

> OPENAI_API_KEY가 `/opt/roboadvisor/.env`에 반드시 설정되어 있어야 합니다. 미설정 시 LangGraph가 동작하지 않아 fallback 응답(GitHub 링크)만 반환됩니다.

---

## 트러블슈팅

### 서비스가 안 뜰 때

```bash
cd /opt/roboadvisor
docker compose -f docker-compose.prod.yml ps -a
docker compose -f docker-compose.prod.yml logs --tail=50 api
```

### ChromaDB 데이터가 비었을 때 (`/research` 출처가 GitHub 링크)

```bash
cd /opt/roboadvisor
# 문서 수 확인
docker compose -f docker-compose.prod.yml run --rm collector \
  python -c "
import chromadb; client = chromadb.PersistentClient(path='/app/chroma_db')
col = client.get_or_create_collection('news'); print('문서 수:', col.count())
"
# 0이면 수동 수집 실행
docker compose -f docker-compose.prod.yml run --rm collector
```

### `/train_curve` 404 또는 빈 응답

서버에 `logs/tensorboard/ppo_*` 디렉터리가 없는 경우입니다. 위 "최초 배포 전 필수 데이터 셋업" 섹션의 tensorboard 업로드 절차를 따르세요.

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
