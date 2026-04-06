#!/usr/bin/env python3
"""
수집 파이프라인 스모크 테스트 (prompt.txt 출력 요청).

1. 네이버 금융 목록에서 최신 뉴스 5건 수집 후 ChromaDB 저장
2. 한국은행 ECOS StatisticTableList로 지정 통계 목록 수집 후 저장
3. '삼성전자 실적' 키워드로 검색해 관련 문서 포함 여부 확인

실행 (프로젝트 루트):
    python scripts/collector_smoke_test.py

기본적으로 ``.env``의 ``CHROMA_PERSIST_DIR``(미설정 시 ``./chroma_db``)에 적재합니다.
LangGraph·API와 **동일한 DB**를 쓰므로, 스모크 후 ``python -m src.agent.graph``에서 RAG hit이 납니다.

격리 DB가 필요하면:
    COLLECTOR_SMOKE_TMP=1 python scripts/collector_smoke_test.py

필요: .env에 BOK_API_KEY(2번). 네트워크 필수.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from apps.api.config import settings  # noqa: E402
from src.agent.news_collector import (  # noqa: E402
    collect_bok_ecos_tables_and_store,
    collect_naver_finance_news_and_store,
)
from src.agent.vectorstore import query_documents  # noqa: E402


def main() -> None:
    if os.getenv("COLLECTOR_SMOKE_TMP", "").strip().lower() in ("1", "true", "yes"):
        persist_dir = tempfile.mkdtemp(prefix="chroma_collector_smoke_")
        logging.info("Chroma 임시 경로(격리): %s", persist_dir)
    else:
        persist_dir = settings.CHROMA_PERSIST_DIR
        logging.info("Chroma 경로(앱·그래프와 동일): %s", os.path.abspath(persist_dir))

    n_news = collect_naver_finance_news_and_store(
        max_items=5,
        persist_dir=persist_dir,
    )
    print(f"[1] 네이버 뉴스 저장: {n_news}건")
    if n_news < 1:
        print("    경고: 네이버에서 가져온 건이 없습니다. 네트워크·파싱을 확인하세요.")

    n_ecos = collect_bok_ecos_tables_and_store(
        stat_codes=["102Y004"],
        start_index=1,
        end_index=10,
        persist_dir=persist_dir,
    )
    print(f"[2] ECOS 통계표 저장: {n_ecos}건")

    query = "삼성전자 실적"
    res = query_documents(
        query_texts=[query],
        n_results=5,
        persist_dir=persist_dir,
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    if not docs:
        print(f"[3] 검색 '{query}': 결과 없음 — 적재·임베딩 실패 가능")
        sys.exit(1)

    blob = " ".join(docs)
    for m in metas:
        blob += " " + str(m.get("title", "")) + str(m.get("summary", ""))

    if "삼성" in blob or "실적" in blob or "삼전" in blob:
        print(
            f"[3] 검색 '{query}': 상위 결과에 '삼성/실적/삼전' 관련 문맥 포함 → OK ({len(docs)}건)"
        )
    else:
        print(
            f"[3] 검색 '{query}': 상위 유사도 결과에 키워드 문자열이 직접 없을 수 있음.\n"
            f"    첫 문서 미리보기: {docs[0][:160]!r}..."
        )

    if not os.getenv("COLLECTOR_SMOKE_TMP"):
        print(
            "\n(팁) 이제 같은 경로로 LangGraph 실행: python -m src.agent.graph"
        )


if __name__ == "__main__":
    main()
