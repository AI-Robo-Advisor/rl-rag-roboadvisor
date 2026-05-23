"""
과거 뉴스 이벤트 parquet → ChromaDB finance_news 컬렉션 벌크 업서트 스크립트.

입력:
  1순위: data/processed/unified_events.parquet  (LLM 라벨 포함 통합본)
  2순위: 개별 raw parquet (FRED / ECOS / manual_seed) — unified 없을 때 폴백

제외:
  GDELT (source='gdelt') — title/summary가 GKG 테마코드 나열(기계 형식)이라
  자연어 임베딩 품질이 극히 낮아 RAG 검색에 부적합. LLM 라벨링은 리스크
  점수(risk_vectors_daily.parquet)용으로만 사용하고 ChromaDB에는 넣지 않음.
  향후 GDELT를 포함하려면 title/summary를 자연어로 재생성하는 전처리 필요.

출력:
  chroma_db/finance_news 컬렉션 upsert

실행:
  python scripts/build_chroma_from_parquet.py
  python scripts/build_chroma_from_parquet.py --dry-run   # 통계만 출력, 실제 upsert 안 함
  python scripts/build_chroma_from_parquet.py --clear     # 기존 컬렉션 초기화 후 재업서트
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────

UNIFIED_PATH = Path("data/processed/unified_events.parquet")

RAW_FALLBACK_PATHS: List[Path] = [
    Path("data/raw/fred/fred_events_2018_2025.parquet"),
    Path("data/raw/ecos/ecos_events_2018_2025.parquet"),
    Path("data/raw/manual_seed/manual_seed_events.parquet"),
]

# GDELT: title/summary가 GKG 기계코드 형식 → 임베딩 품질 낮아 영구 제외.
# LLM 라벨링 완료 후 --clear 재실행해도 포함되지 않음 (의도적 설계).
EXCLUDED_SOURCES = {"gdelt"}

BATCH_SIZE = 500


# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────

def _doc_id(event_id: str) -> str:
    """ChromaDB 고유 ID: event_id 기반.

    실시간 RSS 수집(news_collector.py)은 URL MD5를 사용하므로 ID 공간이 분리됨.
    """
    return f"parquet_{event_id}"


def _risk_label(row: pd.Series) -> str:
    """risk 스코어 컬럼 → 쉼표 구분 태그 문자열."""
    tags: List[str] = []
    for col, label in [
        ("macro_rate_risk",      "macro_rate_risk"),
        ("equity_market_risk",   "equity_market_risk"),
        ("geopolitical_fx_risk", "geopolitical_fx_risk"),
    ]:
        val = row.get(col, 0.0) or 0.0
        if float(val) > 0.0:
            tags.append(label)
    return ",".join(tags)


def _category(row: pd.Series) -> str:
    """primary_tag → ChromaDB category 문자열. 없으면 source 기반 기본값."""
    ptag = str(row.get("primary_tag") or "").strip()
    if ptag and ptag != "none" and ptag != "nan":
        # canonical suffix 보장 (old: macro_rate → macro_rate_risk)
        mapping = {
            "macro_rate":      "macro_rate_risk",
            "equity_market":   "equity_market_risk",
            "geopolitical_fx": "geopolitical_fx_risk",
        }
        return mapping.get(ptag, ptag)
    # primary_tag 없으면 source 기반 fallback
    source = str(row.get("source") or "").lower()
    if source in ("fred", "ecos"):
        return "macro_rate_risk"
    return "equity_market_risk"


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────

def load_events() -> pd.DataFrame:
    """unified_events.parquet 또는 raw 폴백으로 이벤트를 로드합니다."""
    if UNIFIED_PATH.exists():
        df = pd.read_parquet(UNIFIED_PATH)
        df["date"] = pd.to_datetime(df["date"])
        logger.info("unified 로드: %d건", len(df))
    else:
        logger.info("unified parquet 없음 — raw fallback 사용")
        frames: List[pd.DataFrame] = []
        for path in RAW_FALLBACK_PATHS:
            if not path.exists():
                logger.info("스킵 (없음): %s", path)
                continue
            f = pd.read_parquet(path)
            f["date"] = pd.to_datetime(f["date"])
            frames.append(f)
            logger.info("로드: %s (%d건)", path.name, len(f))
        if not frames:
            logger.error("로드할 parquet 파일 없음")
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["event_id"], keep="first")

    # GDELT 제외
    before = len(df)
    df = df[~df["source"].str.lower().isin(EXCLUDED_SOURCES)].copy()
    logger.info("GDELT 제외: %d → %d건", before, len(df))

    # 제목/요약 비어 있으면 제외
    df = df.dropna(subset=["title"]).copy()
    df["title"]   = df["title"].astype(str)
    df["summary"] = df["summary"].fillna("").astype(str)
    df["url"]     = df["url"].fillna("").astype(str)

    df = df.sort_values("date").reset_index(drop=True)
    logger.info("업서트 대상: %d건", len(df))
    return df


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def build_chroma(dry_run: bool = False, clear: bool = False) -> None:
    df = load_events()
    if df.empty:
        logger.error("처리할 이벤트 없음")
        return

    # 소스별 통계 출력
    print("\n=== 소스별 분포 ===")
    print(df["source"].value_counts().to_string())
    print(f"\n날짜 범위: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"총 이벤트: {len(df)}건\n")

    if dry_run:
        logger.info("--dry-run: upsert 없이 종료")
        return

    from src.agent.vectorstore import get_collection, upsert_documents

    if clear:
        import chromadb
        from src.agent.vectorstore import DEFAULT_PERSIST_DIR, COLLECTION_NAME
        client = chromadb.PersistentClient(path=DEFAULT_PERSIST_DIR)
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("기존 컬렉션 삭제 완료")
        except Exception:
            pass

    col = get_collection()
    before_count = col.count()
    logger.info("업서트 전 문서 수: %d", before_count)

    total_upserted = 0
    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start : start + BATCH_SIZE]

        documents: List[str]             = []
        metadatas: List[Dict[str, str]]  = []
        ids:       List[str]             = []

        seen_ids = set()
        for _, row in batch.iterrows():
            doc_id = _doc_id(str(row.get("event_id", "")))
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            doc_text = f"{row['title']} {row['summary']}"[:300]
            date_str = str(row["date"])[:10]

            meta: Dict[str, str] = {
                "title":      row["title"][:200],
                "summary":    row["summary"][:300],
                "url":        row["url"],
                "date":       date_str,
                "category":   _category(row),
                "source":     str(row.get("source") or ""),
                "risk_label": _risk_label(row),
            }
            documents.append(doc_text)
            metadatas.append(meta)
            ids.append(doc_id)

        upsert_documents(documents=documents, metadatas=metadatas, ids=ids)
        total_upserted += len(ids)
        logger.info("배치 업서트: %d/%d건", start + len(batch), len(df))

    after_count = col.count()
    logger.info(
        "완료: %d건 업서트, 컬렉션 %d → %d건",
        total_upserted, before_count, after_count,
    )

    # 간단 검증
    _spot_check(col)


def _spot_check(col: object) -> None:
    """대표 쿼리로 검색 결과를 확인합니다."""
    print("\n=== SPOT CHECK ===")
    queries = [
        "연준 금리 인상 FOMC",
        "미중 무역 전쟁 관세",
        "코스피 증시 급락",
    ]
    for q in queries:
        results = col.query(query_texts=[q], n_results=2)
        print(f"\n쿼리: {q}")
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            print(f"  [{i+1}] dist={dist:.4f} | {meta.get('date','?')} | {doc[:80]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parquet → ChromaDB 벌크 업서트")
    parser.add_argument("--dry-run", action="store_true",
                        help="통계만 출력하고 upsert 안 함")
    parser.add_argument("--clear",   action="store_true",
                        help="기존 컬렉션 삭제 후 재업서트")
    args = parser.parse_args()
    build_chroma(dry_run=args.dry_run, clear=args.clear)
