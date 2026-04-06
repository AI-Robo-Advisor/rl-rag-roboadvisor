"""
ChromaDB 벡터스토어 설정 및 관리 모듈.
finance_news 컬렉션을 로컬 ChromaDB에 연동합니다.

메타데이터 스키마:
    - title   (str): 뉴스 제목
    - summary (str): 요약 (최대 300자)
    - url     (str): 원문 URL
    - date    (str): 발행일 (YYYY-MM-DD)
"""
import os
from typing import Any, Dict, List, Optional

import chromadb

COLLECTION_NAME = "finance_news"
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


def get_client(persist_dir: str = DEFAULT_PERSIST_DIR) -> chromadb.PersistentClient:
    """
    ChromaDB PersistentClient를 반환합니다.

    Args:
        persist_dir: 데이터를 저장할 로컬 디렉터리 경로.

    Returns:
        chromadb.PersistentClient 인스턴스.
    """
    return chromadb.PersistentClient(path=persist_dir)


def get_collection(
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> chromadb.Collection:
    """
    finance_news 컬렉션을 가져오거나, 없으면 생성합니다.

    코사인 유사도(hnsw:space=cosine)를 기본 거리 함수로 사용합니다.

    Args:
        persist_dir: ChromaDB 데이터 저장 경로.

    Returns:
        chromadb.Collection 인스턴스.
    """
    client = get_client(persist_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(
    documents: List[str],
    metadatas: List[Dict[str, str]],
    ids: List[str],
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> None:
    """
    finance_news 컬렉션에 문서를 추가합니다.

    summary 필드는 자동으로 300자로 truncate됩니다.

    Args:
        documents: 임베딩할 원문 텍스트 리스트.
        metadatas: 각 문서의 메타데이터 리스트.
                   필드: title, summary, url, date.
        ids:       각 문서의 고유 ID 리스트.
        persist_dir: ChromaDB 데이터 저장 경로.

    Example:
        add_documents(
            documents=["삼성전자 2분기 실적 발표..."],
            metadatas=[{
                "title": "삼성전자 실적 발표",
                "summary": "...",
                "url": "https://example.com/news/1",
                "date": "2024-07-01",
            }],
            ids=["news_001"],
        )
    """
    collection = get_collection(persist_dir)

    # summary 300자 제한
    for meta in metadatas:
        if "summary" in meta and len(meta["summary"]) > 300:
            meta["summary"] = meta["summary"][:300]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)


def query_documents(
    query_texts: List[str],
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> Dict[str, Any]:
    """
    쿼리 텍스트와 코사인 유사도가 가장 높은 문서를 검색합니다.

    Args:
        query_texts: 검색 쿼리 텍스트 리스트.
        n_results:   반환할 결과 수 (기본값: 5).
        where:       메타데이터 필터 (예: {"date": "2024-07-01"}).
        persist_dir: ChromaDB 데이터 저장 경로.

    Returns:
        ChromaDB 쿼리 결과 딕셔너리.
        키: ids, documents, metadatas, distances.
    """
    collection = get_collection(persist_dir)
    kwargs: Dict[str, Any] = {
        "query_texts": query_texts,
        "n_results": n_results,
    }
    if where:
        kwargs["where"] = where

    return collection.query(**kwargs)


def upsert_documents(
    documents: List[str],
    metadatas: List[Dict[str, str]],
    ids: List[str],
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> None:
    """
    finance_news 컬렉션에 문서를 upsert합니다.

    동일한 ID가 이미 존재하면 덮어쓰고, 없으면 새로 추가합니다.
    중복 뉴스 방지에 사용됩니다.

    Args:
        documents: 임베딩할 원문 텍스트 리스트.
        metadatas: 각 문서의 메타데이터 리스트.
                   필드: title, summary, url, date, category.
        ids:       각 문서의 고유 ID 리스트.
        persist_dir: ChromaDB 데이터 저장 경로.
    """
    collection = get_collection(persist_dir)

    for meta in metadatas:
        if "summary" in meta and len(meta["summary"]) > 300:
            meta["summary"] = meta["summary"][:300]

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)


def delete_documents(
    ids: List[str],
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> None:
    """
    지정한 ID의 문서를 컬렉션에서 삭제합니다.

    Args:
        ids:         삭제할 문서 ID 리스트.
        persist_dir: ChromaDB 데이터 저장 경로.
    """
    collection = get_collection(persist_dir)
    collection.delete(ids=ids)
