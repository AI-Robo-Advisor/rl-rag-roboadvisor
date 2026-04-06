"""
네이버 금융 뉴스 수집 및 ChromaDB 저장 모듈.

네이버 금융 뉴스 목록 페이지에서 기사를 수집하고
finance_news 컬렉션에 upsert합니다.

카테고리:
    - 증시: section_id2=258
    - 실적: section_id2=261
    - 정책: section_id2=263
"""
import hashlib
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.agent.vectorstore import upsert_documents

logger = logging.getLogger(__name__)

NAVER_FINANCE_BASE = "https://finance.naver.com"

CATEGORIES: Dict[str, str] = {
    "증시": "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258",
    "실적": "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=261",
    "정책": "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=263",
}

REQUEST_INTERVAL = 1.0  # seconds (robots.txt 준수)

_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (compatible; rl-rag-roboadvisor/1.0; "
            "+https://github.com/AI-Robo-Advisor/rl-rag-roboadvisor)"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
)


def _make_doc_id(url: str) -> str:
    """
    URL을 기반으로 ChromaDB 문서 ID를 생성합니다.

    중복 방지를 위해 URL의 MD5 해시를 사용합니다.

    Args:
        url: 뉴스 원문 URL.

    Returns:
        32자리 hex 문자열 ID.
    """
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def _parse_date(raw: str) -> str:
    """
    네이버 금융 날짜 문자열을 YYYY-MM-DD 형식으로 변환합니다.

    입력 형식 예시: '2024.07.01 10:30', '2024.07.01'

    Args:
        raw: 원본 날짜 문자열.

    Returns:
        'YYYY-MM-DD' 형식 문자열. 파싱 실패 시 오늘 날짜 반환.
    """
    raw = raw.strip()
    for fmt in ("%Y.%m.%d %H:%M", "%Y.%m.%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return datetime.today().strftime("%Y-%m-%d")


def fetch_news_list(url: str, category: str) -> List[Dict[str, str]]:
    """
    네이버 금융 뉴스 목록 페이지에서 기사 정보를 수집합니다.

    requests + BeautifulSoup으로 HTML을 파싱하여
    각 기사의 title, summary, url, date, category를 추출합니다.

    Args:
        url:      뉴스 목록 페이지 URL.
        category: 카테고리 이름 (증시/실적/정책).

    Returns:
        뉴스 항목 딕셔너리 리스트.
        각 항목 키: title, summary, url, date, category.
    """
    try:
        resp = _SESSION.get(url, timeout=10)
        resp.raise_for_status()
        resp.encoding = "euc-kr"
    except requests.RequestException as e:
        logger.error("뉴스 목록 요청 실패 [%s]: %s", url, e)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    news_items: List[Dict[str, str]] = []

    # 네이버 금융 뉴스 목록: <ul class="newsList"> > <li> > <dl>
    for dl in soup.select("ul.newsList dl"):
        # 제목 + 링크
        title_tag = dl.select_one("dt a")
        if not title_tag:
            continue

        title = title_tag.get_text(strip=True)
        href = title_tag.get("href", "")
        article_url = urljoin(NAVER_FINANCE_BASE, href) if href else ""
        if not article_url:
            continue

        # 요약 및 날짜 (dd.articleSummary)
        dd = dl.select_one("dd.articleSummary")
        date_str = ""
        summary = ""
        if dd:
            date_tag = dd.select_one("span.wdate")
            date_str = _parse_date(date_tag.get_text()) if date_tag else ""
            # 날짜·언론사 span 제거 후 남은 텍스트를 요약으로 사용
            for span in dd.select("span"):
                span.decompose()
            summary = dd.get_text(strip=True)[:300]

        if not date_str:
            date_str = datetime.today().strftime("%Y-%m-%d")

        news_items.append(
            {
                "title": title,
                "summary": summary,
                "url": article_url,
                "date": date_str,
                "category": category,
            }
        )

    logger.info("[%s] %d개 기사 수집", category, len(news_items))
    return news_items


def collect_and_store(
    categories: Optional[Dict[str, str]] = None,
    persist_dir: Optional[str] = None,
) -> int:
    """
    지정한 카테고리의 네이버 금융 뉴스를 수집하여 ChromaDB에 저장합니다.

    요청 간격을 REQUEST_INTERVAL(1초) 이상 유지하여 robots.txt를 준수합니다.
    URL을 ID로 사용하여 중복 뉴스를 방지합니다.

    Args:
        categories: {카테고리명: URL} 딕셔너리.
                    None이면 CATEGORIES(증시/실적/정책) 전체 사용.
        persist_dir: ChromaDB 저장 경로.
                     None이면 vectorstore 기본값 사용.

    Returns:
        총 저장(upsert)된 문서 수.
    """
    if categories is None:
        categories = CATEGORIES

    upsert_kwargs: Dict = {}
    if persist_dir:
        upsert_kwargs["persist_dir"] = persist_dir

    total = 0

    for category, url in categories.items():
        items = fetch_news_list(url, category)
        if not items:
            time.sleep(REQUEST_INTERVAL)
            continue

        documents = [f"{item['title']} {item['summary']}" for item in items]
        metadatas = [
            {
                "title": item["title"],
                "summary": item["summary"],
                "url": item["url"],
                "date": item["date"],
                "category": item["category"],
            }
            for item in items
        ]
        ids = [_make_doc_id(item["url"]) for item in items]

        upsert_documents(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            **upsert_kwargs,
        )
        total += len(items)
        logger.info("[%s] ChromaDB upsert 완료: %d건", category, len(items))

        time.sleep(REQUEST_INTERVAL)

    logger.info("전체 수집 완료: %d건", total)
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    collect_and_store()
