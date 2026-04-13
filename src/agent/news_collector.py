"""
구글 뉴스 RSS(메인) + 한국은행 ECOS(보조) 통합 수집 및 ChromaDB 저장.

- 구글 뉴스 RSS: feedparser로 키워드 기반 수집 (API 키 불필요).
  카테고리: 급등락 / 실적쇼크 / 규제변경
- ECOS StatisticTableList: 통계표 메타데이터 (거시 지표 보조).
  BOK_API_KEY 필요. 없으면 자동 스킵.

공통 스키마(metadata): title, summary(≤300자), url, date, category.
  구글뉴스: source="google_news" 추가.
  통계: risk_label="" 추가.
문서 ID: 뉴스=URL MD5, 통계=STAT_CODE.
"""
from __future__ import annotations

import hashlib
import html as html_lib
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import feedparser
import requests

from apps.api.config import settings
from src.agent.vectorstore import upsert_documents

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 구글 뉴스 RSS 설정
# ---------------------------------------------------------------------------
GOOGLE_NEWS_FEEDS: Dict[str, str] = {
    "급등락": "https://news.google.com/rss/search?q=주식+증시+급등+급락&hl=ko&gl=KR&ceid=KR:ko",
    "실적쇼크": "https://news.google.com/rss/search?q=기업실적+어닝쇼크+실적발표&hl=ko&gl=KR&ceid=KR:ko",
    "규제변경": "https://news.google.com/rss/search?q=금융규제+정책변경+금융당국&hl=ko&gl=KR&ceid=KR:ko",
}

GOOGLE_REQUEST_INTERVAL = 1.0  # 카테고리 간 요청 간격 (초)

# ---------------------------------------------------------------------------
# ECOS 설정
# ---------------------------------------------------------------------------
ECOS_REQUEST_INTERVAL = 1.0

# ---------------------------------------------------------------------------
# 공용 세션 (ECOS HTTP 요청용)
# ---------------------------------------------------------------------------
_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (compatible; rl-rag-roboadvisor/1.0; "
            "+https://github.com/AI-Robo-Advisor/rl-rag-roboadvisor)"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
)


# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------
def _news_doc_id(url: str) -> str:
    """뉴스 URL의 MD5 해시를 ChromaDB 문서 ID로 반환합니다."""
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def _strip_html(text: str) -> str:
    """HTML 태그를 제거하고 HTML 엔티티를 디코딩합니다."""
    text = re.sub(r"<[^>]+>", "", text)
    return html_lib.unescape(text).strip()


def _parse_rss_date(entry: Any) -> str:
    """feedparser 엔트리에서 날짜를 YYYY-MM-DD 형식으로 추출합니다."""
    if getattr(entry, "published_parsed", None):
        try:
            return time.strftime("%Y-%m-%d", entry.published_parsed)
        except (TypeError, ValueError):
            pass
    if getattr(entry, "updated_parsed", None):
        try:
            return time.strftime("%Y-%m-%d", entry.updated_parsed)
        except (TypeError, ValueError):
            pass
    return datetime.today().strftime("%Y-%m-%d")


def infer_risk_label(title: str, summary: str) -> str:
    """
    제목·요약에 리스크 관련 키워드가 있으면 메타데이터용 risk_label 문자열을 만듭니다.

    쉼표로 구분된 라벨(예: '규제변경,급등락'). 해당 없으면 빈 문자열.

    Args:
        title: 뉴스 제목.
        summary: 요약.

    Returns:
        라벨 문자열 또는 "".
    """
    text = f"{title} {summary}"
    found: List[str] = []

    def add(label: str) -> None:
        if label not in found:
            found.append(label)

    if any(k in text for k in ("규제", "제재", "규제강화")):
        add("규제변경")
    if any(k in text for k in ("쇼크", "실적쇼크", "어닝쇼크")):
        add("실적쇼크")
    if any(k in text for k in ("급락", "급등", "폭락", "폭등")):
        add("급등락")
    if any(k in text for k in ("금리인상", "금리 인상", "빅스텝", "기준금리인상")):
        add("금리인상")

    return ",".join(found)


# ---------------------------------------------------------------------------
# 구글 뉴스 RSS 수집
# ---------------------------------------------------------------------------
def fetch_google_news_rss(
    feed_url: str,
    category: str,
    limit: int = 10,
) -> List[Dict[str, str]]:
    """
    구글 뉴스 RSS 피드에서 최신 뉴스를 수집합니다.

    feedparser로 파싱하며 HTML 태그를 제거합니다.

    Args:
        feed_url: 구글 뉴스 RSS URL (키워드 기반).
        category: 뉴스 카테고리 ("급등락" / "실적쇼크" / "규제변경").
        limit:    최대 수집 건수.

    Returns:
        수집된 뉴스 항목 딕셔너리 리스트.
        키: title, url, date, category, source, summary
    """
    if limit <= 0:
        return []

    items: List[Dict[str, str]] = []
    try:
        feed = feedparser.parse(feed_url)
        if getattr(feed, "bozo", False) and not feed.entries:
            logger.warning("RSS 파싱 경고 [%s]: %s", category, feed.bozo_exception)
            return items

        for entry in feed.entries[:limit]:
            title = _strip_html(getattr(entry, "title", "") or "").strip()
            url = (getattr(entry, "link", "") or "").strip()
            if not title or not url:
                continue

            raw_summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
            summary = _strip_html(raw_summary)[:300]
            date = _parse_rss_date(entry)

            items.append(
                {
                    "title": title,
                    "url": url,
                    "date": date,
                    "category": category,
                    "source": "google_news",
                    "summary": summary,
                }
            )
    except Exception as e:
        logger.error("구글 뉴스 RSS 수집 실패 [%s]: %s", category, e)

    logger.info("[구글뉴스:%s] %d건 수집", category, len(items))
    return items


def collect_google_news_and_store(
    max_items: int = 10,
    feeds: Optional[Dict[str, str]] = None,
    persist_dir: Optional[str] = None,
) -> int:
    """
    구글 뉴스 RSS 3개 피드에서 뉴스를 수집하고 ChromaDB에 upsert합니다.

    중복 방지: URL MD5 해시를 문서 ID로 사용.
    요청 간격: 카테고리 사이 1초 이상 유지.

    Args:
        max_items:   카테고리별 최대 수집 건수.
        feeds:       {카테고리: RSS URL} 딕셔너리. None이면 GOOGLE_NEWS_FEEDS 사용.
        persist_dir: ChromaDB 데이터 저장 경로.

    Returns:
        ChromaDB에 upsert된 총 건수.
    """
    if max_items <= 0:
        return 0

    feed_map = feeds or GOOGLE_NEWS_FEEDS
    upsert_kwargs: Dict[str, Any] = {}
    if persist_dir:
        upsert_kwargs["persist_dir"] = persist_dir

    documents: List[str] = []
    metadatas: List[Dict[str, str]] = []
    ids: List[str] = []

    categories = list(feed_map.keys())
    for i, cat in enumerate(categories):
        items = fetch_google_news_rss(feed_map[cat], cat, max_items)
        for item in items:
            doc_id = _news_doc_id(item["url"])
            if doc_id in ids:
                continue
            doc_text = f"{item['title']} {item['summary']}"[:300]
            meta: Dict[str, str] = {
                "title": item["title"],
                "summary": item["summary"],
                "url": item["url"],
                "date": item["date"],
                "category": item["category"],
                "source": item["source"],
            }
            documents.append(doc_text)
            metadatas.append(meta)
            ids.append(doc_id)

        if i < len(categories) - 1:
            time.sleep(GOOGLE_REQUEST_INTERVAL)

    if not documents:
        return 0

    upsert_documents(documents=documents, metadatas=metadatas, ids=ids, **upsert_kwargs)
    logger.info("구글뉴스 ChromaDB upsert: %d건", len(documents))
    return len(documents)


# ---------------------------------------------------------------------------
# ECOS (한국은행 Market Analyzer)
# ---------------------------------------------------------------------------

ECOS_STATISTIC_TABLE_LIST_BASE = "https://ecos.bok.or.kr/api/StatisticTableList"
DEFAULT_STAT_CODES: Tuple[str, ...] = ("102Y004",)
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0


class EcosApiError(Exception):
    """ECOS API가 오류·정보 코드로 응답한 경우."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


def _normalize_result_code(code: str) -> str:
    c = (code or "").strip().upper()
    for prefix in ("INFO-", "ERROR-"):
        if c.startswith(prefix):
            c = c[len(prefix):]
    return c


def _is_retryable_code(code: str) -> bool:
    norm = _normalize_result_code(code)
    if norm in ("400.0", "602.0"):
        return True
    if "400.0" in norm or "602.0" in norm:
        return True
    return False


def _extract_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    stl = payload.get("StatisticTableList") or {}
    raw = stl.get("row")
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    return []


def _stat_table_url(stat_code: str) -> str:
    return f"https://ecos.bok.or.kr/#/SearchStat/{stat_code}"


def map_category(p_stat_code: str, stat_name: str) -> str:
    name = stat_name or ""
    market_kw = ("주가", "증권", "코스피", "코스닥", "주식", "증시", "유가증권")
    for kw in market_kw:
        if kw in name:
            return "증시"
    macro_kw = (
        "GDP",
        "국민소득",
        "국민계정",
        "생산",
        "고용",
        "실업",
        "수출",
        "수입",
        "무역",
        "소비",
        "투자",
    )
    for kw in macro_kw:
        if kw in name:
            return "실적"
    _ = p_stat_code
    return "정책"


def _build_ecos_summary(row: Dict[str, Any]) -> str:
    stat_name = (row.get("STAT_NAME") or "").strip()
    cycle = row.get("CYCLE")
    cycle_s = str(cycle).strip() if cycle not in (None, "") else ""
    org = row.get("ORG_NAME")
    org_s = str(org).strip() if org not in (None, "") else ""
    parts: List[str] = []
    if stat_name:
        parts.append(stat_name)
    if cycle_s:
        parts.append(f"주기: {cycle_s}")
    if org_s:
        parts.append(f"출처: {org_s}")
    text = " | ".join(parts) if parts else stat_name or "한국은행 경제통계"
    return text[:300]


def row_to_chroma_fields(
    row: Dict[str, Any],
    collected_date: str,
) -> Tuple[str, Dict[str, str], str]:
    """
    ECOS StatisticTableList row -> Chroma (document, metadata, id).

    통계 항목은 risk_label을 빈 문자열로 둡니다.
    """
    stat_code = (row.get("STAT_CODE") or "").strip()
    if not stat_code:
        raise ValueError("STAT_CODE가 비어 있습니다.")

    stat_name = (row.get("STAT_NAME") or "").strip()
    p_stat = (row.get("P_STAT_CODE") or "").strip()
    category = map_category(p_stat, stat_name)
    summary = _build_ecos_summary(row)
    meta: Dict[str, str] = {
        "title": stat_name or stat_code,
        "summary": summary,
        "url": _stat_table_url(stat_code),
        "date": collected_date,
        "category": category,
        "risk_label": "",
    }
    doc_text = f"{meta['title']} {summary}"
    return doc_text, meta, stat_code


def build_statistic_table_list_url(
    api_key: str,
    start_index: int,
    end_index: int,
    stat_code: Optional[str] = None,
) -> str:
    base = f"{ECOS_STATISTIC_TABLE_LIST_BASE}/{api_key}/json/kr/{start_index}/{end_index}"
    if stat_code:
        return f"{base}/{stat_code.strip()}"
    return base


def fetch_statistic_table_list(
    api_key: str,
    start_index: int = 1,
    end_index: int = 10,
    stat_code: Optional[str] = None,
    timeout_sec: float = 30.0,
) -> List[Dict[str, Any]]:
    """StatisticTableList 호출. INFO-200 시 빈 리스트."""
    url = build_statistic_table_list_url(api_key, start_index, end_index, stat_code)
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _SESSION.get(url, timeout=timeout_sec)
            resp.raise_for_status()
            payload = resp.json()
        except requests.Timeout as e:
            last_error = e
            logger.warning(
                "ECOS 요청 TIMEOUT (시도 %s/%s): %s", attempt, MAX_RETRIES, url
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
            continue
        except requests.RequestException as e:
            logger.error("ECOS HTTP 오류: %s", e)
            raise

        if "RESULT" in payload:
            result = payload["RESULT"]
            code = str(result.get("CODE", ""))
            message = str(result.get("MESSAGE", ""))
            norm = _normalize_result_code(code)

            if norm == "100" or code.upper().startswith("INFO-100"):
                logger.error("ECOS INFO-100: %s", message)
                raise EcosApiError(code, message)

            if norm == "200" or code.upper().startswith("INFO-200"):
                logger.info("ECOS INFO-200: %s", message)
                return []

            if _is_retryable_code(code):
                logger.warning(
                    "ECOS 재시도 대상 [%s] %s (시도 %s/%s)",
                    code,
                    message,
                    attempt,
                    MAX_RETRIES,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SEC * attempt)
                    continue
                logger.error("ECOS 실패(재시도 소진): [%s] %s", code, message)
                raise EcosApiError(code, message)

            logger.error("ECOS 미처리 RESULT: [%s] %s", code, message)
            raise EcosApiError(code, message)

        return _extract_rows(payload)

    if last_error:
        logger.error("ECOS TIMEOUT 최종 실패: %s", url)
        raise EcosApiError("400.0", str(last_error)) from last_error
    return []


def collect_bok_ecos_tables_and_store(
    stat_codes: Optional[Sequence[str]] = None,
    start_index: int = 1,
    end_index: int = 10,
    persist_dir: Optional[str] = None,
    api_key: Optional[str] = None,
) -> int:
    """
    ECOS StatisticTableList로 통계표 메타를 수집·저장(Market Analyzer 보조 데이터).

    BOK_API_KEY는 .env 또는 인자로 전달.
    """
    key = (api_key or os.getenv("BOK_API_KEY", "") or settings.BOK_API_KEY or "").strip()
    if not key:
        raise ValueError(
            "BOK_API_KEY가 비어 있습니다. .env에 설정하거나 api_key 인자를 넘기세요."
        )

    codes: Sequence[str] = stat_codes if stat_codes is not None else DEFAULT_STAT_CODES
    upsert_kwargs: Dict[str, Any] = {}
    if persist_dir:
        upsert_kwargs["persist_dir"] = persist_dir

    collected_date = datetime.today().strftime("%Y-%m-%d")
    total = 0

    for i, code in enumerate(codes):
        rows = fetch_statistic_table_list(
            key,
            start_index=start_index,
            end_index=end_index,
            stat_code=code.strip() or None,
        )
        if not rows:
            time.sleep(ECOS_REQUEST_INTERVAL)
            continue

        documents: List[str] = []
        metadatas: List[Dict[str, str]] = []
        ids: List[str] = []
        for row in rows:
            try:
                doc, meta, row_id = row_to_chroma_fields(row, collected_date)
            except ValueError as e:
                logger.warning("행 스킵: %s", e)
                continue
            documents.append(doc)
            metadatas.append(meta)
            ids.append(row_id)

        if documents:
            upsert_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                **upsert_kwargs,
            )
            total += len(documents)
            logger.info("ECOS ChromaDB upsert: stat_code=%s, %d건", code, len(documents))

        if i < len(codes) - 1:
            time.sleep(ECOS_REQUEST_INTERVAL)

    logger.info("ECOS 수집·저장 완료: %d건", total)
    return total


# 하위 호환
collect_and_store = collect_bok_ecos_tables_and_store


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    n = collect_google_news_and_store(max_items=10)
    print(f"구글 뉴스 수집: {n}건")
    if (os.getenv("BOK_API_KEY", "") or settings.BOK_API_KEY or "").strip():
        collect_bok_ecos_tables_and_store(
            stat_codes=["102Y004"], start_index=1, end_index=10
        )
    else:
        logger.warning("BOK_API_KEY 없음 — ECOS 수집은 건너뜀")
