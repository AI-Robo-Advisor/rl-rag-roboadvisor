"""
네이버 금융 뉴스(메인) + 한국은행 ECOS(보조) 통합 수집 및 ChromaDB 저장.

- 네이버: 금융 뉴스 목록 HTML(증시·실적·정책). 공개 RSS가 없어 requests 기반이며,
  feedparser는 RSS/Atom URL이 있을 때 확장용으로 사용합니다.
- ECOS StatisticTableList: 통계표 메타데이터(Market Analyzer / 거시 지표 보조).

공통 스키마(metadata): title, summary(≤300자), url, date, category, risk_label(뉴스만, 키워드 기반).
문서 ID: 뉴스=URL MD5, 통계=STAT_CODE.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup

from apps.api.config import settings
from src.agent.vectorstore import upsert_documents

logger = logging.getLogger(__name__)

# --- 공통 -----------------------------------------------------------------
NAVER_REQUEST_INTERVAL = 1.0
ECOS_REQUEST_INTERVAL = 1.0

NAVER_FINANCE_BASE = "https://finance.naver.com"

# 증시 / 실적 / 정책 — 네이버 금융 뉴스 목록(HTML)
NAVER_FINANCE_CATEGORY_URLS: Dict[str, str] = {
    "증시": "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258",
    "실적": "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=261",
    "정책": "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=263",
}

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


def _news_doc_id(url: str) -> str:
    """뉴스 URL 기준 Chroma ID (MD5)."""
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def _parse_naver_date(raw: str) -> str:
    """네이버 금융 날짜 문자열 -> YYYY-MM-DD."""
    raw = raw.strip()
    for fmt in ("%Y.%m.%d %H:%M", "%Y.%m.%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return datetime.today().strftime("%Y-%m-%d")


def fetch_naver_finance_news_html(
    list_url: str,
    category: str,
    limit: int,
) -> List[Dict[str, str]]:
    """
    네이버 금융 뉴스 **목록 HTML**에서 기사를 수집합니다.

    Args:
        list_url: news_list.naver ... LSS2D URL.
        category: 증시/실적/정책.
        limit: 최대 수집 건수.

    Returns:
        title, summary, url, date, category 키를 가진 dict 리스트.
    """
    if limit <= 0:
        return []

    try:
        resp = _SESSION.get(list_url, timeout=15)
        resp.raise_for_status()
        ct = (resp.headers.get("Content-Type") or "").lower()
        if "charset=euc-kr" in ct:
            resp.encoding = "euc-kr"
        elif "charset=utf-8" in ct:
            resp.encoding = "utf-8"
        else:
            resp.encoding = resp.apparent_encoding or "utf-8"
    except requests.RequestException as e:
        logger.error("네이버 금융 목록 요청 실패 [%s]: %s", list_url, e)
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    items: List[Dict[str, str]] = []

    def _add_from_link_and_summary(title_el, dd_summary: Any) -> None:
        nonlocal items
        if len(items) >= limit:
            return
        title = title_el.get_text(strip=True)
        href = title_el.get("href", "") or ""
        article_url = urljoin(NAVER_FINANCE_BASE, href) if href else ""
        if not title or not article_url:
            return

        date_str = ""
        summary = ""
        if dd_summary:
            date_tag = dd_summary.select_one("span.wdate")
            if date_tag:
                date_str = _parse_naver_date(date_tag.get_text(strip=True))
            for span in dd_summary.select("span"):
                span.decompose()
            summary = dd_summary.get_text(strip=True)[:300]

        if len(summary) > 300:
            summary = summary[:300]
        if not date_str:
            date_str = datetime.today().strftime("%Y-%m-%d")

        items.append(
            {
                "title": title,
                "summary": summary,
                "url": article_url,
                "date": date_str,
                "category": category,
            }
        )

    # 2024~ 네이버 금융: ul.realtimeNewsList > li.newsList > dl (한 dl에 여러 기사)
    for dl in soup.select("ul.realtimeNewsList li.newsList dl"):
        title_tags = dl.select("dt.articleSubject a, dd.articleSubject a")
        summary_dds = dl.select("dd.articleSummary")
        for t_a, dd_sum in zip(title_tags, summary_dds):
            _add_from_link_and_summary(t_a, dd_sum)
            if len(items) >= limit:
                break
        if len(items) >= limit:
            break

    # 구 레이아웃: ul.newsList > dl (dl당 1건, dt a)
    if not items:
        for dl in soup.select("ul.newsList dl"):
            if len(items) >= limit:
                break
            title_tag = dl.select_one("dt.articleSubject a, dd.articleSubject a")
            if not title_tag:
                title_tag = dl.select_one("dt a")
            if not title_tag:
                continue
            _add_from_link_and_summary(title_tag, dl.select_one("dd.articleSummary"))

    logger.info("[네이버:%s] %d건 파싱", category, len(items))
    return items


def fetch_items_from_rss_feed(rss_url: str, category: str, limit: int) -> List[Dict[str, str]]:
    """
    feedparser로 RSS/Atom 피드에서 항목을 가져옵니다(확장·보조용).

    Args:
        rss_url: 피드 URL.
        category: 저장 시 category 메타값.
        limit: 최대 건수.

    Returns:
        title, summary, url, date, category.
    """
    if limit <= 0:
        return []

    parsed = feedparser.parse(
        rss_url,
        agent=_SESSION.headers.get("User-Agent", "rl-rag-roboadvisor/1.0"),
    )
    if getattr(parsed, "bozo", False) and not parsed.entries:
        logger.warning("RSS 파싱 경고: %s (%s)", rss_url, getattr(parsed, "bozo_exception", ""))
        return []

    items: List[Dict[str, str]] = []
    for ent in parsed.entries[:limit]:
        title = (ent.get("title") or "").strip()
        link = (ent.get("link") or "").strip()
        if not title or not link:
            continue
        summary = (ent.get("summary") or ent.get("description") or "").strip()
        if summary:
            soup = BeautifulSoup(summary, "lxml")
            summary = soup.get_text(separator=" ", strip=True)[:300]

        published = ""
        if ent.get("published_parsed"):
            try:
                published = time.strftime(
                    "%Y-%m-%d", ent.published_parsed
                )
            except (TypeError, ValueError):
                published = datetime.today().strftime("%Y-%m-%d")
        elif ent.get("updated_parsed"):
            try:
                published = time.strftime("%Y-%m-%d", ent.updated_parsed)
            except (TypeError, ValueError):
                published = datetime.today().strftime("%Y-%m-%d")
        else:
            published = datetime.today().strftime("%Y-%m-%d")

        items.append(
            {
                "title": title,
                "summary": summary,
                "url": link,
                "date": published,
                "category": category,
            }
        )

    logger.info("[RSS:%s] %d건", category, len(items))
    return items


def _naver_items_to_upsert(
    rows: List[Dict[str, str]],
) -> Tuple[List[str], List[Dict[str, str]], List[str]]:
    """뉴스 row -> documents, metadatas, ids."""
    documents: List[str] = []
    metadatas: List[Dict[str, str]] = []
    ids: List[str] = []
    for row in rows:
        title = row["title"]
        summary = row["summary"][:300] if row.get("summary") else ""
        risk = infer_risk_label(title, summary)
        meta = {
            "title": title,
            "summary": summary,
            "url": row["url"],
            "date": row["date"],
            "category": row["category"],
            "risk_label": risk,
        }
        documents.append(f"{title} {summary}")
        metadatas.append(meta)
        ids.append(_news_doc_id(row["url"]))
    return documents, metadatas, ids


def collect_naver_finance_news_and_store(
    max_items: int = 5,
    category_urls: Optional[Dict[str, str]] = None,
    use_rss: bool = False,
    rss_url_by_category: Optional[Dict[str, str]] = None,
    persist_dir: Optional[str] = None,
) -> int:
    """
    네이버 금융(또는 지정 RSS)에서 뉴스를 수집해 ChromaDB에 upsert합니다.

    카테고리 순회하며 max_items에 도달할 때까지 수집합니다. 요청 간격 1초 이상.

    Args:
        max_items: 저장할 최대 건수.
        category_urls: {카테고리: 목록 URL}. None이면 NAVER_FINANCE_CATEGORY_URLS.
        use_rss: True면 rss_url_by_category의 URL로 feedparser 사용.
        rss_url_by_category: use_rss True일 때 필수.
        persist_dir: Chroma 경로.

    Returns:
        upsert한 건수.
    """
    if max_items <= 0:
        return 0

    urls = category_urls or NAVER_FINANCE_CATEGORY_URLS
    upsert_kwargs: Dict[str, Any] = {}
    if persist_dir:
        upsert_kwargs["persist_dir"] = persist_dir

    collected: List[Dict[str, str]] = []
    categories = list(urls.keys())

    for i, cat in enumerate(categories):
        if len(collected) >= max_items:
            break
        need = max_items - len(collected)
        if use_rss and rss_url_by_category and cat in rss_url_by_category:
            batch = fetch_items_from_rss_feed(rss_url_by_category[cat], cat, need)
        else:
            batch = fetch_naver_finance_news_html(urls[cat], cat, need)
        collected.extend(batch)
        if i < len(categories) - 1 and len(collected) < max_items:
            time.sleep(NAVER_REQUEST_INTERVAL)

    collected = collected[:max_items]
    if not collected:
        return 0
    documents, metadatas, ids = _naver_items_to_upsert(collected)
    upsert_documents(documents=documents, metadatas=metadatas, ids=ids, **upsert_kwargs)
    logger.info("네이버 뉴스 ChromaDB upsert: %d건", len(documents))
    return len(documents)


# --- ECOS (한국은행 Market Analyzer) ----------------------------------------

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
            c = c[len(prefix) :]
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


# 하위 호환: 기존 테스트·스크립트에서 사용
collect_and_store = collect_bok_ecos_tables_and_store


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    collect_naver_finance_news_and_store(max_items=5)
    if (os.getenv("BOK_API_KEY", "") or settings.BOK_API_KEY or "").strip():
        collect_bok_ecos_tables_and_store(
            stat_codes=["102Y004"], start_index=1, end_index=10
        )
    else:
        logger.warning("BOK_API_KEY 없음 — ECOS 수집은 건너뜀")
