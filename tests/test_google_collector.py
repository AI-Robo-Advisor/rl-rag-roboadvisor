"""구글 뉴스 RSS 수집·risk_label 단위 테스트."""
from unittest.mock import MagicMock, call, patch

from src.agent.news_collector import (
    _news_doc_id,
    collect_google_news_and_store,
    fetch_google_news_rss,
    infer_risk_label,
)


def test_infer_risk_label_empty():
    assert infer_risk_label("일반 시황", "배당 일정 확인") == ""


def test_infer_risk_label_regulation_and_shock():
    s = infer_risk_label("규제 강화", "실적쇼크 우려")
    assert "geopolitical_fx_risk" in s
    assert "equity_market_risk" in s


def test_infer_risk_label_volatility_and_rate():
    s = infer_risk_label("코스피 급락", "금리인상 부담")
    assert "equity_market_risk" in s
    assert "macro_rate_risk" in s


def test_fetch_google_news_rss_mock():
    """feedparser mock으로 구글 뉴스 RSS 파싱을 검증합니다."""
    import time

    mock_entry = MagicMock()
    mock_entry.title = "삼성전자 실적 발표 어닝쇼크"
    mock_entry.link = "https://news.google.com/articles/test123"
    mock_entry.summary = "<p>삼성전자 2분기 실적이 예상치를 크게 하회했다.</p>"
    mock_entry.published_parsed = time.strptime("2026-04-07", "%Y-%m-%d")

    mock_feed = MagicMock()
    mock_feed.entries = [mock_entry]
    mock_feed.bozo = False

    with patch("src.agent.news_collector.feedparser.parse", return_value=mock_feed):
        items = fetch_google_news_rss(
            "https://news.google.com/rss/search?q=test", "미국주식", 5
        )

    assert len(items) == 1
    assert "삼성전자" in items[0]["title"]
    assert items[0]["category"] == "미국주식"
    assert items[0]["source"] == "google_news"
    assert items[0]["date"] == "2026-04-07"
    assert "<p>" not in items[0]["summary"]  # HTML 태그 제거 확인
    assert len(_news_doc_id(items[0]["url"])) == 32


def test_collect_google_news_and_store_risk_label_present():
    """collect_google_news_and_store()가 upsert에 넘기는 메타데이터에 risk_label 키가 있어야 한다.

    PR #1 머지 이후 infer_risk_label() 호출이 누락돼 risk_label 없이 upsert되던 버그 재발 방지.
    """
    import time

    mock_entry = MagicMock()
    mock_entry.title = "연준 금리 인상 FOMC 빅스텝 발표"
    mock_entry.link = "https://news.google.com/articles/risktest"
    mock_entry.summary = "연준이 0.75%p 자이언트 스텝을 단행했다."
    mock_entry.published_parsed = time.strptime("2026-05-01", "%Y-%m-%d")

    mock_feed = MagicMock()
    mock_feed.entries = [mock_entry]
    mock_feed.bozo = False

    fake_feeds = {"채권금리": "https://news.google.com/rss/search?q=test"}

    with (
        patch("src.agent.news_collector.feedparser.parse", return_value=mock_feed),
        patch("src.agent.news_collector.upsert_documents") as mock_upsert,
        patch("src.agent.news_collector.time.sleep"),
    ):
        count = collect_google_news_and_store(max_items=5, feeds=fake_feeds)

    assert count == 1
    assert mock_upsert.called

    _, kwargs = mock_upsert.call_args
    metadatas = kwargs.get("metadatas") or mock_upsert.call_args[0][1]
    assert len(metadatas) == 1

    meta = metadatas[0]
    # risk_label 키가 반드시 존재해야 함 (누락 시 rag_eval 태그 정확도 측정 불가)
    assert "risk_label" in meta
    assert meta["risk_label"] == "macro_rate_risk"  # 제목·요약 모두 금리 키워드
