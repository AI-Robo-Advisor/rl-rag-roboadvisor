"""구글 뉴스 RSS 수집·risk_label 단위 테스트."""
from unittest.mock import MagicMock, patch

from src.agent.news_collector import (
    _news_doc_id,
    fetch_google_news_rss,
    infer_risk_label,
)


def test_infer_risk_label_empty():
    assert infer_risk_label("일반 시황", "코스피 소폭 상승") == ""


def test_infer_risk_label_regulation_and_shock():
    s = infer_risk_label("규제 강화", "실적쇼크 우려")
    assert "규제변경" in s
    assert "실적쇼크" in s


def test_infer_risk_label_volatility_and_rate():
    s = infer_risk_label("코스피 급락", "금리인상 부담")
    assert "급등락" in s
    assert "금리인상" in s


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
            "https://news.google.com/rss/search?q=test", "실적쇼크", 5
        )

    assert len(items) == 1
    assert "삼성전자" in items[0]["title"]
    assert items[0]["category"] == "실적쇼크"
    assert items[0]["source"] == "google_news"
    assert items[0]["date"] == "2026-04-07"
    assert "<p>" not in items[0]["summary"]  # HTML 태그 제거 확인
    assert len(_news_doc_id(items[0]["url"])) == 32
