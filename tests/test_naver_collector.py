"""네이버 수집·risk_label 단위 테스트."""
from unittest.mock import MagicMock, patch

from src.agent.news_collector import (
    infer_risk_label,
    fetch_naver_finance_news_html,
    _news_doc_id,
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


def test_fetch_naver_html_mock():
    html = """
    <html><body><ul class="newsList">
    <li><dl>
        <dt><a href="/news/read.naver?nid=1">삼성전자 실적 발표</a></dt>
        <dd class="articleSummary">
            <span class="wdate">2026.04.06 09:00</span>
            요약 텍스트입니다.
        </dd>
    </dl></li>
    </ul></body></html>
    """
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.encoding = "utf-8"
    mock_resp.raise_for_status = MagicMock()
    with patch(
        "src.agent.news_collector._SESSION.get",
        return_value=mock_resp,
    ):
        items = fetch_naver_finance_news_html("http://dummy", "증시", 5)
    assert len(items) == 1
    assert "삼성전자" in items[0]["title"]
    assert items[0]["category"] == "증시"
    assert len(_news_doc_id(items[0]["url"])) == 32
