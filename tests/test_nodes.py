import pytest

from src.agent.nodes import analyst_node


@pytest.mark.integration
def test_analyst_node_returns_required_keys():
    state = {
        "query": "삼성전자 반도체 전망은?",
        "context": "[문서 1] 삼성전자 HBM 수요 급증\n날짜: 2025-01-01\n본문: HBM 공급 부족으로 실적 개선 기대\n출처: https://example.com/news/1",
        "documents": [{"content": "HBM 수요 급증 실적쇼크 우려", "metadata": {"url": "https://example.com/news/1"}}],
        "risk_tags": ["변동성_리스크"],
        "messages": ["[THINK][planner] 플랜 완료", "[THINK][researcher] 검색 완료"],
        "retry_count": 0,
        "needs_research_retry": False,
        "sources": [],
        "reasoning_trace": "",
    }

    result = analyst_node(state)

    assert "response" in result
    assert "sources" in result
    assert "rl_risk_tags" in result
    assert "reasoning_trace" in result
    assert isinstance(result["rl_risk_tags"], list)
    assert isinstance(result["sources"], list)
    assert len(result["response"]) > 0
