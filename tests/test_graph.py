import pytest

from src.agent.graph import run_graph


@pytest.mark.integration
def test_run_graph_returns_required_keys():
    result = run_graph("삼성전자 HBM 실적 전망은?")

    assert "response" in result
    assert "sources" in result
    assert "rl_risk_tags" in result
    assert "reasoning_trace" in result
    assert isinstance(result["sources"], list)
    assert isinstance(result["rl_risk_tags"], list)
    assert len(result.get("response", "")) > 0
