import numpy as np

from src.agent.risk_tags import RL_RISK_TAGS, extract_rl_risk_tags, get_risk_vector


def test_regulation_and_surge_detected():
    tags = extract_rl_risk_tags("금융위원회가 가상자산 규제변경을 발표하자 시장이 급락했다")
    assert "규제변경" in tags
    assert "급등락" in tags
    vec = get_risk_vector(tags)
    assert vec[0] == 1.0
    assert vec[2] == 1.0


def test_earnings_shock_and_surge_detected():
    tags = extract_rl_risk_tags("삼성전자 어닝쇼크로 주가 급락")
    assert "실적쇼크" in tags
    assert "급등락" in tags
    vec = get_risk_vector(tags)
    assert vec[1] == 1.0
    assert vec[2] == 1.0


def test_empty_text_returns_no_tags():
    tags = extract_rl_risk_tags("")
    assert tags == []
    vec = get_risk_vector(tags)
    assert list(vec) == [0.0, 0.0, 0.0]


def test_risk_vector_shape_and_dtype():
    vec = get_risk_vector(["규제변경"])
    assert vec.shape == (3,)
    assert vec.dtype == np.float32


def test_rl_risk_tags_order():
    assert RL_RISK_TAGS == ["규제변경", "실적쇼크", "급등락"]
