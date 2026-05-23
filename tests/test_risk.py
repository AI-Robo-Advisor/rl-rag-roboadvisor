import numpy as np
import pandas as pd

from src.agent.risk_tags import RL_RISK_TAGS, extract_rl_risk_tags, get_risk_vector, score_risk_vector
from src.data.market_close import adjust_date_for_market_close


def test_macro_rate_detected():
    text = "Fed 자이언트 스텝 금리 인상 달러 강세"
    tags = extract_rl_risk_tags(text)
    assert "macro_rate_risk" in tags
    macro, _, _ = score_risk_vector(text)
    assert macro > 0


def test_equity_market_detected():
    text = "삼성전자 어닝쇼크로 증시 급락 VIX 급등"
    tags = extract_rl_risk_tags(text)
    assert "equity_market_risk" in tags
    _, equity, _ = score_risk_vector(text)
    assert equity > 0


def test_geopolitical_detected():
    text = "러시아-우크라이나 전쟁 원/달러 환율 급등 공급망 차질"
    tags = extract_rl_risk_tags(text)
    assert "geopolitical_fx_risk" in tags
    _, _, geo = score_risk_vector(text)
    assert geo > 0


def test_empty_text_returns_no_tags():
    assert extract_rl_risk_tags("") == []
    vec = get_risk_vector("")
    assert list(vec) == [0.0, 0.0, 0.0]


def test_risk_vector_shape_and_dtype():
    vec = get_risk_vector("금리 인상 달러 강세")
    assert vec.shape == (3,)
    assert vec.dtype == np.float32


def test_rl_risk_tags_order():
    assert RL_RISK_TAGS == ["macro_rate_risk", "equity_market_risk", "geopolitical_fx_risk"]


def test_score_risk_vector_range():
    macro, equity, geo = score_risk_vector("연준 금리 인상 CPI 예상 상회 달러 강세")
    assert 0.0 <= macro <= 1.0
    assert 0.0 <= equity <= 1.0
    assert 0.0 <= geo <= 1.0


def test_no_false_positive_on_normal_news():
    text = "코스피 외국인 순매수에 0.4% 상승 마감"
    macro, equity, geo = score_risk_vector(text)
    assert macro == 0.0
    assert equity == 0.0
    assert geo == 0.0


# ── market_close 테스트 ─────────────────────────────────────────

def test_market_close_kr_before_close():
    # 15:00 KST → 마감 전 → 당일 반환
    dt = pd.Timestamp("2022-06-15 15:00:00", tz="Asia/Seoul")
    result = adjust_date_for_market_close(dt, "KR")
    assert str(result) == "2022-06-15"


def test_market_close_kr_after_close():
    # 16:00 KST → 마감 후 → 다음 거래일 반환
    dt = pd.Timestamp("2022-06-15 16:00:00", tz="Asia/Seoul")
    result = adjust_date_for_market_close(dt, "KR")
    assert str(result) > "2022-06-15"


def test_market_close_us_after_close():
    # 17:00 ET → 마감 후 → 다음 거래일
    dt = pd.Timestamp("2023-03-10 17:00:00", tz="America/New_York")
    result = adjust_date_for_market_close(dt, "US")
    assert str(result) > "2023-03-10"
