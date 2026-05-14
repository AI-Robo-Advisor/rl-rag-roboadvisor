"""src/rl/shap.py 유닛 테스트.

모델 파일이 필요 없는 get_feature_names()만 검증합니다.
compute_shap_explanation / generate_*_plot 은 PPO 모델 파일과
shap/torch가 필요하므로 integration 마킹 후 별도 실행합니다.
"""

import pytest

from src.rl.shap import RISK_FEATURE_NAMES, get_feature_names

TICKERS_10 = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ", "069500", "114260"]
TICKERS_3 = ["SPY", "QQQ", "IWM"]


def test_feature_names_total_length_default():
    """기본 파라미터(lookback=30, n_assets=10)일 때 총 333개인지 검증합니다."""
    names = get_feature_names(TICKERS_10, lookback=30)
    expected = (30 + 3) * 10 + 3  # 333
    assert len(names) == expected


def test_feature_names_total_length_custom_lookback():
    """lookback=5, n_assets=3 일 때 (5+3)×3+3 = 27개인지 검증합니다."""
    names = get_feature_names(TICKERS_3, lookback=5)
    expected = (5 + 3) * 3 + 3  # 27
    assert len(names) == expected


def test_feature_names_no_duplicates():
    """피처명에 중복이 없는지 검증합니다."""
    names = get_feature_names(TICKERS_10, lookback=30)
    assert len(names) == len(set(names))


def test_feature_names_risk_tags_at_end():
    """마지막 3개 피처명이 RISK_FEATURE_NAMES와 일치하는지 검증합니다."""
    names = get_feature_names(TICKERS_10, lookback=30)
    assert names[-3:] == RISK_FEATURE_NAMES
    assert names[-3] == "risk_규제변경"
    assert names[-2] == "risk_실적쇼크"
    assert names[-1] == "risk_급등락"


def test_feature_names_returns_window_oldest_first():
    """수익률 윈도우가 오래된 것(t-lookback)부터 시작하는지 검증합니다."""
    lookback = 5
    names = get_feature_names(TICKERS_3, lookback=lookback)
    # 첫 번째 피처: 가장 오래된 lag
    assert names[0] == f"{TICKERS_3[0]}_return_t-{lookback}"
    # 수익률 윈도우 마지막 행: t-1 (가장 최근)
    last_return_idx = lookback * len(TICKERS_3) - 1
    assert names[last_return_idx] == f"{TICKERS_3[-1]}_return_t-1"


def test_feature_names_concat_order():
    """[returns | weights | RSI | MACD_signal | risk] 순서를 검증합니다."""
    lookback = 5
    n = len(TICKERS_3)
    names = get_feature_names(TICKERS_3, lookback=lookback)

    returns_end = lookback * n          # 15
    weights_end = returns_end + n       # 18
    rsi_end = weights_end + n           # 21
    macd_end = rsi_end + n              # 24

    # weights 블록 시작
    assert names[returns_end].startswith("weight_")
    # RSI 블록 시작
    assert names[weights_end].endswith("_RSI")
    # MACD_signal 블록 시작
    assert names[rsi_end].endswith("_MACD_signal")
    # risk 블록 시작
    assert names[macd_end] == "risk_규제변경"


def test_feature_names_all_tickers_present_in_returns():
    """모든 티커가 수익률 윈도우 피처명에 포함되어 있는지 검증합니다."""
    names = get_feature_names(TICKERS_10, lookback=30)
    return_names = [n for n in names if "_return_t-" in n]
    for ticker in TICKERS_10:
        assert any(n.startswith(ticker) for n in return_names)


def test_feature_names_length_matches_obs_dim():
    """피처명 수가 PortfolioEnv obs_dim과 일치하는지 환경 생성 없이 수식으로 검증합니다."""
    lookback = 30
    n_assets = len(TICKERS_10)
    expected_obs_dim = (lookback + 3) * n_assets + 3  # 333

    names = get_feature_names(TICKERS_10, lookback=lookback)
    assert len(names) == expected_obs_dim
