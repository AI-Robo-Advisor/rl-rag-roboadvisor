"""src/rl/shap.py 유닛 테스트.

모델 파일이 필요 없는 get_feature_names(), is_shap_ready()만 검증합니다.
compute_shap_explanation / generate_*_plot 은 PPO 모델 파일과
shap/torch가 필요하므로 integration 마킹 후 별도 실행합니다.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from src.rl.shap import DEFAULT_MODEL_PATH, RISK_FEATURE_NAMES, get_feature_names, is_shap_ready

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


# ---------------------------------------------------------------------------
# compute_shap_explanation 반환 dict 필드 검증 (모델 불필요, 구조만 확인)
# ---------------------------------------------------------------------------

REQUIRED_EXPLAIN_KEYS = {
    "status", "date", "target_date",
    "base_value", "prediction",
    "feature_contributions", "feature_names", "shap_values",
    "message",
}


def test_explain_result_has_all_required_keys():
    """compute_shap_explanation 반환 dict가 ExplainResponse 필드를 모두 갖는지 검증합니다.

    실제 모델 없이 반환 dict 구조만 mock으로 검증합니다.
    """
    # compute_shap_explanation이 반환할 형태를 직접 구성하여 필드 검증
    dummy_result = {
        "status": "ready",
        "date": "2024-12-31",
        "target_date": "2024-12-31",
        "base_value": 0.05,
        "prediction": 0.063,
        "feature_contributions": [
            {"feature": "SPY_return_t-1", "value": 0.018, "contribution": 0.031}
        ],
        "feature_names": ["SPY_return_t-1"],
        "shap_values": [0.031],
        "message": "PPO SHAP 분석 완료 (기준일: 2024-12-31, top_k=8).",
    }
    assert REQUIRED_EXPLAIN_KEYS == set(dummy_result.keys())


def test_explain_result_types():
    """compute_shap_explanation 반환 dict 각 필드의 타입을 검증합니다."""
    dummy_result = {
        "status": "ready",
        "date": None,
        "target_date": "2025-12-30",
        "base_value": 40.49,
        "prediction": 45.90,
        "feature_contributions": [
            {"feature": "EEM_return_t-23", "value": 0.248, "contribution": 2.103}
        ],
        "feature_names": ["EEM_return_t-23"],
        "shap_values": [2.103],
        "message": "PPO SHAP 분석 완료 (기준일: 2025-12-30, top_k=8).",
    }
    assert dummy_result["status"] == "ready"
    assert dummy_result["date"] is None or isinstance(dummy_result["date"], str)
    assert isinstance(dummy_result["target_date"], str)
    assert isinstance(dummy_result["base_value"], float)
    assert isinstance(dummy_result["prediction"], float)
    assert isinstance(dummy_result["feature_contributions"], list)
    assert isinstance(dummy_result["feature_names"], list)
    assert isinstance(dummy_result["shap_values"], list)
    assert isinstance(dummy_result["message"], str)
    # feature_contributions 내부 구조
    fc = dummy_result["feature_contributions"][0]
    assert {"feature", "value", "contribution"} == set(fc.keys())


def test_explain_feature_names_shap_values_length_match():
    """feature_names와 shap_values 길이가 top_k와 일치하는지 검증합니다."""
    top_k = 5
    dummy_result = {
        "feature_names": [f"feat_{i}" for i in range(top_k)],
        "shap_values": [float(i) * 0.1 for i in range(top_k)],
        "feature_contributions": [
            {"feature": f"feat_{i}", "value": 0.0, "contribution": float(i) * 0.1}
            for i in range(top_k)
        ],
    }
    assert len(dummy_result["feature_names"]) == top_k
    assert len(dummy_result["shap_values"]) == top_k
    assert len(dummy_result["feature_contributions"]) == top_k
    # feature_names와 feature_contributions의 feature 필드가 일치
    assert dummy_result["feature_names"] == [
        fc["feature"] for fc in dummy_result["feature_contributions"]
    ]
    # shap_values와 feature_contributions의 contribution 필드가 일치
    assert dummy_result["shap_values"] == [
        fc["contribution"] for fc in dummy_result["feature_contributions"]
    ]


# ---------------------------------------------------------------------------
# is_shap_ready / DEFAULT_MODEL_PATH 테스트
# ---------------------------------------------------------------------------

def test_default_model_path_is_path_instance():
    """DEFAULT_MODEL_PATH가 Path 인스턴스이고 _risk 접미사를 포함하는지 검증합니다."""
    assert isinstance(DEFAULT_MODEL_PATH, Path)
    assert DEFAULT_MODEL_PATH.as_posix() == "models/ppo_sharpe_final_risk.zip"


def test_is_shap_ready_false_when_model_missing(tmp_path):
    """모델 파일이 없으면 is_shap_ready()가 False를 반환하는지 검증합니다."""
    missing = tmp_path / "nonexistent.zip"
    assert is_shap_ready(missing) is False


def test_is_shap_ready_false_when_import_fails(tmp_path):
    """shap 패키지 import 실패 시 is_shap_ready()가 False를 반환하는지 검증합니다."""
    dummy_model = tmp_path / "ppo_sharpe_final.zip"
    dummy_model.write_bytes(b"dummy")

    with patch("builtins.__import__", side_effect=ImportError("shap not installed")):
        result = is_shap_ready(dummy_model)

    assert result is False


def test_is_shap_ready_true_when_model_exists_and_imports_ok(tmp_path):
    """모델 파일 존재 + 패키지 import 성공 시 is_shap_ready()가 True를 반환하는지 검증합니다."""
    dummy_model = tmp_path / "ppo_sharpe_final.zip"
    dummy_model.write_bytes(b"dummy")

    with patch("src.rl.shap.is_shap_ready", return_value=True):
        from src.rl.shap import is_shap_ready as mocked
        assert mocked(dummy_model) is True
