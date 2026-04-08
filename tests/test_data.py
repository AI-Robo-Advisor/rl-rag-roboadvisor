"""src/data 모듈 통합 테스트.

data/processed/의 실제 parquet 파일을 읽어 검증한다.
실행 전 `python -m src.data.collector`로 파일이 생성되어 있어야 한다.

실행:
    pytest tests/test_data.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.indicators import compute_indicators

# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------
RETURNS_PATH = Path("data/processed/returns.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")

EXPECTED_TICKERS: list[str] = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ", "069500", "114260"
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def returns_df() -> pd.DataFrame:
    """returns.parquet을 세션 단위로 한 번만 로드한다."""
    return pd.read_parquet(RETURNS_PATH)


@pytest.fixture(scope="session")
def features_df() -> pd.DataFrame:
    """features.parquet을 세션 단위로 한 번만 로드한다."""
    return pd.read_parquet(FEATURES_PATH)


@pytest.fixture(scope="session")
def raw_indicators(returns_df: pd.DataFrame) -> pd.DataFrame:
    """정규화 전 raw 지표 DataFrame (RSI 범위 검증에 사용).

    Z-score 정규화된 features.parquet에서는 RSI 0~100 범위를 검증할 수 없으므로
    compute_indicators()를 직접 호출해 정규화 전 값을 얻는다.
    """
    return compute_indicators(returns_df)


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------
def test_parquet_loadable() -> None:
    """parquet 파일 두 개 모두 정상 로드되어야 한다."""
    df_r = pd.read_parquet(RETURNS_PATH)
    df_f = pd.read_parquet(FEATURES_PATH)
    assert not df_r.empty, "returns.parquet이 비어 있음"
    assert not df_f.empty, "features.parquet이 비어 있음"


def test_returns_shape(returns_df: pd.DataFrame) -> None:
    """행 수 > 1400, 열 수 == 10.

    Note:
        CLAUDE.local.md 추정치는 1650~1700행이나, US-KR inner join에서
        한국 공휴일(추석·설날 등) 차이로 실제 ~1423행이 산출된다.
        inner join 명세는 유지하되 임계값을 1400으로 조정한다.
    """
    assert len(returns_df) > 1400, (
        f"행 수 부족: {len(returns_df)}행 (inner join 후 ~1423행이 정상)"
    )
    assert len(returns_df.columns) == 10, (
        f"열 수 불일치: {len(returns_df.columns)}열"
    )


def test_returns_columns(returns_df: pd.DataFrame) -> None:
    """10개 티커 컬럼이 모두 존재해야 한다."""
    missing = [t for t in EXPECTED_TICKERS if t not in returns_df.columns]
    assert not missing, f"누락된 컬럼: {missing}"


def test_returns_no_missing(returns_df: pd.DataFrame) -> None:
    """결측치가 0개여야 한다."""
    total_missing = int(returns_df.isnull().sum().sum())
    assert total_missing == 0, (
        f"결측치 {total_missing}개 발견:\n{returns_df.isnull().sum()}"
    )


def test_log_return_range(returns_df: pd.DataFrame) -> None:
    """로그수익률 절댓값이 0.5 미만이어야 한다 (서킷 브레이커 수준의 극단값 제외)."""
    max_abs = returns_df.abs().max().max()
    assert max_abs < 0.5, f"극단값 발견: 최대 절댓값={max_abs:.4f}"


def test_date_index_type(returns_df: pd.DataFrame) -> None:
    """index가 datetime64 타입이어야 한다."""
    assert str(returns_df.index.dtype).startswith("datetime64"), (
        f"index 타입 불일치: {returns_df.index.dtype}"
    )


def test_no_duplicate_dates(returns_df: pd.DataFrame) -> None:
    """중복 날짜가 없어야 한다."""
    duplicates = int(returns_df.index.duplicated().sum())
    assert duplicates == 0, f"중복 날짜 {duplicates}개 발견"


def test_features_shape(returns_df: pd.DataFrame, features_df: pd.DataFrame) -> None:
    """features 행 수가 returns보다 적어야 한다.

    MACD 초기 NaN(약 33행)이 dropna()로 제거되기 때문에 정상적인 차이다.
    """
    assert len(features_df) < len(returns_df), (
        f"features({len(features_df)}행) >= returns({len(returns_df)}행): "
        "MACD 초기 NaN dropna가 적용되지 않았을 수 있음"
    )


def test_rsi_range(raw_indicators: pd.DataFrame) -> None:
    """정규화 전 RSI 값이 0~100 범위 안에 있어야 한다.

    features.parquet은 Z-score 정규화 후이므로 raw_indicators fixture를 사용한다.
    """
    rsi_cols = [c for c in raw_indicators.columns if c.endswith("_RSI")]
    assert len(rsi_cols) == len(EXPECTED_TICKERS), (
        f"RSI 컬럼 수 불일치: {len(rsi_cols)}개 (기대: {len(EXPECTED_TICKERS)}개)"
    )
    rsi_values = raw_indicators[rsi_cols].dropna()
    assert (rsi_values >= 0).all().all(), "RSI 0 미만 값 존재"
    assert (rsi_values <= 100).all().all(), "RSI 100 초과 값 존재"


def test_macd_columns_exist(features_df: pd.DataFrame) -> None:
    """모든 자산의 MACD, MACD_signal 컬럼이 존재해야 한다."""
    macd_cols = [c for c in features_df.columns if c.endswith("_MACD")]
    signal_cols = [c for c in features_df.columns if c.endswith("_MACD_signal")]

    assert len(macd_cols) == len(EXPECTED_TICKERS), (
        f"MACD 컬럼 수 불일치: {macd_cols}"
    )
    assert len(signal_cols) == len(EXPECTED_TICKERS), (
        f"MACD_signal 컬럼 수 불일치: {signal_cols}"
    )
