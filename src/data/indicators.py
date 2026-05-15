"""기술적 지표 계산 모듈.

로그수익률 DataFrame을 입력받아 RSI·MACD를 계산하고
정규화 전 raw_features.parquet을 생성한다.

Downstream:
    data/processed/raw_features.parquet → Walk-Forward 정규화 파이프라인
    data/processed/features.parquet     → legacy/EDA 호환용 전체 기간 Z-score

컬럼명 규칙 변경 시 이문정·강유영에게 반드시 사전 공지.
"""

import logging
from pathlib import Path

import pandas as pd
import pandas_ta_classic as ta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_indicators(returns: pd.DataFrame) -> pd.DataFrame:
    """각 자산의 로그수익률로부터 RSI와 MACD를 계산한다.

    Args:
        returns: 로그수익률 DataFrame.
            index: Date (datetime64[ns])
            columns: 티커명 (예: SPY, QQQ, ..., 069500, 114260)

    Returns:
        기술적 지표 DataFrame.
            index: Date (datetime64[ns])
            columns: {ticker}_RSI, {ticker}_MACD, {ticker}_MACD_signal × 자산 수
            RSI 파라미터: period=14
            MACD 파라미터: fast=12, slow=26, signal=9

    Note:
        초기 NaN 행은 제거하지 않는다. 호출자(build_features / __main__)에서 dropna 처리.
        RSI는 앞 14행, MACD는 앞 33행이 NaN으로 생성되는 것이 정상이다.
    """
    frames: list[pd.Series] = []

    for ticker in returns.columns:
        series: pd.Series = returns[ticker]

        # RSI (period=14)
        # clip(0, 100): pandas-ta-classic이 로그수익률 입력 시 부동소수점 오차로
        # 100을 미세하게 초과하는 엣지케이스 방지 (RSI 정의상 반드시 0~100)
        rsi = ta.rsi(series, length=14).clip(0, 100)
        rsi.name = f"{ticker}_RSI"

        # MACD (fast=12, slow=26, signal=9)
        # pandas-ta-classic 반환 컬럼: MACD_12_26_9 (값), MACDh_12_26_9 (히스토그램), MACDs_12_26_9 (시그널)
        macd_df: pd.DataFrame = ta.macd(series, fast=12, slow=26, signal=9)
        macd_val = macd_df["MACD_12_26_9"].rename(f"{ticker}_MACD")
        macd_sig = macd_df["MACDs_12_26_9"].rename(f"{ticker}_MACD_signal")

        frames.extend([rsi, macd_val, macd_sig])
        logger.debug("  %s 지표 계산 완료", ticker)

    result = pd.concat(frames, axis=1)
    result.index.name = "Date"
    logger.info(
        "지표 계산 완료: %d개 자산, %d개 컬럼, shape=%s",
        len(returns.columns),
        len(result.columns),
        result.shape,
    )
    return result


def build_raw_features(returns: pd.DataFrame) -> pd.DataFrame:
    """로그수익률과 기술적 지표를 결합하여 정규화 전 features를 반환한다.

    처리 순서:
        1. compute_indicators(returns)로 RSI·MACD 계산
        2. returns 컬럼을 {ticker}_return으로 rename
        3. returns + indicators concat
        4. dropna (MACD 초기 ~33행 NaN 제거)

    Args:
        returns: raw 로그수익률 DataFrame (index=Date, columns=티커명).

    Returns:
        정규화 전 raw features DataFrame.
            columns: {ticker}_return, {ticker}_RSI, {ticker}_MACD, {ticker}_MACD_signal × 자산 수
            shape: returns보다 약 33행 적음 (정상).

    Note:
        Look-ahead bias 방지를 위해 Z-score 파라미터는 이 함수에서 계산하지 않는다.
        train_walkforward.py/backtest.py가 각 Walk-Forward 학습 구간 통계로 정규화해야 한다.
    """
    indicators = compute_indicators(returns)

    returns_renamed = returns.rename(
        columns={ticker: f"{ticker}_return" for ticker in returns.columns}
    )

    combined = pd.concat([returns_renamed, indicators], axis=1)
    features = combined.dropna()

    logger.info(
        "raw_features 구성 완료: shape=%s (returns %d행 대비 %d행 적음)",
        features.shape,
        len(returns),
        len(returns) - len(features),
    )
    return features


def build_features(returns: pd.DataFrame) -> pd.DataFrame:
    """전체 기간 Z-score 정규화 features를 반환한다.

    이 함수는 기존 downstream 호환과 EDA 확인용으로 유지한다. Walk-Forward
    학습/백테스트에는 전체 기간 통계가 들어간 이 결과를 직접 사용하지 않는다.
    """
    raw_features = build_raw_features(returns)
    return (raw_features - raw_features.mean()) / raw_features.std()


def main() -> None:
    """returns.parquet을 읽어 raw_features.parquet과 features.parquet을 생성한다.

    단독 실행 시 사용: python -m src.data.indicators
    collector.py 실행 후 returns.parquet이 존재해야 한다.
    """
    from config import DATA_PROCESSED_DIR  # 순환 임포트 방지를 위해 지역 import

    returns_path = Path(DATA_PROCESSED_DIR) / "returns.parquet"
    raw_features_path = Path(DATA_PROCESSED_DIR) / "raw_features.parquet"
    features_path = Path(DATA_PROCESSED_DIR) / "features.parquet"

    if not returns_path.exists():
        raise FileNotFoundError(
            f"returns.parquet 없음: {returns_path}\n"
            "먼저 `python -m src.data.collector`를 실행하세요."
        )

    logger.info("returns.parquet 로드: %s", returns_path)
    returns = pd.read_parquet(returns_path)
    logger.info("로드 완료: shape=%s", returns.shape)

    raw_features = build_raw_features(returns)
    features = (raw_features - raw_features.mean()) / raw_features.std()

    raw_features_path.parent.mkdir(parents=True, exist_ok=True)
    raw_features.to_parquet(raw_features_path)
    features.to_parquet(features_path)
    logger.info("raw_features.parquet 저장 완료: %s  shape=%s", raw_features_path, raw_features.shape)
    logger.info("features.parquet 저장 완료: %s  shape=%s", features_path, features.shape)


if __name__ == "__main__":
    main()
