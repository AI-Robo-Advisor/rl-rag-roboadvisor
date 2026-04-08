"""데이터 수집 및 전처리 파이프라인.

해외 ETF(yfinance)와 국내 ETF(pykrx)를 수집·병합하여
returns.parquet(raw 로그수익률)과 features.parquet(Z-score 정규화)를 저장한다.

Downstream:
    data/processed/returns.parquet  → src/rl/env.py (이문정)
    data/processed/features.parquet → src/rl/env.py, src/rl/backtest.py (이문정·강유영)
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock

from config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    END_DATE,
    START_DATE,
    TICKERS_GLOBAL,
    TICKERS_KR,
)
from src.data.indicators import compute_indicators

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def collect_global(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """yfinance로 해외 ETF 수정종가를 수집한다.

    Args:
        tickers: 야후 파이낸스 티커 목록 (예: ["SPY", "QQQ"]).
        start: 시작일 문자열 "YYYY-MM-DD".
        end: 종료일 문자열 "YYYY-MM-DD".

    Returns:
        index=Date(datetime64), columns=티커명, 값=수정종가인 DataFrame.

    Raises:
        ValueError: 수집 결과가 비어 있을 때.
    """
    logger.info("해외 ETF 수집 시작: %s", tickers)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)

    adj_close = raw["Adj Close"]
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame(name=tickers[0])

    adj_close.index = pd.to_datetime(adj_close.index)
    adj_close.index.name = "Date"

    if adj_close.empty:
        raise ValueError("yfinance 수집 결과가 비어 있습니다.")

    logger.info("해외 ETF 수집 완료: shape=%s", adj_close.shape)
    return adj_close[tickers]  # 컬럼 순서 보장


def collect_kr(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """pykrx로 국내 ETF 종가를 수집한다.

    API 호출 사이에 time.sleep(0.5)를 추가하여 서버 부하를 줄인다.

    Args:
        tickers: KRX 종목코드 목록 (예: ["069500", "114260"]).
        start: 시작일 문자열 "YYYY-MM-DD".
        end: 종료일 문자열 "YYYY-MM-DD".

    Returns:
        index=Date(datetime64), columns=종목코드, 값=종가인 DataFrame.

    Raises:
        ValueError: 수집 결과가 비어 있을 때.
    """
    start_yyyymmdd = start.replace("-", "")
    end_yyyymmdd = end.replace("-", "")

    logger.info("국내 ETF 수집 시작: %s", tickers)
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        df = stock.get_etf_ohlcv_by_date(start_yyyymmdd, end_yyyymmdd, ticker)
        close = df[["종가"]].rename(columns={"종가": ticker})
        close.index = pd.to_datetime(close.index)
        close.index.name = "Date"
        frames.append(close)
        logger.info("  %s 수집 완료: %d행", ticker, len(close))
        time.sleep(0.5)

    result = pd.concat(frames, axis=1)
    if result.empty:
        raise ValueError("pykrx 수집 결과가 비어 있습니다.")

    logger.info("국내 ETF 수집 완료: shape=%s", result.shape)
    return result


def merge_prices(
    df_global: pd.DataFrame,
    df_kr: pd.DataFrame,
) -> pd.DataFrame:
    """해외·국내 ETF 가격을 공통 거래일 기준으로 병합한다.

    inner join → Forward Fill → dropna 순서로 처리한다.

    Args:
        df_global: 해외 ETF 수정종가 DataFrame.
        df_kr: 국내 ETF 종가 DataFrame.

    Returns:
        병합·정제된 가격 DataFrame.
    """
    merged = pd.concat([df_global, df_kr], axis=1, join="inner")
    merged = merged.ffill().dropna()
    logger.info("가격 병합 완료: shape=%s (inner join + ffill + dropna)", merged.shape)
    return merged


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """일별 로그수익률을 계산한다.

    Args:
        prices: 가격 DataFrame (index=Date, columns=티커).

    Returns:
        로그수익률 DataFrame. 첫 행(NaN)은 제거된다.
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    logger.info("로그수익률 계산 완료: shape=%s", returns.shape)
    return returns


def save_raw_returns(returns: pd.DataFrame, path: str | Path) -> None:
    """정규화 전 raw 로그수익률을 parquet으로 저장한다.

    ⚠️ 정규화 전 값으로 저장해야 한다.
    trading_env.py가 이 파일을 읽어 자체 정규화하므로
    여기서 정규화하면 이중 정규화가 발생한다. (이문정 합의사항)

    Args:
        returns: raw 로그수익률 DataFrame.
        path: 저장 경로 (디렉토리가 없으면 자동 생성).
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(dest)
    logger.info("returns.parquet 저장 완료: %s  shape=%s", dest, returns.shape)


def normalize_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """전체 기간 기준 Z-score 정규화를 적용한다.

    Args:
        df: 정규화할 DataFrame.

    Returns:
        Z-score 정규화된 DataFrame. ((x - mean) / std)

    Note:
        Look-ahead bias 방지는 trading_env.py의 Walk-Forward 분리에서 처리한다.
        collector 단계에서는 전체 기간 기준 정규화가 합의된 방식이다. (이문정 합의사항)
    """
    return (df - df.mean()) / df.std()


def build_features(
    prices: pd.DataFrame,
    returns_raw: pd.DataFrame,
) -> pd.DataFrame:
    """정규화된 수익률과 기술적 지표를 결합하여 features DataFrame을 만든다.

    처리 순서:
        1. indicators.compute_indicators(prices)로 RSI·MACD 계산
        2. returns_raw 컬럼을 {ticker}_return으로 rename
        3. returns + indicators concat
        4. Z-score 정규화
        5. dropna (MACD 초기 ~33행 NaN 제거)

    Args:
        prices: 원본 가격 DataFrame (지표 계산에 사용).
        returns_raw: raw 로그수익률 DataFrame.

    Returns:
        Z-score 정규화된 features DataFrame.
        columns: {ticker}_return, {ticker}_RSI, {ticker}_MACD, {ticker}_MACD_signal × 10자산
        shape: returns_raw보다 약 33행 적음 (정상).
    """
    indicators = compute_indicators(prices)

    returns_renamed = returns_raw.rename(
        columns={ticker: f"{ticker}_return" for ticker in returns_raw.columns}
    )

    combined = pd.concat([returns_renamed, indicators], axis=1)
    normalized = normalize_zscore(combined).dropna()

    logger.info(
        "features 구성 완료: shape=%s (returns %d행 대비 %d행 적음)",
        normalized.shape,
        len(returns_raw),
        len(returns_raw) - len(normalized),
    )
    return normalized


def save_features(features: pd.DataFrame, path: str | Path) -> None:
    """Z-score 정규화된 features를 parquet으로 저장한다.

    Args:
        features: features DataFrame.
        path: 저장 경로 (디렉토리가 없으면 자동 생성).
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(dest)
    logger.info("features.parquet 저장 완료: %s  shape=%s", dest, features.shape)


def main() -> None:
    """데이터 수집·전처리 파이프라인 전체를 실행한다.

    처리 순서 (CLAUDE.local.md 지정, 변경 금지):
        yfinance 수집 → pykrx 수집 → 병합 → ffill/dropna
        → 로그수익률 → returns.parquet(raw) 저장
        → features 구성(지표 포함) → Z-score 정규화 → features.parquet 저장
    """
    returns_path = Path(DATA_PROCESSED_DIR) / "returns.parquet"
    features_path = Path(DATA_PROCESSED_DIR) / "features.parquet"

    # 1–3. 수집 및 병합
    df_global = collect_global(TICKERS_GLOBAL, START_DATE, END_DATE)
    df_kr = collect_kr(TICKERS_KR, START_DATE, END_DATE)
    prices = merge_prices(df_global, df_kr)

    # 4–5. 로그수익률 계산 및 raw 저장
    returns = compute_log_returns(prices)
    save_raw_returns(returns, returns_path)

    # 6–7. features 구성(지표 포함 + Z-score) 및 저장
    features = build_features(prices, returns)
    save_features(features, features_path)

    logger.info("파이프라인 완료.")
    logger.info("  returns : %s  shape=%s", returns_path, returns.shape)
    logger.info("  features: %s  shape=%s", features_path, features.shape)


if __name__ == "__main__":
    main()
