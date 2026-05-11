"""collector 모듈의 외부 API 호출 경계 테스트."""

from __future__ import annotations

import pandas as pd
import yfinance.cache as yf_cache

from src.data import collector

EXPECTED_GLOBAL_TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ"]
EXPECTED_KR_TICKERS = ["069500", "114260"]


class _FakeTicker:
    seen_sessions: list[object] = []

    def __init__(self, ticker: str, session: object | None = None) -> None:
        self.ticker = ticker
        self.seen_sessions.append(session)

    def history(self, start: str, end: str, auto_adjust: bool) -> pd.DataFrame:
        return pd.DataFrame(
            {"Adj Close": [100.0, 101.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )


class _FakeCookieCache:
    def __init__(self, cookie: object) -> None:
        self.cookie = cookie
        self.stored: list[tuple[str, object | None]] = []

    def lookup(self, strategy: str) -> dict[str, object] | None:
        if strategy != "basic":
            return None
        return {"cookie": self.cookie, "age": pd.Timedelta(seconds=1)}

    def store(self, strategy: str, cookie: object | None) -> None:
        self.stored.append((strategy, cookie))


def test_collect_global_uses_yfinance_default_session(monkeypatch) -> None:
    """yfinance에 curl_cffi 세션을 직접 주입하지 않아야 한다."""
    _FakeTicker.seen_sessions = []
    monkeypatch.setattr(collector.yf, "Ticker", _FakeTicker)
    monkeypatch.setattr(collector.time, "sleep", lambda _: None)

    result = collector.collect_global(["SPY"], "2024-01-01", "2024-01-03")

    assert _FakeTicker.seen_sessions == [None]
    assert result["SPY"].tolist() == [100.0, 101.0]


def test_collect_global_drops_invalid_yfinance_basic_cookie(monkeypatch) -> None:
    """문자열로 저장된 yfinance basic 쿠키 캐시는 재사용 전에 삭제한다."""
    cache = _FakeCookieCache(cookie="A3")
    monkeypatch.setattr(yf_cache, "get_cookie_cache", lambda: cache)
    monkeypatch.setattr(collector.yf, "Ticker", _FakeTicker)
    monkeypatch.setattr(collector.time, "sleep", lambda _: None)

    collector.collect_global(["SPY"], "2024-01-01", "2024-01-03")

    assert cache.stored == [("basic", None)]


def test_collector_uses_final_selected_dataset_config() -> None:
    """collector 기본 설정은 최종 데이터 선정 문서의 자산과 기간을 따라야 한다."""
    assert collector.TICKERS_GLOBAL == EXPECTED_GLOBAL_TICKERS
    assert collector.TICKERS_KR == EXPECTED_KR_TICKERS
    assert collector.START_DATE == "2018-01-01"
    assert collector.END_DATE == "2025-12-31"
