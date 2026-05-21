"""Market close 기준 날짜 보정 유틸리티.

뉴스·이벤트의 발생 시각이 해당 시장의 마감 시각 이후라면,
다음 거래일을 라벨링 기준 날짜로 반환합니다. (Look-ahead bias 방지)

미국 장 마감: 16:00 ET (NYSE 기준)
한국 장 마감: 15:30 KST (KRX 기준)
"""
from __future__ import annotations

from datetime import date
from typing import Literal

import pandas as pd
import pandas_market_calendars as mcal

_CAL_CACHE: dict[str, mcal.MarketCalendar] = {}

_REGION_CONFIG = {
    "US": {
        "calendar": "NYSE",
        "timezone": "America/New_York",
        "close_hour": 16,
        "close_minute": 0,
    },
    "KR": {
        "calendar": "XKRX",
        "timezone": "Asia/Seoul",
        "close_hour": 15,
        "close_minute": 30,
    },
}


def _get_calendar(name: str) -> mcal.MarketCalendar:
    if name not in _CAL_CACHE:
        _CAL_CACHE[name] = mcal.get_calendar(name)
    return _CAL_CACHE[name]


def adjust_date_for_market_close(
    dt: pd.Timestamp,
    region: Literal["US", "KR"],
) -> date:
    """뉴스 발생 시각이 시장 마감 후라면 다음 거래일로 이동합니다.

    Args:
        dt: 이벤트 발생 시각. timezone-aware 또는 naive 모두 허용.
            naive이면 region 기준 timezone으로 가정.
        region: "US" (NYSE) 또는 "KR" (KRX).

    Returns:
        라벨링 기준 날짜 (datetime.date).
    """
    cfg = _REGION_CONFIG[region]
    tz = cfg["timezone"]

    if dt.tzinfo is None:
        dt = dt.tz_localize(tz)
    else:
        dt = dt.tz_convert(tz)

    local_dt = dt
    market_close = local_dt.replace(
        hour=cfg["close_hour"],
        minute=cfg["close_minute"],
        second=0,
        microsecond=0,
    )

    if local_dt <= market_close:
        return local_dt.date()

    # 마감 이후 → 다음 거래일 탐색 (최대 10일 앞)
    cal = _get_calendar(cfg["calendar"])
    next_start = pd.Timestamp(local_dt.date()) + pd.Timedelta(days=1)
    next_end = next_start + pd.Timedelta(days=10)
    schedule = cal.schedule(
        start_date=str(next_start.date()),
        end_date=str(next_end.date()),
    )
    if len(schedule) > 0:
        return schedule.index[0].date()

    # fallback: 달력에서 못 찾으면 원래 날짜 반환
    return local_dt.date()
