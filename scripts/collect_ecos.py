"""
담당자 C (강유영): 한국은행 ECOS API 수집 스크립트.

한국 정량 매크로 이벤트를 수집하고 룰 기반 라벨링까지 한 번에 처리합니다.
출력: data/raw/ecos/ecos_events_2018_2025.parquet

수집 대상:
  - 722Y001 / 0101000: 한국은행 기준금리 (월별)
  - 036Y001 / 0000003: 원/달러 환율 (일별)
  - 021Y126 / AA: 소비자물가지수 YoY (월별)
  - 731Y001 / 0000003: 국고채 3년 수익률 (일별) — 114260 직접 영향
  - 731Y001 / 0000005: 국고채 10년 수익률 (일별)
  - 802Y001 / 0000001: KOSPI (일별)

  ⚠️ 통계코드는 ECOS 사이트에서 직접 확인 권장 (가끔 변경됨).
  https://ecos.bok.or.kr/api/#/StatisticsSearch 에서 코드 검색 가능.
  데이터가 비면 "데이터 없음" 경고 후 해당 시리즈만 스킵.

실행:
  python scripts/collect_ecos.py

사전 준비:
  .env에 ECOS_API_KEY=<한국은행 ECOS API 키> 설정
  API 키 발급: https://ecos.bok.or.kr/api/#/DevGuide/APIKey
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# scripts/ 아래에서 직접 실행할 때 src 패키지를 찾을 수 있도록 루트를 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_close import adjust_date_for_market_close

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────

ECOS_BASE = "https://ecos.bok.or.kr/api/StatisticSearch"
START_DATE = "20180101"
END_DATE   = "20251231"
OUTPUT_PATH = Path("data/raw/ecos/ecos_events_2018_2025.parquet")

# 수집 대상 시계열
SERIES_CONFIG = [
    {
        "stat_code":     "722Y001",
        "item_code":     "0101000",
        "cycle":         "M",
        "name":          "한국은행 기준금리",
        "tag":           "macro_rate",
        "processor_key": "722Y001",
    },
    {
        "stat_code":     "731Y001",
        "item_code":     "0000001",
        "cycle":         "D",
        "name":          "원/달러 환율",
        "tag":           "geopolitical_fx",
        "processor_key": "036Y001",   # processor 재사용
    },
    {
        # CPI YoY 계산을 위해 START_DATE - 1년치 더 필요 → fetch 시 별도 처리
        "stat_code":     "901Y009",
        "item_code":     "0",
        "cycle":         "M",
        "name":          "소비자물가지수 (CPI)",
        "tag":           "macro_rate",
        "processor_key": "021Y126",   # processor 재사용
    },
    {
        # 114260(한국 채권 ETF) 직접 영향
        "stat_code":     "817Y002",
        "item_code":     "010200000",
        "cycle":         "D",
        "name":          "국고채 3년 수익률",
        "tag":           "macro_rate",
        "processor_key": "731Y001_bond3y",
    },
    {
        "stat_code":     "817Y002",
        "item_code":     "010210000",
        "cycle":         "D",
        "name":          "국고채 10년 수익률",
        "tag":           "macro_rate",
        "processor_key": "731Y001_bond10y",
    },
    {
        "stat_code":     "802Y001",
        "item_code":     "0001000",
        "cycle":         "D",
        "name":          "KOSPI",
        "tag":           "equity_market",
        "processor_key": "802Y001",
    },
]

# ─────────────────────────────────────────────
# 룰 기반 라벨링 함수
# ─────────────────────────────────────────────

def label_base_rate_change(diff: float) -> float:
    """한국은행 기준금리 변경폭 → macro_rate_risk 강도."""
    abs_diff = abs(diff)
    if abs_diff >= 0.50:
        return 1.0
    if abs_diff >= 0.25:
        return 0.66
    if abs_diff > 0.0:
        return 0.33
    return 0.0


def label_usdkrw_change(daily_change_pct: float) -> float:
    """원/달러 일변동률 → geopolitical_fx_risk 강도."""
    abs_change = abs(daily_change_pct)
    if abs_change >= 2.0:
        return 1.0
    if abs_change >= 1.5:
        return 0.66
    if abs_change >= 1.0:
        return 0.33
    return 0.0


def label_cpi_yoy(yoy_pct: float) -> float:
    """CPI YoY → macro_rate_risk 강도."""
    if yoy_pct >= 5.0:
        return 1.0
    if yoy_pct >= 4.0:
        return 0.66
    if yoy_pct >= 3.0:
        return 0.33
    return 0.0


def label_bond_yield_change(diff_bp: float, maturity: str = "3y") -> float:
    """국고채 수익률 일변동 → macro_rate_risk 강도.

    Args:
        diff_bp: 일변동 (basis points, 소수점 백분율 × 100).
        maturity: '3y' 또는 '10y'.
    """
    # 3년: 기준 8bp, 10년: 기준 12bp
    threshold = 8.0 if maturity == "3y" else 12.0
    abs_change = abs(diff_bp)
    if abs_change >= threshold * 2.0:
        return 1.0
    if abs_change >= threshold * 1.5:
        return 0.66
    if abs_change >= threshold:
        return 0.33
    return 0.0


def label_kospi_drop(daily_change_pct: float) -> float:
    """KOSPI 일간 하락률 → equity_market_risk 강도."""
    if daily_change_pct <= -3.0:
        return 1.0
    if daily_change_pct <= -2.0:
        return 0.66
    if daily_change_pct <= -1.5:
        return 0.33
    return 0.0


# ─────────────────────────────────────────────
# ECOS API 호출
# ─────────────────────────────────────────────

def fetch_ecos_series(
    api_key: str,
    stat_code: str,
    item_code: str,
    cycle: str,
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    """ECOS StatisticSearch API를 호출해 시계열 데이터를 반환합니다."""
    # 월별 시리즈는 날짜를 YYYYMM 포맷으로 변환
    if cycle == "M":
        fmt_start = start_date[:6]   # "20200101" → "202001"
        fmt_end   = end_date[:6]     # "20251231" → "202512"
    else:
        fmt_start = start_date
        fmt_end   = end_date

    url = (
        f"{ECOS_BASE}/{api_key}/json/kr/1/10000"
        f"/{stat_code}/{cycle}/{fmt_start}/{fmt_end}/{item_code}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as e:
        logger.error("ECOS 요청 실패 [%s/%s]: %s", stat_code, item_code, e)
        return []

    if "RESULT" in payload:
        code = payload["RESULT"].get("CODE", "")
        msg  = payload["RESULT"].get("MESSAGE", "")
        logger.error("ECOS API 오류 [%s]: %s — %s", stat_code, code, msg)
        return []

    search = payload.get("StatisticSearch", {})
    rows = search.get("row", [])
    if isinstance(rows, dict):
        rows = [rows]
    return rows if isinstance(rows, list) else []


# ─────────────────────────────────────────────
# 시리즈별 이벤트 변환
# ─────────────────────────────────────────────

def process_base_rate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """기준금리 시계열 → 변경일 이벤트 리스트."""
    records = []
    prev_rate: Optional[float] = None

    for row in sorted(rows, key=lambda r: r.get("TIME", "")):
        try:
            rate = float(row["DATA_VALUE"])
            date_str = row["TIME"]  # YYYYMM → YYYY-MM-01
            date = pd.Timestamp(date_str + "01")
        except (KeyError, ValueError):
            continue

        if prev_rate is not None:
            diff = rate - prev_rate
            severity = label_base_rate_change(diff)
            if severity > 0:
                direction = "인상" if diff > 0 else "인하"
                # 월말로 이동: 금통위 결정일(월중)보다 앞선 날짜에 이벤트가 노출되는
                # look-ahead bias를 방지. 실제 결정일을 모르므로 해당 월 내 최대값인
                # 월말을 사용해 보수적으로 처리.
                event_date = date + pd.offsets.MonthEnd(0)
                records.append({
                    "event_id":            f"ecos-{event_date.strftime('%Y%m%d')}-baserate",
                    "date":                event_date.date(),
                    "source":              "ecos",
                    "title":               f"한국은행 기준금리 {direction} {abs(diff):.2f}%p",
                    "summary":             (
                        f"한국은행 기준금리 {diff:+.2f}%p 변경. "
                        f"기존 {prev_rate:.2f}% → 신규 {rate:.2f}%"
                    ),
                    "url":                 "https://ecos.bok.or.kr/#/SearchStat/722Y001",
                    "language":            "ko",
                    "raw_data":            json.dumps({"diff": diff, "new_rate": rate}),
                    "label_method":        "rule_based",
                    "macro_rate_risk":     severity,
                    "equity_market_risk":  round(severity * 0.5, 2),
                    "geopolitical_fx_risk": round(severity * 0.2, 2),
                    "primary_tag":         "macro_rate",
                    "reasoning":           f"기준금리 {diff:+.2f}%p 변경",
                    "confidence":          1.0,
                })

        prev_rate = rate

    return records


def process_usdkrw(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """원/달러 환율 시계열 → 급변 이벤트 리스트."""
    records = []
    series: List[tuple[pd.Timestamp, float]] = []

    for row in rows:
        try:
            rate = float(row["DATA_VALUE"])
            date = pd.Timestamp(row["TIME"])  # YYYYMMDD
            series.append((date, rate))
        except (KeyError, ValueError):
            continue

    series.sort(key=lambda x: x[0])

    for i in range(1, len(series)):
        date, rate = series[i]
        _, prev_rate = series[i - 1]
        change_pct = (rate - prev_rate) / prev_rate * 100
        severity = label_usdkrw_change(change_pct)
        if severity > 0:
            direction = "급등" if change_pct > 0 else "급락"
            records.append({
                "event_id":            f"ecos-{date.strftime('%Y%m%d')}-usdkrw",
                "date":                date.date(),
                "source":              "ecos",
                "title":               f"원/달러 환율 {direction} ({change_pct:+.2f}%)",
                "summary":             (
                    f"원/달러 환율 {prev_rate:.1f}원 → {rate:.1f}원 "
                    f"({change_pct:+.2f}%)"
                ),
                "url":                 "https://ecos.bok.or.kr/#/SearchStat/036Y001",
                "language":            "ko",
                "raw_data":            json.dumps({"rate": rate, "change_pct": change_pct}),
                "label_method":        "rule_based",
                "macro_rate_risk":     round(severity * 0.3, 2),
                "equity_market_risk":  round(severity * 0.3, 2),
                "geopolitical_fx_risk": severity,
                "primary_tag":         "geopolitical_fx",
                "reasoning":           f"원/달러 {change_pct:+.2f}% 급변",
                "confidence":          1.0,
            })

    return records


def process_cpi(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """CPI 시계열 → 고인플레 이벤트 리스트."""
    records = []
    series: List[tuple[pd.Timestamp, float]] = []

    for row in rows:
        try:
            val = float(row["DATA_VALUE"])
            date = pd.Timestamp(row["TIME"] + "01")
            series.append((date, val))
        except (KeyError, ValueError):
            continue

    series.sort(key=lambda x: x[0])

    # YoY 계산 (12개월 전 대비)
    for i in range(12, len(series)):
        date, curr = series[i]
        _, prev = series[i - 12]
        yoy = (curr - prev) / prev * 100
        severity = label_cpi_yoy(yoy)
        if severity > 0:
            # 익월 1일로 이동: 한국 CPI는 해당 월 데이터를 익월 초(~4일)에 발표하므로
            # 당월 1일 기준은 약 33일 이른 look-ahead bias가 발생함.
            event_date = date + pd.DateOffset(months=1)
            records.append({
                "event_id":            f"ecos-{event_date.strftime('%Y%m%d')}-cpi",
                "date":                event_date.date(),
                "source":              "ecos",
                "title":               f"한국 CPI YoY {yoy:.1f}% (고인플레)",
                "summary":             f"소비자물가 전년 동월 대비 {yoy:.1f}% 상승",
                "url":                 "https://ecos.bok.or.kr/#/SearchStat/021Y126",
                "language":            "ko",
                "raw_data":            json.dumps({"yoy": yoy, "index": curr}),
                "label_method":        "rule_based",
                "macro_rate_risk":     severity,
                "equity_market_risk":  round(severity * 0.4, 2),
                "geopolitical_fx_risk": round(severity * 0.2, 2),
                "primary_tag":         "macro_rate",
                "reasoning":           f"CPI YoY {yoy:.1f}%",
                "confidence":          1.0,
            })

    return records


def process_bond_3y(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """국고채 3년 수익률 시계열 → 급변 이벤트 리스트 (114260 직접 영향)."""
    records = []
    series: List[tuple[pd.Timestamp, float]] = []

    for row in rows:
        try:
            rate = float(row["DATA_VALUE"])
            date = pd.Timestamp(row["TIME"])
            series.append((date, rate))
        except (KeyError, ValueError):
            continue

    series.sort(key=lambda x: x[0])

    for i in range(1, len(series)):
        date, rate = series[i]
        _, prev_rate = series[i - 1]
        diff_bp = (rate - prev_rate) * 100  # %p → bp
        severity = label_bond_yield_change(diff_bp, "3y")
        if severity > 0:
            direction = "상승" if diff_bp > 0 else "하락"
            records.append({
                "event_id":             f"ecos-{date.strftime('%Y%m%d')}-bond3y",
                "date":                 date.date(),
                "source":               "ecos",
                "title":                f"국고채 3년 {direction} {abs(diff_bp):.1f}bp",
                "summary":              (
                    f"국고채 3년물 수익률 {prev_rate:.2f}% → {rate:.2f}% "
                    f"({diff_bp:+.1f}bp). 114260 직접 영향."
                ),
                "url":                  "https://ecos.bok.or.kr/#/SearchStat/731Y001",
                "language":             "ko",
                "raw_data":             json.dumps({"rate": rate, "diff_bp": diff_bp}),
                "label_method":         "rule_based",
                "macro_rate_risk":      severity,
                "equity_market_risk":   0.0,
                "geopolitical_fx_risk": 0.0,
                "primary_tag":          "macro_rate",
                "reasoning":            f"국고채 3년 {diff_bp:+.1f}bp",
                "confidence":           1.0,
            })

    return records


def process_bond_10y(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """국고채 10년 수익률 시계열 → 급변 이벤트 리스트."""
    records = []
    series: List[tuple[pd.Timestamp, float]] = []

    for row in rows:
        try:
            rate = float(row["DATA_VALUE"])
            date = pd.Timestamp(row["TIME"])
            series.append((date, rate))
        except (KeyError, ValueError):
            continue

    series.sort(key=lambda x: x[0])

    for i in range(1, len(series)):
        date, rate = series[i]
        _, prev_rate = series[i - 1]
        diff_bp = (rate - prev_rate) * 100
        severity = label_bond_yield_change(diff_bp, "10y")
        if severity > 0:
            direction = "상승" if diff_bp > 0 else "하락"
            records.append({
                "event_id":             f"ecos-{date.strftime('%Y%m%d')}-bond10y",
                "date":                 date.date(),
                "source":               "ecos",
                "title":                f"국고채 10년 {direction} {abs(diff_bp):.1f}bp",
                "summary":              (
                    f"국고채 10년물 수익률 {prev_rate:.2f}% → {rate:.2f}% "
                    f"({diff_bp:+.1f}bp)."
                ),
                "url":                  "https://ecos.bok.or.kr/#/SearchStat/731Y001",
                "language":             "ko",
                "raw_data":             json.dumps({"rate": rate, "diff_bp": diff_bp}),
                "label_method":         "rule_based",
                "macro_rate_risk":      severity,
                "equity_market_risk":   round(severity * 0.3, 2),
                "geopolitical_fx_risk": 0.0,
                "primary_tag":          "macro_rate",
                "reasoning":            f"국고채 10년 {diff_bp:+.1f}bp",
                "confidence":           1.0,
            })

    return records


def process_kospi(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """KOSPI 시계열 → 급락 이벤트 리스트."""
    records = []
    series: List[tuple[pd.Timestamp, float]] = []

    for row in rows:
        try:
            val = float(row["DATA_VALUE"])
            date = pd.Timestamp(row["TIME"])
            series.append((date, val))
        except (KeyError, ValueError):
            continue

    series.sort(key=lambda x: x[0])

    for i in range(1, len(series)):
        date, val = series[i]
        _, prev_val = series[i - 1]
        change_pct = (val - prev_val) / prev_val * 100
        severity = label_kospi_drop(change_pct)
        if severity > 0:
            records.append({
                "event_id":             f"ecos-{date.strftime('%Y%m%d')}-kospi",
                "date":                 date.date(),
                "source":               "ecos",
                "title":                f"KOSPI 급락 {change_pct:.2f}%",
                "summary":              (
                    f"KOSPI {prev_val:.1f} → {val:.1f} ({change_pct:+.2f}%)"
                ),
                "url":                  "https://ecos.bok.or.kr/#/SearchStat/802Y001",
                "language":             "ko",
                "raw_data":             json.dumps({"kospi": val, "change_pct": change_pct}),
                "label_method":         "rule_based",
                "macro_rate_risk":      0.0,
                "equity_market_risk":   severity,
                "geopolitical_fx_risk": round(severity * 0.3, 2),
                "primary_tag":          "equity_market",
                "reasoning":            f"KOSPI {change_pct:+.2f}%",
                "confidence":           1.0,
            })

    return records


# 731Y001은 item_code로 구분 (같은 stat_code 두 번 못 들어가므로 key에 maturity 포함)
PROCESSORS = {
    "722Y001":        process_base_rate,
    "036Y001":        process_usdkrw,
    "021Y126":        process_cpi,
    "731Y001_bond3y": process_bond_3y,
    "731Y001_bond10y": process_bond_10y,
    "802Y001":        process_kospi,
}

# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("ECOS_API_KEY") or os.getenv("BOK_API_KEY", "")
    if not api_key:
        raise ValueError(
            "ECOS_API_KEY(또는 BOK_API_KEY)가 없습니다. .env에 설정하세요.\n"
            "발급: https://ecos.bok.or.kr/api/#/DevGuide/APIKey"
        )

    all_events: List[Dict[str, Any]] = []

    for cfg in SERIES_CONFIG:
        logger.info("수집 중: %s (%s/%s)", cfg["name"], cfg["stat_code"], cfg["item_code"])
        # CPI YoY 계산: 12개월 전 데이터가 필요하므로 1년치 더 수집
        fetch_start = "20170101" if cfg["processor_key"] == "021Y126" else START_DATE
        rows = fetch_ecos_series(
            api_key,
            cfg["stat_code"],
            cfg["item_code"],
            cfg["cycle"],
            fetch_start,
            END_DATE,
        )
        if not rows:
            logger.warning("데이터 없음: %s — series ID 확인 필요", cfg["stat_code"])
            continue

        processor = PROCESSORS.get(cfg["processor_key"])
        if processor is None:
            logger.warning("처리 함수 없음: %s", cfg["processor_key"])
            continue

        events = processor(rows)
        logger.info("  → 이벤트 %d건 생성", len(events))
        all_events.extend(events)
        time.sleep(0.5)

    if not all_events:
        logger.error("수집된 이벤트 없음. API 키와 series ID를 확인하세요.")
        return

    df = pd.DataFrame(all_events)
    df["date"] = pd.to_datetime(df["date"])
    # ECOS 데이터는 이미 거래일 기준이지만, 마감 기준 보정 인프라를 통해 정규화
    def _normalize_kr(d: pd.Timestamp) -> pd.Timestamp:
        dt_kst = pd.Timestamp(d.date()).tz_localize("Asia/Seoul").replace(hour=9)
        return pd.Timestamp(adjust_date_for_market_close(dt_kst, "KR"))
    df["date"] = df["date"].apply(_normalize_kr)
    df = df.sort_values("date").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info("저장 완료: %s (%d건)", OUTPUT_PATH, len(df))

    # 샘플 출력
    print("\n=== 샘플 (최근 5건) ===")
    print(df[["date", "title", "macro_rate_risk", "equity_market_risk",
              "geopolitical_fx_risk"]].tail())


if __name__ == "__main__":
    main()
