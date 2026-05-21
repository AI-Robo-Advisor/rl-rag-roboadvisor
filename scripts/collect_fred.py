from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from fredapi import Fred

START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
OUTPUT_PATH = Path("data/raw/fred/fred_events_2018_2025.parquet")


def get_fred_client() -> Fred:
    api_key = os.getenv("FRED_API_KEY")

    if not api_key:
        raise ValueError(
            "FRED_API_KEY가 없습니다. 프로젝트 루트의 .env 파일에 "
            "FRED_API_KEY=발급받은키 를 추가하세요."
        )

    return Fred(api_key=api_key)


def label_fed_rate_change(diff_pct: float) -> float:
    abs_change = abs(diff_pct)

    if abs_change >= 0.50:
        return 1.0
    if abs_change >= 0.25:
        return 0.66
    if abs_change > 0:
        return 0.33
    return 0.0


def label_cpi(yoy_pct: float) -> float:
    if yoy_pct >= 6.0:
        return 1.0
    if yoy_pct >= 4.0:
        return 0.66
    if yoy_pct >= 3.0:
        return 0.33
    return 0.0


def label_unemployment(diff_pct_point: float) -> float:
    abs_change = abs(diff_pct_point)

    if abs_change >= 0.7:
        return 1.0
    if abs_change >= 0.5:
        return 0.66
    if abs_change >= 0.3:
        return 0.33
    return 0.0


def label_treasury_yield(diff_bp: float) -> float:
    abs_change = abs(diff_bp)

    if abs_change >= 25:
        return 1.0
    if abs_change >= 15:
        return 0.66
    if abs_change >= 10:
        return 0.33
    return 0.0


def label_usdkrw(daily_change_pct: float) -> float:
    abs_change = abs(daily_change_pct)

    if abs_change >= 2.0:
        return 1.0
    if abs_change >= 1.5:
        return 0.66
    if abs_change >= 1.0:
        return 0.33
    return 0.0


def label_vix(vix_value: float) -> float:
    if vix_value >= 40:
        return 1.0
    if vix_value >= 30:
        return 0.66
    if vix_value >= 25:
        return 0.33
    return 0.0


def label_oil(daily_change_pct: float) -> float:
    abs_change = abs(daily_change_pct)

    if abs_change >= 8.0:
        return 1.0
    if abs_change >= 5.0:
        return 0.66
    if abs_change >= 3.0:
        return 0.33
    return 0.0


def make_event(
    event_id: str,
    date: pd.Timestamp,
    title: str,
    summary: str,
    url: str,
    raw_data: dict,
    macro_rate_risk: float,
    equity_market_risk: float,
    geopolitical_fx_risk: float,
    primary_tag: str,
    reasoning: str,
) -> dict:
    return {
        "event_id": event_id,
        "date": pd.to_datetime(date).date(),
        "source": "fred",
        "title": title,
        "summary": summary[:300],
        "url": url,
        "language": "en",
        "raw_data": json.dumps(raw_data, ensure_ascii=False),
        "label_method": "rule_based",
        "macro_rate_risk": float(macro_rate_risk),
        "equity_market_risk": float(equity_market_risk),
        "geopolitical_fx_risk": float(geopolitical_fx_risk),
        "primary_tag": primary_tag,
        "reasoning": reasoning,
        "confidence": 1.0,
    }


def collect_fred_events() -> pd.DataFrame:
    fred = get_fred_client()
    events: list[dict] = []

    # 1. Fed Funds Target Rate
    fed_rate = fred.get_series("DFEDTARU", START_DATE, END_DATE).dropna()
    fed_changes = fed_rate.diff().dropna()

    for date, diff in fed_changes[fed_changes != 0].items():
        severity = label_fed_rate_change(float(diff))

        if severity <= 0:
            continue

        events.append(
            make_event(
                event_id=f"fred-{date.strftime('%Y%m%d')}-fedrate",
                date=date,
                title=f"Fed Funds Target Rate {diff:+.2f}%p",
                summary=(
                    f"Fed target rate changed by {diff:+.2f}%p "
                    f"to {fed_rate.loc[date]:.2f}%."
                ),
                url="https://fred.stlouisfed.org/series/DFEDTARU",
                raw_data={
                    "series": "DFEDTARU",
                    "diff_pct_point": float(diff),
                    "new_rate": float(fed_rate.loc[date]),
                },
                macro_rate_risk=severity,
                equity_market_risk=severity * 0.5,
                geopolitical_fx_risk=severity * 0.3,
                primary_tag="macro_rate_risk",
                reasoning=f"Fed rate change {diff:+.2f}%p",
            )
        )

    # 2. CPI YoY
    cpi = fred.get_series("CPIAUCSL", "2017-01-01", END_DATE).dropna()
    cpi_yoy = cpi.pct_change(12) * 100

    for date, yoy in cpi_yoy.dropna().items():
        if date < pd.Timestamp(START_DATE):
            continue

        severity = label_cpi(float(yoy))

        if severity <= 0:
            continue

        # CPIAUCSL 인덱스는 관측월 기준(예: 2022-06-01)이나
        # 실제 BLS 발표일은 ~6주 후. 보수적 lag로 +1개월 적용.
        release_date = date + pd.DateOffset(months=1)
        events.append(
            make_event(
                event_id=f"fred-{date.strftime('%Y%m%d')}-cpi",
                date=release_date,
                title=f"US CPI YoY {yoy:.1f}%",
                summary=f"US Consumer Price Index year-over-year change was {yoy:.2f}%.",
                url="https://fred.stlouisfed.org/series/CPIAUCSL",
                raw_data={"series": "CPIAUCSL", "yoy_pct": float(yoy)},
                macro_rate_risk=severity,
                equity_market_risk=severity * 0.3,
                geopolitical_fx_risk=0.0,
                primary_tag="macro_rate_risk",
                reasoning=f"CPI YoY {yoy:.1f}%",
            )
        )

    # 3. Unemployment Rate
    unrate = fred.get_series("UNRATE", "2017-01-01", END_DATE).dropna()
    unrate_diff = unrate.diff().dropna()

    for date, diff in unrate_diff.items():
        if date < pd.Timestamp(START_DATE):
            continue

        severity = label_unemployment(float(diff))

        if severity <= 0:
            continue

        # UNRATE 인덱스는 관측월 기준. 실제 BLS 발표일은 다음 달 초.
        # 보수적 lag로 +1개월 적용.
        release_date = date + pd.DateOffset(months=1)
        events.append(
            make_event(
                event_id=f"fred-{date.strftime('%Y%m%d')}-unrate",
                date=release_date,
                title=f"US Unemployment Rate {diff:+.1f}%p",
                summary=(
                    f"US unemployment rate changed by {diff:+.1f}%p "
                    f"to {unrate.loc[date]:.1f}%."
                ),
                url="https://fred.stlouisfed.org/series/UNRATE",
                raw_data={
                    "series": "UNRATE",
                    "diff_pct_point": float(diff),
                    "new_value": float(unrate.loc[date]),
                },
                macro_rate_risk=0.0,
                equity_market_risk=severity,
                geopolitical_fx_risk=0.0,
                primary_tag="equity_market_risk",
                reasoning=f"Unemployment rate change {diff:+.1f}%p",
            )
        )

    # 4. 10Y Treasury Yield
    treasury = fred.get_series("DGS10", START_DATE, END_DATE).dropna()
    treasury_diff_bp = treasury.diff().dropna() * 100

    for date, diff_bp in treasury_diff_bp.items():
        severity = label_treasury_yield(float(diff_bp))

        if severity <= 0:
            continue

        events.append(
            make_event(
                event_id=f"fred-{date.strftime('%Y%m%d')}-treasury10y",
                date=date,
                title=f"US 10Y Treasury Yield {diff_bp:+.0f}bp",
                summary=f"US 10Y Treasury yield changed by {diff_bp:+.1f}bp.",
                url="https://fred.stlouisfed.org/series/DGS10",
                raw_data={"series": "DGS10", "diff_bp": float(diff_bp)},
                macro_rate_risk=severity,
                equity_market_risk=severity * 0.4,
                geopolitical_fx_risk=0.0,
                primary_tag="macro_rate_risk",
                reasoning=f"10Y Treasury yield change {diff_bp:+.0f}bp",
            )
        )

    # 5. USD/KRW Exchange Rate
    usdkrw = fred.get_series("DEXKOUS", START_DATE, END_DATE).dropna()
    usdkrw_pct = usdkrw.pct_change().dropna() * 100

    for date, pct in usdkrw_pct.items():
        severity = label_usdkrw(float(pct))

        if severity <= 0:
            continue

        events.append(
            make_event(
                event_id=f"fred-{date.strftime('%Y%m%d')}-usdkrw",
                date=date,
                title=f"USD/KRW {pct:+.2f}%",
                summary=f"USD/KRW exchange rate changed by {pct:+.2f}% in one day.",
                url="https://fred.stlouisfed.org/series/DEXKOUS",
                raw_data={"series": "DEXKOUS", "daily_change_pct": float(pct)},
                macro_rate_risk=0.0,
                equity_market_risk=severity * 0.3,
                geopolitical_fx_risk=severity,
                primary_tag="geopolitical_fx_risk",
                reasoning=f"USD/KRW daily change {pct:+.2f}%",
            )
        )

    # 6. VIX
    vix = fred.get_series("VIXCLS", START_DATE, END_DATE).dropna()

    vix_above_25 = vix >= 25
    vix_entries = vix_above_25 & ~vix_above_25.shift(1, fill_value=False)

    for date in vix[vix_entries].index:
        severity = label_vix(float(vix.loc[date]))

        if severity <= 0:
            continue

        events.append(
            make_event(
                event_id=f"fred-{date.strftime('%Y%m%d')}-vix",
                date=date,
                title=f"VIX spiked to {vix.loc[date]:.1f}",
                summary=f"VIX entered risk zone at {vix.loc[date]:.1f}.",
                url="https://fred.stlouisfed.org/series/VIXCLS",
                raw_data={"series": "VIXCLS", "value": float(vix.loc[date])},
                macro_rate_risk=0.0,
                equity_market_risk=severity,
                geopolitical_fx_risk=0.0,
                primary_tag="equity_market_risk",
                reasoning=f"VIX spike to {vix.loc[date]:.1f}",
            )
        )

    # 7. WTI Oil
    oil = fred.get_series("DCOILWTICO", START_DATE, END_DATE).dropna()
    oil_pct = oil.pct_change().dropna() * 100

    for date, pct in oil_pct.items():
        severity = label_oil(float(pct))

        if severity <= 0:
            continue

        events.append(
            make_event(
                event_id=f"fred-{date.strftime('%Y%m%d')}-oil",
                date=date,
                title=f"WTI Oil {pct:+.1f}%",
                summary=f"WTI crude oil price changed by {pct:+.2f}% in one day.",
                url="https://fred.stlouisfed.org/series/DCOILWTICO",
                raw_data={"series": "DCOILWTICO", "daily_change_pct": float(pct)},
                macro_rate_risk=0.0,
                equity_market_risk=severity * 0.3,
                geopolitical_fx_risk=severity,
                primary_tag="geopolitical_fx_risk",
                reasoning=f"WTI oil daily change {pct:+.1f}%",
            )
        )

    df = pd.DataFrame(events)

    if df.empty:
        raise RuntimeError("수집된 FRED 이벤트가 없습니다. API 키와 시계열 ID를 확인하세요.")

    df["date"] = pd.to_datetime(df["date"])

    columns = [
        "event_id",
        "date",
        "source",
        "title",
        "summary",
        "url",
        "language",
        "raw_data",
        "label_method",
        "macro_rate_risk",
        "equity_market_risk",
        "geopolitical_fx_risk",
        "primary_tag",
        "reasoning",
        "confidence",
    ]

    df = df[columns].sort_values(["date", "event_id"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"FRED 이벤트 수집 완료: {len(df)}건")
    print(f"저장 위치: {OUTPUT_PATH}")
    print()
    print("태그별 이벤트 수")
    print(df["primary_tag"].value_counts())
    print()
    print("리스크 점수 요약")
    print(
        df[
            [
                "macro_rate_risk",
                "equity_market_risk",
                "geopolitical_fx_risk",
            ]
        ].describe()
    )

    return df


def sanity_check(df: pd.DataFrame) -> None:
    print()
    print("Sanity check")

    checks = {
        "2022-06 FOMC/Fed rate": df[
            (df["date"].dt.year == 2022)
            & (df["date"].dt.month == 6)
            & (df["event_id"].str.contains("fedrate"))
        ],
        "2022 high CPI": df[
            (df["date"].dt.year == 2022)
            & (df["event_id"].str.contains("cpi"))
            & (df["macro_rate_risk"] >= 1.0)
        ],
        "VIX panic >= 40": df[
            (df["event_id"].str.contains("vix"))
            & (df["equity_market_risk"] >= 1.0)
        ],
    }

    for name, result in checks.items():
        print(f"\n[{name}] {len(result)}건")
        if not result.empty:
            print(
                result[
                    [
                        "date",
                        "title",
                        "macro_rate_risk",
                        "equity_market_risk",
                        "geopolitical_fx_risk",
                    ]
                ].head(10)
            )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    fred_events = collect_fred_events()
    sanity_check(fred_events)
    