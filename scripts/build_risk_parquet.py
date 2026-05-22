"""
리스크 벡터 일별 Parquet 생성 스크립트 (강유영 담당).

팀원별 수집 parquet을 통합하고 Exponential Decay를 적용해
RL 학습·백테스트용 일별 risk 벡터 파일을 생성합니다.

입력 (없는 파일은 자동 스킵):
  data/raw/gdelt/gdelt_events_2018_2025_filtered.parquet    (담당: 박지민, PR #53 노이즈 필터 결과물)
  data/raw/fred/fred_events_2018_2025.parquet      (담당: 이문정, 2018~ 수집)
  data/raw/ecos/ecos_events_2018_2025.parquet      (담당: 강유영)
  data/raw/manual_seed/manual_seed_events.parquet  (담당: 강유영 D)

출력:
  data/processed/risk_vectors_daily.parquet
  컬럼: date | risk_macro | risk_equity | risk_geo

Decay 기간:
  macro_rate_risk:      30일 (금리·매크로 이벤트)
  equity_market_risk:   10일 (시장 변동성)
  geopolitical_fx_risk: 60일 (전쟁·제재)

실행:
  python scripts/build_risk_parquet.py
  python scripts/build_risk_parquet.py --sanity   # sanity check만
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────

DATE_START = "2018-01-01"
DATE_END   = "2025-12-31"

DECAY_PERIODS: Dict[str, int] = {
    "macro_rate_risk":      30,
    "equity_market_risk":   10,
    "geopolitical_fx_risk": 60,
}

# 입력 컬럼명 (팀원 parquet의 라벨 컬럼)
SEVERITY_COLS: Dict[str, str] = {
    "macro_rate_risk":      "macro_rate_risk",
    "equity_market_risk":   "equity_market_risk",
    "geopolitical_fx_risk": "geopolitical_fx_risk",
}

RAW_PATHS: List[Path] = [
    Path("data/raw/gdelt/gdelt_events_2018_2025_filtered.parquet"),
    Path("data/raw/fred/fred_events_2018_2025.parquet"),
    Path("data/raw/ecos/ecos_events_2018_2025.parquet"),
    Path("data/raw/manual_seed/manual_seed_events.parquet"),
]

OUTPUT_PATH = Path("data/processed/risk_vectors_daily.parquet")

# sanity check: 알려진 고위험 날짜
SANITY_DATES: Dict[str, str] = {
    "2022-06-15": "macro_rate_risk 높아야 함 (FOMC 0.75%p 자이언트 스텝)",
    "2020-03-16": "equity_market_risk 높아야 함 (코로나 패닉)",
    "2022-02-24": "geopolitical_fx_risk 높아야 함 (러-우 전쟁 개전)",
    "2022-10-07": "geopolitical_fx_risk 높아야 함 (미국 반도체 수출 규제)",
}

# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────

UNIFIED_PATH = Path("data/processed/unified_events.parquet")


def load_events() -> pd.DataFrame:
    """이벤트를 로드합니다.

    unified_events.parquet이 있으면 우선 사용 (LLM 라벨 포함).
    없으면 raw parquet 폴백 (LLM 라벨 미반영).
    """
    if UNIFIED_PATH.exists():
        df = pd.read_parquet(UNIFIED_PATH)
        df["date"] = pd.to_datetime(df["date"])
        for col in SEVERITY_COLS.values():
            if col not in df.columns:
                df[col] = 0.0
        df = df[df["date"].notna()]
        df = df[(df["date"] >= DATE_START) & (df["date"] <= DATE_END)]
        logger.info("unified 로드: %s (%d건)", UNIFIED_PATH.name, len(df))
        return df

    logger.info("unified parquet 없음 — raw parquet 폴백")
    frames: List[pd.DataFrame] = []
    for path in RAW_PATHS:
        if not path.exists():
            logger.info("스킵 (파일 없음): %s", path)
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        for col in SEVERITY_COLS.values():
            if col not in df.columns:
                df[col] = 0.0
        frames.append(df)
        logger.info("로드: %s (%d건)", path.name, len(df))

    if not frames:
        logger.warning("로드된 파일 없음 — 전 기간 0으로 채운 parquet 생성")
        return pd.DataFrame(columns=["date"] + list(SEVERITY_COLS.values()))

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["date"].notna()]
    combined = combined[
        (combined["date"] >= DATE_START) & (combined["date"] <= DATE_END)
    ]
    logger.info("통합 이벤트: %d건", len(combined))
    return combined

# ─────────────────────────────────────────────
# Exponential Decay 계산
# ─────────────────────────────────────────────

def compute_daily_decay(events: pd.DataFrame) -> pd.DataFrame:
    """이벤트 parquet에 decay를 적용해 일별 risk 스코어를 반환합니다.

    Args:
        events: date, macro_rate_risk, equity_market_risk,
                geopolitical_fx_risk 컬럼을 가진 이벤트 DataFrame.

    Returns:
        date | risk_macro | risk_equity | risk_geo 컬럼의 일별 DataFrame.
    """
    dates = pd.date_range(DATE_START, DATE_END, freq="D")
    n = len(dates)
    date_to_idx: Dict[pd.Timestamp, int] = {d: i for i, d in enumerate(dates)}

    risk_macro  = np.zeros(n, dtype=np.float64)
    risk_equity = np.zeros(n, dtype=np.float64)
    risk_geo    = np.zeros(n, dtype=np.float64)

    arrays = {
        "macro_rate_risk":      risk_macro,
        "equity_market_risk":   risk_equity,
        "geopolitical_fx_risk": risk_geo,
    }

    for _, event in events.iterrows():
        event_date = pd.Timestamp(event["date"]).normalize()
        if event_date not in date_to_idx:
            continue
        start_idx = date_to_idx[event_date]

        for tag, arr in arrays.items():
            severity = float(event.get(SEVERITY_COLS[tag], 0.0) or 0.0)
            if severity <= 0:
                continue

            period = DECAY_PERIODS[tag]
            # decay < 0.01이 되는 최대 일수까지만 계산 (성능 최적화)
            max_days = int(-period * math.log(max(0.01 / severity, 1e-9))) + 1
            end_idx = min(start_idx + max_days, n)

            days = np.arange(end_idx - start_idx, dtype=np.float64)
            # max pooling: 여러 이벤트 합산 대신 최댓값 유지 (포화 방지)
            arr[start_idx:end_idx] = np.maximum(
                arr[start_idx:end_idx],
                severity * np.exp(-days / period),
            )

    daily = pd.DataFrame({
        "date":        dates,
        "risk_macro":  np.minimum(risk_macro,  1.0).round(4),
        "risk_equity": np.minimum(risk_equity, 1.0).round(4),
        "risk_geo":    np.minimum(risk_geo,    1.0).round(4),
    })

    return daily

# ─────────────────────────────────────────────
# Sanity Check
# ─────────────────────────────────────────────

def run_sanity_check(daily: pd.DataFrame) -> None:
    """알려진 고위험 날짜에서 기대 방향 스코어를 출력합니다."""
    print("\n" + "=" * 60)
    print("SANITY CHECK")
    print("=" * 60)

    n = len(daily)
    print(f"\n=== 전체 분포 ===")
    for col in ["risk_macro", "risk_equity", "risk_geo"]:
        at_max  = (daily[col] >= 0.99).sum()
        above_half = (daily[col] > 0.5).sum()
        print(f"  {col}: ≥0.99={at_max}일({at_max/n:.1%}), >0.5={above_half}일({above_half/n:.1%})")

    daily_indexed = daily.set_index("date")

    for date_str, desc in SANITY_DATES.items():
        date = pd.Timestamp(date_str)
        print(f"\n{date_str} — {desc}")

        if date not in daily_indexed.index:
            print("  [해당 날짜 없음]")
            continue

        row = daily_indexed.loc[date]
        vals = {"macro": row["risk_macro"], "equity": row["risk_equity"], "geo": row["risk_geo"]}
        dominant = max(vals, key=vals.get)
        print(f"  risk_macro  = {row['risk_macro']:.4f}")
        print(f"  risk_equity = {row['risk_equity']:.4f}")
        print(f"  risk_geo    = {row['risk_geo']:.4f}")
        print(f"  → 최고 축: {dominant}")

    print()

# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main(sanity_only: bool = False) -> None:
    if sanity_only:
        if not OUTPUT_PATH.exists():
            logger.error("parquet 없음. 먼저 build를 실행하세요.")
            return
        daily = pd.read_parquet(OUTPUT_PATH)
        daily["date"] = pd.to_datetime(daily["date"])
        run_sanity_check(daily)
        return

    events = load_events()
    daily  = compute_daily_decay(events)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUTPUT_PATH, index=False)
    logger.info("저장 완료: %s (%d일)", OUTPUT_PATH, len(daily))

    # 기본 통계
    nonzero = (daily[["risk_macro", "risk_equity", "risk_geo"]] > 0).sum()
    print("\n=== 기본 통계 ===")
    print(f"전체 날짜: {len(daily)}일")
    print(f"risk_macro  > 0: {nonzero['risk_macro']}일")
    print(f"risk_equity > 0: {nonzero['risk_equity']}일")
    print(f"risk_geo    > 0: {nonzero['risk_geo']}일")
    print(f"\n최댓값:")
    print(daily[["risk_macro", "risk_equity", "risk_geo"]].max())

    run_sanity_check(daily)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sanity", action="store_true",
        help="기존 parquet으로 sanity check만 실행"
    )
    args = parser.parse_args()
    main(sanity_only=args.sanity)
