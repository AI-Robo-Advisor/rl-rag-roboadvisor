"""
담당자 D: 수동 시드 CSV → Parquet 변환 스크립트.

입력:  data/raw/manual_seed/manual_seed_events.csv
출력:  data/raw/manual_seed/manual_seed_events.parquet

실행:
  python scripts/collect_manual_seed.py
  python scripts/collect_manual_seed.py --validate   # 검증만
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# scripts/ 아래에서 직접 실행할 때 src 패키지를 찾을 수 있도록 루트를 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_close import adjust_date_for_market_close

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CSV_PATH     = Path("data/raw/manual_seed/manual_seed_events.csv")
OUTPUT_PATH  = Path("data/raw/manual_seed/manual_seed_events.parquet")

REQUIRED_COLS = [
    "event_id", "date", "source", "title", "summary", "url", "language",
    "macro_rate_risk", "equity_market_risk", "geopolitical_fx_risk",
    "primary_tag", "reasoning", "confidence",
]
VALID_RISK_VALUES = {0.0, 0.33, 0.66, 1.0}
VALID_TAGS        = {"macro_rate_risk", "equity_market_risk", "geopolitical_fx_risk", "none"}

# sanity check — 알려진 빅 이벤트
SANITY_CHECKS = {
    "2022-06-15": ("macro_rate_risk",     1.0, "FOMC 자이언트 스텝"),
    "2022-02-24": ("geopolitical_fx_risk", 1.0, "러-우 침공"),
    "2020-03-16": ("equity_market_risk",  1.0, "코로나 패닉"),
    "2022-10-07": ("geopolitical_fx_risk", 1.0, "반도체 수출 규제"),
    "2025-04-02": ("geopolitical_fx_risk", 1.0, "Liberation Day 관세"),
}


def validate(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    # 필수 컬럼
    for col in REQUIRED_COLS:
        if col not in df.columns:
            issues.append(f"필수 컬럼 없음: {col}")

    # 리스크 값 범위
    for col in ["macro_rate_risk", "equity_market_risk", "geopolitical_fx_risk"]:
        bad = df[~df[col].isin(VALID_RISK_VALUES)]
        if len(bad):
            issues.append(f"{col}에 유효하지 않은 값: {bad[col].unique().tolist()}")

    # primary_tag
    bad_tags = df[~df["primary_tag"].isin(VALID_TAGS)]
    if len(bad_tags):
        issues.append(f"primary_tag 오류: {bad_tags['primary_tag'].unique().tolist()}")

    # 중복 event_id
    dups = df[df["event_id"].duplicated(keep=False)]
    if len(dups):
        issues.append(f"중복 event_id: {dups['event_id'].tolist()}")

    # sanity check
    df_indexed = df.set_index("date")
    for date_str, (col, expected_min, desc) in SANITY_CHECKS.items():
        date = pd.Timestamp(date_str)
        if date in df_indexed.index:
            val = df_indexed.loc[date, col]
            if isinstance(val, pd.Series):
                val = val.max()
            if val < expected_min:
                issues.append(f"SANITY MISS [{date_str}] {desc}: {col}={val} (기대 >={expected_min})")
        else:
            issues.append(f"SANITY 날짜 없음: {date_str} ({desc})")

    return issues


def main(validate_only: bool = False) -> None:
    if not CSV_PATH.exists():
        logger.error("CSV 없음: %s", CSV_PATH)
        return

    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    # 이벤트 날짜를 한국 거래일 기준으로 정규화 (비거래일이면 다음 거래일로 이동)
    def _normalize_kr(d: pd.Timestamp) -> pd.Timestamp:
        dt_kst = pd.Timestamp(d.date()).tz_localize("Asia/Seoul").replace(hour=12)
        return pd.Timestamp(adjust_date_for_market_close(dt_kst, "KR"))
    df["date"] = df["date"].apply(_normalize_kr)
    df["label_method"]     = "manual"
    df["raw_data"]         = "{}"

    for col in ["macro_rate_risk", "equity_market_risk", "geopolitical_fx_risk"]:
        df[col] = df[col].astype(float)

    # 컬럼 순서 통일
    cols = [
        "event_id", "date", "source", "title", "summary", "url", "language",
        "raw_data", "label_method",
        "macro_rate_risk", "equity_market_risk", "geopolitical_fx_risk",
        "primary_tag", "reasoning", "confidence",
    ]
    df = df[cols]
    df = df.sort_values("date").reset_index(drop=True)

    issues = validate(df)
    if issues:
        for issue in issues:
            logger.warning("검증 이슈: %s", issue)
    else:
        logger.info("검증 통과")

    if validate_only:
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info("저장 완료: %s (%d건)", OUTPUT_PATH, len(df))

    print("\n=== 분포 ===")
    print(df.groupby("primary_tag").size().to_string())
    print(f"\n연도별 건수:")
    print(df.groupby(df["date"].dt.year).size().to_string())
    print("\n=== SANITY CHECK ===")
    for date_str, (col, _, desc) in SANITY_CHECKS.items():
        row = df[df["date"] == pd.Timestamp(date_str)]
        if len(row):
            val = row[col].iloc[0]
            status = "OK" if val >= 1.0 else "WARN"
            print(f"  [{status}] {date_str} {desc}: {col}={val}")
        else:
            print(f"  [MISS] {date_str} {desc}: 날짜 없음")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    main(validate_only=args.validate)
