"""원본 이벤트 통합 스크립트.

data/raw/{gdelt,fred,ecos,manual_seed}/ parquet을 통합해
data/processed/unified_events.parquet을 생성합니다.
없는 파일은 자동으로 스킵합니다.

실행:
  python scripts/merge_raw_events.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_PATHS: list[Path] = [
    Path("data/raw/gdelt/gdelt_events_2018_2025_filtered.parquet"),
    Path("data/raw/fred/fred_events_2018_2025.parquet"),
    Path("data/raw/ecos/ecos_events_2018_2025.parquet"),
    Path("data/raw/manual_seed/manual_seed_events.parquet"),
]

OUTPUT_PATH = Path("data/processed/unified_events.parquet")

REQUIRED_COLS = [
    "event_id", "date", "source", "title", "summary", "url", "language",
    "raw_data", "label_method",
    "macro_rate_risk", "equity_market_risk", "geopolitical_fx_risk",
    "primary_tag", "reasoning", "confidence",
]


def main() -> None:
    frames: list[pd.DataFrame] = []

    for path in RAW_PATHS:
        if not path.exists():
            logger.info("스킵 (파일 없음): %s", path.name)
            continue
        df = pd.read_parquet(path)

        # 라벨 컬럼 없는 경우 (llm 라벨링 대기 중) 0으로 채움
        for col in ["macro_rate_risk", "equity_market_risk", "geopolitical_fx_risk"]:
            if col not in df.columns:
                df[col] = 0.0
        for col in ["primary_tag", "reasoning", "confidence"]:
            if col not in df.columns:
                df[col] = None

        frames.append(df)
        logger.info("로드: %s (%d건)", path.name, len(df))

    if not frames:
        logger.error("로드된 파일 없음.")
        return

    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])

    # 수집 범위 초과 날짜 제거 (market close 보정으로 밀린 경우)
    before_clip = len(merged)
    merged = merged[merged["date"] <= pd.Timestamp("2025-12-31")]
    if len(merged) < before_clip:
        logger.info("2025-12-31 초과 날짜 제거: %d건", before_clip - len(merged))

    merged = merged.sort_values("date").reset_index(drop=True)

    # event_id 기준 중복 제거
    before = len(merged)
    merged = merged.drop_duplicates(subset=["event_id"], keep="first")
    after = len(merged)
    if before != after:
        logger.info("event_id 중복 제거: %d건 → %d건", before, after)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)
    logger.info("저장 완료: %s (%d건)", OUTPUT_PATH, len(merged))

    print("\n=== 소스별 건수 ===")
    print(merged.groupby("source").size().to_string())
    print("\n=== label_method별 건수 ===")
    print(merged.groupby("label_method").size().to_string())
    print(f"\n연도별 건수:")
    print(merged.groupby(merged["date"].dt.year).size().to_string())


if __name__ == "__main__":
    main()
