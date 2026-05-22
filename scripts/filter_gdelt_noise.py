"""GDELT 수집 parquet 노이즈 후처리.

원본 ``gdelt_events_2018_2025.parquet`` 는 변경하지 않고
``gdelt_events_2018_2025_filtered.parquet`` 로 저장합니다.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.gdelt_collector import GDELT_THEME_PATTERNS

INPUT_PATH = Path("data/raw/gdelt/gdelt_events_2018_2025.parquet")
OUTPUT_PATH = Path("data/raw/gdelt/gdelt_events_2018_2025_filtered.parquet")

# 우크라이나 지역 뉴스 과다·집계 노이즈 도메인
DOMAIN_BLACKLIST_SUFFIXES = (".ua",)
DOMAIN_BLACKLIST_EXACT = frozenset({"biztoc.com"})

NOISE_THEME_MARKERS = (
    "TAX_WEAPONS",
    "TAX_FNCACT",
    "SOC_GENERALCRIME",
    "CRISISLEX",
    "KILL",
)

NOISE_RE = re.compile("|".join(re.escape(m) for m in NOISE_THEME_MARKERS))
MACRO_RE = re.compile("|".join(re.escape(p) for p in GDELT_THEME_PATTERNS))

QUALITY_KEYWORDS = {
    "ECON_": lambda t: "ECON_" in t,
    "STOCKMARKET": lambda t: "STOCKMARKET" in t,
    "WB_1104": lambda t: "WB_1104" in t,
    "SANCTIONS": lambda t: "SANCTIONS" in t,
}


def extract_domain(title: str) -> str:
    """title 첫 줄 도메인 (``domain | theme`` 형식)."""
    return str(title).split(" | ")[0].strip().lower()


def parse_themes(raw_data: str) -> list[str]:
    """raw_data JSON에서 themes 리스트를 반환합니다."""
    payload = json.loads(raw_data)
    themes = payload.get("themes", [])
    if not isinstance(themes, list):
        return []
    return [str(t) for t in themes]


def is_blacklisted_domain(domain: str) -> bool:
    """도메인 블랙리스트 여부."""
    if domain in DOMAIN_BLACKLIST_EXACT:
        return True
    return any(domain.endswith(suffix) for suffix in DOMAIN_BLACKLIST_SUFFIXES)


def has_noise_theme(themes: list[str]) -> bool:
    """노이즈 테마가 하나라도 포함되면 True."""
    return any(NOISE_RE.search(theme) for theme in themes)


def has_macro_theme(themes: list[str]) -> bool:
    """매크로/금융 관련 테마가 하나라도 포함되면 True."""
    return any(MACRO_RE.search(theme) for theme in themes)


def filter_gdelt_noise(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """3단계 필터를 적용하고 단계별 제거 건수를 반환합니다."""
    stats: dict[str, int] = {}
    working = df.copy()

    domain_mask = working["title"].map(extract_domain).map(is_blacklisted_domain)
    stats["domain_removed"] = int(domain_mask.sum())
    working = working.loc[~domain_mask].copy()

    noise_mask = working["raw_data"].map(
        lambda raw: has_noise_theme(parse_themes(raw))
    )
    stats["noise_removed"] = int(noise_mask.sum())
    working = working.loc[~noise_mask].copy()

    macro_mask = working["raw_data"].map(
        lambda raw: has_macro_theme(parse_themes(raw))
    )
    stats["macro_removed"] = int((~macro_mask).sum())
    working = working.loc[macro_mask].copy()

    stats["final_count"] = len(working)
    return working, stats


def print_quality_report(df: pd.DataFrame, original_count: int) -> None:
    """최종 품질 요약을 출력합니다."""
    n = len(df)
    print("\n=== 최종 데이터 품질 ===")
    print(f"최종 건수: {n}")

    years = pd.to_datetime(df["date"]).dt.year.value_counts().sort_index()
    print("\n연도별 건수:")
    print("year | count")
    for year, count in years.items():
        print(f"{int(year)} | {count}")

    domains = df["title"].map(extract_domain).value_counts().head(10)
    print("\n상위 10개 도메인:")
    for domain, count in domains.items():
        print(f"  {domain}: {count}")

    print("\n키워드별 매칭 비율:")
    for label, matcher in QUALITY_KEYWORDS.items():
        matched = sum(
            1
            for raw in df["raw_data"]
            if any(matcher(t) for t in parse_themes(raw))
        )
        pct = matched / n * 100 if n else 0.0
        print(f"  {label}: {pct:.1f}% ({matched}/{n})")

    noise_matched = sum(
        1 for raw in df["raw_data"] if has_noise_theme(parse_themes(raw))
    )
    noise_pct = noise_matched / n * 100 if n else 0.0
    print(f"\n노이즈 의심 비율: {noise_pct:.1f}% ({noise_matched}/{n})")

    retention = n / original_count * 100 if original_count else 0.0
    print(
        f"\n=== 원본 대비 잔존율 ===\n"
        f"{original_count:,}건 → {n:,}건 ({retention:.1f}%)"
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)
    original_count = len(df)

    filtered, stats = filter_gdelt_noise(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(OUTPUT_PATH, index=False)

    print("=== 단계별 제거 건수 ===")
    print(f"도메인 블랙리스트 제거: {stats['domain_removed']:,}건")
    print(f"노이즈 테마 제거: {stats['noise_removed']:,}건")
    print(f"매크로 테마 없음 제거: {stats['macro_removed']:,}건")
    print(f"최종 남은 건수: {stats['final_count']:,}건")
    print(f"\n저장 완료: {OUTPUT_PATH}")

    print_quality_report(filtered, original_count)


if __name__ == "__main__":
    main()
