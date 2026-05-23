"""GDELT 2.0 BigQuery GKG 수집 CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.gdelt_collector import (
    DEFAULT_END_DATE,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_START_DATE,
    build_gdelt_query,
    collect_gdelt_events,
    dry_run_query,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect GDELT risk events from BigQuery.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="YYYY-MM-DD")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output parquet path")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for testing")
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help="Print BigQuery dry-run bytes and exit without collecting.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the collection after dry-run. Without this flag, the command only dry-runs.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation when --execute is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = build_gdelt_query(
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
    )

    dry_result = dry_run_query(query)
    print(f"예상 처리 데이터량: {dry_result.total_gb_processed:.2f} GB")

    if args.dry_run_only or not args.execute:
        print("dry-run만 수행했습니다. 실제 수집은 --execute를 붙여 실행하세요.")
        return

    if not args.yes:
        answer = input("실제 BigQuery 쿼리를 실행할까요? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("수집을 취소했습니다.")
            return

    events = collect_gdelt_events(
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=Path(args.output),
        limit=args.limit,
    )
    print(f"수집 완료: {len(events)}건")
    print(f"저장 완료: {args.output}")


if __name__ == "__main__":
    main()
