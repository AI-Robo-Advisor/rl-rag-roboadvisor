"""GDELT 2.0 BigQuery GKG 수집 유틸리티.

gdeltv2.gkg 테이블에서 Themes·Tone·기사 메타데이터를 수집해 공통 Parquet 스키마로
정규화합니다. ``published_at`` 기준 US market close 보정, URL 중복 제거를 적용합니다.

event_id의 YYYYMMDD는 발행일(pub_date) 기준이며, ``date`` 컬럼만 거래일로 보정됩니다.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import pandas as pd
from dotenv import load_dotenv

from src.data.market_close import adjust_date_for_market_close

DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_OUTPUT_PATH = Path("data/raw/gdelt/gdelt_events_2018_2025.parquet")

GDELT_COLUMNS = [
    "event_id",
    "date",
    "source",
    "title",
    "summary",
    "url",
    "language",
    "raw_data",
    "label_method",
]

GDELT_QUERY_COLUMNS = [
    "event_id",
    "date",
    "published_at",
    "source",
    "title",
    "summary",
    "url",
    "language",
    "source_name",
    "themes",
    "tone",
    "locations",
    "label_method",
]

GDELT_TABLE = "`gdelt-bq.gdeltv2.gkg_partitioned`"

GDELT_THEME_PATTERNS = (
    "ECON_INFLATION",
    "ECON_INTEREST_RATE",
    "ECON_RECESSION",
    "ECON_TRADE",
    "ECON_EMPLOYMENT",
    "ECON_BANKRUPTCY",
    "ECON_DEBT",
    "WB_1104_MACROECONOMIC",
    "WB_1150_VOLATILITY",
    "WB_1098_FINANCIAL_SECTOR",
    "WB_1099_BANKING",
    "WB_1100_CAPITAL_MARKETS",
    "WB_695_FINANCIAL",
    "WB_2396_TRADE",
    "WB_696_CONFLICT",
    "FINSEC_",
    "STOCKMARKET",
    "UNGP_FINANCE",
    "SANCTIONS",
    "SUPPLY_CHAIN",
)

GDELT_THEME_REGEX = (
    "ECON_INFLATION|ECON_INTEREST_RATE|ECON_RECESSION|ECON_TRADE|"
    "ECON_EMPLOYMENT|ECON_BANKRUPTCY|ECON_DEBT|"
    "WB_1104_MACROECONOMIC|WB_1150_VOLATILITY|"
    "WB_1098_FINANCIAL_SECTOR|WB_1099_BANKING|WB_1100_CAPITAL_MARKETS|"
    "WB_695_FINANCIAL|WB_2396_TRADE|WB_696_CONFLICT|"
    "FINSEC_|STOCKMARKET|UNGP_FINANCE|"
    "SANCTIONS|SUPPLY_CHAIN"
)

GDELT_LOCATION_FILTERS = (
    "United States",
    "South Korea",
    "China",
    "Russia",
    "Ukraine",
    "Europe",
    "Middle East",
)

ThemesFilterMode = Literal["like", "regexp"]


@dataclass(frozen=True)
class BigQueryDryRunResult:
    """BigQuery dry-run 결과."""

    total_bytes_processed: int

    @property
    def total_gb_processed(self) -> float:
        """예상 처리량을 GB 단위로 반환합니다."""
        return self.total_bytes_processed / 1e9


def _themes_filter_clause(mode: ThemesFilterMode) -> str:
    if mode == "regexp":
        return f"REGEXP_CONTAINS(V2Themes, r'{GDELT_THEME_REGEX}')"
    like_parts = "\n        OR ".join(
        f"V2Themes LIKE '%{pattern}%'" for pattern in GDELT_THEME_PATTERNS
    )
    return f"(\n        {like_parts}\n      )"


def _locations_filter_clause() -> str:
    parts = "\n        OR ".join(
        f"V2Locations LIKE '%{region}%'" for region in GDELT_LOCATION_FILTERS
    )
    return f"(\n        {parts}\n      )"


def _date_range_clause(start_date: str, end_date: str) -> str:
    """GKG DATE(INT64 YYYYMMDDHHMMSS) 범위 필터."""
    return f"""
      AND DATE >= CAST(CONCAT(FORMAT_DATE('%Y%m%d', DATE('{start_date}')), '000000') AS INT64)
      AND DATE < CAST(
          CONCAT(
              FORMAT_DATE('%Y%m%d', DATE_ADD(DATE('{end_date}'), INTERVAL 1 DAY)),
              '000000'
          ) AS INT64
      )"""


def build_gdelt_query(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    limit: int | None = None,
    themes_filter: ThemesFilterMode = "regexp",
) -> str:
    """GDELT GKG 본 수집 쿼리를 생성합니다.

    Args:
        start_date: 수집 시작일(YYYY-MM-DD).
        end_date: 수집 종료일(YYYY-MM-DD).
        limit: 테스트용 최대 행 수. None이면 LIMIT을 붙이지 않습니다.
        themes_filter: ``like``(12 OR LIKE) 또는 ``regexp``(REGEXP_CONTAINS).

    Returns:
        BigQuery Standard SQL 쿼리 문자열.
    """
    themes_clause = _themes_filter_clause(themes_filter)
    locations_clause = _locations_filter_clause()
    date_clause = _date_range_clause(start_date, end_date)
    limit_clause = f"\nLIMIT {int(limit)}" if limit is not None else ""

    return f"""
WITH base AS (
    SELECT
        PARSE_TIMESTAMP(
            '%Y%m%d%H%M%S',
            LPAD(CAST(DATE AS STRING), 14, '0')
        ) AS published_at,
        DATE(
            PARSE_TIMESTAMP(
                '%Y%m%d%H%M%S',
                LPAD(CAST(DATE AS STRING), 14, '0')
            )
        ) AS pub_date,
        DocumentIdentifier AS url,
        SourceCommonName AS source_name,
        V2Themes AS themes,
        CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) AS tone,
        V2Locations AS locations,
        CONCAT(
            COALESCE(REGEXP_EXTRACT(DocumentIdentifier, r'https?://([^/]+)'), 'unknown'),
            ' | ',
            COALESCE(
                REGEXP_REPLACE(SPLIT(V2Themes, ';')[SAFE_OFFSET(0)], r',.*', ''),
                ''
            )
        ) AS title,
        CONCAT(
            IFNULL(SourceCommonName, ''),
            ' | ',
            IFNULL(V2Themes, ''),
            ' | ',
            'Tone: ',
            CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS STRING)
        ) AS summary
    FROM {GDELT_TABLE}
    WHERE _PARTITIONTIME >= TIMESTAMP('{start_date}')
      AND _PARTITIONTIME < TIMESTAMP(DATE_ADD(DATE('{end_date}'), INTERVAL 1 DAY))
      {date_clause}
      AND {themes_clause}
      AND CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) < -15.0
      AND {locations_clause}
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY pub_date
            ORDER BY tone ASC, url
        ) AS day_idx
    FROM base
)
SELECT
    CONCAT(
        'gdelt-',
        FORMAT_DATE('%Y%m%d', pub_date),
        '-',
        CAST(day_idx AS STRING)
    ) AS event_id,
    pub_date AS date,
    published_at,
    'gdelt' AS source,
    title,
    summary,
    url,
    'en' AS language,
    source_name,
    themes,
    tone,
    locations,
    'llm' AS label_method
FROM ranked
ORDER BY date, tone ASC{limit_clause}
""".strip()


def _load_bigquery_module() -> Any:
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise ImportError(
            "google-cloud-bigquery is required for GDELT collection. "
            "Install project requirements before running scripts/collect_gdelt.py."
        ) from exc
    return bigquery


def create_bigquery_client() -> Any:
    """환경 인증 정보를 사용해 BigQuery 클라이언트를 생성합니다."""
    load_dotenv()
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds:
        cred_path = Path(creds)
        if not cred_path.is_file():
            resolved = PROJECT_ROOT / creds
            if resolved.is_file():
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(resolved)
    bigquery = _load_bigquery_module()
    return bigquery.Client()


def dry_run_query(query: str, client: Any | None = None) -> BigQueryDryRunResult:
    """BigQuery dry-run으로 예상 처리량을 확인합니다."""
    bigquery = _load_bigquery_module()
    client = client or create_bigquery_client()
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    dry_job = client.query(query, job_config=job_config)
    return BigQueryDryRunResult(total_bytes_processed=int(dry_job.total_bytes_processed))


def compare_themes_filter_dry_run(
    start_date: str = "2024-06-01",
    end_date: str = "2024-06-01",
    limit: int = 100,
    client: Any | None = None,
) -> dict[str, BigQueryDryRunResult]:
    """Themes 필터 like vs regexp dry-run GB를 비교합니다."""
    results: dict[str, BigQueryDryRunResult] = {}
    for mode in ("like", "regexp"):
        query = build_gdelt_query(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            themes_filter=mode,  # type: ignore[arg-type]
        )
        results[mode] = dry_run_query(query, client=client)
    return results


def run_query(query: str, client: Any | None = None) -> pd.DataFrame:
    """BigQuery 쿼리를 실행하고 DataFrame으로 반환합니다."""
    client = client or create_bigquery_client()
    return client.query(query).to_dataframe()


def _build_raw_data_json(
    themes: str,
    tone: float | int | str,
    locations: str,
    source_name: str,
) -> str:
    theme_list = [
        re.sub(r",.*", "", part.strip())
        for part in str(themes).split(";")
        if part.strip()
    ]
    payload = {
        "themes": theme_list,
        "tone": float(tone) if tone != "" and tone is not None else None,
        "locations": str(locations) if locations is not None else "",
        "source_name": str(source_name) if source_name is not None else "",
    }
    return json.dumps(payload, ensure_ascii=False)


def _apply_us_market_close_dates(published_at: pd.Series) -> pd.Series:
    def _to_trading_date(ts: pd.Timestamp) -> pd.Timestamp:
        if pd.isna(ts):
            raise ValueError("published_at contains null values")
        adjusted = adjust_date_for_market_close(ts, "US")
        return pd.Timestamp(adjusted)

    parsed = pd.to_datetime(published_at, utc=True)
    return parsed.apply(_to_trading_date)


def normalize_gdelt_events(df: pd.DataFrame) -> pd.DataFrame:
    """GDELT GKG 쿼리 결과를 공통 raw 이벤트 스키마로 정규화합니다."""
    if df.empty:
        return pd.DataFrame(columns=GDELT_COLUMNS)

    normalized = df.copy()
    missing = [column for column in GDELT_QUERY_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(f"GDELT query result missing columns: {missing}")

    sort_cols = ["date", "tone"]
    normalized = normalized.sort_values(sort_cols, ascending=[True, True])
    normalized = normalized.drop_duplicates(subset=["url"], keep="first")

    normalized["date"] = _apply_us_market_close_dates(normalized["published_at"])
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
    normalized["source"] = "gdelt"
    normalized["language"] = "en"
    normalized["label_method"] = "llm"

    normalized["raw_data"] = normalized.apply(
        lambda row: _build_raw_data_json(
            themes=row["themes"],
            tone=row["tone"],
            locations=row["locations"],
            source_name=row["source_name"],
        ),
        axis=1,
    )

    for column in ("event_id", "title", "summary", "url", "raw_data"):
        normalized[column] = normalized[column].fillna("").astype(str)

    empty_ids = normalized["event_id"].str.len() == 0
    if empty_ids.any():
        raise ValueError("GDELT query result contains empty event_id values")

    return normalized[GDELT_COLUMNS].copy()


def save_events_parquet(df: pd.DataFrame, output_path: Path | str) -> Path:
    """정규화된 이벤트 DataFrame을 Parquet으로 저장합니다."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def collect_gdelt_events(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    client: Any | None = None,
    limit: int | None = None,
    themes_filter: ThemesFilterMode = "regexp",
) -> pd.DataFrame:
    """GDELT GKG 이벤트를 수집하고 Parquet으로 저장합니다."""
    query = build_gdelt_query(
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        themes_filter=themes_filter,
    )
    raw_df = run_query(query, client=client)
    events = normalize_gdelt_events(raw_df)
    save_events_parquet(events, output_path)
    return events


def raw_data_to_json(raw_data: str) -> dict[str, Any]:
    """테스트와 디버깅을 위해 raw_data JSON 문자열을 dict로 변환합니다."""
    value = json.loads(raw_data)
    if not isinstance(value, dict):
        raise ValueError("raw_data must decode to a JSON object")
    return value
