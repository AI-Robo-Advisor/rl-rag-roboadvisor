"""GDELT 2.0 BigQuery 이벤트 수집 유틸리티.

이 모듈은 LLM 라벨링 전 단계의 raw 이벤트 데이터를 공통 Parquet 스키마로
정규화합니다. GDELT events 테이블은 날짜 단위 ``SQLDATE``만 사용하므로, 뉴스
발행 시각 기반 market close 보정은 후속 DOC API/기사 원문 수집 단계에서
적용해야 합니다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from dotenv import load_dotenv

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

GDELT_COUNTRY_CODES = ("USA", "CHN", "KOR", "RUS", "UKR", "EUR", "JPN", "TWN")
GDELT_EVENT_PREFIXES = ("03", "04", "13", "14", "17", "19")
GDELT_TABLE = "`gdelt-bq.gdeltv2.events_partitioned`"


@dataclass(frozen=True)
class BigQueryDryRunResult:
    """BigQuery dry-run 결과."""

    total_bytes_processed: int

    @property
    def total_gb_processed(self) -> float:
        """예상 처리량을 GB 단위로 반환합니다."""
        return self.total_bytes_processed / 1e9


def _quote_sql_values(values: Iterable[str]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def build_gdelt_query(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    limit: int | None = None,
) -> str:
    """GDELT events 본 수집 쿼리를 생성합니다.

    Args:
        start_date: 수집 시작일(YYYY-MM-DD).
        end_date: 수집 종료일(YYYY-MM-DD).
        limit: 테스트용 최대 행 수. None이면 LIMIT을 붙이지 않습니다.

    Returns:
        BigQuery Standard SQL 쿼리 문자열.
    """
    country_codes = _quote_sql_values(GDELT_COUNTRY_CODES)
    event_filters = "\n        OR ".join(
        f"LPAD(CAST(EventCode AS STRING), 3, '0') LIKE '{prefix}%'"
        for prefix in GDELT_EVENT_PREFIXES
    )
    limit_clause = f"\nLIMIT {int(limit)}" if limit is not None else ""

    return f"""
WITH filtered AS (
    SELECT
        GLOBALEVENTID,
        SQLDATE,
        EventCode,
        LPAD(CAST(EventCode AS STRING), 3, '0') AS NormalizedEventCode,
        EventBaseCode,
        Actor1CountryCode,
        Actor1Name,
        Actor2CountryCode,
        Actor2Name,
        GoldsteinScale,
        NumMentions,
        AvgTone,
        SOURCEURL,
        ROW_NUMBER() OVER (
            PARTITION BY SQLDATE, LPAD(CAST(EventCode AS STRING), 3, '0')
            ORDER BY NumMentions DESC, AvgTone ASC, SOURCEURL, GLOBALEVENTID
        ) AS event_rank
    FROM {GDELT_TABLE}
    WHERE _PARTITIONTIME >= TIMESTAMP('{start_date}')
      AND _PARTITIONTIME < TIMESTAMP(DATE_ADD(DATE('{end_date}'), INTERVAL 1 DAY))
      AND SQLDATE BETWEEN CAST(FORMAT_DATE('%Y%m%d', DATE('{start_date}')) AS INT64)
      AND CAST(FORMAT_DATE('%Y%m%d', DATE('{end_date}')) AS INT64)
      AND (
          Actor1CountryCode IN ({country_codes})
          OR Actor2CountryCode IN ({country_codes})
      )
      AND (
        {event_filters}
      )
      AND NumMentions >= 50
      AND GoldsteinScale < -3.0
)
SELECT
    CONCAT(
        'gdelt-',
        FORMAT_DATE('%Y%m%d', PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING))),
        '-',
        NormalizedEventCode,
        '-',
        CAST(event_rank AS STRING)
    ) AS event_id,
    PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) AS date,
    'gdelt' AS source,
    CONCAT(
        IFNULL(Actor1Name, ''),
        ' - ',
        IFNULL(Actor2Name, ''),
        ' (Event ',
        NormalizedEventCode,
        ')'
    ) AS title,
    CONCAT(
        'Goldstein: ',
        CAST(GoldsteinScale AS STRING),
        ', Tone: ',
        CAST(AvgTone AS STRING),
        ', Mentions: ',
        CAST(NumMentions AS STRING)
    ) AS summary,
    SOURCEURL AS url,
    'en' AS language,
    TO_JSON_STRING(STRUCT(
        GLOBALEVENTID,
        EventCode,
        NormalizedEventCode,
        EventBaseCode,
        GoldsteinScale,
        NumMentions,
        AvgTone,
        Actor1CountryCode,
        Actor2CountryCode
    )) AS raw_data,
    'llm' AS label_method
FROM filtered
ORDER BY date, NumMentions DESC{limit_clause}
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
    bigquery = _load_bigquery_module()
    return bigquery.Client()


def dry_run_query(query: str, client: Any | None = None) -> BigQueryDryRunResult:
    """BigQuery dry-run으로 예상 처리량을 확인합니다."""
    bigquery = _load_bigquery_module()
    client = client or create_bigquery_client()
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    dry_job = client.query(query, job_config=job_config)
    return BigQueryDryRunResult(total_bytes_processed=int(dry_job.total_bytes_processed))


def run_query(query: str, client: Any | None = None) -> pd.DataFrame:
    """BigQuery 쿼리를 실행하고 DataFrame으로 반환합니다."""
    client = client or create_bigquery_client()
    return client.query(query).to_dataframe()


def normalize_gdelt_events(df: pd.DataFrame) -> pd.DataFrame:
    """GDELT 쿼리 결과를 공통 raw 이벤트 스키마로 정규화합니다."""
    normalized = df.copy()
    if normalized.empty:
        return pd.DataFrame(columns=GDELT_COLUMNS)

    missing = [column for column in GDELT_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(f"GDELT query result missing columns: {missing}")

    normalized = normalized[GDELT_COLUMNS].copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
    normalized["source"] = "gdelt"
    normalized["language"] = "en"
    normalized["label_method"] = "llm"

    for column in ("event_id", "title", "summary", "url", "raw_data"):
        normalized[column] = normalized[column].fillna("").astype(str)

    empty_ids = normalized["event_id"].str.len() == 0
    if empty_ids.any():
        raise ValueError("GDELT query result contains empty event_id values")

    return normalized


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
) -> pd.DataFrame:
    """GDELT 이벤트를 수집하고 Parquet으로 저장합니다."""
    query = build_gdelt_query(start_date=start_date, end_date=end_date, limit=limit)
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
