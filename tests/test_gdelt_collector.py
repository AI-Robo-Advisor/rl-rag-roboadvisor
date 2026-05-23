import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data import gdelt_collector
from src.data.gdelt_collector import (
    GDELT_COLUMNS,
    GDELT_QUERY_COLUMNS,
    build_gdelt_query,
    collect_gdelt_events,
    normalize_gdelt_events,
    raw_data_to_json,
)


def _sample_gdelt_query_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "gdelt-20220224-1",
                "date": pd.Timestamp("2022-02-24"),
                "published_at": pd.Timestamp("2022-02-24 18:00:00", tz="UTC"),
                "source": "gdelt",
                "title": "www.wsj.com | ECON_INTEREST_RATE",
                "summary": "Wall Street Journal | ECON_INTEREST_RATE;WB_696_CONFLICT | Tone: -5.2",
                "url": "https://www.wsj.com/articles/fed-rates",
                "language": "en",
                "source_name": "Wall Street Journal",
                "themes": "ECON_INTEREST_RATE;WB_696_CONFLICT",
                "tone": -5.2,
                "locations": "United States#...",
                "label_method": "llm",
            }
        ]
    )


def _duplicate_url_query_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "gdelt-20220224-1",
                "date": pd.Timestamp("2022-02-24"),
                "published_at": pd.Timestamp("2022-02-24 10:00:00", tz="UTC"),
                "source": "gdelt",
                "title": "example.com | ECON_INFLATION",
                "summary": "A | ECON_INFLATION | Tone: -4.0",
                "url": "https://example.com/same-story",
                "language": "en",
                "source_name": "Source A",
                "themes": "ECON_INFLATION",
                "tone": -4.0,
                "locations": "United States",
                "label_method": "llm",
            },
            {
                "event_id": "gdelt-20220224-2",
                "date": pd.Timestamp("2022-02-24"),
                "published_at": pd.Timestamp("2022-02-24 11:00:00", tz="UTC"),
                "source": "gdelt",
                "title": "other.com | ECON_INFLATION",
                "summary": "B | ECON_INFLATION | Tone: -3.5",
                "url": "https://example.com/same-story",
                "language": "en",
                "source_name": "Source B",
                "themes": "ECON_INFLATION",
                "tone": -3.5,
                "locations": "United States",
                "label_method": "llm",
            },
        ]
    )


def test_build_gdelt_query_contains_required_safety_filters():
    query = build_gdelt_query("2018-01-01", "2025-12-31", limit=100)

    assert "FROM `gdelt-bq.gdeltv2.gkg_partitioned`" in query
    assert "_PARTITIONTIME >= TIMESTAMP('2018-01-01')" in query
    assert "_PARTITIONTIME < TIMESTAMP(DATE_ADD(DATE('2025-12-31'), INTERVAL 1 DAY))" in query
    assert "REGEXP_CONTAINS(V2Themes" in query
    assert "WB_1104_MACROECONOMIC" in query
    assert "WB_1098_FINANCIAL_SECTOR" in query
    assert "WB_1099_BANKING" in query
    assert "WB_1100_CAPITAL_MARKETS" in query
    assert "WB_696_CONFLICT" in query
    assert "FINSEC_" in query
    assert "TAX_" not in query
    assert "GENERAL_GOVERNMENT" not in query
    assert "CAST(SPLIT(V2Tone" in query
    assert "< -15.0" in query
    assert "V2Locations LIKE '%United States%'" in query
    assert "V2Locations LIKE '%South Korea%'" in query
    assert "%Y%m%d%H%M%S" in query
    assert "PARSE_TIMESTAMP" in query
    assert "LPAD(CAST(DATE AS STRING), 14, '0')" in query
    assert "REGEXP_EXTRACT(DocumentIdentifier" in query
    assert "LIMIT 100" in query


def test_build_gdelt_query_like_themes_filter():
    query = build_gdelt_query("2024-06-01", "2024-06-01", limit=10, themes_filter="like")

    assert "V2Themes LIKE '%ECON_INFLATION%'" in query
    assert "V2Themes LIKE '%SANCTIONS%'" in query
    assert "V2Themes LIKE '%TAX_%'" not in query
    assert "REGEXP_CONTAINS(V2Themes" not in query


def test_build_gdelt_query_has_deterministic_event_rank_ordering():
    query = build_gdelt_query("2018-01-01", "2025-12-31", limit=100)

    assert "PARTITION BY pub_date" in query
    assert "ORDER BY tone ASC, url" in query
    assert "gdelt-'," in query
    assert "day_idx" in query


def test_normalize_gdelt_events_keeps_common_schema():
    events = normalize_gdelt_events(_sample_gdelt_query_df())

    assert list(events.columns) == GDELT_COLUMNS
    assert len(events) == 1
    assert events.loc[0, "event_id"] == "gdelt-20220224-1"
    assert events.loc[0, "source"] == "gdelt"
    assert events.loc[0, "language"] == "en"
    assert events.loc[0, "label_method"] == "llm"
    assert "https://" not in events.loc[0, "title"]
    raw = raw_data_to_json(events.loc[0, "raw_data"])
    assert raw["themes"] == ["ECON_INTEREST_RATE", "WB_696_CONFLICT"]
    assert raw["tone"] == pytest.approx(-5.2)


def test_normalize_deduplicates_by_url():
    events = normalize_gdelt_events(_duplicate_url_query_df())

    assert len(events) == 1
    assert events.loc[0, "event_id"] == "gdelt-20220224-1"
    assert events.loc[0, "summary"].startswith("A |")


def test_normalize_title_uses_domain_not_full_url():
    events = normalize_gdelt_events(_sample_gdelt_query_df())

    assert "https://" not in events.loc[0, "title"]
    assert events.loc[0, "url"].startswith("https://")


def test_normalize_builds_raw_data_themes_list():
    events = normalize_gdelt_events(_sample_gdelt_query_df())
    raw = raw_data_to_json(events.loc[0, "raw_data"])

    assert isinstance(raw["themes"], list)
    assert "ECON_INTEREST_RATE" in raw["themes"]
    assert raw["source_name"] == "Wall Street Journal"


def test_normalize_applies_us_market_close(monkeypatch):
    captured: list[date] = []

    def fake_adjust(ts: pd.Timestamp, region: str) -> date:
        captured.append(region)
        return date(2022, 2, 25)

    monkeypatch.setattr(
        gdelt_collector,
        "adjust_date_for_market_close",
        fake_adjust,
    )
    events = normalize_gdelt_events(_sample_gdelt_query_df())

    assert captured == ["US"]
    assert events.loc[0, "date"] == pd.Timestamp("2022-02-25")


def test_normalize_gdelt_events_returns_empty_schema_for_empty_frame():
    events = normalize_gdelt_events(pd.DataFrame())

    assert list(events.columns) == GDELT_COLUMNS
    assert events.empty


def test_normalize_gdelt_events_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        normalize_gdelt_events(pd.DataFrame({"event_id": ["gdelt-1"]}))


def test_normalize_gdelt_events_rejects_empty_event_id():
    raw = _sample_gdelt_query_df()
    raw.loc[0, "event_id"] = ""

    with pytest.raises(ValueError, match="empty event_id"):
        normalize_gdelt_events(raw)


def test_dry_run_query_uses_bigquery_dry_run_config(monkeypatch):
    calls: dict[str, object] = {}

    class FakeQueryJobConfig:
        def __init__(self, dry_run: bool = False, use_query_cache: bool = True):
            self.dry_run = dry_run
            self.use_query_cache = use_query_cache

    class FakeClient:
        def query(self, query: str, job_config: object | None = None):
            calls["query"] = query
            calls["job_config"] = job_config
            return SimpleNamespace(total_bytes_processed=123_456_789)

    monkeypatch.setattr(
        gdelt_collector,
        "_load_bigquery_module",
        lambda: SimpleNamespace(QueryJobConfig=FakeQueryJobConfig),
    )

    result = gdelt_collector.dry_run_query("SELECT 1", client=FakeClient())

    assert result.total_bytes_processed == 123_456_789
    assert calls["query"] == "SELECT 1"
    job_config = calls["job_config"]
    assert getattr(job_config, "dry_run") is True
    assert getattr(job_config, "use_query_cache") is False


def test_collect_gdelt_events_runs_query_and_writes_parquet(tmp_path):
    output = tmp_path / "gdelt_events.parquet"

    class FakeQueryJob:
        def to_dataframe(self) -> pd.DataFrame:
            return _sample_gdelt_query_df()

    class FakeClient:
        def query(self, query: str):
            assert "gdelt-bq.gdeltv2.gkg_partitioned" in query
            assert all(col in query for col in ("published_at", "themes", "tone"))
            return FakeQueryJob()

    events = collect_gdelt_events(
        output_path=output,
        client=FakeClient(),
        limit=1,
    )

    assert len(events) == 1
    assert output.exists()
    saved = pd.read_parquet(output)
    assert list(saved.columns) == GDELT_COLUMNS


def test_gdelt_query_columns_cover_normalize_inputs():
    assert set(GDELT_COLUMNS) - {"raw_data"} <= set(GDELT_QUERY_COLUMNS) | {"raw_data"}
