import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data import gdelt_collector
from src.data.gdelt_collector import (
    GDELT_COLUMNS,
    build_gdelt_query,
    collect_gdelt_events,
    normalize_gdelt_events,
    raw_data_to_json,
)


def _sample_gdelt_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "gdelt-20220224-190-1",
                "date": "2022-02-24",
                "source": "gdelt",
                "title": "RUSSIA - UKRAINE (Event 190)",
                "summary": "Goldstein: -10.0, Tone: -7.5, Mentions: 250",
                "url": "https://example.com/gdelt/1",
                "language": "en",
                "raw_data": json.dumps(
                    {
                        "EventCode": "190",
                        "EventBaseCode": "19",
                        "GoldsteinScale": -10.0,
                        "NumMentions": 250,
                        "AvgTone": -7.5,
                        "Actor1CountryCode": "RUS",
                        "Actor2CountryCode": "UKR",
                    }
                ),
                "label_method": "llm",
            }
        ]
    )


def test_build_gdelt_query_contains_required_safety_filters():
    query = build_gdelt_query("2018-01-01", "2025-12-31", limit=100)

    assert "FROM `gdelt-bq.gdeltv2.events_partitioned`" in query
    assert "_PARTITIONTIME >= TIMESTAMP('2018-01-01')" in query
    assert "_PARTITIONTIME < TIMESTAMP(DATE_ADD(DATE('2025-12-31'), INTERVAL 1 DAY))" in query
    assert "SQLDATE BETWEEN CAST(FORMAT_DATE('%Y%m%d', DATE('2018-01-01')) AS INT64)" in query
    assert "DATE('2025-12-31')" in query
    assert "NumMentions >= 50" in query
    assert "GoldsteinScale < -3.0" in query
    assert "Actor1CountryCode IN" in query
    assert "LPAD(CAST(EventCode AS STRING), 3, '0') LIKE '03%'" in query
    assert "LPAD(CAST(EventCode AS STRING), 3, '0') LIKE '19%'" in query
    assert "LIMIT 100" in query


def test_build_gdelt_query_has_deterministic_event_rank_ordering():
    query = build_gdelt_query("2018-01-01", "2025-12-31", limit=100)

    assert "PARTITION BY SQLDATE, LPAD(CAST(EventCode AS STRING), 3, '0')" in query
    assert "ORDER BY NumMentions DESC, AvgTone ASC, SOURCEURL, GLOBALEVENTID" in query
    assert "GLOBALEVENTID" in query


def test_normalize_gdelt_events_keeps_common_schema():
    events = normalize_gdelt_events(_sample_gdelt_df())

    assert list(events.columns) == GDELT_COLUMNS
    assert len(events) == 1
    assert events.loc[0, "event_id"] == "gdelt-20220224-190-1"
    assert events.loc[0, "date"] == pd.Timestamp("2022-02-24")
    assert events.loc[0, "source"] == "gdelt"
    assert events.loc[0, "language"] == "en"
    assert events.loc[0, "label_method"] == "llm"
    assert raw_data_to_json(events.loc[0, "raw_data"])["EventCode"] == "190"


def test_normalize_gdelt_events_returns_empty_schema_for_empty_frame():
    events = normalize_gdelt_events(pd.DataFrame())

    assert list(events.columns) == GDELT_COLUMNS
    assert events.empty


def test_normalize_gdelt_events_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        normalize_gdelt_events(pd.DataFrame({"event_id": ["gdelt-1"]}))


def test_normalize_gdelt_events_rejects_empty_event_id():
    raw = _sample_gdelt_df()
    raw.loc[0, "event_id"] = ""

    with pytest.raises(ValueError, match="empty event_id"):
        normalize_gdelt_events(raw)


def test_dry_run_query_uses_bigquery_dry_run_config(monkeypatch):
    calls = {}

    class FakeQueryJobConfig:
        def __init__(self, dry_run=False, use_query_cache=True):
            self.dry_run = dry_run
            self.use_query_cache = use_query_cache

    class FakeClient:
        def query(self, query, job_config=None):
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
    assert calls["job_config"].dry_run is True
    assert calls["job_config"].use_query_cache is False


def test_collect_gdelt_events_runs_query_and_writes_parquet(tmp_path, monkeypatch):
    output = tmp_path / "gdelt_events.parquet"

    class FakeQueryJob:
        def to_dataframe(self):
            return _sample_gdelt_df()

    class FakeClient:
        def query(self, query):
            assert "gdelt-bq.gdeltv2.events_partitioned" in query
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
