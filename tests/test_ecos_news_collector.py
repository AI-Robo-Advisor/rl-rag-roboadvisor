"""ECOS StatisticTableList 수집 로직 단위 테스트(HTTP 모킹) 및 선택적 실연동 테스트."""
import os
from unittest.mock import MagicMock, patch

import pytest

from src.agent.news_collector import (
    EcosApiError,
    build_statistic_table_list_url,
    collect_bok_ecos_tables_and_store,
    fetch_statistic_table_list,
    map_category,
    row_to_chroma_fields,
)
from src.agent.vectorstore import get_collection


def test_build_url_with_and_without_stat_code():
    u = build_statistic_table_list_url("mykey", 1, 10, None)
    assert u == "https://ecos.bok.or.kr/api/StatisticTableList/mykey/json/kr/1/10"
    u2 = build_statistic_table_list_url("k", 2, 5, "102Y004")
    assert u2.endswith("/k/json/kr/2/5/102Y004")


def test_map_category_keywords():
    assert map_category("x", "코스피 시세") == "증시"
    assert map_category("x", "GDP 성장률") == "실적"
    assert map_category("x", "본원통화 구성") == "정책"


def test_row_to_chroma_fields():
    row = {
        "P_STAT_CODE": "0000000622",
        "STAT_CODE": "102Y004",
        "STAT_NAME": "본원통화 구성내역",
        "CYCLE": "M",
        "ORG_NAME": "한국은행",
    }
    doc, meta, doc_id = row_to_chroma_fields(row, "2026-04-06")
    assert doc_id == "102Y004"
    assert meta["title"] == "본원통화 구성내역"
    assert meta["date"] == "2026-04-06"
    assert "주기: M" in meta["summary"]
    assert meta["url"].startswith("https://ecos.bok.or.kr/")
    assert meta["category"] == "정책"
    assert meta.get("risk_label") == ""


def _mock_response(payload):
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def test_info_100_raises():
    with patch(
        "src.agent.news_collector._SESSION.get",
        return_value=_mock_response(
            {
                "RESULT": {
                    "CODE": "INFO-100",
                    "MESSAGE": "인증키가 유효하지 않습니다.",
                }
            }
        ),
    ):
        with pytest.raises(EcosApiError) as ei:
            fetch_statistic_table_list("bad", 1, 1)
        assert "100" in ei.value.code.upper()


def test_info_200_returns_empty():
    with patch(
        "src.agent.news_collector._SESSION.get",
        return_value=_mock_response(
            {
                "RESULT": {
                    "CODE": "INFO-200",
                    "MESSAGE": "해당하는 데이터가 없습니다.",
                }
            }
        ),
    ):
        rows = fetch_statistic_table_list("k", 1, 1, "NOPE")
    assert rows == []


def test_retry_then_success_on_602():
    """602.0 응답 후 재시도하여 성공하는 경우."""
    ok_payload = {
        "StatisticTableList": {
            "list_total_count": 1,
            "row": [{"STAT_CODE": "102Y004", "STAT_NAME": "t", "CYCLE": "M"}],
        }
    }
    fail = _mock_response({"RESULT": {"CODE": "602.0", "MESSAGE": "호출 제한"}})
    ok = _mock_response(ok_payload)
    with patch(
        "src.agent.news_collector._SESSION.get",
        side_effect=[fail, ok],
    ) as m:
        with patch("src.agent.news_collector.time.sleep", return_value=None):
            rows = fetch_statistic_table_list("sample", 1, 10, "102Y004")
    assert len(rows) == 1
    assert m.call_count == 2


def test_success_rows():
    with patch(
        "src.agent.news_collector._SESSION.get",
        return_value=_mock_response(
            {
                "StatisticTableList": {
                    "list_total_count": 1,
                    "row": [
                        {
                            "P_STAT_CODE": "0000000622",
                            "STAT_CODE": "102Y004",
                            "STAT_NAME": "테스트 통계",
                            "CYCLE": "M",
                            "ORG_NAME": None,
                        }
                    ],
                }
            }
        ),
    ):
        rows = fetch_statistic_table_list("sample", 1, 10, "102Y004")
    assert len(rows) == 1
    assert rows[0]["STAT_CODE"] == "102Y004"


@pytest.mark.integration
def test_live_collect_102y004_to_chroma(tmp_path, monkeypatch):
    """
    실제 ECOS(sample 또는 BOK_API_KEY)로 102Y004를 수집해 임시 ChromaDB에 적재합니다.
    네트워크·한국은행 ECOS API 필요. 로컬 확인: pytest -m integration
    """
    key = os.getenv("BOK_API_KEY", "").strip() or "sample"
    monkeypatch.setenv("BOK_API_KEY", key)
    persist = str(tmp_path / "chroma_ecos_test")
    n = collect_bok_ecos_tables_and_store(
        stat_codes=["102Y004"],
        start_index=1,
        end_index=10,
        persist_dir=persist,
    )
    assert n >= 1
    col = get_collection(persist_dir=persist)
    got = col.get(ids=["102Y004"], include=["metadatas", "documents"])
    assert got["ids"] and got["ids"][0] == "102Y004"
    assert got["metadatas"] and got["metadatas"][0].get("title")


@pytest.mark.integration
def test_live_sample_key_list():
    """sample 키 단일 조회 연동."""
    rows = fetch_statistic_table_list("sample", 1, 5, "102Y004")
    assert any(r.get("STAT_CODE") == "102Y004" for r in rows)
