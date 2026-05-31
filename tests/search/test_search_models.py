# tests/search/test_search_models.py
"""Unit tests for :mod:`llmcore.search.models` (provider-agnostic result types)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from llmcore.search.models import (
    DatasetField,
    DatasetInfo,
    DatasetMetadata,
    DatasetSnapshot,
    DiscoverItem,
    DiscoverResult,
    ScrapeResult,
    SearchItem,
    SearchResultBase,
    WebSearchResult,
)


def test_elapsed_ms_computes_when_both_timestamps_present():
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(milliseconds=250)
    r = SearchResultBase(success=True, trigger_sent_at=t0, data_fetched_at=t1)
    assert r.elapsed_ms() == 250.0


def test_elapsed_ms_none_when_missing():
    assert SearchResultBase(success=True).elapsed_ms() is None


def test_base_to_dict_serializes_datetimes():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    r = SearchResultBase(success=True, provider="brightdata", trigger_sent_at=t0)
    d = r.to_dict()
    assert d["provider"] == "brightdata"
    assert d["trigger_sent_at"] == t0.isoformat()
    # round-trips through JSON
    assert json.loads(r.to_json())["success"] is True


def test_web_search_result_to_dict_expands_items():
    r = WebSearchResult(
        success=True,
        provider="brightdata",
        query="q",
        engine="google",
        items=[SearchItem(position=1, title="t", url="u", description="d", displayed_url="du")],
        total_results=42,
        raw={"organic": []},
    )
    d = r.to_dict()
    assert d["engine"] == "google"
    assert d["total_results"] == 42
    assert isinstance(d["items"], list)
    assert d["items"][0] == {
        "position": 1,
        "title": "t",
        "url": "u",
        "description": "d",
        "displayed_url": "du",
    }
    # JSON-serializable end to end
    json.loads(r.to_json())


def test_discover_result_to_dict_expands_items():
    r = DiscoverResult(
        success=True,
        query="q",
        intent="why",
        items=[DiscoverItem(title="t", url="u", description="d", relevance_score=0.5, content="c")],
        total_results=1,
        task_id="task-1",
        duration_seconds=2.0,
    )
    d = r.to_dict()
    assert d["intent"] == "why"
    assert d["items"][0]["relevance_score"] == 0.5
    assert d["items"][0]["content"] == "c"


def test_scrape_result_fields_default():
    r = ScrapeResult(success=True, url="https://x", content="<html/>", response_format="raw")
    assert r.status == "ready"
    assert r.content_char_size is None
    d = r.to_dict()
    assert d["url"] == "https://x"
    assert d["response_format"] == "raw"


def test_dataset_snapshot_record_count():
    empty = DatasetSnapshot(success=True, dataset_id="gd_1", snapshot_id="s")
    assert empty.record_count == 0
    filled = DatasetSnapshot(
        success=True,
        dataset_id="gd_1",
        snapshot_id="s",
        status="ready",
        records=[{"a": 1}, {"a": 2}],
    )
    assert filled.record_count == 2


def test_dataset_metadata_field_names_order_preserved():
    meta = DatasetMetadata(
        id="gd_1",
        fields=[
            DatasetField(name="alpha", type="text"),
            DatasetField(name="beta", type="number"),
            DatasetField(name="gamma", type="url"),
        ],
    )
    assert meta.field_names() == ["alpha", "beta", "gamma"]


def test_dataset_info_defaults():
    di = DatasetInfo(id="gd_1", name="X")
    assert di.size == 0
