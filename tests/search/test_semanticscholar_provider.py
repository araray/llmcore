# tests/search/test_semanticscholar_provider.py
"""Tests for :mod:`llmcore.search.providers.semanticscholar_provider`.

All HTTP is intercepted with ``respx`` (no network). We assert the exact S2
endpoints (``/graph/v1/...``, ``/recommendations/v1/...``, ``/datasets/v1/...``),
that the **optional** ``x-api-key`` header is sent only when a key is configured,
the cross-provider -> S2 parameter mapping (``query``/``limit``/``fields``), the
``search_type`` flavor routing (relevance | bulk | match | snippet), engine-aware
normalization (papers / snippets / citations / references / authors /
autocomplete), the POST batch + recommendations bodies, the Datasets API
helpers, retry-with-backoff on 429/5xx, auth failures, the autocomplete-based
health check, and manager wiring (including keyless loading and the ``s2``
alias) — using fixtures modeled on the real S2 API schemas.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from confy.loader import Config

from llmcore.exceptions import SearchProviderError
from llmcore.search.manager import SearchProviderManager
from llmcore.search.models import WebSearchResult
from llmcore.search.providers.semanticscholar_provider import (
    DEFAULT_BASE_URL,
    SemanticScholarSearchProvider,
    _abstract_or_tldr,
    _coerce_int,
    _venue_label,
)

BASE = DEFAULT_BASE_URL
GRAPH = f"{BASE}/graph/v1"
REC = f"{BASE}/recommendations/v1"
DATASETS = f"{BASE}/datasets/v1"

# --- Representative response fixtures (modeled on the S2 swagger schemas) -----
PAPER_1 = {
    "paperId": "5c5751d45e298cea054f32b392c12c61027d2fe7",
    "corpusId": 215416146,
    "title": "Construction of the Literature Graph in Semantic Scholar",
    "abstract": "We describe a deployed scalable system ...",
    "url": "https://www.semanticscholar.org/paper/5c5751d45e298cea054f32b392c12c61027d2fe7",
    "venue": "Annual Meeting of the Association for Computational Linguistics",
    "year": 2018,
    "citationCount": 453,
    "authors": [{"authorId": "1741101", "name": "Oren Etzioni"}],
}
PAPER_2 = {
    "paperId": "abcdef",
    "title": "A second paper",
    "tldr": {"text": "A concise summary."},
    "year": 2021,
    "journal": {"name": "IETE Technical Review"},
}
RELEVANCE_RESPONSE = {"total": "15117", "offset": 0, "next": 10, "data": [PAPER_1, PAPER_2]}
BULK_RESPONSE = {"total": "98765", "token": "CONTINUE_TOKEN", "data": [PAPER_1]}
MATCH_RESPONSE = {"data": [{**PAPER_1, "matchScore": 174.2}]}
SNIPPET_RESPONSE = {
    "data": [
        {
            "snippet": {"text": "graph neural networks excel at ...", "snippetKind": "body"},
            "score": 0.91,
            "paper": {"corpusId": 42, "title": "GNN paper", "authors": []},
        }
    ],
    "retrievalVersion": "v1",
}
AUTOCOMPLETE_RESPONSE = {
    "matches": [{"id": "649def", "title": "SciBERT", "authorsYear": "Beltagy et al., 2019"}]
}
CITATIONS_RESPONSE = {
    "offset": 0,
    "data": [{"isInfluential": True, "citingPaper": {"paperId": "cite1", "title": "Citing paper"}}],
}
REFERENCES_RESPONSE = {
    "offset": 0,
    "data": [{"isInfluential": False, "citedPaper": {"paperId": "ref1", "title": "Cited paper"}}],
}
AUTHOR_1 = {
    "authorId": "1741101",
    "name": "Oren Etzioni",
    "url": "https://www.semanticscholar.org/author/1741101",
    "affiliations": ["Allen Institute for AI"],
    "paperCount": 300,
}
AUTHOR_SEARCH_RESPONSE = {"total": "1", "offset": 0, "data": [AUTHOR_1]}
RECOMMEND_RESPONSE = {"recommendedPapers": [PAPER_1, PAPER_2]}
RELEASES_RESPONSE = ["2023-03-14", "2023-03-21", "2023-03-28"]
RELEASE_META_RESPONSE = {
    "release_id": "2023-03-28",
    "README": "Subject to terms ...",
    "datasets": [{"name": "abstracts", "description": "Paper abstract text ..."}],
}
DATASET_META_RESPONSE = {
    "name": "abstracts",
    "description": "Paper abstract text, where available.",
    "README": "Semantic Scholar Academic Graph Datasets ...",
    "files": ["https://ai2-s2ag.s3.amazonaws.com/.../abstracts/0001.gz"],
}
DIFFS_RESPONSE = {
    "dataset": "papers",
    "start_release": "2023-08-01",
    "end_release": "2023-08-29",
    "diffs": [
        {
            "from_release": "2023-08-01",
            "to_release": "2023-08-07",
            "update_files": ["http://x/u1"],
            "delete_files": ["http://x/d1"],
        }
    ],
}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Patch asyncio.sleep so retry/backoff loops run instantly."""

    async def _instant(_seconds):
        return None

    monkeypatch.setattr("asyncio.sleep", _instant)
    yield


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Ensure no ambient S2 key leaks into keyless tests."""
    for var in ("SEMANTIC_SCHOLAR_API_KEY", "S2_API_KEY", "SERP_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def provider():
    """A keyless provider instance (the default S2 mode)."""
    return SemanticScholarSearchProvider(
        {
            "_instance_name": "semanticscholar",
            "timeout": 5,
            "max_retries": 3,
            "min_request_interval": 0,
        }
    )


@pytest.fixture
def keyed_provider():
    """A provider configured with an explicit API key."""
    return SemanticScholarSearchProvider(
        {"_instance_name": "semanticscholar", "api_key": "secret-key", "max_retries": 3}
    )


# ---------------------------------------------------------------------------
# Identity / capabilities / construction
# ---------------------------------------------------------------------------
def test_identity_and_capabilities(provider):
    assert provider.get_name() == "semanticscholar"
    assert provider.get_capabilities() == {"web_search", "batch_search"}
    assert provider.supports("web_search") is True
    assert provider.supports("batch_search") is True
    assert provider.supports("scrape") is False
    assert provider.supports("discover") is False
    assert provider.supports("dataset_search") is False


def test_keyless_construction_does_not_raise_and_sends_no_key_header(provider):
    assert provider._api_key is None
    client = provider._get_client()
    assert "x-api-key" not in client.headers
    assert client.headers["User-Agent"] == "llmcore-semanticscholar-search"


def test_key_from_explicit_config_sets_header(keyed_provider):
    assert keyed_provider._api_key == "secret-key"
    assert keyed_provider._get_client().headers["x-api-key"] == "secret-key"


def test_key_from_primary_env(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "env-key")
    p = SemanticScholarSearchProvider({"_instance_name": "s2"})
    assert p._api_key == "env-key"


def test_key_from_s2_fallback_env(monkeypatch):
    monkeypatch.setenv("S2_API_KEY", "s2-fallback")
    p = SemanticScholarSearchProvider({"_instance_name": "s2"})
    assert p._api_key == "s2-fallback"


def test_custom_api_key_env_var(monkeypatch):
    monkeypatch.setenv("MY_S2_TOKEN", "custom-env")
    p = SemanticScholarSearchProvider(
        {"_instance_name": "s2", "api_key_env_var": "MY_S2_TOKEN"}
    )
    assert p._api_key == "custom-env"


def test_invalid_default_search_type_falls_back(provider):
    p = SemanticScholarSearchProvider({"default_search_type": "bogus"})
    assert p._default_search_type == "relevance"


# ---------------------------------------------------------------------------
# web_search: relevance
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_web_search_relevance(provider):
    route = respx.get(f"{GRAPH}/paper/search").mock(
        return_value=httpx.Response(200, json=RELEVANCE_RESPONSE)
    )
    res = await provider.web_search("literature graph", count=10, country="us", language="en")
    assert route.called
    params = route.calls.last.request.url.params
    assert params["query"] == "literature graph"
    assert params["limit"] == "10"
    assert "fields" in params  # default paper fields applied
    # cross-provider geo/device params are ignored (not forwarded to S2)
    assert "gl" not in params and "hl" not in params and "device" not in params

    assert isinstance(res, WebSearchResult)
    assert res.success is True
    assert res.engine == "semanticscholar:relevance"
    assert res.total_results == 15117
    assert len(res.items) == 2
    assert res.items[0].title == PAPER_1["title"]
    assert res.items[0].url == PAPER_1["url"]
    assert res.items[0].description == PAPER_1["abstract"]
    assert res.items[0].displayed_url == PAPER_1["venue"]
    # tldr fallback for the abstract-less second paper
    assert res.items[1].description == "A concise summary."
    assert res.raw == RELEVANCE_RESPONSE


@pytest.mark.asyncio
@respx.mock
async def test_web_search_sends_key_header_when_configured(keyed_provider):
    route = respx.get(f"{GRAPH}/paper/search").mock(
        return_value=httpx.Response(200, json=RELEVANCE_RESPONSE)
    )
    await keyed_provider.web_search("x")
    assert route.calls.last.request.headers["x-api-key"] == "secret-key"


@pytest.mark.asyncio
@respx.mock
async def test_web_search_count_clamped_to_100_for_relevance(provider):
    route = respx.get(f"{GRAPH}/paper/search").mock(
        return_value=httpx.Response(200, json=RELEVANCE_RESPONSE)
    )
    await provider.web_search("x", count=500)
    assert route.calls.last.request.url.params["limit"] == "100"


@pytest.mark.asyncio
@respx.mock
async def test_web_search_passthrough_filters_and_openaccess_flag(provider):
    route = respx.get(f"{GRAPH}/paper/search").mock(
        return_value=httpx.Response(200, json=RELEVANCE_RESPONSE)
    )
    await provider.web_search(
        "x",
        year="2018-2020",
        fieldsOfStudy="Computer Science",
        minCitationCount=100,
        openAccessPdf=True,
        fields="title,url",
    )
    params = route.calls.last.request.url.params
    assert params["year"] == "2018-2020"
    assert params["fieldsOfStudy"] == "Computer Science"
    assert params["minCitationCount"] == "100"
    assert params["fields"] == "title,url"  # caller override wins
    # valueless presence flag
    assert "openAccessPdf" in params
    assert params["openAccessPdf"] == ""


@pytest.mark.asyncio
@respx.mock
async def test_web_search_openaccess_false_dropped(provider):
    route = respx.get(f"{GRAPH}/paper/search").mock(
        return_value=httpx.Response(200, json=RELEVANCE_RESPONSE)
    )
    await provider.web_search("x", openAccessPdf=False)
    assert "openAccessPdf" not in route.calls.last.request.url.params


# ---------------------------------------------------------------------------
# web_search: bulk / match / snippet flavors
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_web_search_bulk(provider):
    route = respx.get(f"{GRAPH}/paper/search/bulk").mock(
        return_value=httpx.Response(200, json=BULK_RESPONSE)
    )
    res = await provider.web_search("deep learning", search_type="bulk", count=1000, sort="citationCount:desc")
    assert route.called
    params = route.calls.last.request.url.params
    assert params["limit"] == "1000"
    assert params["sort"] == "citationCount:desc"
    assert res.engine == "semanticscholar:bulk"
    assert res.total_results == 98765
    assert res.raw["token"] == "CONTINUE_TOKEN"


@pytest.mark.asyncio
@respx.mock
async def test_web_search_match_no_limit_single_result(provider):
    route = respx.get(f"{GRAPH}/paper/search/match").mock(
        return_value=httpx.Response(200, json=MATCH_RESPONSE)
    )
    res = await provider.web_search("Construction of the Literature Graph", search_type="match")
    assert route.called
    # match returns a single best result; no limit param is sent
    assert "limit" not in route.calls.last.request.url.params
    assert res.engine == "semanticscholar:match"
    assert len(res.items) == 1
    assert res.items[0].title == PAPER_1["title"]


@pytest.mark.asyncio
@respx.mock
async def test_web_search_snippet_items_from_snippet_text(provider):
    route = respx.get(f"{GRAPH}/snippet/search").mock(
        return_value=httpx.Response(200, json=SNIPPET_RESPONSE)
    )
    res = await provider.web_search("graph neural networks", search_type="snippet", count=5)
    assert route.called
    params = route.calls.last.request.url.params
    assert params["limit"] == "5"
    # snippet search must NOT inject the default paper fields
    assert "fields" not in params
    assert res.engine == "semanticscholar:snippet"
    assert res.items[0].description == "graph neural networks excel at ..."
    assert res.items[0].title == "GNN paper"
    assert res.items[0].url == "https://www.semanticscholar.org/paper/CorpusID:42"


@pytest.mark.asyncio
async def test_web_search_unknown_search_type_raises(provider):
    with pytest.raises(SearchProviderError, match=r"Unsupported search_type"):
        await provider.web_search("x", search_type="nonsense")


@pytest.mark.asyncio
async def test_web_search_non_string_query_raises(provider):
    with pytest.raises(SearchProviderError, match=r"must be a string"):
        await provider.web_search(123)  # type: ignore[arg-type]


@pytest.mark.asyncio
@respx.mock
async def test_web_search_non_200_soft_fails(provider):
    respx.get(f"{GRAPH}/paper/search").mock(
        return_value=httpx.Response(400, json={"error": "Unrecognized fields: [bad]"})
    )
    res = await provider.web_search("x")
    assert res.success is False
    assert "400" in res.error
    assert "Unrecognized" in res.error


@pytest.mark.asyncio
@respx.mock
async def test_web_search_in_body_error_soft_fails(provider):
    respx.get(f"{GRAPH}/paper/search").mock(
        return_value=httpx.Response(200, json={"error": "Response would exceed maximum size"})
    )
    res = await provider.web_search("x")
    assert res.success is False
    assert "exceed maximum size" in res.error


# ---------------------------------------------------------------------------
# Paper graph: details / batch / citations / references / authors / match / autocomplete
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_paper_details_returns_dict(provider):
    route = respx.get(f"{GRAPH}/paper/{PAPER_1['paperId']}").mock(
        return_value=httpx.Response(200, json=PAPER_1)
    )
    out = await provider.paper(PAPER_1["paperId"], fields="title,year")
    assert route.calls.last.request.url.params["fields"] == "title,year"
    assert isinstance(out, dict)
    assert out["title"] == PAPER_1["title"]


@pytest.mark.asyncio
@respx.mock
async def test_paper_details_404_returns_empty_dict(provider):
    respx.get(f"{GRAPH}/paper/missing").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    assert await provider.paper("missing") == {}


@pytest.mark.asyncio
@respx.mock
async def test_paper_batch_posts_ids_body(provider):
    route = respx.post(f"{GRAPH}/paper/batch").mock(
        return_value=httpx.Response(200, json=[PAPER_1, PAPER_2])
    )
    res = await provider.paper_batch(["id1", "id2"], fields="title")
    assert route.called
    import json as _json

    sent = _json.loads(route.calls.last.request.content)
    assert sent == {"ids": ["id1", "id2"]}
    assert route.calls.last.request.url.params["fields"] == "title"
    # bare-list payload normalized into items
    assert res.success is True
    assert len(res.items) == 2
    assert res.raw == {"data": [PAPER_1, PAPER_2]}


@pytest.mark.asyncio
async def test_paper_batch_empty_raises(provider):
    with pytest.raises(SearchProviderError, match=r"non-empty id list"):
        await provider.paper_batch([])


@pytest.mark.asyncio
@respx.mock
async def test_paper_citations_uses_citing_paper(provider):
    route = respx.get(f"{GRAPH}/paper/p1/citations").mock(
        return_value=httpx.Response(200, json=CITATIONS_RESPONSE)
    )
    res = await provider.paper_citations("p1", limit=50, offset=5)
    params = route.calls.last.request.url.params
    assert params["limit"] == "50"
    assert params["offset"] == "5"
    assert res.engine == "semanticscholar:citations"
    assert res.items[0].title == "Citing paper"
    assert res.items[0].url == "https://www.semanticscholar.org/paper/cite1"


@pytest.mark.asyncio
@respx.mock
async def test_paper_references_uses_cited_paper(provider):
    respx.get(f"{GRAPH}/paper/p1/references").mock(
        return_value=httpx.Response(200, json=REFERENCES_RESPONSE)
    )
    res = await provider.paper_references("p1")
    assert res.engine == "semanticscholar:references"
    assert res.items[0].title == "Cited paper"
    assert res.items[0].url == "https://www.semanticscholar.org/paper/ref1"


@pytest.mark.asyncio
@respx.mock
async def test_paper_authors(provider):
    route = respx.get(f"{GRAPH}/paper/p1/authors").mock(
        return_value=httpx.Response(200, json={"data": [AUTHOR_1]})
    )
    res = await provider.paper_authors("p1")
    # author default fields applied
    assert "name" in route.calls.last.request.url.params["fields"]
    assert res.items[0].title == "Oren Etzioni"
    assert res.items[0].url == AUTHOR_1["url"]


@pytest.mark.asyncio
@respx.mock
async def test_paper_match_helper(provider):
    respx.get(f"{GRAPH}/paper/search/match").mock(
        return_value=httpx.Response(200, json=MATCH_RESPONSE)
    )
    res = await provider.paper_match("Construction of the Literature Graph")
    assert res.engine == "semanticscholar:match"
    assert res.items[0].title == PAPER_1["title"]


@pytest.mark.asyncio
@respx.mock
async def test_paper_match_404_soft_fails(provider):
    respx.get(f"{GRAPH}/paper/search/match").mock(
        return_value=httpx.Response(404, json={"error": "Title match not found"})
    )
    res = await provider.paper_match("a title that does not exist")
    assert res.success is False


@pytest.mark.asyncio
@respx.mock
async def test_autocomplete_returns_matches_list(provider):
    route = respx.get(f"{GRAPH}/paper/autocomplete").mock(
        return_value=httpx.Response(200, json=AUTOCOMPLETE_RESPONSE)
    )
    out = await provider.autocomplete("sciber")
    assert route.calls.last.request.url.params["query"] == "sciber"
    assert isinstance(out, list)
    assert out[0]["title"] == "SciBERT"


@pytest.mark.asyncio
@respx.mock
async def test_snippet_search_helper(provider):
    respx.get(f"{GRAPH}/snippet/search").mock(
        return_value=httpx.Response(200, json=SNIPPET_RESPONSE)
    )
    res = await provider.snippet_search("graph neural networks", limit=3)
    assert res.engine == "semanticscholar:snippet"
    assert res.items[0].description == "graph neural networks excel at ..."


# ---------------------------------------------------------------------------
# Authors: details / batch / papers / search
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_author_details_returns_dict(provider):
    respx.get(f"{GRAPH}/author/1741101").mock(return_value=httpx.Response(200, json=AUTHOR_1))
    out = await provider.author("1741101")
    assert isinstance(out, dict)
    assert out["name"] == "Oren Etzioni"


@pytest.mark.asyncio
@respx.mock
async def test_author_batch_posts_ids(provider):
    route = respx.post(f"{GRAPH}/author/batch").mock(
        return_value=httpx.Response(200, json=[AUTHOR_1])
    )
    res = await provider.author_batch(["1741101"])
    import json as _json

    assert _json.loads(route.calls.last.request.content) == {"ids": ["1741101"]}
    assert res.items[0].title == "Oren Etzioni"


@pytest.mark.asyncio
@respx.mock
async def test_author_papers(provider):
    respx.get(f"{GRAPH}/author/1741101/papers").mock(
        return_value=httpx.Response(200, json={"data": [PAPER_1]})
    )
    res = await provider.author_papers("1741101")
    assert res.engine == "semanticscholar:author_papers"
    assert res.items[0].title == PAPER_1["title"]


@pytest.mark.asyncio
@respx.mock
async def test_author_search_clamps_limit(provider):
    route = respx.get(f"{GRAPH}/author/search").mock(
        return_value=httpx.Response(200, json=AUTHOR_SEARCH_RESPONSE)
    )
    res = await provider.author_search("etzioni", limit=5000)
    params = route.calls.last.request.url.params
    assert params["query"] == "etzioni"
    assert params["limit"] == "1000"  # clamped to author_search cap
    assert res.items[0].title == "Oren Etzioni"


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_recommend_papers_for_paper(provider):
    route = respx.get(f"{REC}/papers/forpaper/{PAPER_1['paperId']}").mock(
        return_value=httpx.Response(200, json=RECOMMEND_RESPONSE)
    )
    res = await provider.recommend_papers(PAPER_1["paperId"], limit=2, pool="all-cs")
    params = route.calls.last.request.url.params
    assert params["from"] == "all-cs"
    assert params["limit"] == "2"
    assert res.engine == "semanticscholar:recommendations"
    assert len(res.items) == 2
    assert res.raw["recommendedPapers"][0]["title"] == PAPER_1["title"]


@pytest.mark.asyncio
@respx.mock
async def test_recommend_papers_limit_clamped_to_500(provider):
    route = respx.get(f"{REC}/papers/forpaper/p1").mock(
        return_value=httpx.Response(200, json=RECOMMEND_RESPONSE)
    )
    await provider.recommend_papers("p1", limit=9999)
    assert route.calls.last.request.url.params["limit"] == "500"


@pytest.mark.asyncio
@respx.mock
async def test_recommend_from_examples_posts_lists(provider):
    route = respx.post(f"{REC}/papers/").mock(
        return_value=httpx.Response(200, json=RECOMMEND_RESPONSE)
    )
    res = await provider.recommend_from_examples(["pos1", "pos2"], ["neg1"], limit=10)
    import json as _json

    body = _json.loads(route.calls.last.request.content)
    assert body == {"positivePaperIds": ["pos1", "pos2"], "negativePaperIds": ["neg1"]}
    assert route.calls.last.request.url.params["limit"] == "10"
    assert len(res.items) == 2


@pytest.mark.asyncio
async def test_recommend_from_examples_requires_positive(provider):
    with pytest.raises(SearchProviderError, match=r"at least one positive"):
        await provider.recommend_from_examples([])


# ---------------------------------------------------------------------------
# Datasets API
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_list_releases(provider):
    respx.get(f"{DATASETS}/release/").mock(
        return_value=httpx.Response(200, json=RELEASES_RESPONSE)
    )
    out = await provider.list_releases()
    assert out == RELEASES_RESPONSE


@pytest.mark.asyncio
@respx.mock
async def test_get_release(provider):
    respx.get(f"{DATASETS}/release/latest").mock(
        return_value=httpx.Response(200, json=RELEASE_META_RESPONSE)
    )
    out = await provider.get_release("latest")
    assert out["release_id"] == "2023-03-28"
    assert out["datasets"][0]["name"] == "abstracts"


@pytest.mark.asyncio
@respx.mock
async def test_get_dataset(provider):
    respx.get(f"{DATASETS}/release/latest/dataset/abstracts").mock(
        return_value=httpx.Response(200, json=DATASET_META_RESPONSE)
    )
    out = await provider.get_dataset("abstracts")
    assert out["name"] == "abstracts"
    assert out["files"][0].endswith("0001.gz")


@pytest.mark.asyncio
@respx.mock
async def test_get_dataset_diffs(provider):
    respx.get(f"{DATASETS}/diffs/2023-08-01/to/latest/papers").mock(
        return_value=httpx.Response(200, json=DIFFS_RESPONSE)
    )
    out = await provider.get_dataset_diffs("2023-08-01", "latest", "papers")
    assert out["diffs"][0]["update_files"] == ["http://x/u1"]


# ---------------------------------------------------------------------------
# batch_search (client-side fan-out)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_batch_search_strings(provider):
    respx.get(f"{GRAPH}/paper/search").mock(
        return_value=httpx.Response(200, json=RELEVANCE_RESPONSE)
    )
    out = await provider.batch_search(["q1", "q2", "q3"], count=5)
    assert len(out) == 3
    assert all(r.success for r in out)
    assert [r.query for r in out] == ["q1", "q2", "q3"]


@pytest.mark.asyncio
@respx.mock
async def test_batch_search_dicts_with_per_query_type(provider):
    respx.get(f"{GRAPH}/snippet/search").mock(
        return_value=httpx.Response(200, json=SNIPPET_RESPONSE)
    )
    out = await provider.batch_search(
        [{"query": "q1", "search_type": "snippet"}, {"query": "q2", "search_type": "snippet"}]
    )
    assert len(out) == 2
    assert all(r.engine == "semanticscholar:snippet" for r in out)


@pytest.mark.asyncio
async def test_batch_search_empty_raises(provider):
    with pytest.raises(SearchProviderError, match=r"non-empty list"):
        await provider.batch_search([])


@pytest.mark.asyncio
async def test_batch_search_dict_without_query_raises(provider):
    with pytest.raises(SearchProviderError, match=r"must include a 'query'"):
        await provider.batch_search([{"search_type": "relevance"}])


# ---------------------------------------------------------------------------
# Retry / backoff / auth / health / close
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_retry_on_429_then_success(provider):
    route = respx.get(f"{GRAPH}/paper/search").mock(
        side_effect=[
            httpx.Response(429, json={"message": "Too Many Requests"}),
            httpx.Response(200, json=RELEVANCE_RESPONSE),
        ]
    )
    res = await provider.web_search("x")
    assert res.success is True
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_retry_on_transport_error_then_success(provider):
    route = respx.get(f"{GRAPH}/paper/search").mock(
        side_effect=[httpx.ConnectError("boom"), httpx.Response(200, json=RELEVANCE_RESPONSE)]
    )
    res = await provider.web_search("x")
    assert res.success is True
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_401_raises_with_status(provider):
    respx.get(f"{GRAPH}/paper/search").mock(
        return_value=httpx.Response(401, json={"message": "bad key"})
    )
    with pytest.raises(SearchProviderError) as exc:
        await provider.web_search("x")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
@respx.mock
async def test_403_raises_with_status(provider):
    respx.get(f"{GRAPH}/paper/search").mock(return_value=httpx.Response(403))
    with pytest.raises(SearchProviderError) as exc:
        await provider.web_search("x")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
@respx.mock
async def test_persistent_5xx_soft_fails(provider):
    route = respx.get(f"{GRAPH}/paper/search").mock(return_value=httpx.Response(503))
    res = await provider.web_search("x")
    assert res.success is False
    assert "503" in res.error
    assert route.call_count == provider._max_retries


@pytest.mark.asyncio
@respx.mock
async def test_persistent_transport_error_raises(provider):
    respx.get(f"{GRAPH}/paper/search").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(SearchProviderError, match=r"Transport error"):
        await provider.web_search("x")


@pytest.mark.asyncio
@respx.mock
async def test_health_check_ok(provider):
    respx.get(f"{GRAPH}/paper/autocomplete").mock(
        return_value=httpx.Response(200, json=AUTOCOMPLETE_RESPONSE)
    )
    assert await provider.health_check() is True


@pytest.mark.asyncio
@respx.mock
async def test_health_check_fails_on_auth(provider):
    respx.get(f"{GRAPH}/paper/autocomplete").mock(return_value=httpx.Response(403))
    assert await provider.health_check() is False


@pytest.mark.asyncio
async def test_close_is_idempotent(provider):
    await provider.close()  # never opened
    provider._get_client()
    await provider.close()
    await provider.close()
    assert provider._client is None


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------
def test_coerce_int_variants():
    assert _coerce_int("15117") == 15117
    assert _coerce_int("1,234") == 1234
    assert _coerce_int(42) == 42
    assert _coerce_int(3.9) == 3
    assert _coerce_int(True) is None
    assert _coerce_int("none") is None
    assert _coerce_int(None) is None


def test_abstract_or_tldr_prefers_abstract():
    assert _abstract_or_tldr({"abstract": "A", "tldr": {"text": "B"}}) == "A"
    assert _abstract_or_tldr({"tldr": {"text": "B"}}) == "B"
    assert _abstract_or_tldr({}) == ""


def test_venue_label_fallback_chain():
    assert _venue_label({"venue": "V"}) == "V"
    assert _venue_label({"journal": {"name": "J"}}) == "J"
    assert _venue_label({"publicationVenue": {"name": "PV"}}) == "PV"
    assert _venue_label({}) == ""


# ---------------------------------------------------------------------------
# Manager wiring
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_manager_loads_keyless_semanticscholar():
    mgr = SearchProviderManager(Config(defaults={"search_providers": {"semanticscholar": {}}}))
    await mgr.initialize()
    assert "semanticscholar" in mgr.get_available_search_providers()
    prov = mgr.get_search_provider("semanticscholar")
    assert isinstance(prov, SemanticScholarSearchProvider)
    assert prov._api_key is None
    await mgr.close_all()


@pytest.mark.asyncio
async def test_manager_loads_s2_alias_keyless():
    mgr = SearchProviderManager(Config(defaults={"search_providers": {"s2": {"type": "s2"}}}))
    await mgr.initialize()
    assert "s2" in mgr.get_available_search_providers()
    assert isinstance(mgr.get_search_provider("s2"), SemanticScholarSearchProvider)
    await mgr.close_all()


@pytest.mark.asyncio
async def test_manager_loads_keyed_from_env(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "env-key")
    mgr = SearchProviderManager(Config(defaults={"search_providers": {"semanticscholar": {}}}))
    await mgr.initialize()
    prov = mgr.get_search_provider("semanticscholar")
    assert prov._api_key == "env-key"
    await mgr.close_all()
