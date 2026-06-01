# LLMCore Search Providers — Usage

Practical, copy‑pasteable usage for `llmcore.search`. For the design rationale,
capability matrix, and configuration reference, see **README.md** in this folder.

All snippets assume an initialized instance:

```python
from llmcore import LLMCore
llm = await LLMCore.create()          # search wiring is automatic & optional
# ... use llm ...
await llm.close()
```

---

## 1. Setup

```bash
pip install "llmcore[brightdata]"     # pulls httpx; the provider needs nothing else
export BRIGHTDATA_API_TOKEN="bd_..."  # a single token authenticates every endpoint
```

Configure zone names once (created in the Bright Data control panel,
<https://brightdata.com/cp/zones>). Either in your config file:

```toml
# ~/.config/llmcore/config.toml
[llmcore]
# default_search_provider = "brightdata"   # optional; auto-adopted if it's the only one

[search_providers.brightdata]
serp_zone     = "my_serp_zone"
unlocker_zone = "my_unlocker_zone"
# default_engine = "google"   # google | bing | yandex
# timeout = 60
```

…or inline at construction time (handy for tests / scripts):

```python
llm = await LLMCore.create(config_overrides={
    "search_providers": {
        "brightdata": {"serp_zone": "my_serp_zone", "unlocker_zone": "my_unlocker_zone"}
    }
})
```

> Datasets and Discover need only the token — no zone required.

---

## 2. Web search (SERP)

```python
res = await llm.web_search(
    "best vector databases 2026",
    count=10,           # desired organic results
    country="US",       # ISO-3166 alpha-2 (geolocation)
    language="en",      # ISO-639-1
    device="desktop",   # or "mobile"
    engine="google",    # google | bing | yandex (defaults to provider's default_engine)
)

if res.success:
    print(f"{len(res.items)} results, total≈{res.total_results}, {res.elapsed_ms():.0f} ms")
    for item in res.items:
        print(item.position, item.title, item.url, item.displayed_url)
else:
    print("error:", res.error)
```

Engine‑specific extras pass through as keyword args (Google): `safe_search=True`,
`time_range="w"` (qdr: day/week/month/year). To get **raw HTML** instead of parsed
JSON, pass `parse=False` (then `res.items == []` and the HTML is in `res.raw`).

### Async (non‑blocking) mode

For the SERP/Unlocker async pipeline (trigger + poll), pass `mode="async"`:

```python
res = await llm.web_search("python 3.13 release notes", count=10, mode="async")
```

Async mode triggers `POST /unblocker/req` and polls `GET /unblocker/get_result`
every `poll_interval` seconds up to `poll_timeout`. (Sync mode is usually faster
for a single query; async shines when you fire many concurrently.)

---

## 3. Scrape a URL (Web Unlocker)

```python
page = await llm.scrape_url(
    "https://example.com/product/42",
    response_format="raw",   # "raw" (HTML/text) or "json" (provider-structured)
    country="DE",            # exit-node country (uppercased automatically)
)
if page.success:
    print(page.root_domain, page.content_char_size)
    html = page.content
else:
    print("scrape failed:", page.error)   # non-200 returns success=False, not an exception
```

---

## 4. Discover (AI‑relevance‑ranked search)

```python
disc = await llm.discover(
    "agentic AI frameworks",
    intent="compare open-source multi-agent orchestration libraries",
    include_content=True,    # include full-page markdown in each item.content
    count=5,
    country="us",
    timeout=60, poll_interval=2,
)
for item in disc.items:                    # ranked by relevance_score
    print(item.relevance_score, item.title, item.url)
    if item.content:
        print(item.content[:200])
```

`discover()` triggers `POST /discover`, then polls `GET /discover?task_id=` until
the task is `done` (or raises on `failed` / timeout).

---

## 5. Datasets (structured records)

```python
# Discover what's available
datasets = await llm.list_datasets()
for ds in datasets[:10]:
    print(ds.id, ds.name, ds.size)

# Inspect a dataset's filterable fields
meta = await llm.get_dataset_metadata("gd_l1viktl72bvl7bjuj0")
print(meta.field_names())

# Filter + wait + download in one call (the convenience method)
snapshot = await llm.dataset_search(
    "gd_l1viktl72bvl7bjuj0",
    {"name": "industry", "operator": "=", "value": "Technology"},
    records_limit=100,
    format="jsonl",      # json | jsonl | csv
    timeout=300, poll_interval=5,
)
if snapshot.success:
    print(snapshot.record_count, "records")
    for row in (snapshot.records or [])[:3]:
        print(row)
```

Compound filters are supported (provider schema):

```python
filt = {
    "operator": "and",
    "filters": [
        {"name": "industry",  "operator": "=", "value": "Technology"},
        {"name": "followers", "operator": ">", "value": 10000},
    ],
}
```

### Manual, two‑phase dataset workflow

If you want to do other work while the snapshot builds, drop to the provider and
split filter / poll / download:

```python
provider = llm.get_search_provider()                     # BaseSearchProvider
snap = await provider.dataset_filter("gd_...", filt, records_limit=500)
print("snapshot:", snap.snapshot_id, snap.status)        # returns immediately
# ... do other work ...
status = await provider.dataset_status(snap.snapshot_id) # "scheduled"|"building"|"ready"|"failed"
ready = await provider.dataset_download(snap.snapshot_id, format="jsonl")
print(ready.record_count)
```

---

## 6. Choosing / inspecting providers

```python
llm.get_available_search_providers()      # -> ["brightdata"] (or [] if unconfigured)
p = llm.get_search_provider()             # default instance
p = llm.get_search_provider("brightdata") # by name
sorted(p.get_capabilities())              # {"web_search","scrape","discover","dataset_search"}
p.supports("crawl")                        # False
await p.health_check()                     # True/False (GET /zone/get_active_zones)
```

### Multiple Bright Data instances

Use distinct section names plus an explicit `type`:

```toml
[llmcore]
default_search_provider = "bd_us"

[search_providers.bd_us]
type = "brightdata"
serp_zone = "serp_us"

[search_providers.bd_eu]
type = "brightdata"
serp_zone = "serp_eu"
```

```python
await llm.web_search("q", provider="bd_eu")
```

---

## 7. Error handling

```python
from llmcore import SearchProviderError, ConfigError

try:
    res = await llm.web_search("q")
    if not res.success:
        ...  # recoverable: inspect res.error
except ConfigError:
    ...      # no search provider configured / none requested and no default
except SearchProviderError as e:
    ...      # transport/auth/protocol fault; e.status_code may be set
```

Rules of thumb:
- **Recoverable** issues (non‑200 scrape, empty results) → `success=False`, no exception.
- **Hard** faults (`401/403`, transport errors after retries, missing `task_id`,
  missing zone, async timeout) → `SearchProviderError`.
- **Configuration** problems (no provider, ambiguous default) → `ConfigError`.

---

## 8. Runtime controls

```python
llm.set_raw_payload_logging(True)   # also logs Bright Data request bodies at DEBUG
# (requires the llmcore logger at DEBUG level)
```

`reload_config()` rebuilds the search manager along with the rest of LLMCore, so
edits to `[search_providers.*]` take effect on reload.

---

## 9. Serper.dev (web search, verticals, batch, scrape)

Serper is a fast/cheap Google Search API. It authenticates with an `X-API-KEY`
header, needs **no zones**, and supports search verticals plus batched array
queries. Capabilities: `web_search`, `batch_search`, `scrape`.

### Setup

```bash
pip install "llmcore[serper]"
export SERPER_API_KEY="your-serper-key"
```

```toml
# ~/.config/llmcore/config.toml
[search_providers.serper]
# api_key via SERPER_API_KEY (recommended)
default_search_type = "search"   # search|news|images|videos|shopping|scholar|patents|maps|places|autocomplete
```

If Serper is your only configured search provider it is auto-adopted as the
default and you can omit `provider="serper"`.

### Web search + verticals

The vertical is chosen with `search_type`; time filtering with `tbs` (raw, e.g.
`"qdr:m"`) or `time_range` (shorthand: `"d"`/`"w"`/`"m"`/`"y"`). `device`,
`engine`, and `mode` are accepted for cross-provider compatibility but ignored.

```python
# standard web search
res = await llm.web_search("apple inc", provider="serper", count=10, country="us", language="en")
for it in res.items:
    print(it.position, it.title, it.url)

# Google News for the past day
news = await llm.web_search("apple inc", provider="serper", search_type="news", time_range="d")

# Scholar / patents, paginated
scholar = await llm.web_search("graph neural networks", provider="serper",
                               search_type="scholar", page=2, tbs="qdr:m")
```

SERP extras Serper returns — `knowledgeGraph`, `peopleAlsoAsk`,
`relatedSearches`, `sitelinks`, `attributes` — are preserved verbatim on
`res.raw` (the normalized `items` carry title/url/snippet/position):

```python
kg = res.raw.get("knowledgeGraph", {})
paa = res.raw.get("peopleAlsoAsk", [])
```

### Batched queries (one request, many searches)

Serper accepts a JSON array of query objects; `batch_web_search` returns one
`WebSearchResult` per input, in order.

```python
# list of strings (inherit count/country/language/time_range)
results = await llm.batch_web_search(
    ["apple inc", "tesla inc", "google inc"],
    provider="serper", country="us", time_range="m",
)

# list of ready Serper query objects (full per-query control)
results = await llm.batch_web_search(
    [
        {"q": "apple inc", "tbs": "qdr:m", "page": 2},
        {"q": "tesla inc", "tbs": "qdr:d"},
    ],
    provider="serper", search_type="patents",
)
for r in results:
    print(r.query, "->", len(r.items), "items")
```

### Scrape (markdown / json)

Serper scraping uses the separate `scrape.serper.dev` host and returns
structured JSON. `response_format="markdown"` (default) requests and extracts
markdown into `content`; the full payload (text/markdown/metadata/jsonld) is
always on `raw`.

```python
page = await llm.scrape_url("https://example.com/article", provider="serper")
print(page.response_format, page.content_char_size)   # "markdown", <n>
md = page.content                                      # extracted markdown
meta = page.raw.get("metadata", {})                    # full payload preserved

# full structured payload instead of extracted markdown
raw = await llm.scrape_url("https://example.com", provider="serper", response_format="json")
```

> Note: Serper exposes no free health/balance endpoint, so
> `provider.health_check()` issues a minimal `/search` (`num=1`) and consumes
> **one credit**.

### Direct provider access (verticals not on the unified API)

```python
serper = llm.get_search_provider("serper")
res = await serper.web_search("nikon z9", search_type="shopping")
batch = await serper.batch_search(["a", "b"], search_type="images")
```

---

## 10. SerpApi (100+ engines, async, archive, free health check)

SerpApi is a real-time **meta-SERP** API: a single `GET /search` endpoint
scrapes 100+ search engines/verticals selected with the `engine` parameter. It
authenticates with an `api_key` **query parameter** (not a header), needs **no
zones**, and exposes a **free** Account API used for health checks.
Capabilities: `web_search`, `batch_search` (client-side fan-out).

### Setup

```bash
pip install "llmcore[serpapi]"
export SERPAPI_API_KEY="your-serpapi-key"   # SERPAPI_KEY is also honored
```

```toml
# ~/.config/llmcore/config.toml
[search_providers.serpapi]
# api_key via SERPAPI_API_KEY (recommended)
default_engine = "google"     # any SerpApi engine string (not enumerated; see docs)
default_output = "json"       # json | html
# no_cache = false            # force-fetch (cached reads are free); dropped in async mode
# zero_trace = false          # Enterprise ZeroTrace mode
# json_restrictor = "organic_results[].{title,link}"   # trim responses
# poll_interval = 2           # async: seconds between Search-Archive polls
# poll_timeout = 60           # async: max seconds to wait
# max_concurrency = 5         # batch_search fan-out width
```

If SerpApi is your only configured search provider it is auto-adopted as the
default and you can omit `provider="serpapi"`.

### Web search across engines

The engine/vertical is chosen with `engine`. Cross-provider arguments map onto
SerpApi parameters: `query`→the engine's query param (`q`/`query`/`p`/`text`/
`search_query`/`term`/`_nkw`/`find_desc`), `count`→`num` (best-effort — the
standard `google` web engine ignores it, use `start`), `country`→`gl`,
`language`→`hl`, `device`→`device`. Any other SerpApi parameter (e.g.
`location`, `uule`, `lat`/`lon`, `google_domain`, `tbm`, `tbs`, `safe`, `start`,
`no_cache`, `output`, `json_restrictor`) is passed through verbatim.

```python
# standard Google web search
res = await llm.web_search("apple inc", provider="serpapi", count=10, country="us", language="en")
for it in res.items:
    print(it.position, it.title, it.url)

# Google News
news = await llm.web_search("apple inc", provider="serpapi", engine="google_news")

# Bing, mobile, geolocated
bing = await llm.web_search("coffee", provider="serpapi", engine="bing",
                            device="mobile", location="Austin, TX")

# Google Scholar, paginated (this engine DOES honor num)
scholar = await llm.web_search("graph neural networks", provider="serpapi",
                               engine="google_scholar", num=20, start=20)
```

The full SerpApi payload — `knowledge_graph`, `answer_box`, `related_questions`,
`local_results`, `serpapi_pagination`, and every engine-specific block — is
preserved verbatim on `res.raw`; normalized `items` carry
title/url/snippet/position (resolved from the engine's primary result array:
`organic_results`, `news_results`, `images_results`, `shopping_results`, …):

```python
kg  = res.raw.get("knowledge_graph", {})
nxt = res.raw.get("serpapi_pagination", {}).get("next")
```

### Async (submit + poll the Search Archive)

`mode="async"` submits the search (`async=true`), extracts
`search_metadata.id`, then polls `GET /searches/{id}` until the status is
`Success`/`Error`. (`async` and `no_cache` are mutually exclusive; `no_cache`
is dropped automatically.)

```python
res = await llm.web_search("apple inc", provider="serpapi", mode="async")
```

### Batched queries (client-side fan-out)

SerpApi has **no** server-side batch endpoint, so `batch_web_search` issues
concurrent requests bounded by `max_concurrency` and returns one
`WebSearchResult` per input, in order. Each query consumes one credit.

```python
# strings (inherit count/country/language; engine via search_type)
results = await llm.batch_web_search(
    ["apple inc", "tesla inc", "google inc"],
    provider="serpapi", country="us", search_type="google_news",
)

# ready SerpApi parameter dicts (full per-query control, mixed engines)
results = await llm.batch_web_search(
    [
        {"engine": "google", "q": "apple inc", "tbm": "nws"},
        {"engine": "google_scholar", "q": "diffusion models", "num": 20},
    ],
    provider="serpapi",
)
for r in results:
    print(r.query, "->", len(r.items), "items")
```

### Direct provider access (archive, account, locations, raw search)

```python
serpapi = llm.get_search_provider("serpapi")

# Faithful low-level pass-through (engine-agnostic; mirrors the official SDK)
res = await serpapi.search({"engine": "google_jobs", "q": "python developer", "location": "Remote"})

# Re-fetch a past search by id (cached archive reads are free)
again = await serpapi.search_archive("64e9...id...")

# Plan / quota (FREE — does not consume a credit)
acct = await serpapi.account()
print(acct["plan_name"], acct["total_searches_left"])

# Canonical locations for the `location` parameter
locs = await serpapi.locations(q="Austin", limit=3)
```

> Note: unlike Serper, SerpApi exposes a **free** `/account.json` endpoint, so
> `provider.health_check()` consumes **zero** credits.

---

## 11. Semantic Scholar (academic papers; **no API key required**)

[Semantic Scholar](https://www.semanticscholar.org/product/api) is a free,
AI‑powered academic search engine over 200M+ papers. The provider wraps the
**Academic Graph**, **Recommendations**, and **Datasets** APIs. It advertises
`web_search` and `batch_search`; the recommendations/datasets/paper‑graph
endpoints are exposed as provider‑specific methods.

The S2 **API key is optional** — the provider works keyless against the shared
public pool (rate‑limited; a few thousand requests per 5 minutes shared across
all anonymous clients). Set `SEMANTIC_SCHOLAR_API_KEY` for a dedicated limit.

### Setup

```toml
# ~/.config/llmcore/config.toml
[search_providers.semanticscholar]
# api_key via SEMANTIC_SCHOLAR_API_KEY (OPTIONAL; S2_API_KEY also honored)
# default_search_type = "relevance"   # relevance | bulk | match | snippet
# default_fields = "title,abstract,url,venue,year,authors,citationCount,externalIds,openAccessPdf"
# timeout = 30
# max_retries = 3                      # S2 requires exponential backoff (applied automatically)
# max_concurrency = 1                  # batch_search fan-out; raise only with a key
# min_request_interval = 0             # optional request spacing (seconds); 1 ≈ 1 RPS
```

> The packaged `default_config.toml` ships this block **commented out** so the
> search subsystem stays empty by default. Uncomment the section header to
> enable it (no key needed).

### Paper search (relevance / bulk / match / snippet)

```python
s2 = llm.get_search_provider("semanticscholar")

# relevance search (default) — rich filters pass straight through
res = await llm.web_search(
    "graph neural networks", provider="semanticscholar", count=20,
    year="2018-2024", fieldsOfStudy="Computer Science", openAccessPdf=True,
)
for item in res.items:
    print(item.title, "—", item.url)        # item.description = abstract or TLDR
print("approx total:", res.total_results)

# bulk search (large result sets; paginate via the continuation token)
page1 = await llm.web_search("transformers", provider="semanticscholar",
                             search_type="bulk", count=1000, sort="citationCount:desc")
token = page1.raw.get("token")              # feed back as token=... for the next page

# title match — resolve a free-text title/citation to one canonical paper
match = await s2.paper_match("Attention is all you need")

# snippet search — text passages ideal for RAG grounding (item.description = passage)
snips = await s2.snippet_search("retrieval augmented generation", limit=10)
```

`count` maps to S2's `limit` (clamped per endpoint: 100 relevance, 1000
bulk/snippet, 1 match). `country` / `language` / `device` / `engine` / `mode`
are accepted for cross‑provider compatibility but ignored (academic search is
not geolocated and is always synchronous).

### Paper & citation graph, authors, autocomplete

```python
paper   = await s2.paper("ARXIV:1706.03762", fields="title,abstract,tldr,citationCount")
papers  = await s2.paper_batch(["DOI:10.1038/nature14539", "CorpusId:215416146"])  # ≤500 ids
citing  = await s2.paper_citations(paper["paperId"], limit=100)   # items ← citingPaper
refs    = await s2.paper_references(paper["paperId"], limit=100)  # items ← citedPaper
authors = await s2.paper_authors(paper["paperId"])

who     = await s2.author("1741101", fields="name,affiliations,hIndex,paperCount")
hits    = await s2.author_search("Yoshua Bengio")
byauth  = await s2.author_papers("1741101", limit=50)
sugg    = await s2.autocomplete("attention is all")   # [{id, title, authorsYear}, ...]
```

### Recommendations

```python
# from a single seed paper (pool: "recent" | "all-cs")
recs = await s2.recommend_papers("ARXIV:1706.03762", limit=20, pool="recent")

# from positive/negative example lists
recs = await s2.recommend_from_examples(
    positive_paper_ids=["CorpusId:215416146", "ARXIV:1810.04805"],
    negative_paper_ids=["ARXIV:1805.02262"], limit=20,
)
```

### Datasets (bulk‑corpus download links)

```python
releases = await s2.list_releases()                      # ["2023-03-14", ...]
release  = await s2.get_release("latest")                # {release_id, README, datasets:[...]}
abstracts = await s2.get_dataset("abstracts")            # {name, README, files:[presigned URLs]}
diffs    = await s2.get_dataset_diffs("2023-08-01", "latest", "papers")
```

> The Datasets API returns pre‑signed S3 URLs for full‑corpus partitions (e.g.
> "100M records in 30 × 1.8 GB files"); these helpers return that metadata/links,
> they do not stream records in‑process. For local querying at higher throughput
> than the API allows, download the partitions and load them yourself.

### Connectivity check

```python
ok = await s2.health_check()   # tiny autocomplete probe; True ⇒ reachable
```

---

## 12. Validation / how to verify it works

1. **Offline (no account):** run the unit tests — they intercept all HTTP with
   `respx` and assert exact endpoints/payloads:
   ```bash
   pip install "llmcore[test,brightdata,serper,serpapi,semanticscholar]"
   pytest tests/search/ -q
   ```
2. **Credential/connectivity smoke test:**
   ```python
   llm = await LLMCore.create()
   ok = await llm.get_search_provider().health_check()   # True ⇒ token valid & reachable
   ```
3. **End‑to‑end:** run `examples/brightdata_search_example.py` (with
   `BRIGHTDATA_API_TOKEN` + zone env vars), `examples/serper_search_example.py`
   (with `SERPER_API_KEY`), `examples/serpapi_search_example.py` (with
   `SERPAPI_API_KEY`), or `examples/semanticscholar_search_example.py` (which
   runs **without** a key — `SEMANTIC_SCHOLAR_API_KEY` is optional).
4. **Config wizard:** the confy‑curator schema exposes "Search Provider: Bright
   Data", "Search Provider: Serper.dev", "Search Provider: SerpApi" and "Search
   Provider: Semantic Scholar" sections for guided setup.
