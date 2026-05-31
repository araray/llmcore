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

## 10. Validation / how to verify it works

1. **Offline (no account):** run the unit tests — they intercept all HTTP with
   `respx` and assert exact endpoints/payloads:
   ```bash
   pip install "llmcore[test,brightdata,serper]"
   pytest tests/search/ -q
   ```
2. **Credential/connectivity smoke test:**
   ```python
   llm = await LLMCore.create()
   ok = await llm.get_search_provider().health_check()   # True ⇒ token valid & reachable
   ```
3. **End‑to‑end:** run `examples/brightdata_search_example.py` (with
   `BRIGHTDATA_API_TOKEN` + zone env vars) or `examples/serper_search_example.py`
   (with `SERPER_API_KEY`). Both perform real, billable calls.
4. **Config wizard:** the confy‑curator schema exposes "Search Provider: Bright
   Data" and "Search Provider: Serper.dev" sections for guided setup.
