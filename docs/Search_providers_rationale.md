# LLMCore Search Providers

**Web / data search for LLMCore — used "just like" the LLM providers.**

This subsystem (`llmcore.search`) adds a second family of pluggable backends
alongside the existing LLM providers (`llmcore.providers`). Where an LLM provider
turns a prompt into a completion, a **search provider** turns a query into search
results, scraped pages, AI‑ranked discoveries, or structured dataset records.

The bundled providers are **Bright Data** and **Serper.dev**.

### Providers at a glance

| Provider       | Type key   | Capabilities                                   | Auth header     | Notable strengths                                  |
| -------------- | ---------- | ---------------------------------------------- | --------------- | -------------------------------------------------- |
| **Bright Data**| `brightdata` | `web_search`, `scrape`, `discover`, `dataset_search` | `Authorization: Bearer` | Web Unlocker, AI Discover, dataset marketplace, async SERP |
| **Serper.dev** | `serper`   | `web_search`, `batch_search`, `scrape`         | `X-API-KEY`     | Fast/cheap Google SERP, verticals (news/scholar/patents/…), **batched** array queries |
| **SerpApi**    | `serpapi`  | `web_search`, `batch_search`                   | *(none — `api_key` **query param**)* | 100+ engines via one `engine` param, async + Search Archive, **free** Account-API health check |
| **Semantic Scholar** | `semanticscholar` | `web_search`, `batch_search` | *(optional — `x-api-key`)* | 200M+ papers; relevance/bulk/title/snippet search; **no key required**; recommendations, citation graph, batch lookups, bulk datasets |

All four return the same provider-agnostic result types, so consumer code that
calls `llm.web_search(...)` / `llm.scrape_url(...)` works against any backend.

---

## 1. Why a parallel subsystem (design rationale)

LLMCore's core invariant is to be a *clean, general‑purpose framework*: each
interface stays cohesive, and the library defines protocols and delegates rather
than coupling to a vendor.

A search provider has **no** concept of `chat_completion`, token counting, or a
context window. Forcing Bright Data into `BaseProvider` (the LLM contract) would
mean stubbing out ~9 abstract methods and would break the Liskov substitution
expectation for every consumer that iterates LLM providers. So search providers
get their own minimal contract, `BaseSearchProvider`, and their own loader,
`SearchProviderManager` — mirroring the LLM side one‑for‑one:

| LLM side (`llmcore.providers`) | Search side (`llmcore.search`)         |
| ------------------------------ | --------------------------------------- |
| `BaseProvider`                 | `BaseSearchProvider`                    |
| `ProviderManager`              | `SearchProviderManager`                 |
| `[providers.<name>]`           | `[search_providers.<name>]`             |
| `llmcore.default_provider`     | `llmcore.default_search_provider`       |
| `ProviderError`                | `SearchProviderError`                   |
| `llm.chat(...)`                | `llm.web_search(...)`, `llm.scrape_url(...)`, `llm.discover(...)`, `llm.dataset_search(...)` |

**Key difference:** search is **optional**. The LLM `ProviderManager` *requires*
a working default provider and raises if none loads. `SearchProviderManager`
happily loads **zero** providers; existing deployments that never configure
`[search_providers]` are completely unaffected, and the search methods raise a
clear `ConfigError` only if you actually call them.

### Native HTTP, no vendor SDK

The Bright Data provider talks to the Bright Data REST API over `httpx` directly
— exactly like the native Gemini/Kimi/DeepInfra LLM providers talk to their
vendors. This keeps llmcore self‑contained (no `brightdata-sdk` dependency) and
keeps the result models provider‑agnostic, so a future second search provider
can return the same `WebSearchResult` / `ScrapeResult` / `DiscoverResult` types.

---

## 2. Capability matrix

Capabilities are declared per provider via `get_capabilities()` and can be probed
with `provider.supports("web_search")`. Unsupported operations raise
`NotImplementedError`. The available capability values are `web_search`,
`batch_search`, `scrape`, `discover`, `dataset_search`, and `crawl`.

**Bright Data** (`brightdata`):

| Capability        | `LLMCore` method            | Bright Data product      | Endpoint(s)                                                                   |
| ----------------- | --------------------------- | ------------------------ | ----------------------------------------------------------------------------- |
| `web_search`      | `web_search()`              | SERP API                 | `POST /request` (sync) · `POST /unblocker/req` + `GET /unblocker/get_result` (async) |
| `scrape`          | `scrape_url()`              | Web Unlocker             | `POST /request` (sync) · async pair                                           |
| `discover`        | `discover()`                | Discover API             | `POST /discover` + `GET /discover?task_id=`                                   |
| `dataset_search`  | `dataset_search()`, `list_datasets()`, `get_dataset_metadata()` | Dataset Marketplace | `POST /datasets/filter` · `GET /datasets/snapshots/{id}` · `…/download` · `GET /datasets/list` · `GET /datasets/{id}/metadata` |
| `crawl`           | *(not implemented)*         | —                        | Reserved on the base class; raises `NotImplementedError`.                     |

Bright Data authenticates with a Bearer token; health via `GET /zone/get_active_zones`.

**Serper.dev** (`serper`):

| Capability       | `LLMCore` method        | Serper surface              | Endpoint(s)                                                       |
| ---------------- | ----------------------- | --------------------------- | ---------------------------------------------------------------- |
| `web_search`     | `web_search()`          | Google SERP + verticals     | `POST /{type}` (`type` ∈ search/news/images/videos/shopping/scholar/patents/maps/places/autocomplete) |
| `batch_search`   | `batch_web_search()`    | Batched array queries       | `POST /{type}` with a JSON **array** of query objects            |
| `scrape`         | `scrape_url()`          | Scrape (markdown/text/json) | `POST https://scrape.serper.dev` `{url, includeMarkdown}`        |

Serper authenticates with an `X-API-KEY` header. It has no Discover/dataset APIs,
so those capabilities are not advertised. The select-a-vertical option is passed
as `search_type=` to `web_search()`/`batch_web_search()`; time filtering via
`tbs=` or `time_range=` (shorthand for `qdr:<x>`). Serper's full payload —
`knowledgeGraph`, `peopleAlsoAsk`, `relatedSearches`, and vertical-specific
fields — is preserved on `WebSearchResult.raw`. (Serper exposes no free health
endpoint, so `health_check()` issues a minimal 1-credit search.)

**SerpApi** (`serpapi`):

| Capability       | `LLMCore` method        | SerpApi surface                 | Endpoint(s)                                                          |
| ---------------- | ----------------------- | ------------------------------- | ------------------------------------------------------------------- |
| `web_search`     | `web_search()`          | 100+ engines via `engine=`      | `GET /search` (sync) · `GET /search?async=true` + `GET /searches/{id}` (async) |
| `batch_search`   | `batch_web_search()`    | Client-side concurrent fan-out  | N× `GET /search` bounded by `max_concurrency` (no server-side batch endpoint) |

SerpApi authenticates with an `api_key` **query parameter** (not a header). The
engine/vertical is selected with `engine=` (free-form string — 100+ engines:
`google`, `bing`, `baidu`, `duckduckgo`, `yahoo`, `yandex`, `google_news`,
`google_images`, `google_shopping`, `google_scholar`, `youtube`, `amazon`,
`ebay`, `walmart`, `google_maps`, …), so it is **not** enumerated as an enum. It
has no arbitrary-URL unlocker, AI-Discover, or dataset marketplace, so `scrape`,
`discover` and `dataset_search` are not advertised. `count` maps to `num`
(best-effort — the standard `google` web engine ignores it, use `start`);
`country`→`gl`, `language`→`hl`. The full payload — `knowledge_graph`,
`answer_box`, `related_questions`, the engine-specific result arrays, and
`serpapi_pagination` — is preserved on `WebSearchResult.raw`. Provider-specific
extras: `search()` (raw pass-through), `search_archive(id)`, `account()`,
`locations()`. Because SerpApi's `/account.json` is **free**, `health_check()`
consumes **zero** credits.

**Semantic Scholar** (`semanticscholar`):

| Capability       | `LLMCore` method        | S2 surface                       | Endpoint(s)                                                          |
| ---------------- | ----------------------- | -------------------------------- | ------------------------------------------------------------------- |
| `web_search`     | `web_search()`          | Academic Graph paper/snippet search | `GET /graph/v1/paper/search` (`search_type=relevance`, default) · `…/search/bulk` (`bulk`) · `…/search/match` (`match`) · `GET /graph/v1/snippet/search` (`snippet`) |
| `batch_search`   | `batch_web_search()`    | Client-side concurrent fan-out   | N× `GET /graph/v1/paper/search` bounded by `max_concurrency` (no server-side batch endpoint) |

Semantic Scholar authenticates with an **optional** `x-api-key` header — the
provider runs **keyless** against the shared public pool, so unlike the other
providers a missing key is not an error (it is the documented default mode). The
S2 search flavor is chosen with `search_type` (`relevance` | `bulk` | `match` |
`snippet`); `count`→`limit` (clamped: 100 relevance, 1000 bulk/snippet, 1 match)
and every S2 filter (`year`, `publicationDateOrYear`, `venue`, `fieldsOfStudy`,
`publicationTypes`, `minCitationCount`, `openAccessPdf`, `sort`, `token`, …)
passes through verbatim; `country`/`language`/`device`/`engine`/`mode` are
accepted but ignored (academic search is not geolocated and is always
synchronous). The full payload is preserved on `WebSearchResult.raw`.

S2's **Recommendations** API (paper → papers) is an item-to-item recommender,
not a text-query "AI search", and its **Datasets** API is a bulk-corpus download
workflow (pre-signed S3 URLs for 100M-record partitions) — neither matches the
cross-provider `discover` (text-query) or `dataset_search` (filter → snapshot →
inline records) contracts, so they are **not** advertised as those capabilities.
Instead they are exposed as provider-specific methods, alongside the rest of the
graph: `paper()`, `paper_batch()` (≤500 ids), `paper_citations()`,
`paper_references()`, `paper_authors()`, `paper_match()`, `autocomplete()`,
`snippet_search()`, `author()` / `author_batch()` / `author_papers()` /
`author_search()`, `recommend_papers()` / `recommend_from_examples()`, and the
Datasets helpers `list_releases()` / `get_release()` / `get_dataset()` /
`get_dataset_diffs()`. S2 has no free quota endpoint, so `health_check()` issues
a minimal autocomplete probe. S2 **requires** exponential backoff; the provider
retries `429`/`5xx` automatically and offers optional proactive request spacing
(`min_request_interval`).

---

## 3. Result models (provider‑agnostic)

Every result inherits `SearchResultBase` (`success`, `provider`, `error`, `cost`,
`trigger_sent_at`, `data_fetched_at`, `.elapsed_ms()`, `.to_dict()`,
`.to_json()`):

- **`WebSearchResult`** — `query`, `engine`, `items: list[SearchItem]`,
  `total_results`, `raw`. Each `SearchItem` has `position`, `title`, `url`,
  `description`, `displayed_url`.
- **`ScrapeResult`** — `url`, `content`, `response_format`, `status`,
  `root_domain`, `content_char_size`, `raw` (full provider payload when the
  provider returns structured data alongside the extracted content, e.g. Serper).
- **`DiscoverResult`** — `query`, `intent`, `items: list[DiscoverItem]`
  (`relevance_score`, `content`), `task_id`, `duration_seconds`, `raw`.
- **`DatasetInfo`**, **`DatasetMetadata`** (`fields: list[DatasetField]`),
  **`DatasetSnapshot`** (`snapshot_id`, `status`, `records`, `.record_count`).

Operations return a result object with `success=False` and a populated `error`
for *recoverable* problems (e.g. a non‑200 scrape). Hard faults (bad credentials,
transport errors, malformed API responses) raise `SearchProviderError`.

---

## 4. Quick start

```bash
pip install "llmcore[brightdata]"
export BRIGHTDATA_API_TOKEN="bd_..."
```

```toml
# ~/.config/llmcore/config.toml
[search_providers.brightdata]
serp_zone     = "my_serp_zone"       # create at https://brightdata.com/cp/zones
unlocker_zone = "my_unlocker_zone"
```

```python
from llmcore import LLMCore

llm = await LLMCore.create()
result = await llm.web_search("best vector databases 2026", count=5, country="US")
for item in result.items:
    print(item.position, item.title, item.url)
await llm.close()
```

A single configured provider is auto‑adopted as the default, so you can omit
`provider=` and `llmcore.default_search_provider`. **If you enable more than one
search provider** (e.g. both Bright Data and Serper), set
`llmcore.default_search_provider` or pass `provider="serper"` per call.

### Serper.dev variant

```bash
pip install "llmcore[serper]"
export SERPER_API_KEY="your-serper-key"
```

```python
llm = await LLMCore.create()
# vertical + time filter via kwargs:
news = await llm.web_search("apple inc", provider="serper", search_type="news", time_range="d")
# batched queries in a single request (Serper strength):
batch = await llm.batch_web_search(["apple inc", "tesla inc"], provider="serper")
await llm.close()
```

See **USAGE.md** for the full surface (verticals, batch, scrape, discover,
datasets, async mode, multiple instances, env‑var indirection, validation).

---

## 5. Configuration reference (`[search_providers.brightdata]`)

| Key              | Type    | Default                       | Notes                                                       |
| ---------------- | ------- | ----------------------------- | ----------------------------------------------------------- |
| `api_key`        | secret  | —                             | Prefer the `BRIGHTDATA_API_TOKEN` env var.                  |
| `api_key_env_var`| string  | `BRIGHTDATA_API_TOKEN`        | Name of the env var holding the token.                      |
| `base_url`       | url     | `https://api.brightdata.com`  | Rarely changed.                                             |
| `serp_zone`      | string  | —                             | Required for `web_search()`. Not auto‑created.              |
| `unlocker_zone`  | string  | —                             | Required for `scrape_url()` (alias: `web_unlocker_zone`).   |
| `default_engine` | enum    | `google`                      | `google` · `bing` · `yandex`.                               |
| `timeout`        | integer | `60`                          | Seconds for sync requests and each poll.                    |
| `poll_interval`  | integer | `2`                           | Seconds between polls (async mode).                         |
| `poll_timeout`   | integer | `60`                          | Max seconds to wait for an async result.                    |
| `max_retries`    | integer | `3`                           | Retries on transient transport / 5xx errors.               |
| `ssl_verify`     | boolean | `true`                        | Disable only for sandbox/proxy environments.                |

**Env‑var indirection** mirrors the LLM `ProviderManager`: if `api_key` is absent
the manager checks `api_key_env_var`, then the conventional `BRIGHTDATA_API_TOKEN`,
then `<SECTION>_API_KEY`. Confy's nested env override also works:
`LLMCORE_SEARCH_PROVIDERS__BRIGHTDATA__SERP_ZONE=...`.

The packaged `default_config.toml` already includes a commented
`[search_providers.brightdata]` block, and the confy‑curator schema
(`tools/llmcore.confy-schema.json`) gains a "Search Provider: Bright Data"
section so the configuration wizard can drive it.

### `[search_providers.serper]`

| Key                  | Type    | Default                      | Notes                                                          |
| -------------------- | ------- | ---------------------------- | -------------------------------------------------------------- |
| `api_key`            | secret  | —                            | Sent as `X-API-KEY`. Prefer the `SERPER_API_KEY` env var.      |
| `api_key_env_var`    | string  | `SERPER_API_KEY`             | Name of the env var holding the key.                           |
| `base_url`           | url     | `https://google.serper.dev`  | Rarely changed.                                                |
| `scrape_base_url`    | url     | `https://scrape.serper.dev`  | Separate host for `scrape_url()`.                              |
| `default_search_type`| enum    | `search`                     | search/news/images/videos/shopping/scholar/patents/maps/places/autocomplete. |
| `timeout`            | integer | `30`                         | Seconds per request.                                           |
| `max_retries`        | integer | `3`                          | Retries on transient transport / 429 / 5xx errors.            |
| `ssl_verify`         | boolean | `true`                       | Disable only for sandbox/proxy environments.                   |

The packaged `default_config.toml` includes a commented
`[search_providers.serper]` block and the schema gains a "Search Provider:
Serper.dev" section. Serper needs **no zones** (unlike Bright Data).

### `[search_providers.serpapi]`

| Key                | Type    | Default                | Notes                                                                 |
| ------------------ | ------- | ---------------------- | --------------------------------------------------------------------- |
| `api_key`          | secret  | —                      | Sent as the `api_key` **query parameter**. Prefer `SERPAPI_API_KEY` (`SERPAPI_KEY` also honored). |
| `api_key_env_var`  | string  | `SERPAPI_API_KEY`      | Name of the env var holding the key.                                  |
| `base_url`         | url     | `https://serpapi.com`  | Rarely changed.                                                       |
| `default_engine`   | string  | `google`               | Free-form SerpApi engine string (not an enum — 100+ engines).         |
| `default_output`   | enum    | `json`                 | `json` (structured) or `html` (raw HTML on `raw["raw_html"]`).        |
| `no_cache`         | boolean | `false`                | Force-fetch (cached reads are free). Dropped automatically in async mode. |
| `zero_trace`       | boolean | `false`                | Enterprise-only ZeroTrace mode.                                       |
| `json_restrictor`  | string  | —                      | Optional response-trimming expression.                                |
| `timeout`          | integer | `60`                   | Seconds per request.                                                  |
| `max_retries`      | integer | `3`                    | Retries on transient transport / 429 / 5xx errors.                    |
| `poll_interval`    | integer | `2`                    | Seconds between Search-Archive polls (async mode).                    |
| `poll_timeout`     | integer | `60`                   | Max seconds to wait for an async result.                              |
| `max_concurrency`  | integer | `5`                    | Fan-out width for `batch_search()` (client-side concurrency).         |
| `ssl_verify`       | boolean | `true`                 | Disable only for sandbox/proxy environments.                          |

The packaged `default_config.toml` includes a commented
`[search_providers.serpapi]` block and the schema gains a "Search Provider:
SerpApi" section. SerpApi needs **no zones** and its health check is **free**.

### `[search_providers.semanticscholar]`

| Key                    | Type    | Default                          | Notes                                                                          |
| ---------------------- | ------- | -------------------------------- | ------------------------------------------------------------------------------ |
| `api_key`              | secret  | —                                | **Optional.** Sent as `x-api-key`. Prefer `SEMANTIC_SCHOLAR_API_KEY` (`S2_API_KEY` also honored). Omit to use the public pool. |
| `api_key_env_var`      | string  | `SEMANTIC_SCHOLAR_API_KEY`       | Name of the env var holding the key.                                           |
| `base_url`             | url     | `https://api.semanticscholar.org`| Shared host for the Graph / Recommendations / Datasets APIs.                   |
| `default_search_type`  | enum    | `relevance`                      | `relevance` · `bulk` · `match` · `snippet`.                                    |
| `default_fields`       | string  | *(useful paper set)*             | Comma-separated paper fields applied when `fields` is omitted.                 |
| `default_author_fields`| string  | *(useful author set)*            | Comma-separated author fields for the author endpoints.                        |
| `timeout`              | integer | `30`                             | Seconds per request.                                                           |
| `max_retries`          | integer | `3`                              | Retries on transient transport / 429 / 5xx (S2 **requires** backoff).          |
| `max_concurrency`      | integer | `1`                              | `batch_search` fan-out width. Kept at 1 — S2 limits are strict.                |
| `min_request_interval` | integer | `0`                              | Optional proactive request spacing (seconds); 0 disables. `1` ≈ 1 RPS.         |
| `ssl_verify`           | boolean | `true`                           | Disable only for sandbox/proxy environments.                                   |

The packaged `default_config.toml` ships the `[search_providers.semanticscholar]`
block **commented out** (so search stays empty by default); uncomment the header
to enable it — **no key required**. The schema gains a "Search Provider:
Semantic Scholar" section. S2 needs **no zones**.

---

## 6. Failure modes & edge cases

- **No token** → the provider raises `ConfigError` during construction; the
  manager catches it and simply does not load the provider (search stays empty).
  This is why a misconfigured token never breaks app startup.
- **No `serp_zone` / `unlocker_zone`** → `web_search()` / `scrape_url()` raise
  `SearchProviderError` *at call time* (datasets/discover still work).
- **`401/403`** → `SearchProviderError(status_code=…)` (no retry).
- **`5xx` / transport errors** → retried up to `max_retries` with capped
  exponential backoff, then surfaced as `SearchProviderError`.
- **Zone returns raw HTML instead of parsed JSON** → `WebSearchResult.items` is
  empty and the HTML is preserved under `raw` (`{"raw_html": …}`); pass
  `parse=False` to request HTML deliberately.
- **Wrapped `{status_code, body}` envelope** → transparently unwrapped.
- **Async polling timeout** → `SearchProviderError` after `poll_timeout`.
- **Datasets are asynchronous** → `dataset_search()` polls the snapshot to
  `ready` before downloading; a `failed` snapshot returns `success=False`.

---

## 7. Tradeoffs & alternatives considered

- **Parallel ABC vs. extending `BaseProvider`** — chosen for cohesion/LSP (see §1).
  Tradeoff: a little duplication (a second manager) in exchange for not polluting
  the LLM contract.
- **Native `httpx` vs. vendoring `brightdata-sdk`** — chosen to keep llmcore
  dependency‑light and the result types vendor‑neutral. Tradeoff: we re‑implement
  a thin slice of the API surface (verified against the official SDK).
- **No zone auto‑creation** — the vendor SDK will create `sdk_serp` / `sdk_unlocker`
  zones for you; llmcore deliberately does not, to avoid silently mutating your
  Bright Data account. Tradeoff: one‑time manual zone setup.
- **SerpApi batch is client-side fan-out** — SerpApi has no server-side batch
  endpoint (unlike Serper's array payload), so `batch_search()` issues N
  concurrent `GET /search` calls bounded by `max_concurrency`. Tradeoff: N
  queries cost N credits and open N connections, but it honors the unified
  `batch_web_search` contract (one ordered result per input) and exploits
  SerpApi's per-hour throughput.
- **SerpApi `engine` is a free string, not an enum** — SerpApi supports 100+
  engines and adds them frequently; enumerating them in config/schema would be
  stale on arrival and would reject valid engines. A `KNOWN_ENGINES` set drives
  only a soft debug warning, never rejection. Tradeoff: typos aren't hard-failed
  (SerpApi ignores unknown params and returns an in-body error, surfaced as
  `success=False`).
- **SerpApi `count`→`num` is best-effort** — `num` is a real SerpApi parameter
  honored by many engines (scholar/patents/naver/…) but the standard `google`
  web engine ignores it (use `start` for pagination). We forward it because
  SerpApi silently ignores unsupported params; the alternative (silently dropping
  the unified `count`) would surprise callers more.
- **Semantic Scholar loads keyless by default** — S2's API key is optional, so
  the provider never raises on a missing key (it uses the shared public pool).
  Tradeoff: to preserve the "search is empty unless configured" invariant, the
  packaged `default_config.toml` ships the S2 block **commented out** (an
  uncommented keyless block would otherwise auto-load and make S2 the default
  provider on a fresh install). One-line opt-in; no key needed.
- **S2 Recommendations are not `discover`; S2 Datasets are not `dataset_search`**
  — Recommendations is an item-to-item recommender (paper → papers), not a
  text-query AI search, and the Datasets API is a bulk-corpus download workflow
  (pre-signed URLs for 100M-record partitions), not a filter → snapshot →
  inline-records flow. Shoehorning either into the cross-provider contract would
  misrepresent it, so both are exposed as provider-specific methods. Tradeoff:
  they aren't reachable through the unified `llm.discover(...)` /
  `llm.dataset_search(...)` surface (call them on the provider object instead).
- **S2 `search_type` is a kwarg, `count→limit` is clamped per endpoint** — one
  `web_search` entry point serves four S2 endpoints (relevance/bulk/match/
  snippet) selected by `search_type`, mirroring Serper's vertical selector. The
  documented per-endpoint `limit` caps (100/1000/1) are enforced client-side so
  an over-large `count` is clamped rather than rejected by the API.
- **cardctl is intentionally not extended** — `cardctl` manages *model cards*
  (token pricing, context length, capabilities). Bright Data, Serper, SerpApi
  and Semantic Scholar have no token‑priced "models"; they bill per request /
  per record / per credit (or are free). Adding any of them to the model‑card
  registry would be a category error. See
  `tools/cardctl/BRIGHTDATA_SKIP_RATIONALE.md`.

---

## 8. Extending: adding another search provider

1. Implement `BaseSearchProvider` in `llmcore/search/providers/<vendor>_provider.py`
   (override `get_name`, `get_capabilities`, and the capability methods you support).
2. Register it in `SEARCH_PROVIDER_MAP` in `llmcore/search/manager.py`.
3. Add a `[search_providers.<vendor>]` block to `default_config.toml` and a section
   to the confy‑curator schema.
4. Add `respx`‑based unit tests under `tests/search/`.

Because the result models are provider‑agnostic, consumer code that already calls
`llm.web_search(...)` keeps working unchanged against the new backend.
