# LLMCore Search Providers

**Web / data search for LLMCore — used "just like" the LLM providers.**

This subsystem (`llmcore.search`) adds a second family of pluggable backends
alongside the existing LLM providers (`llmcore.providers`). Where an LLM provider
turns a prompt into a completion, a **search provider** turns a query into search
results, scraped pages, AI‑ranked discoveries, or structured dataset records.

The first provider is **Bright Data**.

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
`NotImplementedError`.

| Capability        | `LLMCore` method            | Bright Data product      | Endpoint(s)                                                                   |
| ----------------- | --------------------------- | ------------------------ | ----------------------------------------------------------------------------- |
| `web_search`      | `web_search()`              | SERP API                 | `POST /request` (sync) · `POST /unblocker/req` + `GET /unblocker/get_result` (async) |
| `scrape`          | `scrape_url()`              | Web Unlocker             | `POST /request` (sync) · async pair                                           |
| `discover`        | `discover()`                | Discover API             | `POST /discover` + `GET /discover?task_id=`                                   |
| `dataset_search`  | `dataset_search()`, `list_datasets()`, `get_dataset_metadata()` | Dataset Marketplace | `POST /datasets/filter` · `GET /datasets/snapshots/{id}` · `…/download` · `GET /datasets/list` · `GET /datasets/{id}/metadata` |
| `crawl`           | *(not implemented)*         | —                        | Reserved on the base class; raises `NotImplementedError`.                     |

All endpoints authenticate with a single Bearer token. Health/connectivity is
checked via `GET /zone/get_active_zones`.

---

## 3. Result models (provider‑agnostic)

Every result inherits `SearchResultBase` (`success`, `provider`, `error`, `cost`,
`trigger_sent_at`, `data_fetched_at`, `.elapsed_ms()`, `.to_dict()`,
`.to_json()`):

- **`WebSearchResult`** — `query`, `engine`, `items: list[SearchItem]`,
  `total_results`, `raw`. Each `SearchItem` has `position`, `title`, `url`,
  `description`, `displayed_url`.
- **`ScrapeResult`** — `url`, `content`, `response_format`, `status`,
  `root_domain`, `content_char_size`.
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
`provider=` and `llmcore.default_search_provider`.

See **USAGE.md** for the full surface (scrape, discover, datasets, async mode,
multiple instances, env‑var indirection, and validation steps).

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
- **cardctl is intentionally not extended** — `cardctl` manages *model cards*
  (token pricing, context length, capabilities). Bright Data has no token‑priced
  "models"; its products bill per request / per record. Adding it to the model‑card
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
