# Changelog

All notable changes to **llmcore** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.49.12

### Added — SerpApi search provider (`llmcore.search`)

Adds **SerpApi** (https://serpapi.com) as a first-class **search provider**,
joining Bright Data and Serper.dev in the optional `llmcore.search` subsystem.
SerpApi is a real-time *meta-SERP* API: a single endpoint scrapes 100+ search
engines/verticals selected with one `engine` parameter.

- **New provider `SerpApiSearchProvider`** (`llmcore/search/providers/serpapi_provider.py`)
  — native `httpx` client (no vendor SDK), advertising `web_search` and
  `batch_search`. Highlights:
  - **100+ engines** via a free-form `engine` argument (`google`, `bing`,
    `baidu`, `duckduckgo`, `yahoo`, `yandex`, `google_news`, `google_images`,
    `google_shopping`, `google_scholar`, `youtube`, `amazon`, `ebay`, `walmart`,
    `google_maps`, …). Engine is **not** an enum (SerpApi adds engines often); a
    `KNOWN_ENGINES` set drives only a soft debug warning, never rejection.
  - **Engine-aware mapping & normalization:** `query` → the engine's query field
    (`q`/`query`/`p`/`text`/`search_query`/`term`/`_nkw`/`find_desc`), `count` →
    `num` (best-effort), `country` → `gl`, `language` → `hl`, `device` →
    `device`; the primary result array is resolved per engine
    (`organic_results`/`news_results`/`images_results`/`video_results`/
    `shopping_results`/`local_results`/…). The full payload is preserved on
    `WebSearchResult.raw`.
  - **Auth via `api_key` query parameter** (not a header). Key resolved from
    `api_key`/`token`, `api_key_env_var`, or `SERPAPI_API_KEY` (with `SERPAPI_KEY`
    / `SERP_API_KEY` SDK-convention fallbacks); always redacted from logs.
  - **Async mode** (`mode="async"`): submits with `async=true`, then polls the
    Search Archive (`GET /searches/{id}`) until `Success`/`Error`
    (`no_cache` is dropped automatically as it is incompatible with async).
  - **Client-side `batch_search`:** SerpApi has no server-side batch endpoint, so
    queries are run concurrently bounded by `max_concurrency` (one credit each),
    returning one ordered `WebSearchResult` per input.
  - **Provider-specific helpers:** `search()` (raw `/search` pass-through),
    `search_archive(id)`, `account()` and `locations()`.
  - **Free `health_check()`** via the Account API (`/account.json`) — consumes
    **zero** search credits (unlike Serper).
  - Passes through every other SerpApi request parameter verbatim (`location`,
    `uule`, `lat`/`lon`, `google_domain`, `tbm`, `tbs`, `safe`, `start`,
    `no_cache`, `output`, `json_restrictor`, `zero_trace`, …).
- **Registry & exports:** registered in `SEARCH_PROVIDER_MAP` and
  `_SEARCH_PROVIDER_ENV_DEFAULTS` (`serpapi`, aliases `serp_api` /
  `serpapi_search` → `SERPAPI_API_KEY`); exported from `llmcore.search` and the
  top-level `llmcore` package.
- **Configuration:** new commented `[search_providers.serpapi]` block in
  `default_config.toml` (`api_key_env_var`, `base_url`, `default_engine`,
  `default_output`, `no_cache`, `zero_trace`, `json_restrictor`, `timeout`,
  `max_retries`, `poll_interval`, `poll_timeout`, `max_concurrency`,
  `ssl_verify`).
- **confy-curator schema** (`tools/llmcore.confy-schema.json`): added a
  "Search Provider: SerpApi" section (order 48) for the configuration wizard.
- **Packaging:** new `serpapi` extra (`pip install "llmcore[serpapi]"`); added to
  the `all` extra.
- **Tests:** `tests/search/test_serpapi_provider.py` — 47 `respx`-based unit
  tests (no network) covering param mapping/auth, engine-specific query fields,
  vertical normalization, async submit+poll (incl. timeout/error), client-side
  batch fan-out, archive/account/locations, retries, the free health check, and
  manager wiring.
- **Docs:** `docs/Search_providers_usage.md` and
  `docs/Search_providers_rationale.md` updated with a SerpApi section, capability
  matrix row and config reference; new `examples/serpapi_search_example.py`.

> `cardctl` is intentionally **not** extended for SerpApi — it manages *LLM model
> cards* (token pricing/context), which do not apply to a per-credit SERP API.
> See `tools/cardctl/BRIGHTDATA_SKIP_RATIONALE.md`. No public APIs, schemas, or
> existing provider behavior changed; the addition is fully backward-compatible.

## v0.49.11

### Added — Web/Data Search Providers (`llmcore.search`)

A new, **optional** subsystem that adds web/data **search** providers alongside
the existing LLM providers, usable "just like" LLM providers (config‑driven,
discovered through a manager, accessed via a uniform interface). The first
provider is **Bright Data**.

- **New package `llmcore.search`:**
  - `BaseSearchProvider` (ABC) + `SearchCapability` enum — the search‑side
    analogue of `BaseProvider`. Optional capability methods default to
    `NotImplementedError` (same idiom as the LLM provider's optional modalities).
  - `SearchProviderManager` — mirrors `ProviderManager`, but **optional**: loads
    zero or more providers from `[search_providers]` and never fails when the
    section is absent. Auto‑adopts a lone provider as the default.
  - Provider‑agnostic result models: `WebSearchResult`/`SearchItem`,
    `ScrapeResult`, `DiscoverResult`/`DiscoverItem`,
    `DatasetInfo`/`DatasetField`/`DatasetMetadata`/`DatasetSnapshot`
    (all with `to_dict()`/`to_json()`/`elapsed_ms()`).
  - `BrightDataSearchProvider` — native `httpx` client (no vendor SDK). Supports
    SERP web search (sync + async), Web Unlocker scraping, the Discover API
    (AI‑ranked), and the Dataset Marketplace (filter → snapshot → download), plus
    a connectivity `health_check()`.
- **`LLMCore` API:** `web_search()`, `scrape_url()`, `discover()`,
  `list_datasets()`, `get_dataset_metadata()`, `dataset_search()`,
  `get_search_provider()`, `get_available_search_providers()`. A
  `_search_provider_manager` is initialized after the LLM `ProviderManager`,
  closed in `close()`, and rebuilt on `reload_config()`;
  `set_raw_payload_logging()` now also propagates to search providers.
- **New exception:** `SearchProviderError` (search‑side analogue of
  `ProviderError`, with optional `status_code`).
- **Configuration:** new `llmcore.default_search_provider` key and a
  `[search_providers.brightdata]` section in `default_config.toml`
  (token via `BRIGHTDATA_API_TOKEN`; `serp_zone` / `unlocker_zone`;
  `default_engine`, `timeout`, `poll_interval`, `poll_timeout`, `max_retries`,
  `ssl_verify`). Zones are **not** auto‑created.
- **confy‑curator schema** (`tools/llmcore.confy-schema.json`): added a
  "Search Provider: Bright Data" section and a "Default Search Provider" field
  to Core Settings (validated against confy‑curator's `SchemaModel`).
- **Packaging:** new optional extra `brightdata = ["httpx>=0.27.0"]`, also
  included in the `all` extra.
- **Docs & examples:** `docs/search/README.md`, `docs/search/USAGE.md`,
  `examples/brightdata_search_example.py`.
- **Tests:** `tests/search/` (62 tests) — `respx`‑based provider tests asserting
  exact endpoints/payloads/headers, model tests, manager tests, and an
  `LLMCore`‑level wiring test.

### Notes / non‑goals

- **`cardctl` intentionally not extended.** Bright Data has no token‑priced
  "models"; adding it to the model‑card registry would be a category error. See
  `tools/cardctl/BRIGHTDATA_SKIP_RATIONALE.md`.
- **Backward compatible.** The subsystem is additive and optional; deployments
  that do not configure `[search_providers]` are unaffected. The search methods
  raise a clear `ConfigError` only if called with no provider configured.
