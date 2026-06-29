# Changelog

All notable changes to **llmcore** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.51.0

### Added — Z.ai (GLM) provider

- **Z.ai provider**: first-class `ZaiProvider` for the Z.ai Open Platform,
  serving the GLM model family (`glm-5.2`, `glm-5.1`, `glm-4.7`, the `glm-*v`
  vision models, and `embedding-3`). Built on the OpenAI-compatible chat
  endpoint (`https://api.z.ai/api/paas/v4`) via the AsyncOpenAI SDK, with:
  - GLM **thinking mode** (`thinking = {"type": "enabled" | "disabled"}`) and
    `reasoning_effort` (`none|minimal|low|medium|high|xhigh|max`).
  - `reasoning_content` extraction in both streaming and non-streaming modes.
  - Open-interval `(0, 1)` clamping of `temperature`/`top_p` (matching the
    Z.ai API constraint).
  - Platform extras (`do_sample`, `request_id`, `user_id`, `seed`,
    `watermark_enabled`, `sensitive_word_check`, `tool_stream`) routed via
    `extra_body`.
  - Tool calling, cache/reasoning token usage accounting, and `embedding-3`
    embeddings.
  - Region selection (`overseas` default, or `china` for the
    `open.bigmodel.cn` endpoint).
- **Provider registration**: `zai` registered in `ProviderManager` with
  `glm`, `zhipu`, `zhipuai`, and `bigmodel` aliases.
- **Model cards**: builtin cards for `glm-5.2`, `glm-5.1`, `glm-4.7`,
  `glm-4.6v` (vision), and `embedding-3`.
- **Tests**: 57-test offline suite for the Z.ai provider.

## v0.50.0

### Added — June 2026 agent, context, provider, and observability rollup

- **Deepgram voice/audio provider**: native SDK integration for speech-to-text,
  text-to-speech, Flux streaming, Voice Agent, text intelligence, token grants,
  Deepgram model cards, docs, examples, and offline tests.
- **Per-call usage accounting**: `LLMCore.chat_with_usage()` and `ChatUsage`
  expose prompt/completion/total token counts for transient calls without
  requiring session persistence.
- **Search providers**: the optional `llmcore.search` subsystem now includes
  Bright Data, Serper.dev, SerpApi, and Semantic Scholar, with provider-neutral
  result models and manager/facade wiring.
- **Token counting**: native provider fallback paths and OpenAI token counting
  now route through llmcore's shared model-aware token counters.
- **Agent execution**: typed plan-step specs, structured plan-step tool
  execution, loaded-tool validation, activity-protocol routing to loaded tools,
  runtime permission metadata, and preserved resumed history/pending action
  snapshots.
- **Context and memory**: objective-aware compression, semantic citation
  provenance, structured tool-result summaries, typed citation source handling,
  and external backend consolidation hooks.
- **Observability**: semantic retrieval events, context diagnostics after agent
  runs, context failure diagnostics, phase token summaries, iteration summaries,
  and ecosystem federation telemetry.
- **HITL auditability**: OWASP metadata and audit reporting for dangerous action
  patterns.

### Changed

- Bumped package and documentation metadata to `0.50.0`.
- Removed noisy import-time optional SDK warnings by lazy-loading
  `sentence-transformers` and `google-genai` only when their providers are
  instantiated.
- Updated `SingleAgent` environment config loading to use the current unified
  `confy.Config` path instead of the deprecated `load_agents_config(config_path=...)`
  compatibility path.

## v0.49.14

### Added — Deepgram voice/audio provider (STT, TTS, Flux, Voice Agent, Text Intelligence)

Adds a complete, native-SDK **Deepgram** provider — llmcore's first real-time
**voice/audio** provider. Deepgram is fundamentally different from the
text-completion LLM providers: its primary surfaces are WebSocket streams for
speech-to-text (STT), text-to-speech (TTS), and a bidirectional **Voice Agent**
(STT → LLM → TTS over one socket). There is no text chat-completion surface, so
`chat_completion` raises a clear, actionable `ProviderError` (HTTP 400,
non-retryable) and callers use the media methods instead. Tested against
`deepgram-sdk` v7.3.1.

- **New provider `DeepgramProvider`** (`llmcore/providers/deepgram_provider.py`),
  registered in `ProviderManager.PROVIDER_MAP` as `"deepgram"`. Follows the
  Gemini native-SDK template: lazy SDK import with an availability flag, an
  `ImportError` (surfaced as *"install llmcore[deepgram]"*) when the SDK is
  absent, and `ConfigError` for bad config. Wraps the official async-first,
  WebSocket-native `deepgram-sdk` (v7.x).
- **Batch media**: `transcribe_audio` (bytes / file path, or a remote `url=`) →
  `TranscriptionResult`; `generate_speech` (Aura voices) → `SpeechResult`.
- **Streaming**: `transcribe_stream` / `open_transcription_socket` (live STT),
  `transcribe_stream_flux` / `open_flux_socket` (Flux, listen.v2, turn-aware),
  and a dual-mode `stream_speech` / `open_speech_socket` (a string → REST
  streaming; an async iterable of text → a TTS WebSocket). The one-call
  streaming helpers fan microphone audio in and stream events/audio out
  concurrently, always close the socket on completion, and surface producer
  errors as `ProviderError`.
- **Voice Agent**: `open_voice_agent` (manual `DeepgramVoiceAgentSession`) and
  `run_voice_agent` (high-level driver that auto-answers `FunctionCallRequest`s
  via a `function_handler` and exposes an `on_event` hook). Runtime steering:
  `inject_user_message` / `inject_agent_message` / `update_prompt` /
  `update_think` / `update_speak` / `respond_to_function_call` / `keepalive`.
  **No system prompt is ever defaulted** — callers pass `prompt=` explicitly (or
  inject upstream), honouring the ecosystem "no hardcoded prompt" invariant.
- **Text intelligence**: `analyze_text` (read.v1) → `TextAnalysisResult`
  (summary / topics / sentiment / intents).
- **Token auth & account**: `grant_token` (short-lived access token for
  browsers/clients) and `get_projects`; the full management API remains
  available via the `provider.client` escape hatch.
- **New provider-neutral models** (`llmcore/models_multimodal.py`):
  `StreamEventType`, `TranscriptionStreamEvent`, `VoiceAgentEventType`,
  `VoiceAgentFunctionCall`, `VoiceAgentEvent`, and `TextAnalysisResult`.
- **Configuration**: a fully-documented `[providers.deepgram]` block in
  `config/default_config.toml` wires every capability (auth, transport,
  default models, `[stt]`/`[stt.streaming]`, `[flux]`, `[tts]`/`[tts.streaming]`,
  and `[agent]` with the SDK's `provider`-nested `listen`/`think`/`speak`
  shape and top-level `audio`).
- **Model cards**: 11 cards under `model_cards/default_cards/deepgram/` (STT:
  `nova-3`, `nova-3-medical`, `nova-2`, `nova-2-phonecall`, `whisper-large`,
  `flux-general-en`; TTS: `aura-2-thalia-en`, `aura-2-andromeda-en`,
  `aura-2-apollo-en`, `aura-asteria-en`, `aura-luna-en`), generated by
  `tools/generate_deepgram_cards.py`. Each card records Deepgram's published
  pay-as-you-go rates (per audio-minute STT / per-character TTS) in
  `provider_extension.pricing` with units, source URL, and capture date; the
  token-centric `pricing` field stays `null` (tokens are not the billing unit).
- **Packaging**: new optional extra `deepgram = ["deepgram-sdk>=7.0.0",
  "websockets>=12.0"]` (also folded into `all`).
- **Tokens / context**: Deepgram bills per audio-minute (STT) / per-character
  (TTS), so `count_tokens` returns a documented character-count heuristic and
  `get_max_context_length` returns a configurable nominal value
  (`fallback_context_length`, default 2000). These are **not** billing units.
- **Tests**: 54 new tests across `tests/providers/test_deepgram_provider.py`,
  `test_deepgram_streaming.py`, and `test_deepgram_agent.py` (fake
  clients/sockets; no network). Full provider suite: 553 passed, 2 skipped.
- **Docs & examples**: `docs/Deepgram_provider_usage.md` plus seven runnable
  `examples/deepgram_*.py` scripts.

The existing public API is unchanged; this is purely additive.

### Added — Per-call token usage surface (`LLMCore.chat_with_usage`)

Adds a **usage-returning** companion to `chat()` so that *callers* can meter
token consumption per call without enabling session persistence. This is the
foundational dependency for external usage/quota systems built on top of
llmcore (e.g. Convergence's metering bridge): previously the prompt/completion
token counts llmcore computes internally were only persisted when
`save_session=True`, and were never returned to the caller of a transient
`chat()` call.

- **New method `LLMCore.chat_with_usage(message, *, ...) -> tuple[str, ChatUsage]`**
  (`llmcore/api.py`). Non-streaming only; mirrors `chat()`'s full keyword
  signature. It runs the *exact* same code path as `chat()` (provider
  resolution, context preparation, the provider call, and llmcore's own
  prompt/completion token counting) and additionally returns the per-call
  token usage. The existing `chat() -> str` contract is **unchanged** — this is
  purely additive and opt-in.
- **New public value object `ChatUsage`** (`llmcore/usage.py`, exported from
  `llmcore`). A frozen dataclass carrying `prompt_tokens` / `completion_tokens`
  / `total_tokens` / `provider` / `model`, with `tokens_in` / `tokens_out`
  read-only aliases (so it is a drop-in for either naming convention) and an
  `is_available` flag. When usage cannot be determined every count is `None`,
  letting downstream meters degrade to a no-op rather than recording a
  zero-token event.
- **Concurrency-safe & residue-free.** Usage is read back via the existing
  per-session introspection cache (`get_last_interaction_context_info`) under a
  *call-local* session id, so concurrent calls never read each other's usage.
  When the caller passes no `session_id`, an ephemeral one is synthesised and
  its transient caches are dropped on return.
- **Tests** (`tests/api/test_chat_with_usage.py`) — `ChatUsage` value
  semantics, the signature/protocol contract, and offline end-to-end behaviour
  against an injected fake provider (no network).
- **Docs** — `docs/USAGE_chat_with_usage.md`.

## v0.49.13

### Added — Semantic Scholar search provider (`llmcore.search`)

Adds **Semantic Scholar** (https://www.semanticscholar.org/product/api) as a
first-class **search provider**, joining Bright Data, Serper.dev and SerpApi in
the optional `llmcore.search` subsystem. Semantic Scholar is a free, AI-powered
academic search engine over 200M+ papers; the provider wraps all three public S2
APIs (Academic Graph, Recommendations, Datasets), which share a host
(`https://api.semanticscholar.org`) under different path prefixes.

- **New provider `SemanticScholarSearchProvider`**
  (`llmcore/search/providers/semanticscholar_provider.py`) — native `httpx`
  client (no vendor SDK), advertising `web_search` and `batch_search`.
  Highlights:
  - **Optional API key / keyless by default.** The S2 key is optional; the
    provider operates against the shared public pool when no key is set (a
    missing key is **not** an error — unlike the other providers). When present,
    the key is sent via the `x-api-key` header and resolved from `api_key` /
    `token`, `api_key_env_var`, or `SEMANTIC_SCHOLAR_API_KEY` (with `S2_API_KEY`
    fallback); never logged.
  - **Four search flavors via `search_type`:** `relevance` (default,
    `/paper/search`), `bulk` (`/paper/search/bulk`, with continuation `token`),
    `match` (`/paper/search/match`, single best title match), and `snippet`
    (`/snippet/search`, text passages for RAG — `item.description` is the
    passage). `count` → `limit` (clamped per endpoint: 100 / 1000 / 1); all S2
    filters (`year`, `publicationDateOrYear`, `venue`, `fieldsOfStudy`,
    `publicationTypes`, `minCitationCount`, `openAccessPdf`, `sort`, `token`, …)
    pass through verbatim (`openAccessPdf` handled as a valueless presence flag).
    `country` / `language` / `device` / `engine` / `mode` are accepted but
    ignored (academic search is not geolocated and is always synchronous). Full
    payload preserved on `WebSearchResult.raw`.
  - **Mandatory exponential backoff** on `429` / `5xx` (S2 requires it), plus an
    optional proactive `min_request_interval` request spacer and a conservative
    default batch concurrency of 1.
  - **Client-side `batch_search`** fan-out (S2 has no multi-query endpoint),
    returning one ordered result per input query.
  - **Rich provider-specific methods** (not shoehorned into the cross-provider
    `discover`/`dataset_search` contracts, which don't fit S2's item-to-item
    recommender or bulk-corpus workflow): `paper`, `paper_batch` (≤500 ids),
    `paper_citations`, `paper_references`, `paper_authors`, `paper_match`,
    `autocomplete`, `snippet_search`, `author`, `author_batch`, `author_papers`,
    `author_search`, `recommend_papers`, `recommend_from_examples`, and the
    Datasets helpers `list_releases`, `get_release`, `get_dataset`,
    `get_dataset_diffs`.
  - **Free-ish `health_check()`** via a minimal autocomplete probe (S2 has no
    quota endpoint).
- **Registry & exports:** registered in `SEARCH_PROVIDER_MAP` and
  `_SEARCH_PROVIDER_ENV_DEFAULTS` (`semanticscholar`, aliases `semantic_scholar`
  / `semantic-scholar` / `s2` → `SEMANTIC_SCHOLAR_API_KEY`); exported from
  `llmcore.search` and the top-level `llmcore` package.
- **Configuration:** a `[search_providers.semanticscholar]` block added to
  `default_config.toml` **commented out** — because the provider loads keyless,
  an uncommented block would auto-load and break the "search is empty unless
  configured" invariant; it is a one-line opt-in (no key required). Keys:
  `api_key_env_var`, `base_url`, `default_search_type`, `default_fields`,
  `timeout`, `max_retries`, `max_concurrency`, `min_request_interval`,
  `ssl_verify`.
- **confy-curator schema** (`tools/llmcore.confy-schema.json`): added a
  "Search Provider: Semantic Scholar" section (order 49) for the wizard.
- **Packaging:** new `semanticscholar` extra (`pip install
  "llmcore[semanticscholar]"`); added to the `all` extra.
- **Tests:** `tests/search/test_semanticscholar_provider.py` — 61 `respx`-based
  unit tests (no network) covering keyless vs keyed auth (header presence), the
  four search flavors and per-endpoint clamps, filter pass-through &
  `openAccessPdf` flag, snippet/citation/reference/author normalization, POST
  batch & recommendations bodies, the Datasets helpers, retry-on-429/5xx +
  transport retry, 401/403 raises, the health check, client-side batch fan-out,
  and manager wiring (keyless + `s2` alias + keyed). Full search suite: 199
  passed, 0 regressions.
- **Docs:** `docs/Search_providers_usage.md` (new §11 Semantic Scholar) and
  `docs/Search_providers_rationale.md` (glance row, capability matrix, config
  reference, tradeoffs) updated; new `examples/semanticscholar_search_example.py`.

> `cardctl` is intentionally **not** extended for Semantic Scholar — it manages
> *LLM model cards* (token pricing/context), which do not apply to a free,
> per-request academic API. See `tools/cardctl/BRIGHTDATA_SKIP_RATIONALE.md`. No
> public APIs, schemas, or existing provider behavior changed; the addition is
> fully backward-compatible.

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
