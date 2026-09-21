# Changelog

All notable changes to **llmcore** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added — FriendliAI provider

- **FriendliAI provider**: first-class `FriendliProvider` covering all three
  Friendli inference surfaces through one `[providers.friendli]` section,
  selected with `endpoint_type`:
  `"serverless"` (Friendli Model APIs — the hosted pay-per-token catalog),
  `"dedicated"` (Dedicated Endpoints; the `model` field is the **endpoint ID**,
  or `ENDPOINT_ID:ADAPTER_ROUTE` for Multi-LoRA), and `"container"`
  (self-hosted Friendli Engine; `base_url` required, API key optional).
- **Dual transport**: `backend = "openai" | "httpx" | "sdk"`, auto-resolving
  **openai → httpx → sdk**. The vendor `friendli` SDK is supported but ranked
  last on purpose: its generated response models ignore unknown fields, so
  `reasoning_content` / `reasoning` are silently dropped and there is no
  `extra_body` escape hatch. The provider warns at startup when `backend =
  "sdk"` is combined with `parse_reasoning`.
- **Reasoning controls**: `reasoning_effort`
  (`minimal|low|medium|high|xhigh|max|ultracode`), `reasoning_budget`,
  `parse_reasoning`, `include_reasoning`, plus the chat-template switches
  `enable_thinking` / `clear_thinking` folded into `chat_template_kwargs`.
  Parsed chains of thought are surfaced by `extract_reasoning_content()` and
  `extract_delta_reasoning_content()` in both streaming and non-streaming mode.
- **Friendli Engine sampling**: `top_k`, `min_p`, `min_tokens`,
  `repetition_penalty`, `eos_token`, and XTC (`xtc_threshold` /
  `xtc_probability`) routed through `extra_body`; mutually exclusive body
  fields (`tools` vs `min_tokens`/`response_format`) are dropped with a warning
  instead of 422-ing.
- **Structured output** including Friendli's `regex` `response_format`, tool
  calling with first-class `Message.tool_calls` (R-2), and multimodal input via
  `metadata["inline_images"|"inline_audio"|"inline_videos"|"content_parts"]`.
- **Rich model discovery**: `GET /models` reports context length, max completion
  tokens, per-token pricing, a `functionality` block, modalities, reasoning
  options, `base_model` and `mode`; the catalog is cached, primed by
  `warm_up()`, and drives `get_max_context_length()`.
- **Auxiliary surfaces**: `tokenize()` / `detokenize()` / `render_chat()`,
  `text_completion()`, `transcribe_audio()`, and — on dedicated/container only
  — `create_embeddings()` / `generate_image()`, each gated with an actionable
  error on the wrong endpoint type. `get_team_cost()` / `get_team_usage()` read
  the Friendli Suite billing APIs for the configured team.
- **Team scoping**: `team_id` (or `FRIENDLI_TEAM_ID` / `FRIENDLIAI_TEAM_ID`) is
  sent as `X-Friendli-Team` on every request.
- **Token counting**: local by default (tiktoken `cl100k_base`, then a
  character-ratio estimate); `native_token_count = true` routes counts through
  Friendli's exact `/tokenize` endpoint at the cost of one API request per
  count, with a local fallback when that call fails.
- **Config**: new `[providers.friendli]` section in `default_config.toml`
  (`api_key`/`api_key_env_var` → `FRIENDLI_TOKEN` → `FRIENDLIAI_API_KEY` →
  `FRIENDLI_API_KEY`, `team_id`/`team_id_env_var`, `endpoint_type`, `backend`,
  `base_url`, `suite_base_url`, `default_model`, `timeout`, the reasoning
  defaults, `native_token_count`, `fallback_context_length`) and a matching
  `provider_friendli` section in the confy schema.
- **Model cards**: `FriendliAdapter` for cardctl derives context, pricing,
  capabilities, modalities and reasoning options straight from the live
  catalog, with a `friendli.toml` enrichment overlay for architecture and
  display names; seven generated cards under
  `model_cards/default_cards/friendli/`.
- **Packaging**: `llmcore[friendli]` extra (`openai`, `httpx`, and the optional
  `friendli` SDK), included in `llmcore[all]`; `friendli` registered in
  `ProviderManager` with the `friendliai` / `friendli_ai` aliases.
- **Errors**: 401/403/404/429 map to actionable `ProviderError`s (429 marked
  retryable so `chat_completion_with_retry` applies), and context-overflow
  wording on 400/422 maps to `ContextLengthError`.
- **Docs, examples & tests**: `docs/Friendli_provider_usage.md`,
  `examples/friendli_example.py`, README updates, and a 120-test offline suite
  (`tests/providers/test_friendli_provider.py`) covering credential/backend
  resolution, parameter splitting, payload building, both direct backends,
  discovery, tokenization, endpoint gating, Suite APIs and error mapping.

### Fixed — `ContextLengthError` construction in three providers

- **OpenAI, DeepSeek and Z.ai raised `TypeError` instead of
  `ContextLengthError` on every context overflow.** All three constructed the
  exception with a keyword set it has never accepted
  (`provider_name` / `model` / `max_tokens` / `requested_tokens`) rather than
  its real signature `(model_name, limit, actual, message)`, so the `raise`
  statement itself blew up inside `__init__`. Callers catching
  `ContextLengthError` — including llmcore's own context-management and agent
  retry paths — never saw it, and the user got an opaque `TypeError` with no
  model or limit attached. Fixing `OpenAIProvider` also fixes its subclasses
  (DeepInfra, vLLM, Poe, OpenRouter). Anthropic, Mistral, Gemini, Kimi and the
  new Friendli provider already used the correct signature.
- **Regression coverage** (`tests/providers/test_context_length_error_mapping.py`):
  a static AST check asserts that *every* `ContextLengthError(...)` call site
  in `src/llmcore` uses keywords the constructor accepts — covering providers
  with no error-path tests and any added later — plus behavioural tests that
  drive the real `chat_completion()` failure path of each fixed provider and
  assert the mapped exception carries the model name and context limit. Both
  guards were verified to fail against the pre-fix code.

### Notes

- Friendli's documented `/detokenize` and `/chat/render` routes currently
  return 404 on Model APIs (verified 2026-09-20); they are implemented and work
  on Dedicated Endpoints / Container, and every caller degrades gracefully.
- Model APIs rate limits are tier-based; tier 0 is "adaptive" and in practice
  allows only a couple of requests per minute, which is why native token
  counting is opt-in and `examples/friendli_example.py` paces its calls.

## v0.53.0

### Added — TypeSafe.ai (System One) provider

- **TypeSafe.ai provider**: first-class `TypeSafeProvider` for TypeSafe's
  System One typed-judgment API (Jev models). It is **not a chat model**:
  `POST /v1/systemone` evaluates a `state` (string / JSON object / array)
  against typed questions and returns calibrated, structured answers —
  `noul` (yes/no probability), `choice` (pick one option; per-option
  probabilities + `confidence`), `score` (rubric level; per-level
  probabilities + `confidence`).
- **Typed surface**: `await provider.system_one(state, questions, model=…)`
  with the `Noul` / `Choice` / `Score` builders (raw dicts accepted;
  `normalize_questions()` validates locally), a `SystemOneResult` with
  `.nouls` / `.choices` / `.scores` accessors, `usage`, `request_id`
  (`x-typesafe-request-id`) and `raw`; `await provider.list_models()`
  (`GET /v1/models`).
- **Chat bridge**: `llm.chat(msg, provider_name="typesafe", questions={…})`
  sends the conversation as the state (or an explicit `state=`) and returns
  the answers as a JSON string; usage flows into cost tracking. Streaming
  and tool calling raise a non-retryable `ProviderError` (400).
- **Transport**: plain `httpx` against the two REST endpoints (no vendor
  SDK). In-provider retries for `408/429/5xx/529`, timeouts and connection
  errors with exponential backoff honouring `Retry-After` / `retry-after-ms`;
  errors map to `ProviderError` with `status_code`, `retryable`,
  `retry_after_seconds`, the request id and the server's validation detail.
- **Config**: new `[providers.typesafe]` section (`api_key`/`api_key_env_var`
  → `TYPESAFE_API_KEY`, `base_url` ← `TYPESAFE_BASE_URL`, `default_model`
  ← `TYPESAFE_DEFAULT_MODEL`, `timeout`, `max_retries`,
  `retry_backoff_initial`/`_max`, `fallback_context_length`) in
  `default_config.toml`, with a matching `provider_typesafe` section in the
  confy schema (whose `app.version` now tracks the package again).
- **Model cards**: new `ModelType.DECISION` (`"decision"`) and a builtin
  `typesafe/jev-1.13.0` card carrying the `jev-latest` / `jev-preview`
  aliases, the 64k-per-request / 32k state+longest-question budget, the
  $0.042 per 1M input tokens price (output free), rate limits and question
  types. `get_models_details()` lists the versioned id and its aliases.
- **cardctl**: `TypeSafeAdapter` (collapses the listed aliases onto the
  versioned id; registered as `typesafe` / `jev`) and a `typesafe.toml`
  enrichment overlay so regenerated cards match the builtin one.
- **Packaging**: `llmcore[typesafe]` extra (`httpx`), included in
  `llmcore[all]`; `typesafe` registered in `ProviderManager` with the `jev`
  alias.
- **Docs & examples**: `docs/TypeSafe_provider_usage.md` (surface, builders,
  answers, confidence-gated routing, chat bridge, errors/retries, limits),
  README / `CONFIG_REFERENCE.md` / `model_cards.md` updates, and
  `examples/typesafe_example.py`.
- **Tests**: 92-test offline provider suite (config/env precedence, builders,
  `system_one`, error mapping + retry loop, `list_models`, model cards, chat
  bridge, tokens, lifecycle, registration), builtin-card + registry alias
  tests, cardctl adapter tests, and a key-gated live smoke
  (`tests/integration/test_typesafe_live.py`).

### Fixed

- DeepSeek DSML text-format tool calls are normalized at the provider
  boundary; OpenRouter model cards refreshed (both landed after the 0.52.0
  entry was written).

## v0.52.0

### Changed — Grimoire is THE control plane (breaking)

- **Hard dependency on `grimoire>=0.4.0`.** llmcore ships a packaged
  `llmcore-builtin` grimoire pack (`src/llmcore/grimoire_pack/`) covering
  EVERY agent prompt as spells — the cognitive phases (`plan`, `think`,
  `validate`, `reflect`, plus the new `finalize`), the activity (XML) protocol,
  the goal classifier, the fast path (+ canned responses as
  `llmcore/fast_path/*` promptlets), the five builtin personas (definitions in
  spell `attributes.persona`), the Darwin arbiter/TDD prompts, and autonomous
  goal decomposition. Zero config → the bundled pack loads; user layers
  override by id.
- **Fail-loud, no silent fallbacks (breaking).** Every inline/f-string prompt
  fallback is DELETED. `prompt_registry` is required by
  `SingleAgentMode`/`CognitiveCycle` (raise on `None`);
  `EnhancedAgentManager` self-builds a bundled-only adapter when constructed
  directly. New `[grimoire]` config section (layers, prompt_map, strict,
  metrics) with startup validation that renders every required template and
  raises `ConfigError` naming template, spell, winning layer, and cause.
- **Prompt-consumer tail routed through the registry**: goal classifier LLM
  fallback, fast-path executor (system+user via `render_messages`; canned
  responses via promptlets), `PersonaManager.load_from_grimoire()` (spells
  tagged `llmcore.persona`; hardcoded builtins remain only for legacy
  no-grimoire construction), `MultiAttemptArbiter` + `TestGenerator`/
  `TDDManager` (`darwin_arbiter_*`/`darwin_tdd_*` templates; the
  `ArbiterPrompts` class and TDD prompt constants are gone),
  `GoalManager._decompose_goal` (`goal_decomposition` — also fixes a latent
  `MessageRole` ImportError that silently disabled LLM decomposition),
  `agents/activities/prompts.py` DELETED (activity prompts render via
  `activity_system`/`activity_execute`), and the four cognitive default
  templates removed from `template_loader` (the variable-mismatch source;
  `PromptRegistry.with_defaults()` now seeds snippets only).
- **Builtin tools are catalog-driven**: `GrimoireToolCatalog` builds the five
  builtins (`finish`, `human_approval`, `semantic_search`, `episodic_search`,
  `calculator`) from bundled rune contracts with real parameter schemas and
  risk/approval/OWASP metadata; unbound runes are visible in `contracts()`
  but never registered.
- **Deprecations**: the legacy `agents/cognitive_cycle.py` + `prompt_utils.py`
  stack (outside the control plane) — `AgentManager.run_agent_loop()` now
  emits a `DeprecationWarning`; removal next minor. Unwired
  `reasoning/react.py`/`reflexion.py` carry deprecation notes.

### Added/Fixed — Darwin convergence & accounting (plan Phase 2)

- **Finish-tool convergence** (2.1): native `finish`/`final_answer` tool calls
  terminate THINK (with act-phase defense in depth); `finish` gets a real
  `{answer}` schema; new `TerminationReason` enum stamped at every stop site
  and mirrored on streaming results (`termination_reason`).
- **`remaining_steps` + in-cycle forced finalize** (2.2): the cycle can no
  longer exit un-converged on budget paths — a tool-less finalize pass (new
  `finalize_prompt` spell, `tool_choice="required"` where supported) or the
  synthesis fallback guarantees an answer; hosts' "grace" crutches are
  redundant.
- **Deterministic guards under `skip_validation`** (2.3): tool-registry +
  dangerous-pattern prechecks always run; only the LLM judge is skipped
  (`ValidationConfig.deterministic_guards` escape hatch).
- **Redundant-call detector** (2.4): repeated tool signatures skip
  VALIDATE/ACT with a corrective observation; ≥3 repeats → forced finalize.
- **Conditional PLAN** (2.5): `PlanningConfig.mode`
  (`always|first|complex_only|on_failure`, default `complex_only`) + replan
  budget — planning no longer precedes observation on knowledge tasks.
- **Grounded structured REFLECT** (2.6): JSON reflection (where the provider
  supports it) with text fallback; `action_success=False` clamps
  self-judgment; reflection gated to action iterations; THINK's
  `expected_outcome` finally reaches OBSERVE.
- **End-to-end token/cost accounting** (2.7): `PhaseUsage` per LLM phase,
  iteration/state totals, circuit-breaker cost double-count fixed, and new
  `api.record_agent_usage(session_id, records)` so `get_session_token_stats`
  covers agent runs (hosts flush per run/segment).

## v0.51.0

### Added — Z.ai (GLM) provider

- **Z.ai provider**: first-class `ZaiProvider` for the Z.ai Open Platform,
  serving the GLM model family (`glm-5.2`, `glm-5.1`, `glm-4.7`, the `glm-*v`
  vision models, and `embedding-3`).
- **Selectable transport backend** (`backend` config): the provider prefers
  the official synchronous `zai-sdk` (`"sdk"`, the default when installed,
  bridged to async via threads), and falls back to the `openai` SDK
  (`"openai"`, OpenAI-compatibility mode) and/or direct `httpx` REST calls
  (`"httpx"`). Unset/`"auto"` auto-detects in that order. All three backends
  share one request-building path and normalize responses to a common shape.
  Chat, embeddings, and every media API honor the selected backend. Built on
  the chat endpoint (`https://api.z.ai/api/paas/v4`) with:
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
- **Z.ai multimodal media APIs**: the provider implements llmcore's full
  media surface against the Z.ai endpoints:
  - `generate_image` (CogView / GLM-Image, `/images/generations`)
  - `generate_speech` (GLM-TTS, `/audio/speech`)
  - `transcribe_audio` (GLM-ASR, `/audio/transcriptions`)
  - `ocr` (GLM-OCR layout parsing, `/layout_parsing`)
  - `generate_video` + `retrieve_video_result` (CogVideoX,
    `/videos/generations` with async task polling)
  - `web_search` (Z.ai Web Search API, `/web_search`)
- **cardctl**: new `ZaiAdapter` (OpenAI-compatible `/models` listing) and
  `zai.toml` enrichment overlay registered in the model-card tool, with
  `glm`/`zhipu`/`zhipuai`/`bigmodel` aliases; media/generation model ids are
  filtered out of chat cards.
- **Packaging**: new `llmcore[zai]` extra (`openai` + `httpx`), included in
  `llmcore[all]`.
- **Pricing**: GLM model cards and the `zai.toml` enrichment now carry
  verified USD per-1M-token pricing (input/output/cached-input) from the
  official Z.ai pricing page, plus per-unit reference notes for image/video/
  web-search services.
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
