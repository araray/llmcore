# Changelog

All notable changes to **llmcore** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added — `llmcore.routing`: pools, lanes, failover and proxy mode

Five composable layers, all off until configured. With no `[routing]` section
every call resolves exactly as it did before. Design and the reasoning behind
each decision: [`ROUTING_SUBSYSTEM_SPEC.md`](docs/ROUTING_SUBSYSTEM_SPEC.md);
usage: [`Routing_usage.md`](docs/Routing_usage.md).

**Config stops being an allow-list.** `[providers.*]` sections were acting as
one: a provider registered in `PROVIDER_MAP`, with its key in the environment
and its base URL already known, was still unreachable. Any provider+model is
now addressable by spec string, and llmcore builds the instance on demand:

```python
await llm.chat("hi", target="xai:grok-4.1-20251117?effort=high")
await llm.chat("hi", target="ollama:llama3.3:70b")
await llm.chat("hi", target="vllm:Qwen/Qwen3-30B#my-box")
```

`routing.autoprovision = false` restores the closed set for deployments that
want one. (Found along the way: `xai`, `groq` and `together` had no
`[providers.*]` section at all, so all three were unreachable despite being
registered — sections added.)

**Pools, with a failure taxonomy rather than one retry rule.** Conflating
failures is how naive failover burns money or loops, so each gets its own
handling: a 429 cools down briefly and honours `Retry-After`; an empty wallet
cools down for minutes and records a balance of zero; a 5xx is retried once
*in place* before moving, since it is usually one bad node and moving would
discard a warm prompt cache; a bad key benches the target for the process; a
400 does not fail over at all; a content refusal does not either, unless
`on_refusal = "failover"` — retrying a refusal elsewhere is "shop until
someone says yes", which should be opted into. A prompt that overflows the
context window is treated as a routing signal, not an error, and is sent to a
model with a larger window; llmcore also skips a target *before* calling it
when the card already proves the prompt will not fit.

Seven selection strategies: `priority`, `round_robin`, `weighted`,
`lowest_latency`, `lowest_cost`, `least_busy`, `most_credits`. Order tiers
(`?order=1`) express "my own GPU, and only pay a vendor if it is down".
Session affinity is on by default, because failing over mid-conversation drops
the cached prefix, shifts output style under few-shot expectations, and
invalidates preserved reasoning blocks.

Three places where the strategies refuse to guess, because treating "unknown"
as a number inverts them: an **unpriced** target ranks after every priced one;
an **unknown balance** ranks after a known one but ahead of a known-*empty*
one; `lowest_latency` explores an unmeasured target first, since it cannot
prefer low latency without a measurement. Self-hosted providers price at zero
rather than unknown, so a local model wins outright.

**Lanes and classifiers.** A classifier names a lane, never a model, so
swapping models is a config edit. That one indirection covers complexity
tiers, speed tiers, domain routing and a privacy class with one mechanism.
Nine classifiers ship, from a free `hint` to a local 350M zero-shot encoder to
TypeSafe's `choice` primitive. llmcore orders the chain itself: cheapest
first, and within a cost band instructions before guesses — a `lane=` argument
and a length heuristic are both free, and running the guess first would
override what the caller asked for.

Authority has four levels (`caller` > `policy` > `prompt` > `inferred`)
because a routing marker found in *content* is not as trustworthy as an
argument on the call. Markers are the point of the feature — in an agent
harness the model's text is the only channel that passes through, so that is
how an agent routes itself — but in a RAG path that text may have come from a
retrieved document, where `[[lane:deep]]` would be a one-line prompt
injection, or a way out of the private lane. Markers are always stripped
before egress, acted on or not.

**Cascades** (opt-in): answer cheaply, verify, escalate only if the answer
fell short. Verdicts are three-valued, and *could not judge* is kept distinct
from both — read as a fail it escalates every unjudgeable answer and inverts
the saving; read as a pass it silently disables the quality floor the moment
the judge breaks.

**The privacy path, where routing is the guarantee and redaction is not.** A
detector that misses one identifier has leaked it, and none catches
everything, so `on_detect = "constrain"` changes the *destination*: a prompt
with personal data goes to a pool that never leaves the machine, and a miss
stays on your own hardware. Redaction can be stacked and is documented as
defence in depth. Three ways it refuses to degrade quietly: `constrain` with
no pool configured blocks rather than sending; a constrain pool that does not
exist blocks; and a detector that raises blocks (`fail_closed`), because a
detector that crashed has not cleared the prompt. Findings carry a 16-char
hash, never the value — including in the exception message.

**Proxy mode** (`llmcore-bridge proxy`): an OpenAI-compatible endpoint, so an
unmodified agent harness gets all of the above by setting a base URL and a
model name. `model` accepts `lane:deep`, `pool:main`, `profile:frugal`, a
target spec, a bare model name or `auto`; lanes and pools appear in
`GET /v1/models` so the harness's own picker selects routing policy. Usage
reports the target that *actually* answered, because under a pool that is not
the model requested and a harness logging spend should not be lied to. It
binds to loopback and **refuses** a non-loopback bind without a bearer token,
since the process holds every provider credential in the config.

**Effort and parameters.** The existing effort vocabulary is extended rather
than replaced, with precedence card → provider config → target → lane →
profile → per-call keyword. Under a pool an unsupported parameter is dropped
with a warning, because members genuinely have different parameter surfaces;
outside a pool it still raises.

**Everything is overridable.** Config is a warm-up, not a cage: every
`[routing]` setting resolves config → environment → request, and the request
wins. A misspelled override raises rather than being silently dropped.

**Explaining itself.** `llm.routing.explain()`/`why()` report the lane, the
classifier that chose it, the chosen target and *why each other candidate was
not used*, without making a call. `health()` shows per-target cooldowns,
latency and balance. Every decision also emits a structured event.

New skills in the bundled grimoire pack: `skills/llmcore/proxy`,
`skills/llmcore/routing`, `skills/llmcore/cost`.

### Fixed — Anthropic rejected `thinking.budget_tokens` with a 400

Claude 4.6 deprecated `thinking.budget_tokens` and 5.x rejects it outright,
while pre-4.6 models require it. llmcore forwarded whatever the caller passed,
so a reasonable request became an API error purely because of which model
served it. Recorded in `PROVIDER_MODERNIZATION_PLAN.md` and flagged again in
the routing spec §6.4 -- pools make this routine, since members span
generations.

The provider now branches on the model generation parsed from its id, and
`effort` is a first-class parameter that means the same thing either way:

* **4.6 and later** get `thinking: {"type": "adaptive"}` plus
  `output_config.effort`; a caller's `budget_tokens` is dropped with a warning
  naming the 400 it would have caused.
* **Earlier models** get `thinking: {"type": "enabled", "budget_tokens": N}`,
  with the budget mapped from the effort level; a request for `adaptive` is
  converted rather than failing.
* **An unparseable model id is assumed modern**, because the pre-4.6 family is
  the shrinking set and defaulting the other way would break new models.
* `effort="none"` **disables** thinking on both generations rather than
  requesting adaptive-at-low, which would quietly spend reasoning tokens the
  caller explicitly asked not to spend.

`xhigh` folds to `high` on 4.6+, following the documented policy of folding to
the nearest supported rung rather than dropping an effort level and losing the
caller's intent.

### Added — the Colab runtime backend (R2-R6)

`llmcore.runtimes` can now actually provision: sizing, the Colab backend, a
supervisor that enforces the deadlines, and `llmcore-runtimes` for the
commands you need when something has gone wrong.

**Sizing** (R2) is read-only and free, and prints its own arithmetic — a sizer
that answers "use an A100" and shows nothing is impossible to argue with.
Every unknown rounds toward *needing more*: overestimating buys a bigger GPU,
underestimating OOMs on the VM after billing has started. Context shrinks
before a bigger GPU is chosen, and when nothing fits the sizer refuses with a
concrete alternative rather than sending the caller to guess — the next guess
is also a launch.

**The Colab backend** (R3) enforces four rules in code: state is written
before compute can be assigned; any bootstrap failure releases the VM; nothing
is connected to until the session actually exists; deadlines are set at
creation. If a release *also* fails, the log says MAY STILL BE BILLING in
those words. The VM-side server is started with `setsid`, because as a child
of the kernel a kernel restart would kill it silently while the VM kept
billing. Secrets travel on stdin or through `colab exec --env`, never in argv.

**Supervision** (R4). `reap()` existed but nothing called it, which made the
idle and hard deadlines documentation rather than limits; `up()` now starts a
supervisor. Liveness marks a runtime DEGRADED after three consecutive
failures, not one, and never tears it down — letting a transient network
problem destroy an expensive VM would be worse than the problem. The idle
reaper needs a last-used time and vLLM exposes no such metric, so `attach()`
wraps the provider's `chat_completion`.

**`llmcore-runtimes`** (R6) — estimate, up, status, down, logs, adopt, bake,
cache. `up` requires `--yes`; `estimate` works while the subsystem is
disabled, because deciding whether to spend should not require enabling spend.
`status` lists orphans with the command that adopts them.

Three corrections the real tools forced on the spec:

* `colab new --gpu` accepts T4, L4, G4, H100, A100 and nothing else, so the
  spec's `A100-40`/`A100-80` rungs could never have been provisioned. The
  ladder now uses the CLI's names (A100 sized conservatively at 40 GB) and a
  test asserts every SKU maps to a value the CLI accepts. The old spellings
  still resolve.
* `Quantization` had one `GGUF` member, but Q4 and Q8 differ by 2x in weight
  bytes — routinely the difference between fitting a 24 GB card and not.
* The spec's KV fallback estimated 0.5 GB for a 70B model at 8k context, about
  5x under. It now scales from the parameter count at a figure calibrated
  against models where the real numbers are available.

**Three gates are left explicitly unmet**, and the spec's phase table says so
rather than claiming completion: R3's "one real model served end to end" and
R5's "cold start is seconds" both require provisioning a real GPU VM and
spending real compute units, and R6's agent-lens migration guide is not
written because it would be telling another project to depend on an unproven
path. `cache gc` is also not implemented — `cache` lists, nothing deletes.

### Fixed — the test suite wrote into the real `~/.llmcore/runtimes`

That directory is the record of what is currently costing money — the spec
calls it a safety mechanism rather than a cache, and `llmcore-runtimes status`
reads it. Running the test suite left a phantom entry claiming a READY L4 VM
was running, which is exactly the false signal the subsystem exists to
prevent: someone checking whether they were being billed would have been told
yes, by their own test run.

### Fixed — two config keys promised behaviour that does not exist

A full audit of `default_config.toml` (660 keys) against the code found two
with no reader anywhere. Both are now marked **NOT ENFORCED** in the config
rather than quietly removed, because a control that silently does nothing is
worse than one that is absent — someone has to be able to find out:

* `llmcore.admin_api_key` documented itself as protecting administrative
  endpoints "such as live configuration reloading". Nothing reads it, and the
  bridge's `ControlService/ReloadConfig` has no reference to it, so a
  deployment that set it believing its reload endpoint was protected was
  wrong. What does protect the bridge is transport-level: mTLS and
  `--auth authflow`.
* `context_management.minimum_history_messages` — truncation does not honour
  it.

Everything else audited clean. All 48 `[routing]` keys and all 14
`[runtimes]` keys are wired; the 116 unread `semantiscan.*` keys belong to
that package, which llmcore only carries defaults for.

### Fixed — Gemini targets had no pricing or context window, silently

The model-card alias map ran the wrong way: provider type `gemini` was mapped
*to* a `gemini` card namespace, but the cards are filed under `google/`.
Nothing raised — the lookup returned `None`, `None` means "unknown", and every
caller handles unknown quietly — so `lowest_cost` could not price any Gemini
target and the pre-call context-window check never fired, for one of the most
used providers in the library. `tests/routing/test_cards.py` now audits every
provider type against the packaged card tree, read from disk rather than from
the registry singleton that other suites reset.

### Fixed — tests could load the repo's `.env` and reach real vendors

The test fixtures built confy `Config` objects with its default
`load_dotenv_file=True`, which exports `.env` into `os.environ`. That made a
credential-discovery assertion depend on the developer's machine, and it meant
a test run could reach a real vendor and spend real money. Fixtures now pass
`load_dotenv_file=False`.

### Changed — measured corrections to the routing spec

Two claims in the design document were wrong and are corrected in place rather
than quietly dropped:

- §4.4 asserted that a local classifier adds "<50 ms p50 on CPU". Measured on
  8 CPU threads with `LFM2.5-Encoder-350M-Prompt-Router`: **191 ms for 2
  lanes, 246 ms for 5, 314 ms for 9**, plus ~40 s once to load. Wrong by about
  5x, which is why the gate said *measured, not assumed*. The encoder is off
  by default, runs its forward pass in a worker thread so it cannot stall the
  event loop, and is documented as a batch/agent feature.
- The same model's raw top score is meaningless without reading it against
  chance: 0.20 across five lanes is exactly uniform, i.e. *no opinion*, and
  taking it as 20% confidence would route on noise. Confidence is now reported
  chance-corrected, so one floor means the same thing at any lane count.

The heuristic classifier also lost its short-prompt rule, because prompt length
does not predict request complexity — "write a 2000-word essay on X" is ten
tokens. Length now only ever *vetoes* the trivial lane.

### Added — `LLMCore.discard_transient_state()`

Drops the cached per-turn introspection and raw response for a session. Needed
by any long-running embedder — the routing proxy uses one synthetic session per
request, and without this those caches would grow for the life of the process.


### Added — dual transport for every provider that has a vendor SDK

llmcore's rule is to call each API directly and fall back to the vendor SDK
where one exists. Six providers did not follow it. All six now do, taking
coverage to **21 of 23**; the two that remain single-transport have no official
vendor SDK.

**Fixed a false claim first.** The transport audit asserted that "TypeSafe
publishes no Python SDK". `typesafe-sdk` 0.7.2 is official and was already
cloned in the vendor repos — the evidence was in llmcore's own support matrix.
TypeSafe now has an SDK fallback, and the remaining exemptions cite what was
actually checked: the PyPI names `deepseek` (Deskpai.com), `deepseek-sdk` (Sifat
Hasan) and `kimi-sdk` (no stated author or repository) are **not** vendor
packages, which is why DeepSeek and Kimi stay direct-only.

**SDK fallbacks added** (direct stays the default):

- **`typesafe`** — `typesafe-sdk`, via a response shim so neither existing
  parser changes. One honest limitation: request ids are unavailable on that
  path, because the SDK does not surface response headers.
- **`mistral`** — `mistralai` v3 for chat, streaming, models and embeddings.
  OCR, audio, classification, moderation and FIM stay on direct REST, because
  the SDK models those with its own typed resources rather than OpenAI-shaped
  dicts. That split is logged per call, so `backend = "sdk"` never silently
  does nothing for half the surface.

**Direct REST paths added** (the SDK stays the default, for stated reasons):

- **`anthropic`** — `/v1/messages`, streaming included. The SDK keeps the
  default because it owns prompt-caching and beta headers plus retry behaviour.
- **`gemini`** — Developer-API `generateContent` and `streamGenerateContent`.
  Vertex mode is *forced* onto the SDK, since Vertex authenticates with Google
  ADC rather than an API key.
- **`deepgram`** — `/v1/listen` and `/v1/speak`. The realtime surfaces
  (streaming STT/TTS, voice agent, Flux v2) are duplex WebSocket protocols and
  stay on the SDK, logged per call.
- **`ollama`** — `/api/chat` with NDJSON streaming, and `/api/tags`.

Throughout, there is **one normalization path per provider rather than one per
transport**: Anthropic's stream normalizer consumes event dicts from either
source, Gemini's reads a wire shim that presents REST JSON the way the SDK's
typed objects look, and Deepgram's batch parser reads a JSON attribute wrapper.
Each direct path maps failures to the same exceptions the SDK path raises,
because a dual transport that reports failures differently is not really dual.

### Fixed — five bugs found by live calls rather than by reading

- The **Mistral SDK appends its own version prefix**, so passing llmcore's
  `base_url` verbatim produced `/v1/v1/...` and a "no Route match" 404.
- The two **Mistral streaming paths returned different shapes** — the SDK path
  an async generator, the existing httpx path a coroutine resolving to one.
  Matched to the existing contract rather than "improved", since callers depend
  on it.
- **Gemini's `finish_reason`** arrives as a plain string but readers call
  `.name`, so enum-valued wire fields are now wrapped.
- **Gemini's `text` means different things at different levels** of the
  response graph — the joined non-thought parts on a response, the part's own
  string on a part. A shim property that only did the join returned `""` for
  every part and made streaming yield empty deltas.
- **Deepgram's credential attribute** is `api_key`, not `_api_key`, and the
  scheme differs between an API key (`Token`) and an access token (`Bearer`).
- **Ollama double-wrapped its own errors**: a `ProviderError` raised by the
  direct path was re-caught by the generic handler and reported as "An
  unexpected error occurred", burying the actionable message and making the two
  transports describe the same condition differently.

### Changed — the transport audit now checks more

`test_transport_duality.py` gained checks that a provider whose SDK stays the
default still has a direct client, a transport selector and its own error
mapper. It also walks the MRO, since `deepinfra` and `vllm` inherit their
selector from `OpenAIProvider` — reading only a provider's own module
misclassified both as single-transport.

### Added — Higgsfield provider

- **New `higgsfield` provider**: generative image (Soul) and video (hosted Kling
  and MiniMax Hailuo) behind one async request API. Media-only;
  `chat_completion()` raises. Direct REST by default, following the fal pattern
  deliberately rather than inventing a second one for the same shape.
- **Credentials are a pair.** The scheme is `Authorization: Key {id}:{secret}`,
  not one opaque token, so `api_key_id` / `api_key_secret` can be configured
  separately and a single-token credential warns at construction rather than
  failing with a 401 much later.
- **`nsfw` is handled as its own terminal state.** Higgsfield reports it
  alongside `failed`, but one is a content refusal and the other a malfunction.
  Both end the job; the error says explicitly that a refusal will not succeed on
  retry, and the raw state is preserved in `provider_metadata`.
- **Text-to-video and image-to-video are separate endpoints**, so supplying a
  conditioning image switches the path instead of posting an image to a
  text-only endpoint and getting a 422. An unmapped model warns rather than
  silently failing.
- Opts into the webhook receiver and parses its own deliveries; a mismatched
  `request_id` falls back to polling.
- **Dual transport, per the house rule**: direct REST by default, with
  `higgsfield-client` 0.2.0 as the `backend = "sdk"` fallback covering the whole
  `submit` / `status` / `result` / `cancel` lifecycle. Live-validated end to end
  on **both** transports.

### Added — `test_transport_duality.py`: the direct-vs-SDK audit, enforced

llmcore prefers calling each API directly with the vendor SDK as a fallback, but
nothing checked it. The Higgsfield provider initially shipped with a
`backend = "sdk"` option that its resolver accepted, an SDK import, and an
`self._sdk` attribute that was **never assigned and never called** — so
selecting it silently did nothing.

The audit now encodes the rule: a provider offering one transport must declare
why, dual-transport providers must actually construct *and call* their SDK
client, direct must be the default, and every SDK fallback must be installable
from its extra. Known gaps are listed with a `GAP:` prefix so they read as
outstanding work; a provider that gains a second transport and stays on the list
fails the audit.

The audit walks the MRO rather than a provider's own module, because
`deepinfra` and `vllm` inherit their selector from `OpenAIProvider` and have
dual transport without a line of their own about it — the first version read
only the own-module source and misclassified both.

Result: **15 dual-transport**, 3 single by design (`deepseek`, `kimi`,
`typesafe` — no vendor SDK published), and **5 recorded gaps**: `anthropic`,
`gemini`, `deepgram` and `ollama` are **SDK-only with no direct path**, and
`mistral` is **direct-only with an unused mistralai v3 SDK**.

### Fixed — cardctl was missing adapters for eight providers

`fal`, `elevenlabs`, `replicate`, `deepgram`, `groq`, `together`, `vllm` and
`higgsfield` were registered in `PROVIDER_MAP` with **no cardctl adapter**, so
their model cards could never be generated. `openai_compat`'s own docstring had
claimed Groq and Together since it was written, but neither was ever registered,
so `cardctl generate groq` simply failed.

Nothing caught this because `generate` only reports on the provider you name and
`stats` only sees providers that already have cards — a provider with no adapter
was invisible to both.

### Added — `cardctl doctor`, and a test that enforces coverage

- **`cardctl doctor`** cross-checks llmcore's provider registry against the
  adapter registry and the cards on disk, separating *errors* (a registered
  provider with no adapter — cards can never be generated) from *warnings* (no
  cards yet, or a key not set) and *info* (an adapter with no provider, usually
  an alias). `--strict` fails on warnings too.
- `tests/tools/test_cardctl_coverage.py` asserts the same invariant, so adding a
  provider without an adapter now fails the suite instead of shipping quietly.

### Added — cardctl support for providers with no catalog endpoint

- **`CuratedAdapter`**: emits cards from a declared set for providers that
  publish no listing route — fal and Higgsfield address models by endpoint path,
  Replicate's catalog is tens of thousands of community models, and vLLM serves
  whatever a deployment loaded. Every such card is tagged **`curated`**, so a
  declared entry is never mistaken for a discovered one.
- **`ReplicateAdapter` enriches** each curated entry from the model's live
  schema, recording the owner's description and the model's real input field
  names — the same schema the provider reads at runtime, so the card documents
  what will actually be sent.
- **`VLLMAdapter`** requires `--base-url` and says why: there is no vendor
  catalog, and defaulting to localhost would silently card whatever happens to
  be running.

### Added — media capabilities on model cards

- `ModelCapabilities` gains `image_generation`, `image_edit`, `image_upscale`,
  `video_generation`, `video_interpolation`, `speech_synthesis`,
  `transcription`, `music_generation`, `sfx_generation` and `voice_design`,
  mirroring `llmcore.media.MediaCapability`. `ModelType` gains
  `video-generation` and `media`. All default off, so existing chat cards are
  unaffected.
- The card builder maps them through. Without this the media cards claimed **no
  capabilities at all**, which reads as "this model does nothing" rather than
  "nobody filled this in".

### Fixed — two cardctl bugs found while generating

- **Adapter aliases built duplicate card trees.** `cardctl generate gemini`
  created a second `gemini/` directory alongside `google/`, because the
  directory is named after whatever string the caller passed.
  `cards_dir_for_provider()` now canonicalizes aliases, so there is exactly one
  directory per provider.
- **Deepgram cards were written once per model *and language*.** Its
  `/v1/models` lists a record per language, so 553 records collapsed into 144
  files with whichever variant came last deciding each card's language metadata.
  Records are now grouped by canonical name with their language coverage merged.

### Changed — all model cards regenerated

2319 cards across 22 providers, 0 validation failures.

### Added — `llmcore.runtimes` core: remote GPU runtimes (R1)

- **New `llmcore.runtimes` subsystem**, reachable as `llm.runtimes`: provision
  compute elsewhere, serve an open-weights model on it, and attach the endpoint
  as a provider instance — so a remotely served model is reachable through the
  same `llm.chat()` as any hosted API. Phase R1 of
  `COLAB_RUNTIME_SPEC.md`: core types, the `ComputeRuntime` protocol, the state
  store, `RuntimeManager`, and a `FakeRuntime`. **No real backend yet, so
  nothing can spend money.**
- **The safety model is enforced, not just documented.** Unlike every other
  provider, a runtime bills per minute from the moment it is assigned, whether
  or not anyone calls it. So: the subsystem is **off by default**;
  `LLMCore.create()` builds the manager without contacting any backend; `up()`
  raises `SpendNotConfirmedError` unless confirmation is explicit; and
  `estimate()` is free and works **while disabled**, because deciding whether to
  spend should not require enabling spend.
- **State is written before provisioning returns**, because the dangerous window
  is a crash between assignment and bookkeeping — money burning with nothing
  tracking it. One indented-JSON file per runtime under `~/.llmcore/runtimes`,
  so anyone who suspects they are being billed can find out with `ls` and `cat`.
- **Bounded by default**: idle (45 min) and hard-lifetime (240 min) deadlines
  come from config, not from the caller remembering. A **compute-unit ceiling**
  is checked *before* either, because an idle reaper does not protect against a
  runtime that is busy in a loop.
- **Fail closed**: if attach fails after `up()` succeeded, the runtime is
  released rather than left burning, and the error says so.
- **`close()` detaches; it does not tear down.** A process exiting is not a
  reason to destroy compute someone is paying for and may still want, so
  `LLMCore.close()` unregisters the provider instances and leaves the state
  files for the next session. `down_all()` is the explicit way to stop spending.
- Provider attachment needs **no new provider class**: the endpoint is
  OpenAI-compatible, so `attach()` registers a `vllm`-type instance
  (`ephemeral=True`, `replace=True`) and `api_style` decides the type, so a
  future TGI or llama.cpp recipe attaches a different one without touching the
  runtime layer.
- Robustness where the state directory is already wrong: an unknown `phase`
  parses as `DEGRADED` rather than raising, and one corrupt record is skipped
  with a warning rather than failing the listing — a bad file must not hide the
  runtimes still running.
- New config section `[runtimes]`, disabled by default.

### Added — direct `httpx` transport for OpenAI and its four subclasses

- **`OpenAIProvider` now has a direct REST transport** alongside the `openai`
  SDK, selected with `transport = "httpx"`. Because this class is the base for
  **`deepinfra`, `vllm`, `poe` and `openrouter`**, one transport gives five
  providers a dual approach at once — the highest-leverage item in Phase 3 of
  `PROVIDER_MODERNIZATION_PLAN.md` (§5.8).
- **The key is `transport`, not `backend`.** OpenRouter and Poe already use
  `backend` to choose between their native vendor SDK and OpenAI-compatible
  mode; overloading it would have made one of the two settings unreachable.
- **The default stays `"sdk"`, deliberately.** Four subclasses' test suites
  mock `AsyncOpenAI`, so flipping the default would route five providers past
  their own tests — the same failure the Z.ai SDK backend caused when it
  changed auto-resolution and broke 21 tests. Direct transport is opt-in per
  instance.
- **The transports are interchangeable**, which is the actual requirement:
  every `extract_*` method parses response *dicts*, so the direct path returns
  exactly what `model_dump(exclude_none=True)` produces and maps failures to
  the same typed exceptions — `ContextLengthError`, and the actionable
  model-not-found `ProviderError` that names the provider's default model.
  Verified live against OpenAI: identical response keys, identical extracted
  tool calls, identical 137-model listings, streaming on both. Also verified
  live on DeepInfra, confirming the base-class leverage is real.
- Request shaping — parameter validation, native search, reasoning-model
  parameter naming, tool payloads — happens **before** the transport branch, so
  the direct path inherits it rather than reimplementing it. That is what keeps
  the two from drifting.
- **OpenRouter's `HTTP-Referer` / `X-Title` headers reach the direct transport
  too**, so both identify the application to OpenRouter the same way.
- Streaming parses server-sent events, treats `[DONE]` as a sentinel rather
  than JSON, and skips an unparseable chunk instead of failing the stream.
- New config: `transport` and `[providers.<name>.default_headers]`.

### Added — Hugging Face media adapter and Inference Endpoints (M8)

- **The `huggingface` provider is now a media adapter** for `image_generate`,
  `tts` and `asr`, behind the media protocols. It remains a chat provider too —
  the first adapter to be both.
- **Dedicated Inference Endpoints are the custom-weights / private-repo path**,
  which is this phase's gate. A private model is not a model id on a shared
  router; it is a deployment you own. Configure one under
  `[providers.huggingface.endpoints]` and it is used verbatim — no routing
  lookup, no model id in the path — and that capability switches to direct HTTP.
- **Provider routing is discovered, not guessed.** A model is not served for
  every task by every provider, and each provider knows the model by *its own
  id*: the Hub says `black-forest-labs/FLUX.1-schnell`, fal-ai says
  `fal-ai/flux/schnell`. llmcore reads the Hub's `inferenceProviderMapping`
  (cached) and resolves provider, id and URL shape. A failed lookup degrades to
  `hf-inference` rather than raising.
- **TTS artifacts carry no `VoiceConsent`**: HF serves open-weight voices and
  tracks no per-voice consent, so claiming anything there would be inventing it.

### Changed — Hugging Face is llmcore's one exception to direct-REST-first

Every other media adapter defaults to direct REST. Hugging Face defaults to the
SDK for router traffic, and the reason was measured rather than assumed: **the
router hands third-party providers their own request shape.** `hf-inference`
takes `{"inputs": ...}`, while the same model routed to fal-ai wants
`{"prompt": ...}` and answers `422 Field required` otherwise. That shape belongs
to the provider and changes on their schedule, so reimplementing the mapping
would mean tracking N third-party schemas forever — absorbing it is what
`huggingface_hub` exists for.

The dual approach is intact: both transports ship and `media_backend` selects.
The split is per capability — SDK for router JSON bodies, direct HTTP for
dedicated endpoints and for binary-input tasks such as ASR, where the SDK sends
raw audio with **no `Content-Type`** and hf-inference rejects it outright.

### Added — Replicate provider: one adapter for the whole catalog (M7)

- **New `replicate` provider**: image generate/edit/upscale, video generation,
  ASR, TTS and music — **seven capabilities, zero per-model classes**, which is
  what the spec required. Media-only; `chat_completion()` raises.
- **Model-schema descriptors are the design.** Replicate hosts tens of thousands
  of community models, so nothing can be hardcoded. Every model publishes an
  OpenAPI schema naming its own inputs, and the adapter maps llmcore's canonical
  protocol arguments onto whatever that model actually calls them. Verified
  live: `n` became `num_outputs` for `flux-schnell` while `audio` stayed `audio`
  for `whisper`. A model llmcore has never heard of works.
- Explicit keyword arguments always beat the mapping — the caller knows their
  model better than a candidate list does. A schema lookup failure **degrades**
  to canonical spellings rather than raising, because a 422 naming the field is
  more useful than a silently dropped input.
- **Ready for the webhook receiver**: declares `accepts_webhook_url` and parses
  its own deliveries; a payload whose `id` does not match falls back to polling.
- **Dual transport**: direct REST by default, `replicate` SDK opt-in. New
  `replicate` extra.

### Fixed — Replicate community models were unreachable

Found by live validation: Replicate has **two** prediction-creation routes and
the model reference does not say which one applies. *Official* models run
unversioned at `/v1/models/{owner}/{name}/predictions`; *community* models —
including `openai/whisper` — `404` there and must be run by version at
`/v1/predictions`.

The obvious fix, trying the first and falling back on the 404, costs **two**
creation requests. Replicate throttles accounts under $5 of credit to a burst of
**1**, so that fallback reliably turned a working call into a `429`. The version
is now resolved from the model lookup already made for the input schema — a
`GET`, which does not count against prediction-creation limits — so a community
model runs in a single request. *A retry-based fallback is not free when the
operation being retried is the rate-limited one.*

Also: `402` is reported as a billing stop that states the token is valid, and
`429` explains the under-$5 burst limit, which otherwise reads as a bug.

### Changed — Replicate MIME types fall back to the requested format

Replicate output URLs often carry no file extension, so MIME detection returned
`None`. It now falls back to the `output_format` that was *requested* — grounded
in the call rather than guessed — and stays `None` when neither is available,
because callers key decode paths off this field and a plausible lie is worse
than an honest unknown.

### Added — generic webhook receiver for media jobs (M9a)

- **`llmcore.media.webhooks`**: a vendor-neutral callback receiver. `WebhookRegistry`
  issues signed, single-use callback URLs; `create_webhook_app()` returns a plain
  ASGI app (no web framework is imposed) that can be mounted anywhere.
- **Polling remains the fallback, by design.** `wait()` races the callback
  against its existing backoff sleep and polls anyway, so webhook and poll
  delivery converge on one code path instead of two. A missing, late, duplicated
  or malformed callback costs latency, never correctness — and
  `webhook_base_url = ""` (the default, and the normal case with no public
  ingress) stays a fully supported mode.
- **Reserve-then-bind.** Vendors want the callback URL at submission, before the
  job exists, so a token is reserved, handed over, then bound to the job that
  comes back. A delivery arriving in that window is refused. Tokens are
  HMAC-signed, verified in constant time, single-use, and bound server-side so
  they cannot be transplanted onto another job.
- **Callback URLs are only offered to adapters that opt in** via
  `accepts_webhook_url`. Most media adapters forward unknown keyword arguments
  into the vendor payload — fal does so deliberately — so passing one blindly
  would post the callback URL to a model as a generation parameter.
- **fal now parses its own callbacks** and receives a per-job URL. A delivery
  whose `request_id` does not match falls back to polling: anyone who learns a
  callback URL can POST to it, and a wrong artifact is worse than a slow one.
- Providers that cannot parse a callback still fall back to a single poll — the
  delivery says *when* to look even when it cannot say *what* happened.
- New config: `media.jobs.webhook_base_url`, `media.jobs.webhook_secret`.

### Added — ElevenLabs provider, and consent as first-class metadata (M6)

- **New `elevenlabs` provider** (alias `eleven_labs`): TTS, streaming TTS, batch
  STT, sound effects, music and voice design. Media-only — `chat_completion()`
  raises with a pointer to `llm.media`.
- **New `VoiceConsent` type on `MediaProvenance`.** Synthetic speech raises a
  question no other media kind does: a generated image resembles no one in
  particular, but a cloned voice belongs to a person who either did or did not
  agree to it. ElevenLabs tracks that state only on the *voice* resource, so
  llmcore resolves it (cached per voice) and attaches it to the artifact —
  a caller can refuse audio from an unverified clone without a second API call.
- **`verification_satisfied` is deliberately tri-state.** `None` means *the
  provider said nothing*, which is not `False`. Collapsing the two would force a
  default — silently clearing unknown voices, or refusing audio from every
  provider that reports nothing — and that is policy belonging to the caller.
  The same reasoning drives three related choices: a failed consent lookup
  yields `provider_declared=False` with `None` fields rather than raising (you
  keep your audio, and the `None`s read as *we do not know*); designed voices
  state `category="generated"` explicitly rather than leaving consent open; SFX
  and music carry no consent record at all, because nothing there is a voice.
- **New `VoiceDesignProvider` protocol.** `VOICE_DESIGN` had been mapped to
  `TTSProvider` as a placeholder since M1. Voice design is not TTS — it returns
  candidate *voices* from a description, each preview carrying the id needed to
  keep it. The M1 protocol-coverage invariant caught the placeholder the moment
  the real protocol landed.
- **Dual transport**: direct REST against `api.elevenlabs.io` by default, with
  the `elevenlabs` SDK as an opt-in backend. New `elevenlabs` extra.
- Default models verified against the live `/v1/models` lineup rather than
  assumed: `eleven_v4` for TTS (GA, standard rate, 85 languages), `scribe_v1`
  for STT, `music_v2_5`, `eleven_text_to_sound_v2`, `eleven_ttv_v3`.

### Fixed — ElevenLabs plan gating was reported as an authentication failure

Found by live validation: a perfectly valid API key on a free plan returns
`402 paid_plan_required` or `403 feature_not_available` for music and voice
design. The first implementation mapped that 403 to *"authentication failed —
check ELEVENLABS_API_KEY"*, which would send a caller hunting a credential
problem they do not have. Both are now reported as plan gating, stating
explicitly that the key is valid. A genuine `403` still reads as auth.

### Changed — ElevenLabs SFX refuses a video reference rather than ignoring it

The `SFXProvider` protocol accepts a `video` reference for video-conditioned
foley; ElevenLabs sound generation is text-conditioned only. Passing one raises,
because silently dropping it would return audio unrelated to the footage the
caller supplied, with nothing to indicate why.

### Not included — ElevenLabs realtime STT

`ASR_STREAM` routing names ElevenLabs, but realtime transcription is duplex —
the caller pushes audio *and* consumes events — and needs the session shape
Deepgram established. The adapter deliberately does **not** declare
`asr_stream`, so routing falls through to Deepgram rather than advertising a
capability that would fail.

### Added — fal provider: the media marketplace adapter (M5)

- **New `fal` provider** (alias `fal_ai`), llmcore's broadest single media
  adapter: `image_generate`, `image_edit`, `image_upscale`, `video_generate`,
  `video_interpolate`, `sfx`, `music`, `tts` and `asr`. It is **media-only** —
  `chat_completion()` raises with a pointer to `llm.media`.
- **The provider-neutrality gate passed: no core type changed.** fal is
  structurally unlike the first four adapters — a marketplace rather than a
  vendor, everything queued rather than only the slow things, inputs addressed
  by URL rather than by bytes, output schemas that vary per model. It needed no
  new `MediaJob` field, no new `MediaExecution` member and no change to
  `MediaJobManager`. See `MEDIA_SUBSYSTEM_SPEC.md` §5.2.
- **Every capability is an async job**, including image generation. The same
  `llm.media.images.generate(...)` call returns a `MediaResult` on OpenAI and a
  `MediaJob` on fal; `llm.media.wait()` absorbs both, because execution class is
  declared per capability rather than per provider.
- **Per-capability endpoints are configurable** under `[providers.fal.models]`.
  fal model ids are endpoint paths and the gallery moves faster than a release
  cycle, so llmcore ships a small starting set rather than a catalog. Unknown
  keyword arguments are forwarded verbatim, so model-specific fields work
  without an llmcore change.
- **Dual transport**: direct REST against `queue.fal.run` by default, with the
  `fal-client` SDK as an opt-in backend. New `fal` extra.
- **Cancellation is reported honestly.** fal answers `202
  CANCELLATION_REQUESTED` and work already running may still complete and still
  bill, so the job is marked cancelled *and* the caveat is recorded in
  `provider_metadata`. A `400 ALREADY_COMPLETED` is treated as success — the
  result is fetched and the job succeeds. This is the third distinct
  cancellation semantic across adapters (Gemini refuses, Deepgram n/a, fal
  best-effort) and the shared handle models all three.

### Fixed — three fal API contract bugs, all found by live validation

None of these were visible from the docs or from mocked tests:

- **Queue routes are namespaced by application, not by model path.** A request
  submitted to `fal-ai/flux/schnell` is tracked at `fal-ai/flux/requests/{id}`;
  polling the full model path returns `405`. The adapter now prefers the
  absolute `status_url` / `response_url` / `cancel_url` fal returns at
  submission and only falls back to a reconstructed, app-scoped path.
- **Uploads go to a different host, with two backends.** `fal.run` reads
  `storage/upload` as an owner/app pair and 404s; storage lives at
  `rest.fal.ai`. There, `storage_type=gcs` answers *"Invalid storage type"* for
  newer accounts. The adapter now mirrors the official client: CDN v3 first,
  signed-URL flow as fallback, both causes reported if both fail.
- **FILM takes an explicit image pair**, not a frame list — a frames list
  returned `422 Field required` for `start_image_url` / `end_image_url`.
  `interpolate_video_media()` now maps the frames it is given onto that pair.

### Added — Gemini media adapter, the first async-job provider (M4)

- **Gemini is now a media adapter** for `image_generate`, `image_edit`,
  `image_upscale`, `tts` and `video_generate`, plus provider-level embeddings.
- **Veo makes it the first true async-job provider.** `generate_video_media()`
  returns a `MediaJob`; `poll_media_job()` refreshes it through the
  long-running-operations API. **The `MediaJob` abstraction absorbed a real
  vendor's operation shape without modification** — which is what this phase
  was meant to find out.
- **Capability declaration is now mode-aware.** Imagen's `generate_images`,
  `edit_image` and `upscale_image` are Vertex-only: the Developer API rejects
  them with *"only supported in Gemini Enterprise Agent Platform mode"*
  (verified live). `image_edit` / `image_upscale` are therefore advertised only
  when `vertex_ai = true`. This is the first provider whose capability set
  depends on configuration rather than on its class.
- **Image generation has two transports.** Imagen on Vertex;
  `generate_content` with an `IMAGE` response modality on the Developer API.
  Same capability, different call, invisible to the caller.
- **`cancel_media_job()` raises rather than lying.** Veo offers no
  cancellation, and reporting success would let a caller believe billing had
  stopped when it has not.
- Uses google-genai's non-deprecated `source=` argument for `generate_videos`
  (`prompt=`/`image=` are deprecated, removal no earlier than 2026-07-31).
- Generated imagery and video carry `MediaProvenance(watermarked=True)`, since
  Google watermarks its generative output.
- Unlike OpenAI, Imagen supports `negative_prompt` natively, so it is forwarded
  rather than dropped — the protocol parameter is honoured wherever the vendor
  honours it.

Live-validated: a 2 MB PNG through the Developer-API path, 112 KB of 24 kHz PCM
from native TTS, 256-dimension embeddings, and a Veo job submitted and polled
through `MediaJobManager`. 52 new tests; full unit suite 5470 passed.

### Added — OpenAI media adapter (M3)

- **OpenAI is now a media adapter** for `image_generate`, `image_edit`, `tts`,
  `tts_stream` and `asr`. Image generation, TTS and ASR delegate to the existing
  provider methods; **image editing (`POST /v1/images/edits`) and streaming TTS
  are new**.
- **`create_embeddings()` on the provider** — previously OpenAI embeddings were
  reachable only through the separate `[embedding.openai]` subsystem, so a
  caller holding a provider could not embed with it. Closes the gap recorded in
  the support matrix.
- **Sora is deliberately not offered.** `openai` 3.1 deprecated the video APIs,
  so `video_generate` is absent and a test asserts it stays that way.
- Parameters with no OpenAI equivalent (`seed`, `negative_prompt`,
  `sample_rate_hz`) are dropped with a debug log rather than forwarded, where
  forwarding would 400. Supplying `reference_images` routes to the edit
  endpoint, which is how OpenAI expresses reference-conditioned generation.

### Fixed — audio format was hard-coded on upload

- **`OpenAIProvider.transcribe_audio()` labelled every raw-bytes upload
  `audio.wav`.** OpenAI infers the container format from the upload filename, so
  passing mp3 bytes was rejected with *"This model does not support the format
  you provided"*. Found by feeding a TTS artifact straight back in as an ASR
  input — the exact chaining the media subsystem makes natural.
- `transcribe_audio()` gained an optional `filename` parameter (defaulting to
  the previous `"audio.wav"`, so existing callers are unaffected), and the media
  adapter derives the right name from the `MediaRef`'s mime type, filename or
  URL.

### Changed — capability-less providers are no longer registered as adapters

- Four providers subclass `OpenAIProvider` and therefore inherit its media
  protocol *methods* — but not the endpoints behind them. Each now declares its
  own `_MEDIA_CAPABILITIES` (DeepInfra: image/TTS/ASR; vLLM, Poe and OpenRouter:
  none), and a test asserts **every** subclass declares explicitly, so a future
  one cannot silently inherit and advertise endpoints that 404.
- `MediaManager.from_provider_manager()` now skips providers that implement the
  protocols but declare no capabilities, so `adapter_names` keeps meaning "can
  actually do something".

Live-validated: TTS (55 KB mp3) → artifact → ASR round trip transcribed
correctly, streaming TTS, and 256-dimension embeddings. 42 new tests; full unit
suite 5410 passed.

### Added — Deepgram behind the media protocols (M2)

- **Deepgram is now a media adapter**, implementing `MediaCapableProvider`,
  `ASRProvider`, `TTSProvider`, `StreamingTTSProvider` and
  `StreamingASRProvider`. It declares exactly the five capabilities it can
  serve (`asr`, `asr_stream`, `tts`, `tts_stream`, `voice_agent`) and no
  execution class it does not have — Deepgram is request/response or live
  stream only, never an async job.
- Deepgram was chosen as the **reference migration** because it is the only
  integration that already exercises batch STT, realtime WebSocket STT *and* a
  bidirectional voice agent, so it stress-tests the hard parts of the
  abstraction before any new vendor lands.
- **The new methods delegate; they do not duplicate.** `transcribe_media`,
  `synthesize_speech_media`, `stream_speech_media` and
  `open_transcription_session` translate `MediaRef` in and `MediaArtifact` out,
  then call the existing implementations — one code path per operation rather
  than two that can drift. A remote `MediaRef` is handed to Deepgram's own
  `transcribe_url` path rather than downloaded locally.
- **All twelve provider-specific methods are untouched** and still return the
  legacy types; the three existing Deepgram test suites pass unchanged.

### Added — `models_multimodal` ↔ `MediaArtifact` bridge

- `SpeechResult`, `TranscriptionResult`, `OCRResult`, `GeneratedImage` and
  `ImageGenerationResult` gained `to_artifact()` / `to_artifacts()`, and the two
  round-trippable ones gained `from_artifact()`. These types are public API
  returned by seven providers, so they are **bridged, not replaced** (spec §4.3).
- Data that must survive the conversion does: audio format → MIME type,
  diarization segments and timings, `revised_prompt`, OCR page structure. Base64
  image payloads are decoded to real bytes, because the media layer deals in
  bytes; malformed base64 degrades to the URI path instead of raising.

Live-validated end to end: TTS through the router produced 146 KB of WAV, the
resulting artifact was fed straight back in as an ASR input via
`MediaRef.from_artifact()` and transcribed correctly, and streaming TTS yielded
39 chunks. 74 new tests; full unit suite 5372 passed.

### Added — media subsystem core (M1)

- **`llmcore.media`**, reached through `llm.media`: a sibling subsystem to chat
  providers and search providers for generative image, audio and video.
  Implements phase M1 of `docs/MEDIA_SUBSYSTEM_SPEC.md`; no vendor adapters yet,
  which is the gate the spec requires before any provider work lands.
- **Three execution classes, not one** — `MediaResult` for request/response,
  `AsyncIterator[bytes]` for streams, `MediaJob` for long-running work. Image
  generation, TTS and video generation genuinely differ, and collapsing them
  into one shape is the modelling mistake the design avoids.
- **Types**: `MediaKind`, `MediaCapability` (19 capabilities), `MediaExecution`,
  `MediaJobStatus`, `MediaRef` (url/path/bytes/artifact inputs, so callers never
  hand-roll base64), `MediaArtifact` (with `expires_at` + `checksum_sha256`,
  because every aggregator returns short-lived URLs), `MediaProvenance`,
  `MediaUsage` (keeps the vendor's native billing units rather than inventing a
  token count), `MediaResult` and `MediaJob`.
- **Capability protocols** (`typing.Protocol`, runtime-checkable) — routers
  discover what an adapter can do with `isinstance`, so routing logic never
  names a provider. A capability declared but not backed by its protocol is
  dropped with a warning rather than failing at call time.
- **`MediaManager`** with per-modality routers (`images`, `audio`, `video`),
  capability discovery (`capabilities()`, `who_can()`), and a documented
  resolution order: explicit provider → explicit model → `[media.routing]` →
  built-in defaults → any capable adapter. **Adapters are the chat providers**:
  any `[providers.*]` instance implementing the protocols becomes a media
  adapter, so there is one credential per vendor and nothing to duplicate.
- **`MediaJobManager`** owns polling, capped exponential backoff with jitter,
  timeouts and cancellation, so no adapter writes its own poll loop. A timeout
  raises **without cancelling the job** — the handle stays valid and can be
  waited on again, because an expensive video generation must not be discarded
  over a client-side deadline.
- **`ArtifactStore`** — content-addressed, sharded by SHA-256, atomic publish,
  with `always` / `on_expiry` (default) / `never` materialization policies. The
  byte fetcher is injected, so the store has no hard dependency on `httpx`.
- **`FakeMediaProvider`** ships inside the package (not under `tests/`) so
  downstream projects building adapters can use it too. It implements every
  protocol, and is what the 134 new tests run against — no network, no account.
- **`[media]` config section** (artifact policy/path, `[media.routing]`
  preferences per capability, `[media.jobs]` poll/timeout policy). Entirely
  optional: omit it and `llm.media` still works on built-in defaults.
- **Backward compatible**: `BaseProvider`'s five media methods and the
  `models_multimodal` result types are untouched. Providers gain routing when
  they are migrated in M2 onward; until then `llm.media.adapter_names` is empty
  and reports so honestly.

### Added — dynamic provider registration

- **`ProviderManager.register_instance()` / `unregister_instance()`** plus
  `is_ephemeral()` / `ephemeral_instances`. Providers were previously only
  constructible during `__init__`; subsystems that *create* endpoints need to
  add one afterwards. This is the single capability shared by the media program
  and the remote-runtime program (`docs/COLAB_RUNTIME_SPEC.md`), where a Colab
  VM's OpenAI-compatible endpoint is registered as a `vllm` instance.
- Guards that matter: a name collision raises unless `replace=True` (so a live
  provider is never silently swapped out from under its callers), the configured
  default provider cannot be unregistered, construction failures surface as
  `ConfigError`, and a failing `close()` during unregister is logged rather than
  blocking teardown.

### Fixed

- **Job-polling backoff could overflow.** `2 ** attempt` stops converting to
  float past ~1024 polls, so the wait loop would die with `OverflowError` — a
  multi-hour video job polled every few seconds would actually reach that. The
  exponent is now capped; regression tested at 10,000 polls.

### Added — media and remote-runtime specifications

- `docs/MEDIA_SUBSYSTEM_SPEC.md` — design and specification for a first-class
  `llmcore.media` subsystem: `MediaArtifact` / `MediaUsage` / `MediaJob`,
  capability `Protocol`s per modality, the three execution classes
  (request/response, byte stream, async job), capability-oriented model cards
  with the aggregator sourcing split, generic webhooks with polling fallback,
  an artifact store, a selection policy, and a nine-phase vendor rollout
  (Deepgram refactor → OpenAI → Google/Veo → fal → ElevenLabs → Replicate → HF
  Endpoints → direct specialists). Includes a backward-compatibility path that
  keeps `BaseProvider`'s five media methods and the `models_multimodal` types
  working. Notably corrects the source research: **OpenAI's Sora video APIs
  were deprecated in `openai` 3.1**, so frontier video comes from Veo and
  fal-hosted models instead.
- `docs/COLAB_RUNTIME_SPEC.md` — design and specification for a
  `llmcore.runtimes` subsystem that provisions and controls remote GPU runtimes
  (Google Colab first) and attaches the resulting OpenAI-compatible endpoint as
  a provider instance, so a remotely served model is reachable through the
  normal `llm.chat(provider_name=...)` path. Built on the study of agent-lens's
  implemented design and BellaVox's process; proposes that llmcore own the
  abstraction and agent-lens delegate to it. Adds a spend-ceiling
  (`max_lifetime_minutes`) on top of the reference idle reaper, since an idle
  reaper does not protect against a busy runaway runtime.

Both documents are specification only — no implementation.

### Changed — provider SDK majors (Phase 0 of the modernization plan)

- **`openai` `>=3.0.0,<4`** (was `>=2.31.0`), **`anthropic` `>=1,<2`** (was
  `>=0.94.0`), **`google-genai` `>=2,<3`** (was `>=1.72.0`), plus
  `ollama>=0.6.3`, `deepgram-sdk>=7.11.0`, `zai-sdk>=0.2.3`. All three majors
  were adopted at once because `openai` 3.x and `anthropic` 1.x share the same
  breaking change: their HTTP layer moved from `httpx` to **httpx2**.
- **New extras for six providers that had none**: `mistral`, `kimi`, `poe`,
  `openrouter`, `vllm`, `huggingface`. These import `httpx` but previously
  worked only because `openai` installed it transitively — under `openai>=3`
  they would have failed at import. All six are in `[all]`.
- **TLS behaviour change documented**: httpx2 verifies against the OS trust
  store, not `certifi`, which can break minimal containers and TLS-inspecting
  proxies. `CONFIG_REFERENCE.md` gained an "HTTP transport and TLS" section
  with the `SSL_CERT_FILE` / `SSL_CERT_DIR` escape hatches.
- **No provider code changes were needed for httpx2**: llmcore only ever passes
  numeric timeouts to the vendor clients, never `httpx` objects, and no `respx`
  test routes traffic through a vendor SDK. Verified before bumping.
- **Z.ai's native SDK backend is now exercised.** `zai-sdk` was never installed,
  so the provider's preferred transport was dead code in CI. Its tests now pin
  `backend` explicitly (the pattern the plan's §9 mandates) instead of depending
  on what happens to be installed, so the SDK can be installed safely — and the
  SDK backend is validated live for the first time.
- **CI installs `.[dev,all]`** instead of a hand-maintained extras subset, so a
  new extra is exercised the moment it is added. The previous `zai-sdk`
  carve-out is gone, since tests can no longer be bypassed by an installed SDK.

Live-validated after the upgrade: OpenAI (3.22.1), Google Gemini (2.25.0, 47
models discovered), Z.ai (SDK backend), DeepSeek. Full unit suite green (5164
passed). **Anthropic 1.9.0 is import- and test-verified but not live-validated —
no `ANTHROPIC_API_KEY` is available in this environment.**

### Added — provider audit documents

- `docs/PROVIDER_SUPPORT_MATRIX.md` — the ongoing tracker: per provider, the
  vendor SDK clone with tag/commit/date, our pin, the installed version, the
  transport shape, and a capability matrix extracted from the provider classes.
  Section 6 is a runnable refresh procedure.
- `docs/PROVIDER_MODERNIZATION_PLAN.md` — the phased program that closes the
  gaps, built on the dual-transport and one-contract principles.

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
