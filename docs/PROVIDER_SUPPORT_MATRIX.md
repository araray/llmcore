# Provider Support Matrix

Tracking document for every provider `llmcore` curates: which upstream API/SDK
version we are known-good against, how we talk to it, and which of the
provider's capabilities we actually expose.

**This file is the source of truth for "are we current?".** Update it in the
same commit as any provider change — see [Refreshing this document](#refreshing-this-document).

- **Last full audit:** 2026-09-29
- **Phase 0 landed:** 2026-09-29 — pins bumped to the majors below, extras corrected, live-validated
- **llmcore version at audit:** 0.53.0
- **Vendor SDK clones:** `/av/avalon/xrepos/<sdk-repo>` (paths in the table below)

---

## 1. Legend

| Mark | Meaning |
|:---:|---|
| ✅ | Implemented in llmcore and covered by tests |
| 🟡 | Partially implemented (see the note) |
| ❌ | Provider supports it; llmcore does **not** |
| — | Provider does not offer it (nothing to do) |
| ❓ | Provider surface not yet verified against upstream docs — audit pending |

**Transport** column values:

- `sdk` — vendor SDK only
- `openai` — the `openai` SDK pointed at the vendor's OpenAI-compatible base URL
- `httpx` — direct REST calls from llmcore
- Multiple values joined by `→` are the runtime auto-resolution order (the
  "dual approach"): first available wins, and each is individually selectable
  via the provider's `backend` config key.

---

## 2. SDK / API version tracking

Upstream columns are the **vendor SDK clone in `/av/avalon/xrepos`** as of the
audit date. `llmcore pin` is from `pyproject.toml`; `installed` is the shared
dev venv. A pin that trails the upstream **major** version is a red flag.

| Provider | SDK package | Clone (`/av/avalon/xrepos/…`) | Upstream tag | Upstream commit | Tag date | llmcore pin | Installed | Status |
|---|---|---|---|---|---|---|---|:---:|
| OpenAI | `openai` | `openai-python` | **v3.22.1** | `58aca1dcfd8d` | 2026-09-30 | `>=3.0.0,<4` | 3.22.1 | ✅ current (live ✓) |
| Anthropic | `anthropic` | `anthropic-sdk-python` | **v1.9.0** | `a7285e919ab7` | 2026-09-28 | `>=1,<2` | 1.9.0 | 🟡 transport ✓, **no completion** — account credit balance is zero |
| Google Gemini | `google-genai` | `python-genai` | **v2.25.0** | `f15d1482d747` | 2026-09-29 | `>=2,<3` | 2.25.0 | ✅ current (live ✓, 47 models) |
| Mistral | `mistralai` | `mistral-client-python` | **v3.0.0** | `e8dfa1c8a2d0` | 2026-09-28 | *(none — httpx only)* | not installed | 🟠 SDK unused (httpx path live ✓, 46 models) |
| OpenRouter | `openrouter` | `openrouter_python_sdk` | **v1.3.9** | `fd5ffce2995d` | 2026-09-30 | *(optional backend)* | not installed | 🟠 major behind + absent |
| Ollama | `ollama` | `ollama-python` | v0.6.3 | `8785556559ec` | 2026-09-28 | `>=0.6.3` | 0.6.3 | ✅ current |
| Deepgram | `deepgram-sdk` | `deepgram-python-sdk` | v7.11.0 | `a379a7f37b11` | 2026-09-28 | `>=7.11.0` | 7.11.0 | ✅ current |
| Hugging Face | `huggingface-hub` | `huggingface_hub` | `main` @ v0.9.0.rc1 | `1092497a9b65` | 2026-09-29 | `>=1.12.0` | 1.12.0 | ✅ pinned |
| Z.ai (GLM) | `zai-sdk` | `z-ai-sdk-python` | v0.2.3 | `ca5109c0aa9b` | 2026-06-16 | `>=0.2.3` | 0.2.3 | ✅ current (SDK backend live ✓) |
| FriendliAI | `friendli` | `friendli-python` | v0.15.1 (repo pyproject reads 0.15.2, unreleased) | `f3039e22ec0d` | 2026-09-28 | `>=0.15.1` | 0.15.1 | ✅ current |
| Replicate | `replicate` | `replicate-python` | **v1.0.7** | `d2956ff9c3e2` | 2025-08-26 | `>=1.0.7` *(optional backend)* | not installed | ✅ current (live ✓, direct REST default) |
| ElevenLabs | `elevenlabs` | `elevenlabs-python` | **v2.70.0** | `963b4a59bc0d` | 2026-09-28 | `>=2.70.0` *(optional backend)* | not installed | ✅ current (live ✓, direct REST default) |
| fal | `fal-client` | `fal` (monorepo: `projects/fal_client`) | **v1.0.3** | `ec46b79` | 2026-09-22 | `>=1.0.0` *(optional backend)* | 1.0.3 | ✅ current (live ✓, direct REST default) |
| TypeSafe.ai | `typesafe-sdk` | `typesafe-sdk-python` | v0.7.2 | `f078f1e208a0` | 2026-09-26 | *(none — httpx only)* | not installed | ✅ by design |
| Poe | `fastapi_poe` | `fastapi_poe` | 0.0.83 | `41ffd02e16f2` | 2026-01-21 | *(optional backend)* | not installed | 🟡 SDK path untested |
| vLLM (self-hosted) | `vllm` (server) | `vllm` | v0.19.1rc0 | `219bb5b8c0dc` | 2026-04-16 | *(server, not a client dep)* | n/a | 🟡 clone stale |
| DeepSeek | *(none published)* | — | — | — | — | uses `openai` | — | ✅ by design |
| Kimi (Moonshot) | *(none published)* | — | — | — | — | uses `openai` | — | ✅ by design |
| DeepInfra | *(none published)* | — | — | — | — | uses `openai` | — | ✅ by design |
| xAI (Grok) | `xai-sdk` | `xai-sdk-python` | **v1.20.0** | `1d9e1dffc9a0` | 2026-09-28 | *(none)* | not installed | 🔴 native SDK unused |
| Groq | `groq` | `groq-python` | **v1.7.0** | `55066d94acca` | 2026-09-04 | *(none)* | not installed | 🔴 native SDK unused |
| Together | `together` | `together-python` | **v1.5.35** | `cc9f25369987` | 2026-03-18 | *(none)* | not installed | 🔴 native SDK unused |

### 2.1 Cross-cutting: the HTTPX2 migration

`openai` 3.0.0 (2026-08-12) and `anthropic` 1.0.0 (2026-08-20) both moved their
HTTP layer from `httpx` to [`httpx2`](https://httpx2.pydantic.dev/) (Pydantic's
maintained fork) — see `openai-python/httpx2.md` and
`anthropic-sdk-python/MIGRATION.md` in the clones.

What this means for llmcore, **verified against the source**:

| Concern | Status |
|---|---|
| Does llmcore pass `httpx` objects *into* the OpenAI/Anthropic clients (`http_client=`, `httpx.Timeout`)? | **No.** Only numeric timeouts are passed. Nothing to port. |
| Does llmcore use `httpx` for its *own* clients? | **Yes** — 9 providers (`deepinfra`, `poe`, `mistral`, `friendli`, `kimi`, `openrouter`, `vllm`, `zai`, `typesafe`) plus all search providers. These keep using `httpx` and are unaffected. |
| Do `respx`-based tests break? | **No.** Every `respx` mock in `tests/` targets llmcore's own `httpx` clients, never traffic routed through a vendor SDK. (Upstream vendored `tests/respx2` for their own suite; we don't need it.) |
| **`httpx` is no longer installed transitively by `openai`** | 🔴 **Action required** — see below. |
| `certifi` is no longer installed by `openai`; httpx2 uses the **OS trust store** | ✅ Documented in `CONFIG_REFERENCE.md` § HTTP transport and TLS. |

**The concrete break:** six providers import `httpx` but have **no extra of
their own**, so today they only work because `openai` happened to install
`httpx` transitively. Under `openai>=3` they fail at import:

| Provider | Imports `httpx` | Own extra |
|---|:---:|:---:|
| `mistral` | yes | ✅ added (Phase 0) |
| `kimi` | yes | ✅ added (Phase 0) |
| `poe` | yes | ✅ added (Phase 0) |
| `openrouter` | yes | ✅ added (Phase 0) |
| `vllm` | yes | ✅ added (Phase 0) |
| `huggingface` | — | ✅ added (Phase 0) |

(`deepinfra`, `zai`, `friendli`, `typesafe` already declare `httpx` explicitly.)

---

## 3. Default models configured in `default_config.toml`

Stale defaults are a correctness problem, not cosmetics — they drive model-card
lookups, context budgets and cost estimates.

| Provider | Configured default | Assessment |
|---|---|---|
| openai | `gpt-4o` | 🔴 several generations stale |
| anthropic | `claude-sonnet-4-6` | 🔴 current lineup is Opus 5.5 / Sonnet 5.5 (see §5.2) |
| gemini | `gemini-3.1-flash-lite-preview` | ❓ verify against current lineup |
| deepseek | `deepseek-v4-pro` | ✅ current at last provider audit |
| zai | `glm-5.2` | 🟡 GLM-5.3 is served (Friendli lists it) |
| friendli | `zai-org/GLM-5.3` | ✅ verified live 2026-09-20 |
| ollama | `llama3` | 🔴 very stale |
| openrouter | `openai/gpt-4o-mini` | 🔴 stale |
| poe | `GPT-4o-Mini` | 🔴 stale |
| vllm | `meta-llama/Llama-3.1-8B-Instruct` | 🟡 example value, self-hosted |
| mistral | `mistral-large-latest` | 🔴 alias resolves, but **403 — not in this account's tier**; `open-mistral-nemo` verified working |
| huggingface | `meta-llama/Llama-3.3-70B-Instruct` | 🟡 stale example |
| deepinfra | `deepseek-ai/DeepSeek-V3` | 🟡 V3.2 is served |
| kimi | `kimi-k2.6` | ✅ current at last provider audit |
| typesafe | `jev-latest` | ✅ alias, self-updating |

---

## 4. Capability matrix

What llmcore **exposes today**, extracted from the provider classes (not from
vendor docs). Columns are the `BaseProvider` surface plus the media APIs.

| Provider | Transport | Chat | Stream | Tools | Structured out | Reasoning extract | Vision in | Audio in (STT) | Audio out (TTS) | Image gen | Video gen | Embeddings | OCR | Native search | Exact tokenizer |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| openai | `sdk` | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | — | ✅ | ✅ |
| anthropic | `sdk` | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | — | — | — | — | — | — | ❌ | ❌ |
| gemini | `sdk` | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | — | ✅ | ❌ |
| deepseek | `sdk`(openai) | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — | — | ❌ | — | — | ❌ |
| zai | `sdk → openai → httpx` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| friendli | `openai → httpx → sdk` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | — | ✅ | — | — | ✅ |
| mistral | `httpx` | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | — | — | ✅ | ✅ | — | ❌ |
| kimi | `openai + httpx` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — | ❌ | — | — | ✅ |
| deepinfra | `openai + httpx` | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | — | ✅ | — | — | ❌ |
| ollama | `sdk` | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | — | — | — | — | ❌ | — | — | 🟡 |
| huggingface | `sdk + httpx` | ✅ | ✅ | ✅ | 🟡 | ❌ | ✅ | ✅ | ✅ | ✅ | — | ✅ | — | — | ❌ |
| openrouter | `openai (+sdk)` | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | — | — | — | — | ❌ | — | ❓ | ❌ |
| poe | `openai (+native)` | ✅ | ✅ | ✅ | 🟡 | ❌ | ✅ | ❓ | ❓ | ❓ | ❓ | — | — | — | ❌ |
| vllm | `openai + httpx` | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | — | — | — | — | ❌ | — | — | ❌ |
| replicate | `httpx (+sdk)` | — | — | — | — | — | — | ✅ | ✅ | ✅ | ✅ | — | — | — | — |
| elevenlabs | `httpx (+sdk)` | — | — | — | — | — | — | ✅ | ✅ | — | — | — | — | — | — |
| fal | `httpx (+sdk)` | — | — | — | — | — | — | ✅ | ✅ | ✅ | ✅ | — | — | — | — |
| deepgram | `sdk` | — | — | — | — | — | — | ✅ | ✅ | — | — | — | — | — | — |
| typesafe | `httpx` | — | — | — | ✅ | — | — | — | — | — | — | — | — | — | — |

Notes on the 🟡 cells:

- **anthropic / gemini / ollama / mistral reasoning** — reasoning *is* surfaced,
  but through provider-specific channels (`message["thinking"]`,
  `extract_thinking_content()`) instead of the
  `extract_reasoning_content()` / `extract_delta_reasoning_content()` contract
  that deepseek, zai, kimi and friendli implement. **Naming should be unified.**
- **ollama exact tokenizer** — tiktoken approximation, not the model's tokenizer.
- **huggingface / poe structured output** — passthrough only, unvalidated.
- **deepgram** is a voice/audio provider by design; `chat_completion()`
  intentionally raises.
- **typesafe** is a typed-judgment API, not a chat model, by design.

### 4.1 Extractor contract coverage

| Provider | `reasoning_content` | `delta_reasoning_content` | `tool_calls` | `usage_details` | `finish_reason` |
|---|:-:|:-:|:-:|:-:|:-:|
| deepseek, zai, kimi, friendli | ✅ | ✅ | ✅ | ✅ | ✅ |
| openai, mistral | ❌ | ❌ | ✅ | ✅ | ❌ |
| anthropic, gemini, ollama, huggingface | ❌ | ❌ | ✅ | ❌ | ❌ |
| openrouter, poe, vllm, deepinfra | ❌ | ❌ | inherited | inherited | ❌ |

Only four of sixteen providers implement the full extractor contract. This is
the single biggest consistency gap in the provider layer.

---

## 5. Known capability gaps per provider

Vendor-side surfaces that exist upstream and are **not** in llmcore. Items
marked ❓ still need verification against current vendor docs.

### 5.1 OpenAI (`openai` v3.x)

- ❌ **Embeddings** at provider level (`create_embeddings()` is not overridden —
  only the separate `[embedding.openai]` subsystem covers it)
- ❌ Responses API (llmcore uses Chat Completions only)
- ❌ Batch API, Files API, Vector Stores, Containers
- ❌ Realtime / WebSocket sessions (v3.1 added WebSocket stream IDs)
- ❌ `reasoning_content` extractor for the reasoning-model families
- ❌ "Ultrafast" service tier (added v3.1)
- ⚠️ **Sora video APIs were deprecated in v3.1** — do *not* add them
- ❓ Structured MCP / separate websocket error events (v3.1)

### 5.2 Anthropic (`anthropic` v1.x)

Verified against the bundled `claude-api` skill reference (2026-09-25 cache):

- 🔴 **Model lineup is stale.** Current: `claude-opus-5-5` (default,
  1M ctx / 128K out), `claude-opus-5`, `claude-opus-4-8/4-7/4-6`,
  `claude-sonnet-5-5`, `claude-sonnet-5`, `claude-sonnet-4-6`,
  `claude-haiku-4-5`, `claude-fable-5-1`/`claude-fable-5`.
  llmcore defaults to `claude-sonnet-4-6` and its cards stop at the 4.x family.
- 🔴 **`budget_tokens` is rejected (400) on Opus 5.x / Sonnet 5.x / Fable 5.x.**
  llmcore's `thinking_budget_tokens` config would hard-fail on current models.
  `thinking: {type: "adaptive"}` + `output_config.effort` is the current API.
- ❌ `output_config.effort` (`low|medium|high|xhigh|max`) not plumbed
- ❌ `stop_reason: "refusal"` / `stop_details` handling
- ❌ Server tools: `web_search_20260209`, `web_fetch_20260209`,
  `code_execution_20260521`, tool search
- ❌ Batches, Files, Skills, Models API, `messages.count_tokens`
- ❌ Citations on document blocks
- ❌ Compaction, context editing, mid-conversation system messages
- ❌ Fast mode, task budgets, memory tool, Tool Runner
- ❌ Bedrock / Vertex / Foundry provider clients
- ❌ Preserved-thinking (history-editing) compliance — relevant to llmcore's
  context-management rewrites

### 5.3 Google Gemini (`google-genai` v2.x)

- ❌ Image generation (Imagen family)
- ❌ Video generation (Veo family)
- ❌ Native TTS / audio output
- ❌ Live API (bidirectional realtime) — a `*-live-preview` model is already in
  the context table but unused
- ❌ Provider-level `create_embeddings()`
- ❌ Multimodal file search (added v1.75)
- ❌ Interactions API (v2.0's breaking surface; `GenerateContent` unaffected)
- ⚠️ Legacy `response_format` deprecated for a new polymorphic field (v2.0)
- 🟡 Reasoning exposed as `thinking`, not via the extractor contract

### 5.4 Mistral (`mistralai` v3.0.0)

- 🟠 llmcore is httpx-only; the **v3 SDK is not used at all** → no dual approach
- ❓ v3.0 breaking changes unknown (no CHANGELOG in the repo) — needs review
- ❌ Agents / conversations API ❓
- ❌ Batch API ❓

### 5.5 xAI / Groq / Together

All three are wired as bare `OpenAIProvider` + `base_url`. Each now ships a
**native Python SDK** (`xai-sdk` v1.20.0, `groq` v1.7.0, `together` v1.5.35),
so none has a dual approach and none exposes vendor-specific surfaces
(xAI Live Search is referenced by `supports_native_search()` but there is no
native-SDK path).

### 5.6 Others

| Provider | Gaps |
|---|---|
| ollama | ❌ provider-level embeddings (`/api/embed`); 🟡 tokenizer is approximate; ❓ newer `/api/*` surface |
| openrouter | 🟠 v1.x SDK unused; ❌ embeddings; ❓ provider routing/ZDR/prompt-caching controls |
| poe | ❓ media bots (image/video/audio) not surfaced through the media APIs |
| vllm | ❌ `/v1/embeddings`, `/pooling`, `/score`, `/rerank`; clone is 5 months stale |
| huggingface | 🟠 unpinned dep; ❌ provider-level rerank; ❓ Inference Providers routing surface |
| deepgram | 🟡 SDK 7.3.1 vs 7.11.0 — review new surface |
| zai | ✅ Phase 0 installed `zai-sdk` and made the tests backend-hermetic; the SDK backend is now exercised in CI and validated live |
| friendli | ⚠️ `/detokenize` + `/chat/render` 404 on Model APIs (upstream gap, tracked) |

---

## 6. Refreshing this document

Run from the repo root. **Never print secret values.**

```bash
# 1. Sync every vendor SDK clone (fast-forward only)
cd /av/avalon/xrepos
for d in openai-python anthropic-sdk-python python-genai ollama-python \
         mistral-client-python huggingface_hub z-ai-sdk-python \
         deepgram-python-sdk typesafe-sdk-python friendli-python \
         openrouter_python_sdk fastapi_poe xai-sdk-python groq-python \
         together-python vllm; do
  git -C "$d" pull --ff-only --tags origin >/dev/null 2>&1
  printf "%-26s %-12s %s %s\n" "$d" \
    "$(git -C $d describe --tags --abbrev=0 2>/dev/null)" \
    "$(git -C $d rev-parse --short=12 HEAD)" \
    "$(git -C $d log -1 --format=%cs)"
done

# 2. Compare against llmcore's pins and the installed venv
cd /media/araray/kilgharrah/repos/llmcore
./venv/bin/python -c "
import importlib.metadata as md, tomllib, pathlib
ex = tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['optional-dependencies']
print({k: v for k, v in ex.items() if k in ('openai','anthropic','gemini','ollama','zai','deepgram','friendli')})
for p in ('openai','anthropic','google-genai','ollama','deepgram-sdk','friendli','httpx','httpx2'):
    try: print(p, md.version(p))
    except Exception: print(p, 'NOT INSTALLED')
"

# 3. Re-extract the implemented-capability matrix (§4) from the source
#    (the AST walk used for the audit lives in the plan doc's appendix)

# 4. Refresh model cards for every provider with an adapter
./venv/bin/python -m tools.cardctl generate <provider>   # no --force
./venv/bin/python -m tools.cardctl diff <provider>
```

Then update §2, §3, §4 and the audit date at the top **in the same commit** as
any provider change.

---

## 7. Live validation log

Recorded per audit so "current" always means "we called it".

| Date | Provider | SDK | Result |
|---|---|---|---|
| 2026-09-29 | OpenAI | `openai` 3.22.1 | ✅ completion + usage |
| 2026-09-29 | Google Gemini | `google-genai` 2.25.0 | ✅ completion; 47 models discovered |
| 2026-09-29 | Z.ai | `zai-sdk` 0.2.3 (**native SDK backend**) | ✅ completion + reasoning tokens |
| 2026-09-29 | DeepSeek | via `openai` 3.22.1 | ✅ completion + cache/reasoning usage |
| 2026-09-29 | Mistral | httpx path | ✅ 46 models; `open-mistral-nemo` completion. `mistral-large-latest` → 403 (tier) |
| 2026-09-29 | Anthropic | `anthropic` 1.9.0 | 🟡 auth + error mapping ✓ through the new major, but every request returns `invalid_request_error: credit balance is too low` — **no completion validated** |
| 2026-09-30 | Replicate | direct REST (`httpx`) | ✅ schema-driven field mapping (`n`→`num_outputs` for flux, `audio` for whisper), image generation, community-model ASR, cancel. Found 2 real bugs: community models need a version pin (and the naive 404-fallback trips the burst-1 rate limit under $5 credit), MIME unknowable from extensionless URLs |
| 2026-09-30 | ElevenLabs | direct REST (`httpx`) | ✅ TTS + consent metadata, streaming TTS, STT round trip, SFX, model discovery (11 models). 🟡 music + voice design **plan-gated** (free tier) — implemented, not live-validated. Found 1 real bug: 403 plan gating was reported as an auth failure |
| 2026-09-30 | fal | direct REST (`httpx`) | ✅ image, TTS → ASR round trip, CDN upload, FILM interpolation, upscale, music, cancel. Found 3 real bugs: app-scoped queue paths, storage host + backend, FILM's input schema |
| 2026-09-20 | FriendliAI | `openai`/`httpx`/`friendli` | ✅ all three backends, streaming, tools, team billing |

---

## 8. Related documents

- [`PROVIDER_MODERNIZATION_PLAN.md`](PROVIDER_MODERNIZATION_PLAN.md) — the phased
  plan that closes the gaps listed in §5
- [`model_cards.md`](model_cards.md) — card schema, the canonical
  reasoning-effort vocabulary, and the cardctl workflow
- [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md) — every provider config key
- [`MEDIA_SUBSYSTEM_SPEC.md`](MEDIA_SUBSYSTEM_SPEC.md) — design/spec for the
  image/audio/video subsystem and its provider adapters
- [`COLAB_RUNTIME_SPEC.md`](COLAB_RUNTIME_SPEC.md) — design/spec for remote GPU
  runtimes (Colab first), so a remotely served model is just another provider
- Per-provider guides: [`Friendli_provider_usage.md`](Friendli_provider_usage.md),
  [`Deepgram_provider_usage.md`](Deepgram_provider_usage.md),
  [`TypeSafe_provider_usage.md`](TypeSafe_provider_usage.md),
  [`Fal_provider_usage.md`](Fal_provider_usage.md),
  [`ElevenLabs_provider_usage.md`](ElevenLabs_provider_usage.md),
  [`Replicate_provider_usage.md`](Replicate_provider_usage.md)
