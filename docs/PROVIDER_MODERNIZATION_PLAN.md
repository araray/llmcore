# Provider Modernization Plan

The program that closes the gaps recorded in
[`PROVIDER_SUPPORT_MATRIX.md`](PROVIDER_SUPPORT_MATRIX.md), so that `llmcore`
stays at the bleeding edge of the APIs it curates.

- **Written:** 2026-09-29, from the audit of the same date
- **Scope:** all 16 providers + the 3 OpenAI-compatible aliases (xai, groq, together)
- **Status:** plan only — no phase has landed yet

---

## 1. Principles

These are the rules the phases below are designed against. They are also the
review checklist for any new provider.

1. **Dual transport, always.** Every provider should reach its API by at least
   two independent paths: direct REST from llmcore (`httpx`) and the vendor SDK
   (or the `openai` SDK in compatibility mode). One `backend` config key
   selects; unset auto-resolves. Rationale: vendor SDKs lag their own APIs,
   drop fields outside their published schema (proven for `friendli`), and add
   breaking dependency churn (proven for `openai` 3.x / `anthropic` 1.x).
2. **Direct-first where the SDK is lossy.** Auto-resolution should prefer
   whichever path preserves the most response data. Record the reason in the
   provider docstring, as `friendli_provider.py` does.
3. **One contract, every provider.** The `BaseProvider` extractor surface
   (`extract_reasoning_content`, `extract_delta_reasoning_content`,
   `extract_tool_calls`, `extract_usage_details`, `extract_finish_reason`) is
   not optional-in-practice. Provider-specific names like
   `extract_thinking_content()` are bugs in disguise: callers cannot use them
   polymorphically.
4. **Capabilities are declared, not guessed.** Anything a provider can do must
   be discoverable from `get_models_details()` and from its model cards, so
   routing and validation work without hardcoded model lists.
5. **Live-validated.** No provider change lands without a real call against the
   real API using the keys in `/av/data/dbs/.env`, plus offline tests that mock
   the transport.
6. **Additive.** Config keys and methods are added with safe defaults; existing
   callers keep working.

---

## 2. Phase 0 — Unblock the dependency upgrade (P0, breaking)

**Why first:** `openai>=3` and `anthropic>=1` cannot be adopted until this is
done, and every later phase needs those SDKs. Nothing else in the plan is
blocked by anything but this.

### 2.1 Declare `httpx` where it is actually used

`openai` 3.x no longer installs `httpx` transitively. Six providers import it
with no extra of their own and would fail at import (matrix §2.1). Add extras:

```toml
mistral     = ["httpx>=0.27.0"]
kimi        = ["openai>=3.0.0", "httpx>=0.27.0", "tiktoken>=0.9.0"]
poe         = ["openai>=3.0.0", "httpx>=0.27.0"]              # + fastapi-poe optional
openrouter  = ["openai>=3.0.0", "httpx>=0.27.0"]              # + openrouter optional
vllm        = ["openai>=3.0.0", "httpx>=0.27.0"]
huggingface = ["huggingface-hub>=1.12.0"]                     # currently unpinned
```

Add all six to `[all]`, and to the CI install list.

### 2.2 Bump the pins

| Extra | From | To | Notes |
|---|---|---|---|
| `openai` | `openai>=2.31.0` | `openai>=3.0.0,<4` | pulls `httpx2`, drops `httpx`/`certifi` |
| `anthropic` | `anthropic>=0.94.0` | `anthropic>=1,<2` | min Python 3.10 — llmcore is already ≥3.11 |
| `gemini` | `google-genai>=1.72.0` | `google-genai>=2,<3` | `GenerateContent` unaffected by the 2.0 break |
| `ollama` | `>=0.6.0` | `>=0.6.3` | |
| `deepgram` | `>=7.0.0` | `>=7.11.0` | |
| `friendli` | `>=0.15.1` | `>=0.15.2` | |

### 2.3 Port the breaking changes

- **openai 3.x** — verified: llmcore never passes `httpx` objects into the
  client, so there is *no code* to port. Only packaging and docs change.
- **anthropic 1.x** — audit for: removed deprecated request params, removed type
  aliases/exports, `.with_raw_response` shape (async now awaited), byte-valued
  headers, and the removed legacy Text Completions API. llmcore uses none of
  these as far as the audit could tell — **confirm with a type-checker pass**,
  which the upstream guide recommends as the checklist.
- The bundled `claude-api` skill has a `/claude-api upgrade python` subcommand
  and a `python/claude-api/sdk-upgrade.md` guide written for exactly this
  migration — use it rather than improvising.

### 2.4 Document the TLS change

httpx2 verifies against the **OS trust store**, not `certifi`. Add a note to
`CONFIG_REFERENCE.md` and the provider guides: minimal containers and
TLS-inspecting corporate proxies need CA certs installed, or
`SSL_CERT_FILE` / `SSL_CERT_DIR` set.

### 2.5 Verification

- Install `[all]` into a scratch venv and import every provider module.
- Full unit suite green.
- One live call per SDK-backed provider.
- Confirm `respx` still intercepts llmcore's own clients (it should — the audit
  found no test routes traffic through a vendor SDK).

**Deliverable:** one PR, packaging + docs + any anthropic 1.x fixes. No new features.

---

## 3. Phase 1 — Contract consistency (P1, high value / low risk)

Cheap, purely additive, and it makes every later phase easier.

### 3.1 Unify the extractor contract

Implement the full five-method surface on every chat provider (matrix §4.1 —
only 4 of 16 have it today). Specifically:

- `ollama`: rename/alias `extract_thinking_content()` →
  `extract_reasoning_content()` (keep the old name as a deprecated alias).
- `gemini`: expose the `thinking` parts it already parses through
  `extract_reasoning_content()` / `extract_delta_reasoning_content()` instead of
  only `message["thinking"]`.
- `anthropic`: expose `thinking` content blocks through the same contract.
- `mistral`: same for Magistral reasoning.
- `openai`: add reasoning extraction for the reasoning-model families.
- All: add `extract_usage_details()` and `extract_finish_reason()` where missing.

Add a **cross-provider contract test** — parametrized over every registered
provider — asserting each method exists, is defensive against `{}`, and returns
the documented type. This is the same shape as the static guard added in
`tests/providers/test_context_length_error_mapping.py`, which caught a bug three
providers had silently carried.

### 3.2 Provider-level embeddings

`create_embeddings()` is missing on `openai`, `gemini`, `ollama`, `vllm`,
`openrouter`, `deepseek`, `kimi`. The separate `[embedding.*]` subsystem covers
some of these, but the provider surface should be complete and consistent so
callers can embed through whichever provider they already hold.

### 3.3 New `BaseProvider` surfaces

Add, defaulting to `NotImplementedError` so nothing existing changes:

| Method | Rationale |
|---|---|
| `generate_video()` | only `zai` has one today, under a provider-specific name; Gemini Veo and others need a home |
| `rerank()` | vLLM, HF and several hosted providers expose rerank/score endpoints |
| `create_batch()` / `retrieve_batch()` | OpenAI + Anthropic + Mistral all have batch APIs at ~50% cost |
| `upload_file()` / `list_files()` | OpenAI, Anthropic, Gemini, Kimi all have Files APIs; `kimi` already has a bespoke `upload_file()` |
| `count_tokens_native()` | distinguish exact provider tokenizers from local estimates; `friendli` and `kimi` already have them under bespoke names |

---

## 4. Phase 2 — Anthropic (P1, largest single gap)

The audit found the widest divergence here. Split into reviewable PRs:

1. **Models + cards.** Refresh the lineup to Opus 5.5 / Opus 5 / Sonnet 5.5 /
   Sonnet 5 / Haiku 4.5 / Fable 5.x, update the default from
   `claude-sonnet-4-6`, regenerate cards, and record per-model thinking rules.
2. **Thinking and effort — correctness fix, not a feature.** `budget_tokens` is
   **rejected with a 400** on Opus 5.x / Sonnet 5.x / Fable 5.x, yet
   `thinking_budget_tokens` is still a documented llmcore config key. Move to
   `thinking: {type: "adaptive"}` + `output_config.effort`
   (`low|medium|high|xhigh|max`), mapped from llmcore's canonical
   reasoning-effort vocabulary (`model_cards.md` §Reasoning-Effort Vocabulary).
   Keep `budget_tokens` only for the pre-4.6 models that still accept it.
3. **Refusals.** Handle `stop_reason: "refusal"` + `stop_details` (currently
   unhandled — a refusal looks like an empty response), and expose the
   server-side `fallbacks` parameter.
4. **Server tools.** `web_search_20260209`, `web_fetch_20260209`,
   `code_execution_20260521`, tool search — and wire `web_search` into
   `supports_native_search()`, which today returns `False` for Anthropic.
5. **Batches + Files + count_tokens + Models API.** `messages.count_tokens`
   replaces llmcore's local estimate for Anthropic; the Models API gives live
   context/capability discovery for `get_models_details()`.
6. **Context lifecycle.** Compaction, context editing, mid-conversation system
   messages — these interact directly with llmcore's own context manager, and
   **preserved thinking** means llmcore's history rewriting can invalidate
   thinking blocks. Treat as a design task, not a passthrough.
7. **Optional/lower priority.** Fast mode, task budgets, memory tool, citations,
   Bedrock/Vertex/Foundry client variants, Admin API.

> Anthropic publishes no separate REST client, so "dual approach" here means
> SDK + llmcore's own `httpx` path against `/v1/messages`. Worth building for
> the same reason as Friendli: it removes the SDK from the critical path.

---

## 5. Phase 3 — OpenAI (P1)

1. **Provider-level `create_embeddings()`** (currently absent).
2. **Responses API** as a second chat surface alongside Chat Completions,
   selected per call/config. This is where OpenAI ships new capability first.
3. **Reasoning extraction** for the reasoning-model families, via the Phase 1
   contract.
4. **Batch, Files, Vector Stores, Containers.**
5. **Refresh the default model** off `gpt-4o`, refresh cards.
6. **Do not add Sora video** — deprecated upstream in `openai` 3.1.
7. Evaluate the "Ultrafast" tier and structured MCP errors (both v3.1).
8. **✅ Direct `httpx` transport on `OpenAIProvider`** — landed 2026-09-30.
   It is the base class for `deepinfra`, `vllm`, `poe` and `openrouter`, so one
   transport gave **five** providers a dual approach at once. Select per
   instance with `transport = "httpx"`; the default stays `"sdk"`.

   Three things worth recording:

   - **The key is `transport`, not `backend`.** OpenRouter and Poe already use
     `backend` for native-SDK-vs-OpenAI-compatible selection, and overloading
     it would have made one of the two settings unreachable.
   - **The default had to stay `sdk`.** Four subclasses' suites mock
     `AsyncOpenAI`; flipping the default would have routed five providers past
     their own tests — the same failure the Z.ai SDK backend caused when it
     changed auto-resolution and broke 21 tests.
   - **Interchangeability is the real requirement.** Every `extract_*` method
     parses response *dicts*, so the direct path returns exactly what
     `model_dump(exclude_none=True)` produces, and maps errors to the same
     typed exceptions (`ContextLengthError`, the actionable model-not-found
     `ProviderError`). Verified live: identical response keys, identical tool
     calls, identical model listings, streaming on both.

   Request shaping (parameter validation, native search, reasoning-model
   parameter naming, tool payloads) happens *before* the transport branch, so
   the direct path inherits it rather than reimplementing it — which is what
   keeps the two from drifting.

---

## 6. Phase 4 — Gemini media (P2)

Gemini has the largest *unexposed* media surface of any provider llmcore
curates:

| Capability | Target |
|---|---|
| Image generation (Imagen family) | `generate_image()` |
| Video generation (Veo family) | `generate_video()` (new, Phase 1.3) |
| Native TTS | `generate_speech()` |
| Audio understanding | `transcribe_audio()` |
| Embeddings | `create_embeddings()` |
| Live API (bidirectional realtime) | new streaming surface, mirroring the Deepgram voice-agent socket pattern |
| Multimodal file search | evaluate |

Also: port off the deprecated `response_format` to the v2 polymorphic field, and
verify whether the v2.0 "interactions" surface is worth adopting (its breaking
changes do not affect `GenerateContent`, which is what llmcore uses).

---

## 7. Phase 5 — Native SDKs for the compatibility aliases (P2)

`xai`, `groq` and `together` are bare `OpenAIProvider` + `base_url`. Each now
has a native SDK. Promote each to a first-class provider with the dual-transport
shape, which also unlocks vendor-specific surfaces (xAI Live Search is already
referenced by `supports_native_search()` with no native path behind it).

Same phase: give **Mistral** a dual approach. llmcore is httpx-only there while
`mistralai` v3.0.0 sits unused — and v3.0's breaking changes are undocumented in
the repo, so review the SDK source before pinning.

---

## 8. Phase 6 — Long tail (P3)

- **vLLM**: `/v1/embeddings`, `/pooling`, `/score`, `/rerank`; refresh the clone (5 months stale).
- **OpenRouter**: adopt the v1.x SDK as the `sdk` backend; expose routing/ZDR/caching controls.
- **Ollama**: provider-level embeddings; audit the newer `/api/*` surface.
- **Hugging Face**: pin the dep; rerank; audit Inference Providers routing.
- **Poe**: surface media bots through the media APIs.
- **Deepgram**: review 7.3.1 → 7.11.0 for new surface.
- **Z.ai**: install `zai-sdk` in CI so the preferred SDK backend is exercised.

---

## 9. Testing and validation strategy

Per provider touched:

1. **Offline** — mock at the transport boundary (`respx` for llmcore's own
   `httpx` clients; `AsyncMock` on the SDK client object for SDK paths). Pin the
   `backend` explicitly in tests so results do not depend on what is installed —
   the pattern established in `tests/providers/test_friendli_provider.py`.
2. **Contract** — the parametrized cross-provider tests from Phase 1.
3. **Static guards** — AST checks for whole-package invariants, in the style of
   `tests/providers/test_context_length_error_mapping.py`. Candidates: every
   provider registered in `PROVIDER_MAP` has a `[providers.*]` config section, a
   confy schema section, and a matrix row.
4. **Live** — a key-gated smoke per provider under `tests/integration/`,
   skipped when the key is absent (the `test_typesafe_live.py` pattern).
5. **CI** — keep vendor SDKs whose presence would bypass mocks out of the
   install list, and say why in the workflow comment.

---

## 10. Risks and open questions

| Risk | Mitigation |
|---|---|
| httpx2's OS trust store breaks deployments in minimal containers | document `SSL_CERT_FILE`/`SSL_CERT_DIR`; call it out in release notes |
| `anthropic` 1.x removals not caught by the audit | run mypy/pyright over the anthropic provider after bumping — upstream recommends exactly this |
| `mistralai` v3.0 breaking changes undocumented | read the SDK source before pinning; keep httpx as the default backend |
| Phase 2.6 (compaction / context editing / preserved thinking) touches llmcore's own context manager | design task with its own review, not a passthrough PR |
| Provider APIs drift again | §6 of the matrix is a runnable refresh procedure; re-run it per release |

**Open questions for the maintainer:**

1. **Phase order** — this plan front-loads Phase 0 (blocking) then Anthropic
   (biggest gap). Prefer breadth-first instead (every provider to parity before
   any provider gets new capabilities)?
2. **Minimum SDK floors** — pin to `>=3,<4` style ranges as proposed, or track
   exact versions for reproducibility?
3. **`openai` direct backend** (§5.8) is the highest-leverage single item — five
   providers gain a dual approach at once. Promote it ahead of Anthropic?
4. Should the compatibility aliases (`xai`/`groq`/`together`) become first-class
   providers, or stay thin aliases with a documented native-SDK escape hatch?

---

## 11. Companion specifications

Two capability programs are specified separately because they add subsystems
rather than extending providers:

- [`MEDIA_SUBSYSTEM_SPEC.md`](MEDIA_SUBSYSTEM_SPEC.md) — a first-class
  `llmcore.media` subsystem (image/audio/video), the capability protocols
  providers implement, async `MediaJob` lifecycle, capability-oriented model
  cards, and the vendor rollout (Deepgram refactor → OpenAI → Google/Veo → fal →
  ElevenLabs → Replicate → HF Endpoints). This **supersedes** Phase 4 of this
  plan (Gemini media) and Phase 1.3's `generate_video()` placeholder, which
  become M4 and part of M1 there.
- [`COLAB_RUNTIME_SPEC.md`](COLAB_RUNTIME_SPEC.md) — a `llmcore.runtimes`
  subsystem that provisions and controls remote GPU runtimes (Colab first) and
  attaches the resulting OpenAI-compatible endpoint as a provider instance.
  Requires the only new `ProviderManager` capability either program needs:
  dynamic instance registration.

---

## Appendix A — capability extraction script

Regenerates matrix §4/§4.1 from the source rather than by hand:

```python
import ast, pathlib

BASE_OPTIONAL = ["generate_speech", "transcribe_audio", "generate_image", "ocr",
                 "create_embeddings", "warm_up", "supports_native_search"]
EXTRACTORS = ["extract_reasoning_content", "extract_delta_reasoning_content",
              "extract_tool_calls", "extract_usage_details", "extract_finish_reason"]
CORE = {"get_name", "get_models_details", "get_supported_parameters",
        "get_max_context_length", "chat_completion", "count_tokens",
        "count_message_tokens", "extract_response_content",
        "extract_delta_content", "close"}

for f in sorted(pathlib.Path("src/llmcore/providers").glob("*_provider.py")):
    tree = ast.parse(f.read_text())
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef)), None)
    if cls is None:
        continue
    methods = {n.name for n in cls.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    extra = sorted(m for m in methods if not m.startswith("_")
                   and m not in BASE_OPTIONAL and m not in EXTRACTORS and m not in CORE)
    print(f"\n## {f.stem.replace('_provider', '')} ({cls.name} <- "
          f"{', '.join(ast.unparse(b) for b in cls.bases)})")
    print("   media/opt :", ", ".join(m for m in BASE_OPTIONAL if m in methods) or "-")
    print("   extractors:", ", ".join(e.replace("extract_", "")
                                      for e in EXTRACTORS if e in methods) or "-")
    print("   extra API :", ", ".join(extra) or "-")
```
