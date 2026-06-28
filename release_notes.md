# LLMCore Release Notes - June 2026 Ecosystem Integration

Prepared for merging `av/improvements_june-2026` into `main`.

Baseline: changes since merge base `0da25909d260` with `origin/main`.

## Release Story

This release turns LLMCore from a Python-first provider abstraction into the runtime backbone for the broader agent ecosystem. It adds a stable language-binding contract, richer live audio support, stronger cognitive-agent execution semantics, and the observability hooks needed for Wairu, Semantiscan, Grimoire, and Convergence to exchange traceable context.

## Highlights

- Ships the new bridge layer with protobuf contracts, a Python sidecar, gRPC, HTTP/SSE, and WebSocket transports.
- Adds Tier 0/Tier 1/Tier 2 client surfaces across TypeScript, Go, Rust, C, and C++, plus an in-process PyO3 Rust embedding path.
- Introduces native Deepgram support for speech-to-text, text-to-speech, Flux, Voice Agent, and Text Intelligence workflows.
- Hardens cognitive-agent planning with typed plan-step specs, structured tool execution, unloaded-tool rejection, resumed-history preservation, and pending-action snapshot restore.
- Adds objective-aware context compression, model-aware token counting, context budgeting, semantic citation provenance, and memory backend delegation.
- Expands observability with context diagnostics, semantic retrieval events, federation telemetry, phase token summaries, and iteration summaries.
- Adds HITL/OWASP metadata for dangerous-pattern auditing and safer human escalation review.
- Connects Grimoire runes into LLMCore as runtime tools.

## Developer Impact

The bridge now gives non-Python clients a first-class way to call LLMCore without embedding Python directly. The generated clients and contract guard make the API shape explicit, while the sidecar keeps runtime behavior centralized.

Agent developers get better plan execution fidelity, clearer permission metadata, more useful tool-result summaries, and resumable state that preserves both planned work and historical context. Provider authors get a shared token-estimation path and a warm-up lifecycle hook for expensive embedding/provider initialization.

## Operator Impact

Operators can observe failed context assembly, semantic retrieval, federation events, and per-phase token consumption from the outside. Audio deployments gain Deepgram examples and model cards, while bridge deployments can choose the transport that fits the client environment.

## Compatibility Notes

- New bridge extras introduce optional gRPC, Starlette, Uvicorn, protobuf, and WebSocket dependencies.
- Deepgram support is opt-in through the `deepgram` extra.
- Generated binding trees are included in the branch; consumers should treat the protobuf contract as the compatibility anchor.
- Token estimation now routes through LLMCore's own counters where native provider counts are unavailable.

## Validation Focus

Reviewers should focus on:

- Bridge conformance tests for gRPC, HTTP/SSE, WebSocket, sessions, vectors, presets, and audio.
- Provider tests around Deepgram and token fallback behavior.
- Cognitive-agent tests covering plan execution, resumed history, state snapshots, Grimoire tools, and context diagnostics.
- Binding smoke tests in TypeScript, Go, Rust, C, and C++.

## By the Numbers

- 71 commits since the merge base.
- 357 files changed.
- 59,708 insertions and 775 deletions before this release-note commit.

## Representative Commits

- `a4a6890` - add the gRPC and HTTP/SSE bridge.
- `99363c4` through `4e8a1e5` - build out Tier 2 audio over bridge and client bindings.
- `6ae71ea` through `2bc338d` - add Tier 1 sessions, vectors, presets, and generated client surfaces.
- `ad86ba3` - add the native Deepgram provider.
- `f94360f`, `3b1a785`, `050ad15`, and `d1af8b5` - improve context compression and diagnostics.
- `557610c`, `7f1097d`, and `cf1b511` - strengthen tool execution, permissions, and Grimoire integration.
