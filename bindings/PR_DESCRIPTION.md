# B1 — gRPC + HTTP/SSE language-binding bridge (Tier 0)

## Summary

Adds an **out-of-process bridge** that exposes the `llmcore` Python facade to
non-Python clients over a single versioned wire contract (`llmcore.v1`). It is
shipped behind the `llmcore[bridge]` optional extra and the `llmcore-bridge`
console command. **gRPC** is the primary transport; **HTTP + SSE** is a faithful
secondary projection of the same protos.

This is phase **B1** (B0 = the architecture spec, already delivered). It
implements the **Tier‑0** surface and establishes the language‑neutral contract
that the foreign‑language clients in **B2** will generate from.

## Motivation

`llmcore` is async‑first, confy‑configured, and pulls in heavy optional deps.
Embedding a CPython runtime into every target language would multiply build and
packaging complexity and couple each client to llmcore's Python ABI. A sidecar
that speaks gRPC/HTTP keeps clients thin and language‑idiomatic: they depend on
the *contract*, not on Python. (The in‑process PyO3 path for Rust is a deliberate
separate track — B5.)

## Implementation notes

* **One core, two transports.** `grpc_server.py` and `http_app.py` are thin
  adapters over the *same* `BridgeCore`; identical canonical results are
  guaranteed (and tested). `BridgeCore` is pure translation — decode → call the
  facade → encode — and passes `enable_rag=False` on every chat call to preserve
  the External‑RAG invariant.
* **Explicit, minimal dependency on llmcore.** `BridgeCore` is written against
  `LLMCoreFacade`, a `typing.Protocol` capturing exactly the methods it calls.
  The real `LLMCore` satisfies it structurally; the **contract guard**
  (`bindings/scripts/contract_guard.py`, AST‑only) verifies that against the live
  source so the bridge can't silently drift.
* **Errors** are normalized to a single `LlmcoreError` proto with a coarse
  category, a stable dotted code, **secret‑redacted** message, and provider
  metadata. gRPC carries it as `llmcore-error-bin` trailing metadata (no
  `grpcio-status` dep); HTTP returns `{"error": {...}}` with the mapped status.
  Cancellation is first‑class on both transports.
* **Capability negotiation.** Clients call `ControlService.GetInfo` and gate on
  `capabilities` (e.g. `tier0`, `transport.grpc`). Provisional features
  (`chat.tool_calls`) and Tier‑2 (`tier2.*`) are deliberately absent so clients
  degrade gracefully.
* **CLI / lifecycle.** `llmcore-bridge serve` supports gRPC+HTTP over UDS or TCP,
  TLS/mTLS, an auth gate, Linux `PR_SET_PDEATHSIG`, and graceful shutdown.

## Scope decisions (called out for review)

* **`Embed` → UNIMPLEMENTED.** `LLMCore` has no public embeddings method in this
  release; the provider‑level `create_embeddings` sits behind the private
  provider manager. Rather than reach into internals (violating the
  public‑surface rule), `Embed` is gated and documented for a follow‑up once a
  public path exists.
* **`AudioService` designed, not implemented.** The Tier‑2 contract is complete
  and compiles; the RPCs return UNIMPLEMENTED until **B3**, and `tier2.*`
  capabilities are not advertised.

## Testing

| Check | Result |
|---|---|
| `buf lint` | clean (3 documented RPC‑naming exceptions) |
| `ruff check` (changed files) | clean |
| `pytest tests/bridge` | **44 passed, 1 skipped** |
| `contract_guard.py` | OK — all mapped symbols consistent |
| `e2e_smoke.py` (real subprocess, gRPC + HTTP over TCP) | **PASS** |

The conformance suite drives a **real `grpc.aio` channel over a Unix socket** and
the **Starlette app over `httpx.ASGITransport`**, both backed by a deterministic
`FakeFacade`: chat / streaming (concatenation == unary) / count / cost / catalog
/ control, `Embed` UNIMPLEMENTED, the full error‑category table (incl.
`retry_after_ms` and redaction), mid‑stream errors, and **cancellation**
(gRPC local cancel → `CancelledError`, `call.code() == CANCELLED`). The golden
e2e additionally proves the **managed‑subprocess lifecycle** (readiness + clean
SIGTERM shutdown).

The **real‑`LLMCore`** integration test (`test_integration_real_llmcore.py`)
constructs the actual `LLMCore.create()` with an offline `FakeProvider` wired via
`PROVIDER_MAP` and drives `BridgeCore` against it. It is **skip‑guarded** on the
sovereign `confy>=0.4.2` dependency (run it in the llmcore dev env). We
deliberately did **not** stub confy to manufacture a green result; the FakeFacade
suite + AST guard are the authoritative, dependency‑free proof.

## Risks / mitigations

* **Contract drift** → `buf breaking` (FILE) + the AST contract guard in CI.
* **Secret leakage in errors/logs** → centralized `redact()` on every
  `LlmcoreError.message`.
* **Plaintext TCP exposure** → refused unless TLS+auth or an explicit
  `--insecure` (dev) opt‑out; UDS uses filesystem trust.
* **gRPC error fidelity without extra deps** → structured error in binary
  trailing metadata, decoded by clients.

## Files

* Added: `src/llmcore/bridge/**`, `bindings/**`, `tests/bridge/**`.
* Changed (additive): `pyproject.toml` (extras, script, ruff extend‑exclude).
* No changes to existing `llmcore` modules.

## Follow‑ups

* **B2**: C/C++/Rust/Go/TS clients generated from `bindings/proto` (via
  `buf generate` / `buf.gen.yaml`).
* **B3**: enable Tier‑2 audio (one‑shot + duplex; gRPC bidi + HTTP‑WS).
* `Embed`: enable once a public embeddings path exists in `llmcore`.
