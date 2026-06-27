# llmcore language bindings

Out-of-process **bridge** that exposes the `llmcore` Python facade to non-Python
clients (C, C++, Rust, Go, TypeScript) over a single, versioned wire contract.

This directory holds the language-neutral pieces:

```
bindings/
├── proto/llmcore/v1/      # the wire contract (source of truth for every client)
│   ├── common.proto       #   Role, Message, Tool, ToolCall, ToolResult, Usage, FloatVector
│   ├── errors.proto       #   ErrorCategory, LlmcoreError
│   ├── inference.proto    #   InferenceService (Chat, ChatStream, Embed, CountTokens, EstimateCost)
│   ├── catalog.proto      #   CatalogService (ListProviders, ListModels, GetProviderDetails)
│   ├── control.proto      #   ControlService (GetInfo, Health, ReloadConfig)
│   └── audio.proto        #   AudioService (Tier-2, designed; implemented in B3)
├── buf.yaml               # buf module + lint/breaking config
├── buf.gen.yaml           # multi-language codegen plugins (B2)
├── contract_map.yaml      # machine-readable map: bridge assumptions -> llmcore symbols
├── scripts/
│   ├── gen_proto.sh       # generate the Python stubs (grpc_tools.protoc + import rewrite)
│   ├── contract_guard.py  # AST guard: fail CI if llmcore drifts from the contract
│   └── e2e_smoke.py       # golden end-to-end smoke (real subprocess, both transports)
├── examples/python/
│   └── client.py          # reference client (B1); foreign-language clients land in B2
├── README.md  USAGE.md  CONTRACT.md
```

The Python half of the bridge (the server) ships **inside** the `llmcore`
package at `src/llmcore/bridge/`, behind the `llmcore[bridge]` optional extra and
the `llmcore-bridge` console entry point.

## Why a sidecar (not an in-process embedding)

`llmcore` is async-first, config-driven (confy), and pulls in heavy optional
deps. Embedding a CPython runtime into each foreign language would multiply
build/packaging complexity and couple every client to llmcore's Python ABI. A
**sidecar process** that speaks gRPC/HTTP keeps clients thin, language-idiomatic,
and decoupled: they depend on the *contract*, not on Python. (An in-process PyO3
path for Rust is a deliberate, separate later track — phase B5.)

## Architecture

```
┌────────────┐   gRPC (primary)         ┌─────────────────────────────────────┐
│  client    │ ───────────────────────► │ llmcore-bridge (sidecar process)     │
│ (Go/Rust/  │   HTTP + SSE (secondary) │                                      │
│  C/C++/TS) │ ───────────────────────► │  grpc_server.py   http_app.py        │
└────────────┘                          │        \           /                 │
                                         │      ┌───────────────┐               │
                                         │      │  BridgeCore   │  (one adapter) │
                                         │      └───────────────┘               │
                                         │              │ proto in / proto out  │
                                         │      ┌───────────────┐               │
                                         │      │ LLMCoreFacade │  (Protocol)    │
                                         │      └───────────────┘               │
                                         │              │                       │
                                         │      ┌───────────────┐               │
                                         │      │  LLMCore (api)│               │
                                         │      └───────────────┘               │
                                         └─────────────────────────────────────┘
```

* **Dual transport, one core.** Both `grpc_server.py` and `http_app.py` are thin
  adapters over the *same* `BridgeCore`, so the two transports are guaranteed to
  produce identical canonical results (asserted by the transport-parity tests).
* **`BridgeCore` is pure translation** — decode → call the facade → encode. No
  retrieval/RAG/routing logic lives in the bridge; `enable_rag=False` is passed
  on every chat call to preserve the ecosystem's External-RAG invariant.
* **`LLMCoreFacade`** (a `typing.Protocol`) is the narrow, explicit surface the
  bridge depends on. The real `LLMCore` satisfies it structurally; the contract
  guard verifies that against the live source so the bridge can never silently
  drift.

## Scope (phased)

| Phase | Contents | Status |
|------:|----------|--------|
| **B0** | Architecture spec (`LLMCORE_BINDINGS_SPEC.md`) | ✅ delivered |
| **B1** | Proto contract (T0+T2) · Python bridge (gRPC + HTTP/SSE) · contract guard · golden e2e | ✅ **this deliverable** |
| B2 | Foreign-language clients (C/C++/Rust/Go/TS) generated from `proto/` | ⏳ next |
| B3 | Tier-2 audio: one-shot + duplex (gRPC bidi + HTTP-WS) | ⏳ |
| B4 | Tier-1 sessions/vector store | ⏳ |
| B5 | In-process PyO3 Rust binding | ⏳ |

### What B1 implements (Tier 0)

`Chat`, `ChatStream`, `CountTokens`, `EstimateCost`, `ListProviders`,
`ListModels`, `GetProviderDetails`, `GetInfo`, `Health`, `ReloadConfig`.

`Embed` is part of the contract but returns **UNIMPLEMENTED**: `LLMCore` exposes
no public embeddings method in this release (the provider-level
`create_embeddings` sits behind the private provider manager, and reaching into
internals would violate the public-surface rule). It is enabled in a later phase
once a public path exists. `AudioService` is **designed** in the contract but
returns UNIMPLEMENTED until B3.

## Setup

```bash
# Server (the bridge) + its runtime deps:
pip install "llmcore[bridge]"

# Dev/codegen tooling (stubs + contract guard). NOTE: `buf` is installed
# separately (https://buf.build), it is not a pip package.
pip install "llmcore[bridge-dev]"
```

Generate / regenerate the Python stubs after editing the contract:

```bash
sh bindings/scripts/gen_proto.sh
```

See **USAGE.md** for running the server and calling it; see **CONTRACT.md** for
the wire contract, capability negotiation, error model, and versioning policy.

## Verification

```bash
# 1. Contract lint (requires buf)
( cd bindings && buf lint )

# 2. Contract-vs-reality guard (AST only; no confy needed)
python bindings/scripts/contract_guard.py

# 3. Conformance suite (real gRPC channel + ASGI HTTP, deterministic FakeFacade)
pytest tests/bridge -q

# 4. Golden end-to-end smoke (spawns the real sidecar over TCP, both transports)
python bindings/scripts/e2e_smoke.py
```
