# Continuation Guide — llmcore bindings (post-B1 → B2)

Ariadne's thread for a cold-start agent. B1 (proto contract + Python gRPC/HTTP
bridge + contract guard + golden e2e) is **complete and green**. Next is **B2**:
the foreign-language clients (C/C++/Rust/Go/TS) generated from the frozen
`llmcore.v1` contract.

## 0. Locked decisions (unchanged)

D1 out-of-process sidecar (not in-process). D2 surface order T0→T2→T1→T3; v1 =
T0(+T2 designed). D3 dual transport: gRPC primary + HTTP/SSE secondary, one core
adapter. D4 bridge at `src/llmcore/bridge/`, `llmcore[bridge]` extra, entry point
`llmcore-bridge`. D5 managed-subprocess (default) + shared-daemon (optional). D6
C gets gRPC + REST/SSE; live duplex audio in C = gRPC best-effort. D7 TCP needs
mTLS + AuthFlow. D8 PyO3 in-process Rust = mandatory later track (B5). D9
per-language idiomatic distribution.

## 1. Environment facts (critical)

- Python 3.12. Build tree: `/home/claude/work/` (extracted llmcore v0.49.14,
  tarball `b1bb207`). venv: `/home/claude/bridge-venv/`
  (`. /home/claude/bridge-venv/bin/activate`).
- **confy >= 0.4.2 is sovereign / tarball-only** — NOT on PyPI (tops at 0.3.8,
  which fails to build). So `pip install -e .` fails dependency resolution.
  Install pattern that works here:
  `pip install -e . --no-deps` + the real runtime deps (aiohttp, aiofiles,
  python-dotenv, typing_extensions, toml, psutil, aiosqlite, deepdiff,
  sqlalchemy[asyncio], pydantic, tiktoken, httpx) + bridge toolchain (grpcio,
  grpcio-tools, protobuf, starlette, uvicorn, websockets, anyio, pyyaml, pytest,
  pytest-asyncio, ruff).
- **`import llmcore` works WITHOUT confy** (top-of-file confy imports are
  TYPE_CHECKING-guarded). confy is only needed at RUNTIME inside
  `LLMCore.create()`. So FakeFacade conformance + AST guard run green here;
  real-LLMCore construction is skip-guarded.
- **buf is available** via `npm i -g @bufbuild/buf` (1.71.0). `buf lint` runs for
  real and passes. chromadb import warning is expected/ignored.
- Test invocation that neutralizes the parent conftest/addopts:
  `python -m pytest tests/bridge --confcutdir=tests/bridge -o addopts="" -p no:cacheprovider -q`

## 2. B1 inventory (all under /home/claude/work, mirrored to outputs)

Contract (`bindings/proto/llmcore/v1/`): common, errors, inference, catalog,
control, audio `.proto`. `bindings/buf.yaml` (lint STANDARD + 3 RPC-naming
excepts), `bindings/buf.gen.yaml` (B2 plugins).

Bridge (`src/llmcore/bridge/`): `__init__.py` (BRIDGE_VERSION="0.1.0",
CONTRACT_VERSION="llmcore.v1"), `redact.py`, `errors.py`, `facade.py`
(LLMCoreFacade Protocol + build_facade), `info.py`, `core.py` (BridgeCore),
`grpc_server.py`, `http_app.py`, `ws_app.py` (B3 stub), `cli.py`,
`_testing/{fake_facade,fake_provider}.py`, `_generated/llmcore/v1/*` (stubs).

Tooling/tests: `bindings/contract_map.yaml`,
`bindings/scripts/{gen_proto.sh,contract_guard.py,e2e_smoke.py}`,
`bindings/examples/python/client.py`, `tests/bridge/{conftest,
test_conformance_grpc,test_conformance_http,test_error_mapping,test_redaction,
test_contract_guard,test_integration_real_llmcore}.py`.

Packaging: `pyproject.toml` additive edits — `bridge`/`bridge-dev` extras,
`bridge` in `all`, `llmcore-bridge` script, ruff `extend-exclude` for
`_generated`.

Docs: `bindings/{README,USAGE,CONTRACT}.md`, `bindings/COMMIT_MESSAGE.txt`,
`bindings/PR_DESCRIPTION.md`, this guide, the SHA256 manifest.

## 3. Verify B1 (all currently green)

```bash
cd /home/claude/work && . /home/claude/bridge-venv/bin/activate
( cd bindings && buf lint )                       # clean
ruff check src/llmcore/bridge bindings tests/bridge  # clean
python bindings/scripts/contract_guard.py         # OK
python -m pytest tests/bridge --confcutdir=tests/bridge -o addopts="" -p no:cacheprovider -q  # 44 passed, 1 skipped
python bindings/scripts/e2e_smoke.py              # E2E SMOKE: PASS
```

If stubs are missing/regenerated: `sh bindings/scripts/gen_proto.sh`.

## 4. Contract state (frozen for v1; additive-only from here)

T0 implemented: Chat, ChatStream, CountTokens, EstimateCost, ListProviders,
ListModels, GetProviderDetails, GetInfo, Health, ReloadConfig.

Capability flags advertised: `tier0`, `inference.chat`, `inference.chat_stream`,
`inference.count_tokens`, `inference.estimate_cost`, `catalog.providers`,
`catalog.models`, `control.info`, `transport.{grpc,http}`.

B1 pins:
- **Embed → UNIMPLEMENTED** (UNSUPPORTED). No public LLMCore embeddings path;
  provider `create_embeddings` is behind the private manager. To enable later:
  either add a public `LLMCore` embeddings method (reviewed change request) or
  wire `EmbeddingManager`; then turn on an `inference.embed` capability and
  implement `BridgeCore.embed`.
- **chat.tool_calls / finish_reason** = PROVISIONAL proto fields, unset, no
  capability. Pin against the real provider response (api.py get_last_raw_response
  at :1566) when building B3/tool support.
- **AudioService** = designed, returns UNIMPLEMENTED; `tier2.*` absent.

Error model: `LlmcoreError{category,code,message(redacted),provider,model,
http_status,retryable,retry_after_ms,details}`. gRPC → abort + binary trailing
metadata key `llmcore-error-bin`. HTTP → `{"error":{...}}` + mapped status. Full
category→status table in `bindings/CONTRACT.md` and `src/llmcore/bridge/errors.py`.

## 5. B2 plan (foreign-language clients)

Goal: per-language client libraries generated from `bindings/proto`, each a full
deliverable (code + tests + README/USAGE + commit/PR + manifest). The Python
bridge is the server they all talk to; reuse `e2e_smoke.py`'s pattern (spawn
`LLMCORE_BRIDGE_FAKE=1` bridge, drive the client, assert) per language.

Codegen: `buf generate` against `bindings/buf.gen.yaml` (already stubs Go, TS
Connect-ES, Python). Add plugins/targets per language. Suggested order: Go and TS
first (best gRPC tooling), then Rust (tonic/prost), then C++ (grpc++), then C
(grpc C-core OR protobuf-c + libcurl for REST/SSE; live duplex audio in C is
gRPC-only/best-effort per D6).

Distribution (D9): Go = module + git tags; Rust = crates.io (`llmcore-proto` +
`llmcore-client`); TS = npm (`@llmcore/client`); C++ = CMake `find_package`; C =
CMake/pkg-config.

Each client must: (1) negotiate via `GetInfo` and check `contract_version` +
required capabilities; (2) decode `LlmcoreError` (gRPC trailing metadata
`llmcore-error-bin`; HTTP `error` body); (3) support cancellation; (4) treat
`Embed`/AudioService as UNIMPLEMENTED.

## 6. Gotchas (learned in B1)

- **Python stub imports** collide with the real `llmcore` package; `gen_proto.sh`
  sed-rewrites `from llmcore.v1 import` → `from llmcore.bridge._generated.
  llmcore.v1 import`. Keep that rewrite for any regen.
- **gRPC local cancel** raises `asyncio.CancelledError` (not AioRpcError) on the
  client; `call.cancelled()` is True and `await call.code()` == CANCELLED.
- **HTTP JSON keys are snake_case** (`MessageToDict(preserving_proto_field_name=
  True)`), e.g. `retry_after_ms`, not `retryAfterMs`.
- **ruff**: generated stubs are `extend-exclude`d; `raise X from exc` inside
  `except` (B904); the repo rule set excludes BLE/N so don't add those noqas.
- **Don't shim confy** to force a green integration test — keep it skip-guarded.
- **Packaging env is `sh`** (no brace expansion); build manifests in
  `/home/claude` then `cp` (the outputs mount fails on streamed multi-command
  redirects).

## 7. Gate (must continue to hold)

Zero changes under `src/llmcore/` outside `src/llmcore/bridge/`; only additions
under `bridge/`, `bindings/`, `tests/bridge/`, and additive `pyproject.toml`
edits. `buf lint`/`buf breaking` clean; conformance + e2e green.
