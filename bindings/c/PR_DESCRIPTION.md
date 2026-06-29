# B2 (C): libcurl + cJSON Tier-0 client (HTTP/SSE)

## Summary
Adds the fifth and final foreign-language client of phase **B2**: a C library
(`llmcore_c`) that drives a running `llmcore-bridge` over its **HTTP + SSE**
transport (JSON wire), using **libcurl** + **cJSON**. Pure addition under
`bindings/c/`. This completes the B2 client matrix (TypeScript, Go, Rust, C++, C).

## Transport choice (per D6)
The contract's gRPC transport is the primary path, but gRPC-in-C is verbose and,
by decision **D6**, is reserved for **duplex audio** in a later phase. For Tier 0
the bridge's HTTP/SSE surface is the pragmatic, idiomatic target for C — the same
surface the TypeScript HTTP client already exercises (snake_case JSON, SSE frames,
`{"error": {...}}` error bodies).

## Honest build status
The authoring sandbox had **gcc/g++ 13 and make, but no libcurl/cJSON dev
packages** (and apt was blocked), so:

- The C source is **complete** and **passes `gcc -std=c11 -Wall -Wextra
  -fsyntax-only`** against minimal stub headers for `<curl/curl.h>` and
  `<cjson/cJSON.h>`.
- It was **not linked or executed** here. Syntax-checking proves well-formedness
  (no typos, correct arity against my usage), **not** linkage or runtime behavior.
- The e2e suite is written and ready; run it with `ctest` / `make test` where
  libcurl + cJSON are installed.

Reported exactly as it is, per the project's anti-fabrication rule.

## What's included
- **`include/llmcore_client.h`** — opaque handle, `llmcore_error`, the Tier-0
  surface, and explicit per-function ownership documentation.
- **`src/llmcore_client.c`** — libcurl POST + cJSON; error decode; an SSE parser
  (`event:`/`data:` lines, blank-line framing) handling mid-stream `event: error`
  and callback-driven cancellation (nonzero return → `CURLE_WRITE_ERROR` →
  reported as success).
- **`CMakeLists.txt`** (CURL via `find_package`; cJSON via config → pkg-config →
  `-lcjson`) and a **plain `Makefile`** (pkg-config) for environments without
  CMake.
- `examples/quickstart.c`, `tests/e2e.c`, README/USAGE, `.gitignore`.

## Implementation notes
- **Error model parity:** unary errors parsed from `{"error": {...}}`; SSE errors
  from the `event: error` frame whose `data` is the bare `LlmcoreError` object
  (not wrapped). Both populate the same `llmcore_error` fields, matching TS/Go/
  Rust/C++ (incl. `retry_after_ms` as a double).
- **Streaming:** one growable buffer accumulates bytes; complete `\n\n`-delimited
  events are parsed incrementally and dispatched to the callback; the terminal
  `event: done` sets `done=1`.
- **Memory:** caller-frees model documented in the header and enforced by
  dedicated `*_free` helpers; all sources `-Wextra`-clean.

## Testing (run where libcurl + cJSON exist)
```bash
apt install libcurl4-openssl-dev libcjson-dev
cd bindings/c
cmake -S . -B build && cmake --build build -j      # or: make
export LLMCORE_BRIDGE_PYTHON=/path/to/venv/bin/python
ctest --test-dir build --output-on-failure          # or: make test
```
Expected: `ALL TESTS PASSED` (chat, streaming, count+cost, catalog, negotiation
accept/reject, Embed UNSUPPORTED, provider rate-limit decode 429 / retryable /
retry_after_ms = 2000.0, cancellation).

## Risks / mitigations
- **Not linked/run here.** Mitigated by `-Wall -Wextra -fsyntax-only` against
  stubs, a conservative single-handle design, and reuse of the exact fixtures the
  other four clients already validate against this bridge.
- **cJSON packaging variance** (header at `<cjson/cJSON.h>` vs `<cJSON.h>`) —
  handled with `__has_include`; library discovery via config/pkg-config/fallback.
- **Unix-only test harness** (fork/exec/kill).

## Scope
Zero changes outside `bindings/c/`. B1 and the TS/Go/Rust/C++ packages untouched.

## Follow-ups
B3 — Tier-2 audio (gRPC bidi + HTTP-WS), implementable and testable on the bridge
side; the C audio path would use gRPC C-core per D6.
