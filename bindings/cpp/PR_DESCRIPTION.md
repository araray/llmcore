# B2 (C++): grpc++ Tier-0 client

## Summary
Adds the fourth foreign-language client of phase **B2**: a C++ library
(`llmcore_client`) that drives a running `llmcore-bridge` over gRPC, generated
from the frozen `llmcore.v1` contract. Pure addition under `bindings/cpp/`.

## Honest build status
The authoring sandbox had **gcc/g++ 13 but no gRPC/Protobuf dev stack** (no
`protoc`, no `grpc_cpp_plugin`, no `libgrpc++`), and they were unfetchable (apt
blocked). Therefore:

- The C++ source is **hand-written and complete**, against the stable grpc++ API.
- Generated `*.pb.*` / `*.grpc.pb.*` are produced by CMake at build time (not
  committed).
- I **could not** compile or run it here. The CTest e2e suite is written and
  ready; run it with `ctest` in an environment that provides gRPC/Protobuf.
- Every referenced proto symbol was verified against the `.proto` sources
  (message/field names; proto3 `optional` → `has_*()`/value accessors).

Consistent with the project's anti-fabrication rule: nothing is claimed green
that wasn't actually executed.

## What's included
- **`include/llmcore/client.hpp`** — pimpl'd public API (`Client`, `ChatStream`,
  `BridgeError`). Methods return generated messages by value and throw on error.
- **`src/client.cpp`** — grpc++ stubs; `ThrowFromStatus` decodes
  `llmcore-error-bin` (or falls back to the gRPC status); cancellable reader.
- **`CMakeLists.txt`** — `find_package(gRPC/Protobuf CONFIG)`, protoc+grpc plugin
  codegen, static lib + example + CTest.
- `examples/quickstart.cpp`, `tests/e2e.cpp`, README/USAGE, `.gitignore`.

## Implementation notes
- **Pimpl** keeps `<grpcpp/grpcpp.h>` out of the public header (only generated
  message headers leak). `ChatStream`'s `ClientContext`+`ClientReader` live
  behind a `unique_ptr<Impl>` at a stable address, so returning the stream by
  move from the factory keeps the reader's context pointer valid.
- **Error decode** uses `ErrorCategory_Name(...)` for the category string and
  `has_*()` for the optional fields (matching the TS/Go/Rust shapes).
- **Cancellation** = `ClientContext::TryCancel()`; the next `Read` returns false
  or throws a CANCELLED `BridgeError`.

## Testing (run with gRPC installed)
```bash
apt install protobuf-compiler protobuf-compiler-grpc libgrpc++-dev libprotobuf-dev
cd bindings/cpp
cmake -S . -B build && cmake --build build -j
export LLMCORE_BRIDGE_PYTHON=/path/to/venv/bin/python
ctest --test-dir build --output-on-failure
```
Expected: the e2e binary prints `ALL TESTS PASSED` (chat, streaming, count+cost,
catalog, negotiation accept/reject, Embed UNSUPPORTED, provider rate-limit decode
HTTP 429 / retryable / retry_after_ms = 2000.0, cancellation).

## Risks / mitigations
- **Uncompiled here.** Mitigated by stable-API authorship, per-symbol proto
  verification, and a conventional pimpl/move design.
- **gRPC packaging variance** (CMake config vs pkg-config; plugin path) —
  resolved via imported targets (`$<TARGET_FILE:protobuf::protoc>`,
  `gRPC::grpc_cpp_plugin`); documented apt deps.
- **Unix-only test harness** (fork/exec/kill).

## Scope
Zero changes outside `bindings/cpp/`. B1 and the TS/Go/Rust packages untouched.

## Follow-ups
C client (libcurl + cJSON over HTTP/SSE; gRPC for duplex audio per D6). Then B3.
