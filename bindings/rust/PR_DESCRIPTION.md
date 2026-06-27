# B2 (Rust): `llmcore-proto` + `llmcore-client` — tonic Tier-0 client

## Summary
Adds the third foreign-language client of phase **B2**: an async Rust workspace
(tonic/prost) that drives a running `llmcore-bridge` over gRPC, generated from the
frozen `llmcore.v1` contract. Pure addition under `bindings/rust/`.

## Honest build status
The authoring sandbox had **no `cargo`/`rustc`** and **no reachable Rust toolchain
source** (rustup's host isn't allow-listed; the Ubuntu package path was blocked).
Therefore:

- The Rust source is **hand-written and complete**, against the stable
  `tonic` 0.12 / `prost` 0.13 public APIs.
- Generated protobuf code is produced by `llmcore-proto`'s `build.rs` at build
  time (never committed) and requires `protoc`.
- I **could not** run `cargo build`/`cargo test` here. The e2e suite is written
  and ready; run it with `cargo test` in a Rust environment.
- Every referenced proto symbol was verified against the `.proto` sources
  (message/field names; `optional` → `Option<T>`; `retry_after_ms` is
  `optional double` → `Option<f64>`).

Consistent with the project's anti-fabrication rule: nothing is claimed green
that wasn't actually executed.

## What's included
- **`llmcore-proto`** — `build.rs` (tonic-build) compiles `bindings/proto` into
  messages + tonic clients; `src/lib.rs` exposes `pub mod v1`.
- **`llmcore-client`** — `LlmcoreClient` over a single `Channel`: `connect` /
  `with_channel`, `ensure_compatible`, `chat`, cancellable `chat_stream`,
  `count_tokens`, `estimate_cost`, `embed` (UNSUPPORTED), catalog + control.
- **`BridgeError`** decoded from the gRPC binary trailing metadata
  `llmcore-error-bin` (`category/code/http_status/retryable/retry_after_ms/
  provider/model/grpc_code`).
- Workspace `Cargo.toml`, `rust-toolchain.toml`, `.gitignore`,
  `examples/quickstart.rs`, `tests/e2e.rs`, README/USAGE.

## Implementation notes
- **Cancellation.** `ChatStream` holds `Option<tonic::Streaming<ChatChunk>>`;
  `cancel()` sets it to `None` (drops the stream → HTTP/2 reset → server sees
  CANCELLED), and `message()` then yields `Ok(None)`.
- **Trailer-based error detail.** tonic surfaces our error as trailing metadata,
  read via `status.metadata().get_bin("llmcore-error-bin")` and prost-decoded;
  enum names via prost's `as_str_name()`.
- **Two-crate split** (`llmcore-proto` generated + `llmcore-client` ergonomic)
  matches the crates.io distribution plan.

## Testing (run in a Rust env)
```bash
# protoc required by the build script:
#   apt install protobuf-compiler   (or brew/dnf)
cd bindings/rust
export LLMCORE_BRIDGE_PYTHON=/path/to/venv/bin/python
cargo build && cargo test
```
Expected: the 8 e2e tests pass against a spawned bridge (chat, streaming,
count+cost, catalog, negotiation accept/reject, Embed UNSUPPORTED, provider
rate-limit decode with HTTP 429 / retryable / retry_after_ms = 2000.0,
cancellation).

## Risks / mitigations
- **Uncompiled here.** Mitigated by writing against stable APIs, verifying every
  proto symbol against source, and keeping the wrapper thin.
- **tonic-build API drift**: `compile_protos` (0.12) vs `.compile` (<0.11) —
  documented in `build.rs`/README.
- **`protoc` dependency** for codegen — documented.
- **Unix-oriented test harness** (kills the child via the OS).

## Scope
Zero changes outside `bindings/rust/`. The B1 gate, B2-TS, and B2-Go packages are
untouched.

## Follow-ups
C++ (grpc++/CMake), then C (protobuf-c + libcurl; gRPC for duplex audio per D6).
Then B3 (Tier-2 audio).
