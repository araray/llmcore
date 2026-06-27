# B2 (Go): `llmcore-go` — gRPC Tier-0 client

## Summary
Adds the second foreign-language client of phase **B2**: an idiomatic Go module
(`llmcore-go`) that drives a running `llmcore-bridge` over gRPC, generated from
the frozen `llmcore.v1` contract. Pure addition under `bindings/go/`.

## Honest build status
The authoring sandbox had **no Go toolchain** and **no reachable Go module
proxy** (`proxy.golang.org` / `go.googlesource.com` are not allow-listed; the
Ubuntu `golang-go` package was also unreachable). Therefore:

- The Go source is **hand-written and complete**, against the stable
  `google.golang.org/grpc` + `protoc-gen-go`/`protoc-gen-go-grpc` public APIs.
- The generated stubs (`gen/`) are **not committed** — they require the Go
  protoc plugins, which can't run here. `make gen` produces them.
- I **did** validate `buf.gen.go.yaml` with the real `buf` 1.71: it parsed the
  managed-mode template and failed only on the absent `protoc-gen-go` binary
  (expected), so the codegen template itself is correct for this `buf`.
- I **could not** run `go build`/`go test` here. The e2e suite is written and
  ready; run it in any Go environment with `make test`.

This is consistent with the project's anti-fabrication rule: nothing is claimed
green that wasn't actually executed.

## What's included
- **Client** (`client.go`) — `Dial`, capability negotiation
  (`EnsureCompatible`), `Chat`, cancellable `ChatStream`, `CountTokens`,
  `EstimateCost`, `Embed` (UNSUPPORTED), catalog + control, `Close`.
- **Errors** (`errors.go`) — `BridgeError` decoded from the gRPC binary trailing
  metadata `llmcore-error-bin` (`Category/Code/HTTPStatus/Retryable/
  RetryAfterMs/Provider/Model/GRPCCode`).
- **Options** (`options.go`) — insecure default, TLS/mTLS, raw dial options.
- **Codegen** — `buf.gen.go.yaml` (managed mode, no `.proto` edits) +
  `scripts/gen.sh`; `Makefile` with `gen/tidy/build/vet/test/all`.
- **Tests** — `harness_test.go` (spawns the bridge) + `client_test.go` (mirrors
  the TS assertions against the same FakeFacade fixtures).
- `go.mod`, `.gitignore`, `examples/quickstart`, `README.md`, `USAGE.md`.

## Implementation notes
- **Generator-version independence.** `protoc-gen-go-grpc` renamed the
  server-streaming client type from a named interface (<v1.5) to
  `grpc.ServerStreamingClient[T]` (>=v1.5). `ChatStream` stores the stream behind
  a local `chatStreamRecv` interface (`Recv() (*ChatChunk, error)` +
  `Trailer()`), which both satisfy — so the client compiles against either.
- **Trailer-based error detail.** grpc-go surfaces our error via trailing
  metadata, so every call passes `grpc.Trailer(&md)` and decodes
  `llmcore-error-bin` on failure (streaming uses `stream.Trailer()` after
  `Recv`).
- **Optional fields** are pointers (`proto.String(...)`); `provider_kwargs` is
  `*structpb.Struct` (`structpb.NewStruct`).
- **Single rename point**: module path in `go.mod` + `go_package_prefix` in
  `buf.gen.go.yaml` + imports.

## Testing (run in a Go env)
```bash
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
cd bindings/go
export LLMCORE_BRIDGE_PYTHON=/path/to/venv/bin/python
make all   # gen tidy build vet test
```
Expected: the 8 e2e tests pass against a spawned bridge (chat, streaming,
count+cost, catalog, negotiation accept/reject, Embed UNSUPPORTED, provider
rate-limit decode with HTTP 429 / retryable / RetryAfterMs=2000, cancellation).

## Risks / mitigations
- **Uncompiled here.** Mitigated by writing against stable APIs, the
  generator-version abstraction, buf template validation, and verifying every
  referenced proto symbol against source (field names, optional→pointer).
- **Unix-only test harness** (SIGTERM). The library itself is platform-neutral.
- **grpc/protobuf versions** pinned in `go.mod`; `go mod tidy` fills `go.sum`.

## Scope
Zero changes outside `bindings/go/`. B1 gate and the B2-TS package are untouched.

## Follow-ups
Rust (tonic/prost), then C++ (grpc++), then C (protobuf-c + libcurl; gRPC for
duplex audio per D6). Then B3 (Tier-2 audio).
