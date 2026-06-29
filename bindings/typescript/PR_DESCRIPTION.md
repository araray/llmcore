# B2 (TypeScript): `@llmcore/client` — Node Tier-0 client over gRPC + HTTP/SSE

## Summary
Adds the first foreign-language client of phase **B2**: a TypeScript/Node package
(`@llmcore/client`) that drives a running `llmcore-bridge` over **both**
transports, generated from the frozen `llmcore.v1` contract. Pure addition under
`bindings/typescript/`; nothing else in the tree changes.

## Motivation
B1 delivered the contract + the Python bridge (server). B2 delivers the clients.
TypeScript leads because it is fully buildable **and** verifiable in CI here
(npm-reachable; the bridge runs as a subprocess with the deterministic
FakeFacade), letting us prove the wire contract end to end before tackling
toolchains that aren't present in this environment (Go/Rust/C/C++).

## What's included
- **Typed gRPC client** (`src/grpcClient.ts`) over `@grpc/grpc-js` + ts-proto
  stubs: promise-based unary, an async-iterable `ChatStream` with `cancel()`,
  and `ensureCompatible()` capability negotiation.
- **HTTP/SSE client** (`src/httpClient.ts`) over the global `fetch`: JSON unary +
  an SSE `chatStream` parser. Snake_case wire shapes (matching the bridge's
  `preserving_proto_field_name=True`).
- **Unified errors** (`src/errors.ts`): one `BridgeError` decoded from the gRPC
  binary trailing metadata `llmcore-error-bin` **or** the HTTP `{error:{…}}`
  body, exposing `category/code/httpStatus/retryable/retryAfterMs/provider/model`.
- **Committed codegen** (`src/gen/**`) + `buf.gen.ts.yaml` + `scripts/gen.sh`
  (buf's bundled protoc + local ts-proto plugin; no standalone `protoc`).
- Docs (`README.md`, `USAGE.md`), `examples/quickstart.ts`, and full project
  config (`tsconfig.json`, `package.json`, `package-lock.json`, `.gitignore`).

## Implementation notes
- **Two casings, one contract.** gRPC rides binary protobuf (camelCase TS types,
  casing irrelevant on the wire); HTTP rides JSON (snake_case). The two client
  classes are deliberately parallel rather than forced into one shape, so each is
  idiomatic for its transport.
- **Cancellation.** `ChatStream.cancel()` maps to gRPC `CANCELLED`; the async
  iterator then rejects with a `BridgeError`. Verified by test.
- **Embed / Audio.** `embed(...)` rejects `UNSUPPORTED` (Embed is UNIMPLEMENTED in
  `llmcore.v1`); AudioService (Tier 2) is intentionally not surfaced yet (B3).
- **No install-time toolchain.** Generated stubs are committed; `npm run gen`
  only needed when the proto changes.

## Testing
`test/bridge.ts` spawns a real `llmcore-bridge` (FakeFacade) on localhost TCP and
waits for both transports; `test/grpc.test.ts` (11) and `test/http.test.ts` (7)
drive the clients against it.

```
tsc --strict build: clean
npm test          : 18 passed, 0 failed
```

Covered: chat; streaming (token concatenation equals the unary text); count;
cost (USD 3.0 on the 1M/1M fake fixture); catalog (providers/models/details);
control/health; capability negotiation (accept + reject of a missing cap); Embed
UNIMPLEMENTED on both transports; structured provider rate-limit decode
(`category=PROVIDER`, `code=provider.rate_limited`, `httpStatus=429`,
`retryable=true`, `retryAfterMs=2000`, `provider=fake`); mid-stream error event;
gRPC stream cancellation.

To run locally:
```bash
export LLMCORE_BRIDGE_PYTHON=/path/to/venv/bin/python   # python with llmcore[bridge]
cd bindings/typescript && npm install && npm test
```

## Risks / mitigations
- **grpc-js stream async-iteration** depends on the installed `@grpc/grpc-js`
  exposing `Symbol.asyncIterator` on `ClientReadableStream` (it does for ^1.10);
  pinned via `package-lock.json`. The iterator is additionally guarded so errors
  rethrow as `BridgeError`.
- **`fetch`/`ReadableStream`/`TextDecoder`** require Node ≥ 18 (declared in
  `engines`).
- **Test isolation flag** `--test-isolation` is unavailable on some Node 22
  builds; the test script uses `node --import tsx --test <files>`, which
  propagates the loader without it.

## Scope / guarantees
- Zero changes outside `bindings/typescript/`. B1 gate still holds (no changes
  under `src/llmcore/` outside `bridge/`; additive `pyproject.toml` only).
- The client never imports Python; it speaks only the `llmcore.v1` wire contract.

## Follow-ups (B2 continued)
Go, Rust, C++, and C clients (code-complete packages; build/verify where the
toolchain exists). Then B3 (Tier-2 audio: gRPC bidi + HTTP-WS).
