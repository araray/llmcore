# @llmcore/client (TypeScript / Node)

TypeScript client for the **llmcore bridge**, generated from the frozen
`llmcore.v1` contract. Talks to a running `llmcore-bridge` over **both**
transports:

* **gRPC** (primary) — `LlmcoreGrpcClient`, built on `@grpc/grpc-js` +
  ts-proto-generated stubs (binary protobuf over HTTP/2).
* **HTTP + SSE** (secondary) — `LlmcoreHttpClient`, built on the global `fetch`
  (Node ≥ 18), with Server-Sent-Events streaming.

This is the first foreign-language client of phase **B2**. It depends only on the
contract (`bindings/proto`), never on Python.

## Install

```bash
npm install @llmcore/client
# peer runtime: a running `llmcore-bridge serve ...` (see the bindings USAGE.md)
```

## Quick start

```ts
import { LlmcoreGrpcClient } from "@llmcore/client";

const client = new LlmcoreGrpcClient("127.0.0.1:50151");
await client.ensureCompatible(["tier0"]);          // capability negotiation

const res = await client.chat({ message: "hello" });
console.log(res.text, res.usage?.totalTokens);

for await (const chunk of client.chatStream({ message: "stream me" })) {
  if (!chunk.done) process.stdout.write(chunk.text);
}
client.close();
```

HTTP is symmetric:

```ts
import { LlmcoreHttpClient } from "@llmcore/client";
const http = new LlmcoreHttpClient("http://127.0.0.1:50152");
await http.ensureCompatible();
console.log((await http.chat({ message: "hi" })).text);
```

See **USAGE.md** for the full API, error handling, and cancellation.

## What it covers (Tier 0)

`chat`, `chatStream`, `countTokens`, `estimateCost`, `listProviders`,
`listModels`, `getProviderDetails`, `getInfo`/`ensureCompatible`, `health`,
`reloadConfig`. `embed(...)` throws a `BridgeError` (UNSUPPORTED) — Embed is
UNIMPLEMENTED in `llmcore.v1`. AudioService is not yet surfaced (Tier 2 / B3).

## Errors

Every failure is a `BridgeError` with the same fields on both transports:
`category` (e.g. `"ERROR_CATEGORY_PROVIDER"`), `code`
(e.g. `"provider.rate_limited"`), `message`, `httpStatus`, `retryable`,
`retryAfterMs`, `provider`, `model`. On gRPC it is decoded from the binary
trailing metadata `llmcore-error-bin`; on HTTP from the `{ "error": {...} }`
body.

## Design notes

* **gRPC types are camelCase** (ts-proto), encoded as binary protobuf — casing is
  irrelevant on the wire. **HTTP types are snake_case**, matching the bridge's
  `preserving_proto_field_name=True` JSON. The two client classes therefore use
  different (but parallel) request/response shapes; pick the transport you want.
* **Cancellation:** `client.chatStream(req)` returns a `ChatStream` with a
  `cancel()` method (maps to gRPC `CANCELLED`); the async iterator rejects with a
  `BridgeError` after cancel.
* **No codegen toolchain needed at install time** — generated stubs under
  `src/gen/` are committed. Regenerate with `npm run gen` (uses buf's bundled
  protoc + the local `ts-proto` plugin; no standalone `protoc`).

## Codegen

```bash
npm run gen     # buf generate (template: buf.gen.ts.yaml) -> src/gen/**
npm run build   # tsc -> dist/
```

## Test

The suite spawns a real `llmcore-bridge` (with the deterministic FakeFacade) over
localhost TCP and drives both transports end to end.

```bash
# point at a python that has llmcore[bridge] importable:
export LLMCORE_BRIDGE_PYTHON=/path/to/venv/bin/python
npm test
```

Coverage: chat / streaming (concatenation == unary) / count / cost / catalog /
control, capability negotiation (accept + reject), `Embed` UNIMPLEMENTED,
structured error decode (incl. `retryAfterMs`), mid-stream error, and
cancellation — **18 tests, all green**.

## Distribution (planned)

Published to npm as `@llmcore/client` (ESM + CJS via the `dist/` build). The
generated stubs and source are included so consumers can rebuild.
