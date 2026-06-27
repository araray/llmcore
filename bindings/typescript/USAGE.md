# Using @llmcore/client

## Prerequisites

A running bridge. For local dev with the deterministic fake backend:

```bash
LLMCORE_BRIDGE_FAKE=1 python -m llmcore.bridge.cli serve \
  --transport grpc,http \
  --grpc-address 127.0.0.1:50151 --http-address 127.0.0.1:50152 --insecure
```

(For production, run the real bridge with TLS/auth — see `bindings/USAGE.md`.)

## gRPC client — `LlmcoreGrpcClient`

```ts
import { LlmcoreGrpcClient, BridgeError } from "@llmcore/client";

const client = new LlmcoreGrpcClient("127.0.0.1:50151");

// 1) Negotiate. Throws BridgeError if the contract or a capability is missing.
const info = await client.ensureCompatible(["tier0"]);
console.log(info.llmcoreVersion, info.bridgeVersion, info.capabilities);

// 2) Unary chat
const res = await client.chat({
  message: "Summarize the contract.",
  providerName: "openai",        // optional
  modelName: "gpt-4o-mini",      // optional
  providerKwargs: { temperature: 0.2 }, // arbitrary non-secret kwargs (Struct)
});
console.log(res.text, res.usage?.promptTokens, res.usage?.totalTokens);

// 3) Streaming chat (+ cancellation)
const stream = client.chatStream({ message: "Write a long answer." });
setTimeout(() => stream.cancel(), 1000); // optional: stop early
try {
  for await (const chunk of stream) {
    if (!chunk.done) process.stdout.write(chunk.text);
  }
} catch (err) {
  if (err instanceof BridgeError && err.grpcCode /* CANCELLED */) {
    // cancelled
  } else {
    throw err;
  }
}

// 4) Tokens, cost, catalog
await client.countTokens({ text: "a b c" });          // { tokens: 3 }
await client.estimateCost({
  providerName: "openai", modelName: "gpt-4o-mini",
  promptTokens: 1000, completionTokens: 500,
});
await client.listProviders();                          // { providers: [...] }
await client.listModels("openai");                     // { models: [...] }
await client.getProviderDetails("openai");             // ModelDetails

client.close();
```

### Method reference (gRPC)

| Method | Returns | Notes |
|---|---|---|
| `ensureCompatible(caps?)` | `ServerInfo` | checks `contractVersion` + capabilities |
| `getInfo()` / `health()` | `ServerInfo` / `HealthStatus` | |
| `chat(req)` | `ChatResponse` | unary |
| `chatStream(req)` | `ChatStream` (async-iterable, `.cancel()`) | terminal chunk has `done:true` |
| `countTokens(req)` | `CountTokensResponse` | |
| `estimateCost(req)` | `CostEstimate` | |
| `embed(req)` | rejects `BridgeError` | UNIMPLEMENTED in v1 |
| `listProviders()` | `ListProvidersResponse` | |
| `listModels(name)` | `ListModelsResponse` | |
| `getProviderDetails(name?)` | `ModelDetails` | |
| `reloadConfig(req?)` | `ReloadConfigResponse` | |
| `close()` | `void` | closes channels |

Request fields are **camelCase** (`sessionId`, `providerName`, `saveSession`, …);
all are optional via `fromPartial` except `message`.

### TLS / mTLS

```ts
import { credentials } from "@grpc/grpc-js";
import { readFileSync } from "node:fs";

const creds = credentials.createSsl(
  readFileSync("ca.crt"),
  readFileSync("client.key"),  // for mTLS
  readFileSync("client.crt"),  // for mTLS
);
const client = new LlmcoreGrpcClient("host:50151", { credentials: creds });
```

## HTTP/SSE client — `LlmcoreHttpClient`

```ts
import { LlmcoreHttpClient, BridgeError } from "@llmcore/client";

const http = new LlmcoreHttpClient("http://127.0.0.1:50152");
await http.ensureCompatible(["tier0"]);

const res = await http.chat({ message: "hello", provider_name: "openai" });
console.log(res.text, res.usage?.total_tokens);

for await (const chunk of http.chatStream({ message: "stream me" })) {
  if (!chunk.done && chunk.text) process.stdout.write(chunk.text);
}

await http.countTokens("a b c");                       // { tokens: 3 }
await http.listProviders();                            // { providers: [...] }
await http.health();                                   // { ok: true }
```

HTTP request/response fields are **snake_case** (`provider_name`, `total_tokens`,
…), matching the bridge's JSON projection.

## Audio (Tier 2) — `LlmcoreGrpcClient` + `LlmcoreWsAudioClient`

Available when the bridge advertises `tier2.audio` (e.g. a fake bridge started
with `LLMCORE_BRIDGE_FAKE_AUDIO=1`). Negotiate with
`ensureCompatible(["tier2.audio"])`.

### One-shot (unary), over gRPC

```ts
const speech = await client.synthesize({ text: "hello", voice: "nova", responseFormat: "wav" });
//   speech.audioData: Uint8Array, speech.format, speech.model, speech.voice
const tr = await client.transcribe({ audioData: bytes, language: "en" });
//   tr.text, tr.language, tr.segments[].{text,start,end,speaker}
const img = await client.generateImage({ prompt: "a cat", n: 2 });
//   img.images[].data is base64 (b64_json): Buffer.from(data, "base64")
const doc = await client.ocr({ data: pdfBytes });           // or { url: "https://…" }
//   doc.pages[], doc.documentAnnotation, doc.pagesProcessed, doc.docSizeBytes
const an = await client.analyzeText({ text: "…", features: { summarize: true, topics: true } });
//   an.summary, an.topics[], an.intents[], an.sentiments
```

### Live duplex, over gRPC

```ts
import { SttControl, TtsControl, StreamEventType, VoiceAgentEventType } from "@llmcore/client";

// STT: write AudioIn frames, iterate TranscriptionStreamEvents
const stt = client.transcribeStream();
stt.write({ open: { model: "nova-3", language: "en" } });
stt.write({ audio: pcmChunk });                       // Uint8Array; repeat as needed
stt.write({ control: SttControl.STT_CONTROL_CLOSE }); // optional; end() also half-closes
stt.end();
for await (const ev of stt) {
  if (ev.type === StreamEventType.STREAM_EVENT_TYPE_FINAL) console.log(ev.text);
}

// TTS: write SynthControl frames, iterate AudioOut chunks (ordered by .seq)
const tts = client.synthesizeStream();
tts.write({ open: { voice: "nova", format: "linear16" } });
tts.write({ text: "hello world" });
tts.write({ control: TtsControl.TTS_CONTROL_CLOSE });
tts.end();
for await (const out of tts) sink.write(out.audio);   // out.audio: Uint8Array, out.seq: number

// Voice agent: write VoiceAgentClientEvents, iterate VoiceAgentEvents
const va = client.voiceAgent();
va.write({ settings: { provider_name: "deepgram" } }); // leading settings opens the session
va.write({ injectUserMessage: "what's the weather?" });
va.write({ audio: micChunk });
va.end();
for await (const ev of va) {
  if (ev.type === VoiceAgentEventType.VOICE_AGENT_EVENT_TYPE_AUDIO) play(ev.audio);
}
```

`AudioDuplexStream` also exposes `cancel()` (maps to gRPC CANCELLED).

### Live duplex, over WebSocket

Same three streams, transport-swapped. `write()`/`end()` are async (they await
the socket open); `end()` sends the `{}` end-of-input sentinel (WebSocket has no
half-close), letting the server finish streaming (e.g. the voice-agent `CLOSE`).

```ts
import { LlmcoreWsAudioClient } from "@llmcore/client";

const wsAudio = new LlmcoreWsAudioClient("http://127.0.0.1:50152"); // http(s) -> ws(s)
const stt = wsAudio.transcribeStream();
await stt.write({ open: { model: "nova-3" } });
await stt.write({ audio: pcmChunk });
await stt.end();
for await (const ev of stt) console.log(ev.type, ev.text);
```

When audio is disabled the socket is accepted then closed `1011`; iterating the
stream throws a `BridgeError` (UNSUPPORTED).

## Error handling

```ts
import { BridgeError } from "@llmcore/client";

try {
  await client.chat({ message: "…" });
} catch (err) {
  if (err instanceof BridgeError) {
    console.error(err.category, err.code, err.httpStatus, err.retryable, err.retryAfterMs);
    if (err.retryable && err.retryAfterMs) {
      await new Promise((r) => setTimeout(r, err.retryAfterMs));
      // …retry
    }
  } else {
    throw err;
  }
}
```

Common `code`s: `provider.rate_limited` (429, retryable, `retryAfterMs` set),
`provider.unauthenticated` (401), `context.too_long` (413),
`not_found.session` (404), `unsupported.capability` (501),
`invalid_argument` (400), `internal` (500).

## Common workflows

```bash
npm run gen      # regenerate stubs from bindings/proto (buf + ts-proto)
npm run build    # tsc -> dist/
LLMCORE_BRIDGE_PYTHON=/path/to/venv/bin/python npm test   # e2e vs a real bridge
```
