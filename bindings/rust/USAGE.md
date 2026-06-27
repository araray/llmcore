# Using the llmcore Rust client

## Prerequisites

A running bridge (dev / fake backend):

```bash
LLMCORE_BRIDGE_FAKE=1 python -m llmcore.bridge.cli serve \
  --transport grpc --grpc-address 127.0.0.1:50151 --insecure
```

## Connect

```rust
use llmcore_client::LlmcoreClient;

let mut client = LlmcoreClient::connect("http://127.0.0.1:50151").await?;
```

`LlmcoreClient` is `Clone` (cheap — clones share the underlying channel). Methods
take `&mut self` (tonic clients are `&mut`); clone per task if you need
concurrency.

### TLS / mTLS

```rust
use tonic::transport::{Channel, ClientTlsConfig, Certificate, Identity};
use llmcore_client::LlmcoreClient;

let tls = ClientTlsConfig::new()
    .ca_certificate(Certificate::from_pem(ca_pem))
    .identity(Identity::from_pem(client_cert_pem, client_key_pem)); // mTLS
let channel = Channel::from_static("https://host:50151")
    .tls_config(tls)?
    .connect()
    .await?;
let mut client = LlmcoreClient::with_channel(channel);
```

## Capability negotiation

```rust
let info = client.ensure_compatible(&["tier0"]).await?;
// Err is a BridgeError with code "contract.mismatch" or "capability.missing".
```

## Inference

```rust
use llmcore_client::v1;

// Unary
let res = client.chat(v1::ChatRequest {
    message: "Summarize the contract.".into(),
    provider_name: Some("openai".into()),     // optional fields are Option<T>
    model_name: Some("gpt-4o-mini".into()),
    ..Default::default()
}).await?;
let _ = res.text;
let _ = res.usage.and_then(|u| u.total_tokens);

// Streaming (+ cancellation)
let mut stream = client.chat_stream(v1::ChatRequest {
    message: "Write a long answer.".into(),
    ..Default::default()
}).await?;
while let Some(chunk) = stream.message().await? {
    if !chunk.done { print!("{}", chunk.text); }
    // stream.cancel(); // stop early — drops the stream (server sees CANCELLED)
}
```

### `provider_kwargs` (google.protobuf.Struct)

```rust
use prost_types::{Struct, Value, value::Kind};
use std::collections::BTreeMap;

let mut fields = BTreeMap::new();
fields.insert("temperature".to_string(), Value { kind: Some(Kind::NumberValue(0.2)) });
let req = v1::ChatRequest {
    message: "hi".into(),
    provider_kwargs: Some(Struct { fields }),
    ..Default::default()
};
```

## Tokens, cost, catalog

```rust
client.count_tokens(v1::CountTokensRequest { text: "a b c".into(), ..Default::default() }).await?; // .tokens
client.estimate_cost(v1::EstimateCostRequest {
    provider_name: "openai".into(), model_name: "gpt-4o-mini".into(),
    prompt_tokens: 1000, completion_tokens: 500, ..Default::default()
}).await?;
client.list_providers().await?;          // .providers: Vec<String>
client.list_models("openai").await?;     // .models:    Vec<String>
client.get_provider_details(Some("openai".into())).await?; // v1::ModelDetails
```

## Audio (Tier 2)

Available when the bridge advertises `tier2.audio`; negotiate with
`client.ensure_compatible(&["tier2.audio"]).await?`.

### One-shot

```rust
let sp = client.synthesize(v1::SynthesizeRequest { text: "hello".into(), ..Default::default() }).await?;
// sp.audio_data: Vec<u8>, sp.format/model/voice: String
let tr = client.transcribe(v1::TranscribeRequest { audio_data: pcm, ..Default::default() }).await?;
// tr.text: String, tr.language: Option<String>, tr.segments: Vec<TranscriptionSegment>
let img = client.generate_image(v1::GenerateImageRequest { prompt: "a cat".into(), n: 2, ..Default::default() }).await?;
// img.images[i].data: Option<String> (base64 b64_json)
let doc = client.ocr(v1::OcrRequest {
    source: Some(v1::ocr_request::Source::Data(pdf)),      // or ::Url(url)
    ..Default::default()
}).await?; // doc.pages: Vec<Struct>, doc.pages_processed: i32, doc.doc_size_bytes: Option<i64>
let an = client.analyze_text(v1::AnalyzeTextRequest { text: "…".into(), ..Default::default() }).await?;
// an.summary/model: Option<String>, an.topics/intents: Vec<Struct>
```

### Live duplex (bidi)

Each call takes the request side as an `impl Stream<Item = …>` (e.g.
`tokio_stream::iter(frames)` for a fixed sequence, or a channel-backed stream for
dynamic sending) and returns an `AudioStream<T>`. The request stream ending
half-closes the call; `message()` yields the next response (`Ok(None)` at the
clean end), `cancel()` aborts.

```rust
use tokio_stream::iter;

// STT — AudioIn frames -> TranscriptionStreamEvent
let reqs = iter(vec![
    v1::AudioIn { frame: Some(v1::audio_in::Frame::Audio(pcm_chunk)) },
    v1::AudioIn { frame: Some(v1::audio_in::Frame::Control(3)) }, // STT_CONTROL_CLOSE
]);
let mut s = client.transcribe_stream(reqs).await?;
while let Some(ev) = s.message().await? {
    if ev.r#type == 2 /* STREAM_EVENT_TYPE_FINAL */ { println!("{}", ev.text); }
}

// TTS — SynthControl frames -> AudioOut (ordered by .seq)
let reqs = iter(vec![
    v1::SynthControl { frame: Some(v1::synth_control::Frame::Text("hello world".into())) },
    v1::SynthControl { frame: Some(v1::synth_control::Frame::Control(3)) }, // TTS_CONTROL_CLOSE
]);
let mut s = client.synthesize_stream(reqs).await?;
while let Some(out) = s.message().await? { sink.write_all(&out.audio)?; } // out.seq: i64

// Voice agent — VoiceAgentClientEvent -> VoiceAgentEvent
// (a leading `Event::Settings(struct)` selects the provider; omit for the default)
let reqs = iter(vec![
    v1::VoiceAgentClientEvent { event: Some(v1::voice_agent_client_event::Event::InjectUserMessage("hi".into())) },
]);
let mut s = client.voice_agent(reqs).await?;
while let Some(ev) = s.message().await? {
    if ev.r#type == 8 /* VOICE_AGENT_EVENT_TYPE_AUDIO */ { if let Some(a) = ev.audio { play(&a); } }
}
```

Enum fields are plain `i32` (prost); compare against the contract's numeric
values (or the generated `v1::StreamEventType` variants). `tokio-stream` is the
stream source used above.

## Error handling

```rust
use llmcore_client::BridgeError;

match client.chat(req).await {
    Ok(res) => { /* ... */ }
    Err(BridgeError { code, http_status, retryable, retry_after_ms, .. }) => {
        eprintln!("code={code} http={http_status:?} retryable={retryable}");
        if retryable {
            if let Some(ms) = retry_after_ms {
                tokio::time::sleep(std::time::Duration::from_millis(ms as u64)).await;
                // ...retry
            }
        }
    }
}
```

Common `code`s: `provider.rate_limited` (HTTP 429, retryable, `retry_after_ms`
set), `provider.unauthenticated` (401), `context.too_long` (413),
`not_found.session` (404), `unsupported.capability` (UNSUPPORTED),
`invalid_argument` (400), `internal` (500).

## Commands

```bash
cargo build
cargo test         # set LLMCORE_BRIDGE_PYTHON first (spawns a real bridge)
cargo run -p llmcore-client --example quickstart
cargo clippy --all-targets
```
