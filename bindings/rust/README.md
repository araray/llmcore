# llmcore Rust bindings

Async Rust client for the **llmcore bridge** over gRPC, generated from the frozen
`llmcore.v1` contract. Crates (per the per-language distribution plan):

* **`llmcore-proto`** — generated messages + tonic gRPC clients (codegen via a
  `build.rs`).
* **`llmcore-client`** — ergonomic async wrapper: capability negotiation,
  structured errors, cancellable streaming.
* **`llmcore-embedded`** — the **B5 in-process (PyO3) binding**: embeds CPython
  and calls `llmcore` directly for single-binary Rust deployments, reusing the
  same `llmcore.v1` types so its Tier-0 API mirrors `llmcore-client`. Excluded
  from the default build (it links libpython); see
  [`llmcore-embedded/README.md`](llmcore-embedded/README.md).

The gRPC crates are built on **tonic 0.12 / prost 0.13** and depend only on the
contract, never on Python. `llmcore-embedded` is the deliberate exception (D8).

> **Build status / sandbox note.** This was **not compiled in the authoring
> sandbox**: no `cargo`/`rustc` was available and no Rust toolchain source was
> reachable. The code is written against the stable tonic/prost public APIs and
> every referenced proto symbol was verified against the `.proto` sources
> (field names, `optional` → `Option<T>`). Build + test it with `cargo` as
> below.

## Prerequisites

* A Rust toolchain (stable).
* **`protoc`** on PATH — required by `llmcore-proto`'s build script
  (`apt install protobuf-compiler`, `brew install protobuf`, or
  `dnf install protobuf-compiler`).

## Build & test

```bash
cd bindings/rust
cargo build
# e2e tests spawn a real bridge; point at a python with llmcore[bridge]:
export LLMCORE_BRIDGE_PYTHON=/path/to/venv/bin/python
cargo test
cargo run -p llmcore-client --example quickstart
```

## Quick start

```rust
use llmcore_client::{v1, LlmcoreClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = LlmcoreClient::connect("http://127.0.0.1:50151").await?;
    client.ensure_compatible(&["tier0"]).await?;

    let res = client
        .chat(v1::ChatRequest { message: "hello".into(), ..Default::default() })
        .await?;
    println!("{} ({:?} tokens)", res.text, res.usage.and_then(|u| u.total_tokens));

    let mut stream = client
        .chat_stream(v1::ChatRequest { message: "stream me".into(), ..Default::default() })
        .await?;
    while let Some(chunk) = stream.message().await? {
        if !chunk.done { print!("{}", chunk.text); }
    }
    Ok(())
}
```

See **USAGE.md** for the full API, TLS, error handling, and cancellation.

## Surface (Tier 0)

`chat`, `chat_stream` (cancellable), `count_tokens`, `estimate_cost`,
`list_providers`, `list_models`, `get_provider_details`,
`get_info`/`ensure_compatible`, `health`, `reload_config`. `embed(...)` returns a
`BridgeError` (UNSUPPORTED) — Embed is UNIMPLEMENTED in `llmcore.v1`.

## Surface (Tier 1 — sessions, vector store & presets)

Available when the bridge advertises `tier1.sessions` and/or `tier1.vector`
(negotiate with `client.ensure_compatible(&["tier1.sessions", "tier1.vector"])`).

* **Sessions:** `create_session`, `get_session`, `list_sessions`,
  `delete_session`, `update_session_name`, `fork_session`, `clone_session`,
  `delete_messages`, `get_messages_by_range`.
* **Context items:** `add_context_item`, `get_context_item`,
  `remove_context_item`.
* **Vector store / RAG:** `add_documents`, `search_vector_store`,
  `list_vector_collections`, `list_rag_collections`, `get_rag_collection_info`,
  `delete_rag_collection`.
* **Context presets:** `save_context_preset`, `get_context_preset`,
  `list_context_presets`, `delete_context_preset`.

Run the end-to-end demo against a fake-backed bridge (the fake gates make the
bridge advertise the Tier-1 capabilities without a real backend):

```bash
# bridge
LLMCORE_BRIDGE_FAKE=1 LLMCORE_BRIDGE_FAKE_SESSIONS=1 LLMCORE_BRIDGE_FAKE_VECTOR=1 \
  python -m llmcore.bridge.cli serve --transport grpc \
  --grpc-address 127.0.0.1:50151 --insecure
# example (note the http:// endpoint scheme)
LLMCORE_GRPC=http://127.0.0.1:50151 cargo run -p llmcore-client --example sessions
```

See **USAGE.md** for request shapes and the top-level **../CONTRACT.md** for the
authoritative capability/contract reference.

## Surface (Tier 2 — audio)

Available when the bridge advertises `tier2.audio`. One-shot: `synthesize`,
`transcribe`, `generate_image`, `ocr`, `analyze_text`. Live duplex — each takes
the request side as an `impl Stream<Item = …>` and returns an
[`AudioStream`]`<T>` (`message()` yields events, `cancel()` aborts):
`transcribe_stream`, `synthesize_stream`, `voice_agent`.

```rust
use llmcore_client::{v1, LlmcoreClient};
# async fn run(mut client: LlmcoreClient) -> Result<(), Box<dyn std::error::Error>> {
let reqs = tokio_stream::iter(vec![
    v1::AudioIn { frame: Some(v1::audio_in::Frame::Audio(pcm)) },
    v1::AudioIn { frame: Some(v1::audio_in::Frame::Control(/* STT_CONTROL_CLOSE */ 3)) },
]);
let mut s = client.transcribe_stream(reqs).await?;
while let Some(ev) = s.message().await? {
    if ev.r#type == /* STREAM_EVENT_TYPE_FINAL */ 2 { println!("{}", ev.text); }
}
let sp = client.synthesize(v1::SynthesizeRequest { text: "hello".into(), ..Default::default() }).await?;
let _ = sp.audio_data; // Vec<u8>
# Ok(()) }
```

## Errors

Every method returns `Result<_, BridgeError>`. `BridgeError` is decoded from the
gRPC binary trailing metadata `llmcore-error-bin` and carries `category`, `code`,
`http_status`, `retryable`, `retry_after_ms`, `provider`, `model`, `grpc_code`.

## Notes

* `compile_protos` is the tonic-build 0.12 API; for tonic-build < 0.11 rename it
  to `.compile(...)` in `llmcore-proto/build.rs`.
* `provider_kwargs` is `Option<prost_types::Struct>`; build one with
  `prost_types::Struct { fields: ... }` or a helper (see USAGE).
* The e2e harness is Unix-oriented (kills the child via the OS).

## Distribution

Publish `llmcore-proto` then `llmcore-client` to crates.io; the client depends on
the proto crate by version. Tag releases per the bindings distribution plan.
