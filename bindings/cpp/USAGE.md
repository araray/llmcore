# Using the llmcore C++ client

## Prerequisites

A running bridge (dev / fake backend):

```bash
LLMCORE_BRIDGE_FAKE=1 python -m llmcore.bridge.cli serve \
  --transport grpc --grpc-address 127.0.0.1:50151 --insecure
```

## Connect

```cpp
#include "llmcore/client.hpp"
auto client = llmcore::Client::Create("127.0.0.1:50151");  // insecure (dev)
```

### TLS / mTLS

The default `Create` uses insecure credentials. To use TLS, build the channel
yourself and construct the stubs over it. The simplest path is to add a `Create`
overload in `src/client.cpp`:

```cpp
// in client.cpp
std::unique_ptr<Client> Client::CreateSecure(
    const std::string& target, std::shared_ptr<grpc::ChannelCredentials> creds) {
  auto c = std::unique_ptr<Client>(new Client());
  c->impl_->channel = grpc::CreateChannel(target, creds);
  /* NewStub(...) as in Create */
  return c;
}
```

with `grpc::SslCredentials(grpc::SslCredentialsOptions{ /* pem_root_certs,
pem_private_key, pem_cert_chain for mTLS */ })`.

## Capability negotiation

```cpp
auto info = client->EnsureCompatible({"tier0"});
// throws BridgeError (code "contract.mismatch" or "capability.missing") on failure
```

## Inference

```cpp
// Unary
llmcore::v1::ChatRequest req;
req.set_message("Summarize the contract.");
req.set_provider_name("openai");                 // optional fields: set_* sets presence
auto* kwargs = req.mutable_provider_kwargs();     // google.protobuf.Struct
(*kwargs->mutable_fields())["temperature"].set_number_value(0.2);
auto res = client->Chat(req);
res.text();
res.usage().total_tokens();

// Streaming (+ cancellation)
auto stream = client->ChatStreamCall(req);
llmcore::v1::ChatChunk chunk;
while (stream.Read(&chunk)) {
  if (!chunk.done()) std::cout << chunk.text();
  // stream.Cancel();  // stop early -> gRPC CANCELLED
}
```

## Tokens, cost, catalog

```cpp
llmcore::v1::CountTokensRequest ct; ct.set_text("a b c");
client->CountTokens(ct).tokens();

llmcore::v1::EstimateCostRequest ec;
ec.set_provider_name("openai"); ec.set_model_name("gpt-4o-mini");
ec.set_prompt_tokens(1000); ec.set_completion_tokens(500);
client->EstimateCost(ec);

client->ListProviders();              // .providers(i), .providers_size()
client->ListModels("openai");         // .models(i)
client->GetProviderDetails("openai"); // llmcore::v1::ModelDetails
```

## Audio (Tier 2)

Available when the bridge advertises `tier2.audio`; negotiate with
`client->EnsureCompatible({"tier2.audio"})`.

### One-shot

```cpp
llmcore::v1::SynthesizeRequest sreq; sreq.set_text("hello");
auto sp = client->Synthesize(sreq);              // sp.audio_data(), sp.model(), sp.voice()

llmcore::v1::TranscribeRequest treq; treq.set_audio_data(pcm);
auto tr = client->Transcribe(treq);              // tr.text(), tr.language(), tr.segments(i)

llmcore::v1::GenerateImageRequest ireq; ireq.set_prompt("a cat"); ireq.set_n(2);
auto img = client->GenerateImage(ireq);          // img.images(i).data() is base64 (b64_json)

llmcore::v1::OcrRequest oreq; oreq.set_data(pdf); // or oreq.set_url(url)
auto doc = client->Ocr(oreq);                    // doc.pages(i), doc.pages_processed(), doc.doc_size_bytes()

llmcore::v1::AnalyzeTextRequest areq; areq.set_text("…");
// areq.mutable_features() -> google::protobuf::Struct to set summarize/topics/...
auto an = client->AnalyzeText(areq);             // an.summary(), an.topics(i), an.intents(i)
```

### Live duplex (bidi)

Each call returns a stream: `Write` frames, `WritesDone` to half-close, then
`Read` until it returns false (`Read` throws `BridgeError` on a non-OK terminal
status). `Cancel()` aborts (gRPC CANCELLED).

```cpp
// STT — AudioIn frames -> TranscriptionStreamEvent
auto s = client->TranscribeStreamCall();
llmcore::v1::AudioIn f; f.set_audio(pcm_chunk); s.Write(f);
llmcore::v1::AudioIn c; c.set_control(llmcore::v1::STT_CONTROL_CLOSE); s.Write(c);
s.WritesDone();
llmcore::v1::TranscriptionStreamEvent ev;
while (s.Read(&ev)) {
  if (ev.type() == llmcore::v1::STREAM_EVENT_TYPE_FINAL) std::cout << ev.text();
}

// TTS — SynthControl frames -> AudioOut (ordered by .seq())
auto t = client->SynthesizeStreamCall();
llmcore::v1::SynthControl tf; tf.set_text("hello world"); t.Write(tf);
llmcore::v1::SynthControl tc; tc.set_control(llmcore::v1::TTS_CONTROL_CLOSE); t.Write(tc);
t.WritesDone();
llmcore::v1::AudioOut out;
while (t.Read(&out)) sink.write(out.audio());

// Voice agent — VoiceAgentClientEvent -> VoiceAgentEvent
// (a leading event with mutable_settings() selects the provider; omit for default)
auto v = client->VoiceAgentCall();
llmcore::v1::VoiceAgentClientEvent ie; ie.set_inject_user_message("hi"); v.Write(ie);
v.WritesDone();
llmcore::v1::VoiceAgentEvent vev;
while (v.Read(&vev)) {
  if (vev.type() == llmcore::v1::VOICE_AGENT_EVENT_TYPE_AUDIO) play(vev.audio());
}
```

## Error handling

```cpp
try {
  client->Chat(req);
} catch (const llmcore::BridgeError& e) {
  // e.category, e.code, e.http_status, e.retryable, e.retry_after_ms, e.provider, e.model
  if (e.retryable && e.retry_after_ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<long>(*e.retry_after_ms)));
    // ...retry
  }
}
```

Common `code`s: `provider.rate_limited` (HTTP 429, retryable, `retry_after_ms`
set), `provider.unauthenticated` (401), `context.too_long` (413),
`not_found.session` (404), `unsupported.capability` (UNSUPPORTED),
`invalid_argument` (400), `internal` (500).

## Build commands

```bash
cmake -S . -B build && cmake --build build -j
export LLMCORE_BRIDGE_PYTHON=/path/to/venv/bin/python
ctest --test-dir build --output-on-failure
```
