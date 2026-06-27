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
