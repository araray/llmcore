# llmcore C++ client

C++ client for the **llmcore bridge** over gRPC, generated from the frozen
`llmcore.v1` contract. Depends only on the contract, never on Python.

> **Build.** `cmake -B build && cmake --build build -j` works out of the box: if
> gRPC isn't installed it is fetched and built from source (see
> [Prerequisites](#prerequisites)). The Tier-1 `sessions` example below was built
> and run against a fake bridge.

## Prerequisites

* CMake ≥ 3.16, a C++17 compiler, and (for the source-build fallback) network
  access.
* **gRPC + Protobuf are resolved automatically** — no system packages required:
  * If an installed gRPC CMake config is found, it is used (fastest), e.g.
    `apt install protobuf-compiler protobuf-compiler-grpc libgrpc++-dev
    libprotobuf-dev`, `brew install grpc`, or vcpkg/conan.
  * Otherwise gRPC (with its bundled protobuf, abseil, and `grpc_cpp_plugin`) is
    fetched and **built from source** via CMake `FetchContent` — just a compiler
    and a network. The first configure is slow because gRPC is large, but the
    build then works on any machine out of the box.
  * Force the source build with `-DLLMCORE_FETCH_GRPC=ON`; pin the version with
    `-DLLMCORE_GRPC_TAG=vX.Y.Z`.

## Build & test

The plain build works with no flags:

```bash
cd bindings/cpp
cmake -B build && cmake --build build -j
# e2e spawns a real bridge; point at a python with llmcore[bridge]:
export LLMCORE_BRIDGE_PYTHON=/path/to/venv/bin/python
ctest --test-dir build --output-on-failure
./build/quickstart
```

Codegen runs as part of the build (protoc + grpc_cpp_plugin into
`build/generated/`); nothing is committed.

## API sketch

```cpp
#include "llmcore/client.hpp"

auto client = llmcore::Client::Create("127.0.0.1:50151");   // throws on transport setup
auto info = client->EnsureCompatible({"tier0"});            // throws BridgeError on mismatch

llmcore::v1::ChatRequest req;
req.set_message("hello");
auto res = client->Chat(req);                               // res.text(), res.usage()...

auto stream = client->ChatStreamCall(req);
llmcore::v1::ChatChunk chunk;
while (stream.Read(&chunk)) {                               // false at end; throws on error
  if (!chunk.done()) std::cout << chunk.text();
}
// stream.Cancel();  // maps to gRPC CANCELLED
```

Methods return generated `llmcore::v1::*` messages by value and **throw**
`llmcore::BridgeError` on failure. See **USAGE.md** for the full API, TLS, error
fields, and cancellation.

## Surface (Tier 0)

`Chat`, `ChatStreamCall` (cancellable), `CountTokens`, `EstimateCost`,
`ListProviders`, `ListModels`, `GetProviderDetails`, `GetInfo` /
`EnsureCompatible`, `Health`, `ReloadConfig`. `Embed(...)` throws
`BridgeError` (UNSUPPORTED) — Embed is UNIMPLEMENTED in `llmcore.v1`.

## Surface (Tier 1 — sessions, vector store & presets)

Available when the bridge advertises `tier1.sessions` (sessions + context items)
and/or `tier1.vector` (vector store + RAG); negotiate with
`client->EnsureCompatible({"tier1.sessions", "tier1.vector"})`. All return
generated `llmcore::v1::*` types by value and throw `BridgeError`.

* **Sessions:** `CreateSession`, `GetSession`, `ListSessions`, `DeleteSession`,
  `UpdateSessionName`, `ForkSession`, `CloneSession`, `DeleteMessages`,
  `GetMessagesByRange`.
* **Context items:** `AddContextItem`, `GetContextItem`, `RemoveContextItem`.
* **Vector store & RAG:** `AddDocuments`, `SearchVectorStore`,
  `ListVectorCollections`, `ListRagCollections`, `GetRagCollectionInfo`,
  `DeleteRagCollection`.
* **Context presets:** `SaveContextPreset`, `GetContextPreset`,
  `ListContextPresets`, `DeleteContextPreset`.

The fake backend gates these behind `LLMCORE_BRIDGE_FAKE_SESSIONS=1` and
`LLMCORE_BRIDGE_FAKE_VECTOR=1`. Run the end-to-end demo (`examples/sessions.cpp`):

```bash
# 1. serve a fake bridge that advertises tier1.sessions + tier1.vector
LLMCORE_BRIDGE_FAKE=1 LLMCORE_BRIDGE_FAKE_SESSIONS=1 LLMCORE_BRIDGE_FAKE_VECTOR=1 \
  python -m llmcore.bridge.cli serve --transport grpc \
  --grpc-address 127.0.0.1:50151 --insecure

# 2. build + run
cmake -B build && cmake --build build -j
LLMCORE_GRPC=127.0.0.1:50151 ./build/sessions
```

See **USAGE.md** for per-method field details.

## Surface (Tier 2 — audio)

Available when the bridge advertises `tier2.audio`. One-shot: `Synthesize`,
`Transcribe`, `GenerateImage`, `Ocr`, `AnalyzeText` (return generated result
types by value, throw `BridgeError`). Live duplex — each returns a cancellable
stream (`Write` frames, `WritesDone`, then `Read` until false; throws on a non-OK
terminal status): `TranscribeStreamCall`, `SynthesizeStreamCall`,
`VoiceAgentCall`.

```cpp
auto s = client->TranscribeStreamCall();
llmcore::v1::AudioIn f; f.set_audio(pcm);
s.Write(f);
llmcore::v1::AudioIn c; c.set_control(llmcore::v1::STT_CONTROL_CLOSE);
s.Write(c);
s.WritesDone();
llmcore::v1::TranscriptionStreamEvent ev;
while (s.Read(&ev)) {
  if (ev.type() == llmcore::v1::STREAM_EVENT_TYPE_FINAL) std::cout << ev.text();
}
```

## Errors

`llmcore::BridgeError` (a `std::runtime_error`) carries `category`, `code`,
`message`, `http_status`, `retryable`, `retry_after_ms`, `provider`, `model`,
`grpc_code`, decoded from the gRPC binary trailing metadata `llmcore-error-bin`.

## Notes

* The public header is pimpl'd so it pulls in only the generated message headers
  (not `<grpcpp/grpcpp.h>`); link against `llmcore_client`.
* TLS: the wrapper uses insecure credentials by default. For TLS, construct a
  channel with `grpc::SslCredentials(...)` — extend `Client::Create` (or add an
  overload) to accept a `std::shared_ptr<grpc::ChannelCredentials>`.
* The e2e harness is Unix-only (fork/exec/kill).
