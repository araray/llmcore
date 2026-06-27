# llmcore C++ client

C++ client for the **llmcore bridge** over gRPC, generated from the frozen
`llmcore.v1` contract. Depends only on the contract, never on Python.

> **Build status / sandbox note.** This was **not compiled in the authoring
> sandbox**: gRPC/Protobuf development packages (and `protoc`) were unavailable
> and unfetchable there (gcc/g++ 13 *are* present). The code is written against
> the stable gRPC C++ API and every referenced proto symbol was verified against
> the `.proto` sources. Build + test it with CMake as below.

## Prerequisites

* CMake ≥ 3.16, a C++17 compiler.
* gRPC + Protobuf with CMake config, `protoc`, and `grpc_cpp_plugin`:
  ```bash
  apt install protobuf-compiler protobuf-compiler-grpc libgrpc++-dev libprotobuf-dev
  # or build gRPC from source; vcpkg ("grpc") / conan also work.
  ```

## Build & test

```bash
cd bindings/cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
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
AudioService (Tier 2) is generated but not wrapped (B3).

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
