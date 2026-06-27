// Quickstart: drive the bridge over gRPC.
//
//   LLMCORE_BRIDGE_FAKE=1 python -m llmcore.bridge.cli serve \
//     --transport grpc --grpc-address 127.0.0.1:50151 --insecure
//   ./quickstart            # or set LLMCORE_GRPC=host:port
#include <cstdlib>
#include <iostream>
#include <string>

#include "llmcore/client.hpp"

int main() {
  const char* env = std::getenv("LLMCORE_GRPC");
  const std::string target = env ? env : "127.0.0.1:50151";
  auto client = llmcore::Client::Create(target);

  try {
    auto info = client->EnsureCompatible({"tier0"});
    std::cout << "contract=" << info.contract_version() << " caps=";
    for (const auto& c : info.capabilities()) std::cout << c << " ";
    std::cout << "\n";

    llmcore::v1::ChatRequest req;
    req.set_message("hello from c++");
    auto res = client->Chat(req);
    std::cout << "chat -> " << res.text() << " (tokens=" << res.usage().total_tokens() << ")\n";

    llmcore::v1::ChatRequest sreq;
    sreq.set_message("stream me");
    auto stream = client->ChatStreamCall(sreq);
    std::cout << "stream -> ";
    llmcore::v1::ChatChunk chunk;
    while (stream.Read(&chunk)) {
      if (!chunk.done()) std::cout << chunk.text();
    }
    std::cout << "\n";
  } catch (const llmcore::BridgeError& e) {
    std::cerr << "error: " << e.category << " " << e.what() << "\n";
    return 1;
  }
  return 0;
}
