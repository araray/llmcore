// llmcore C++ client implementation (grpc++).
#include "llmcore/client.hpp"

#include <grpcpp/grpcpp.h>

#include <string>
#include <utility>

#include "llmcore/v1/catalog.grpc.pb.h"
#include "llmcore/v1/control.grpc.pb.h"
#include "llmcore/v1/inference.grpc.pb.h"

namespace llmcore {
namespace {

// Decode a non-OK gRPC status into a BridgeError, preferring the structured
// payload from the "llmcore-error-bin" trailing metadata.
[[noreturn]] void ThrowFromStatus(const grpc::Status& status, const grpc::ClientContext& ctx) {
  const auto& md = ctx.GetServerTrailingMetadata();
  auto it = md.find("llmcore-error-bin");
  if (it != md.end()) {
    llmcore::v1::LlmcoreError pb;
    if (pb.ParseFromArray(it->second.data(), static_cast<int>(it->second.size()))) {
      BridgeError e(pb.code(), pb.message());
      e.category = llmcore::v1::ErrorCategory_Name(pb.category());
      e.retryable = pb.retryable();
      if (pb.has_http_status()) e.http_status = pb.http_status();
      if (pb.has_retry_after_ms()) e.retry_after_ms = pb.retry_after_ms();
      if (pb.has_provider()) e.provider = pb.provider();
      if (pb.has_model()) e.model = pb.model();
      e.grpc_code = static_cast<int>(status.error_code());
      throw e;
    }
  }
  BridgeError e("grpc." + std::to_string(static_cast<int>(status.error_code())),
                status.error_message());
  e.category = "ERROR_CATEGORY_INTERNAL";
  e.grpc_code = static_cast<int>(status.error_code());
  throw e;
}

BridgeError LocalError(const char* category, const char* code, std::string message) {
  BridgeError e(code, std::move(message));
  e.category = category;
  return e;
}

}  // namespace

// ---------------- ChatStream ----------------

struct ChatStream::Impl {
  grpc::ClientContext ctx;
  std::unique_ptr<grpc::ClientReader<llmcore::v1::ChatChunk>> reader;
  bool finished = false;
};

ChatStream::ChatStream() : impl_(std::make_unique<Impl>()) {}
ChatStream::~ChatStream() = default;
ChatStream::ChatStream(ChatStream&&) noexcept = default;
ChatStream& ChatStream::operator=(ChatStream&&) noexcept = default;

bool ChatStream::Read(llmcore::v1::ChatChunk* out) {
  if (impl_->finished) return false;
  if (impl_->reader->Read(out)) return true;
  grpc::Status status = impl_->reader->Finish();
  impl_->finished = true;
  if (!status.ok()) ThrowFromStatus(status, impl_->ctx);
  return false;
}

void ChatStream::Cancel() { impl_->ctx.TryCancel(); }

// ---------------- Client ----------------

struct Client::Impl {
  std::shared_ptr<grpc::Channel> channel;
  std::unique_ptr<llmcore::v1::InferenceService::Stub> inference;
  std::unique_ptr<llmcore::v1::CatalogService::Stub> catalog;
  std::unique_ptr<llmcore::v1::ControlService::Stub> control;
};

Client::Client() : impl_(std::make_unique<Impl>()) {}
Client::~Client() = default;

std::unique_ptr<Client> Client::Create(const std::string& target) {
  auto c = std::unique_ptr<Client>(new Client());
  c->impl_->channel = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
  c->impl_->inference = llmcore::v1::InferenceService::NewStub(c->impl_->channel);
  c->impl_->catalog = llmcore::v1::CatalogService::NewStub(c->impl_->channel);
  c->impl_->control = llmcore::v1::ControlService::NewStub(c->impl_->channel);
  return c;
}

// -- control --

llmcore::v1::ServerInfo Client::GetInfo() {
  grpc::ClientContext ctx;
  llmcore::v1::Empty req;
  llmcore::v1::ServerInfo resp;
  grpc::Status s = impl_->control->GetInfo(&ctx, req, &resp);
  if (!s.ok()) ThrowFromStatus(s, ctx);
  return resp;
}

llmcore::v1::HealthStatus Client::Health() {
  grpc::ClientContext ctx;
  llmcore::v1::Empty req;
  llmcore::v1::HealthStatus resp;
  grpc::Status s = impl_->control->Health(&ctx, req, &resp);
  if (!s.ok()) ThrowFromStatus(s, ctx);
  return resp;
}

llmcore::v1::ReloadConfigResponse Client::ReloadConfig(std::optional<std::string> path) {
  grpc::ClientContext ctx;
  llmcore::v1::ReloadConfigRequest req;
  if (path.has_value()) req.set_path(*path);
  llmcore::v1::ReloadConfigResponse resp;
  grpc::Status s = impl_->control->ReloadConfig(&ctx, req, &resp);
  if (!s.ok()) ThrowFromStatus(s, ctx);
  return resp;
}

llmcore::v1::ServerInfo Client::EnsureCompatible(
    const std::vector<std::string>& required_capabilities) {
  llmcore::v1::ServerInfo info = GetInfo();
  if (info.contract_version() != "llmcore.v1") {
    throw LocalError("ERROR_CATEGORY_INVALID_ARGUMENT", "contract.mismatch",
                     "server contract " + info.contract_version() + " != llmcore.v1");
  }
  for (const auto& want : required_capabilities) {
    bool found = false;
    for (const auto& have : info.capabilities()) {
      if (have == want) {
        found = true;
        break;
      }
    }
    if (!found) {
      throw LocalError("ERROR_CATEGORY_UNSUPPORTED", "capability.missing",
                       "server lacks required capability: " + want);
    }
  }
  return info;
}

// -- inference --

llmcore::v1::ChatResponse Client::Chat(const llmcore::v1::ChatRequest& req) {
  grpc::ClientContext ctx;
  llmcore::v1::ChatResponse resp;
  grpc::Status s = impl_->inference->Chat(&ctx, req, &resp);
  if (!s.ok()) ThrowFromStatus(s, ctx);
  return resp;
}

ChatStream Client::ChatStreamCall(const llmcore::v1::ChatRequest& req) {
  ChatStream cs;  // private ctor; Client is a friend
  cs.impl_->reader = impl_->inference->ChatStream(&cs.impl_->ctx, req);
  return cs;  // moved; pimpl keeps ctx/reader at a stable address
}

llmcore::v1::CountTokensResponse Client::CountTokens(const llmcore::v1::CountTokensRequest& req) {
  grpc::ClientContext ctx;
  llmcore::v1::CountTokensResponse resp;
  grpc::Status s = impl_->inference->CountTokens(&ctx, req, &resp);
  if (!s.ok()) ThrowFromStatus(s, ctx);
  return resp;
}

llmcore::v1::CostEstimate Client::EstimateCost(const llmcore::v1::EstimateCostRequest& req) {
  grpc::ClientContext ctx;
  llmcore::v1::CostEstimate resp;
  grpc::Status s = impl_->inference->EstimateCost(&ctx, req, &resp);
  if (!s.ok()) ThrowFromStatus(s, ctx);
  return resp;
}

llmcore::v1::EmbedResponse Client::Embed(const llmcore::v1::EmbedRequest& req) {
  grpc::ClientContext ctx;
  llmcore::v1::EmbedResponse resp;
  grpc::Status s = impl_->inference->Embed(&ctx, req, &resp);
  if (!s.ok()) ThrowFromStatus(s, ctx);  // UNIMPLEMENTED -> UNSUPPORTED
  return resp;
}

// -- catalog --

llmcore::v1::ListProvidersResponse Client::ListProviders() {
  grpc::ClientContext ctx;
  llmcore::v1::Empty req;
  llmcore::v1::ListProvidersResponse resp;
  grpc::Status s = impl_->catalog->ListProviders(&ctx, req, &resp);
  if (!s.ok()) ThrowFromStatus(s, ctx);
  return resp;
}

llmcore::v1::ListModelsResponse Client::ListModels(const std::string& provider_name) {
  grpc::ClientContext ctx;
  llmcore::v1::ListModelsRequest req;
  req.set_provider_name(provider_name);
  llmcore::v1::ListModelsResponse resp;
  grpc::Status s = impl_->catalog->ListModels(&ctx, req, &resp);
  if (!s.ok()) ThrowFromStatus(s, ctx);
  return resp;
}

llmcore::v1::ModelDetails Client::GetProviderDetails(const std::string& provider_name) {
  grpc::ClientContext ctx;
  llmcore::v1::GetProviderRequest req;
  if (!provider_name.empty()) req.set_provider_name(provider_name);
  llmcore::v1::ModelDetails resp;
  grpc::Status s = impl_->catalog->GetProviderDetails(&ctx, req, &resp);
  if (!s.ok()) ThrowFromStatus(s, ctx);
  return resp;
}

}  // namespace llmcore
