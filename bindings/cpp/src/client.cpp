// llmcore C++ client implementation (grpc++).
#include "llmcore/client.hpp"

#include <grpcpp/grpcpp.h>

#include <string>
#include <utility>

#include "llmcore/v1/audio.grpc.pb.h"
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

// ---------------- Audio duplex streams ----------------

struct TranscribeStream::Impl {
  grpc::ClientContext ctx;
  std::unique_ptr<grpc::ClientReaderWriter<llmcore::v1::AudioIn,
                                           llmcore::v1::TranscriptionStreamEvent>>
      stream;
  bool finished = false;
};

TranscribeStream::TranscribeStream() : impl_(std::make_unique<Impl>()) {}
TranscribeStream::~TranscribeStream() = default;
TranscribeStream::TranscribeStream(TranscribeStream&&) noexcept = default;
TranscribeStream& TranscribeStream::operator=(TranscribeStream&&) noexcept = default;

bool TranscribeStream::Write(const llmcore::v1::AudioIn& frame) {
  if (impl_->finished) return false;
  return impl_->stream->Write(frame);
}

void TranscribeStream::WritesDone() {
  if (!impl_->finished) impl_->stream->WritesDone();
}

bool TranscribeStream::Read(llmcore::v1::TranscriptionStreamEvent* out) {
  if (impl_->finished) return false;
  if (impl_->stream->Read(out)) return true;
  grpc::Status status = impl_->stream->Finish();
  impl_->finished = true;
  if (!status.ok()) ThrowFromStatus(status, impl_->ctx);
  return false;
}

void TranscribeStream::Cancel() { impl_->ctx.TryCancel(); }

struct SynthesizeStream::Impl {
  grpc::ClientContext ctx;
  std::unique_ptr<grpc::ClientReaderWriter<llmcore::v1::SynthControl,
                                           llmcore::v1::AudioOut>>
      stream;
  bool finished = false;
};

SynthesizeStream::SynthesizeStream() : impl_(std::make_unique<Impl>()) {}
SynthesizeStream::~SynthesizeStream() = default;
SynthesizeStream::SynthesizeStream(SynthesizeStream&&) noexcept = default;
SynthesizeStream& SynthesizeStream::operator=(SynthesizeStream&&) noexcept = default;

bool SynthesizeStream::Write(const llmcore::v1::SynthControl& frame) {
  if (impl_->finished) return false;
  return impl_->stream->Write(frame);
}

void SynthesizeStream::WritesDone() {
  if (!impl_->finished) impl_->stream->WritesDone();
}

bool SynthesizeStream::Read(llmcore::v1::AudioOut* out) {
  if (impl_->finished) return false;
  if (impl_->stream->Read(out)) return true;
  grpc::Status status = impl_->stream->Finish();
  impl_->finished = true;
  if (!status.ok()) ThrowFromStatus(status, impl_->ctx);
  return false;
}

void SynthesizeStream::Cancel() { impl_->ctx.TryCancel(); }

struct VoiceAgentStream::Impl {
  grpc::ClientContext ctx;
  std::unique_ptr<grpc::ClientReaderWriter<llmcore::v1::VoiceAgentClientEvent,
                                           llmcore::v1::VoiceAgentEvent>>
      stream;
  bool finished = false;
};

VoiceAgentStream::VoiceAgentStream() : impl_(std::make_unique<Impl>()) {}
VoiceAgentStream::~VoiceAgentStream() = default;
VoiceAgentStream::VoiceAgentStream(VoiceAgentStream&&) noexcept = default;
VoiceAgentStream& VoiceAgentStream::operator=(VoiceAgentStream&&) noexcept = default;

bool VoiceAgentStream::Write(const llmcore::v1::VoiceAgentClientEvent& event) {
  if (impl_->finished) return false;
  return impl_->stream->Write(event);
}

void VoiceAgentStream::WritesDone() {
  if (!impl_->finished) impl_->stream->WritesDone();
}

bool VoiceAgentStream::Read(llmcore::v1::VoiceAgentEvent* out) {
  if (impl_->finished) return false;
  if (impl_->stream->Read(out)) return true;
  grpc::Status status = impl_->stream->Finish();
  impl_->finished = true;
  if (!status.ok()) ThrowFromStatus(status, impl_->ctx);
  return false;
}

void VoiceAgentStream::Cancel() { impl_->ctx.TryCancel(); }

// ---------------- Client ----------------

struct Client::Impl {
  std::shared_ptr<grpc::Channel> channel;
  std::unique_ptr<llmcore::v1::InferenceService::Stub> inference;
  std::unique_ptr<llmcore::v1::CatalogService::Stub> catalog;
  std::unique_ptr<llmcore::v1::ControlService::Stub> control;
  std::unique_ptr<llmcore::v1::AudioService::Stub> audio;
};

Client::Client() : impl_(std::make_unique<Impl>()) {}
Client::~Client() = default;

std::unique_ptr<Client> Client::Create(const std::string& target) {
  auto c = std::unique_ptr<Client>(new Client());
  c->impl_->channel = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
  c->impl_->inference = llmcore::v1::InferenceService::NewStub(c->impl_->channel);
  c->impl_->catalog = llmcore::v1::CatalogService::NewStub(c->impl_->channel);
  c->impl_->control = llmcore::v1::ControlService::NewStub(c->impl_->channel);
  c->impl_->audio = llmcore::v1::AudioService::NewStub(c->impl_->channel);
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

// -- audio: one-shot --

llmcore::v1::SpeechResult Client::Synthesize(const llmcore::v1::SynthesizeRequest& req) {
  grpc::ClientContext ctx;
  llmcore::v1::SpeechResult resp;
  grpc::Status s = impl_->audio->Synthesize(&ctx, req, &resp);
  if (!s.ok()) ThrowFromStatus(s, ctx);
  return resp;
}

llmcore::v1::TranscriptionResult Client::Transcribe(const llmcore::v1::TranscribeRequest& req) {
  grpc::ClientContext ctx;
  llmcore::v1::TranscriptionResult resp;
  grpc::Status s = impl_->audio->Transcribe(&ctx, req, &resp);
  if (!s.ok()) ThrowFromStatus(s, ctx);
  return resp;
}

llmcore::v1::ImageGenerationResult Client::GenerateImage(
    const llmcore::v1::GenerateImageRequest& req) {
  grpc::ClientContext ctx;
  llmcore::v1::ImageGenerationResult resp;
  grpc::Status s = impl_->audio->GenerateImage(&ctx, req, &resp);
  if (!s.ok()) ThrowFromStatus(s, ctx);
  return resp;
}

llmcore::v1::OCRResult Client::Ocr(const llmcore::v1::OcrRequest& req) {
  grpc::ClientContext ctx;
  llmcore::v1::OCRResult resp;
  grpc::Status s = impl_->audio->Ocr(&ctx, req, &resp);
  if (!s.ok()) ThrowFromStatus(s, ctx);
  return resp;
}

llmcore::v1::TextAnalysisResult Client::AnalyzeText(const llmcore::v1::AnalyzeTextRequest& req) {
  grpc::ClientContext ctx;
  llmcore::v1::TextAnalysisResult resp;
  grpc::Status s = impl_->audio->AnalyzeText(&ctx, req, &resp);
  if (!s.ok()) ThrowFromStatus(s, ctx);
  return resp;
}

// -- audio: live duplex --

TranscribeStream Client::TranscribeStreamCall() {
  TranscribeStream st;  // private ctor; Client is a friend
  st.impl_->stream = impl_->audio->TranscribeStream(&st.impl_->ctx);
  return st;  // moved; pimpl keeps ctx/stream at a stable address
}

SynthesizeStream Client::SynthesizeStreamCall() {
  SynthesizeStream st;
  st.impl_->stream = impl_->audio->SynthesizeStream(&st.impl_->ctx);
  return st;
}

VoiceAgentStream Client::VoiceAgentCall() {
  VoiceAgentStream st;
  st.impl_->stream = impl_->audio->VoiceAgent(&st.impl_->ctx);
  return st;
}

}  // namespace llmcore
