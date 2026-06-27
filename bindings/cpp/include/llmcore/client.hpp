// llmcore C++ client — public API.
//
// Talks to the llmcore bridge over gRPC (the bridge's primary transport),
// generated from the frozen llmcore.v1 contract. Depends only on the contract.
//
// Methods return generated message types by value and THROW llmcore::BridgeError
// on failure. The structured error is decoded from the gRPC binary trailing
// metadata "llmcore-error-bin".
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// Generated protobuf message types (produced into the build's generated/ dir).
#include "llmcore/v1/audio.pb.h"
#include "llmcore/v1/catalog.pb.h"
#include "llmcore/v1/common.pb.h"
#include "llmcore/v1/control.pb.h"
#include "llmcore/v1/errors.pb.h"
#include "llmcore/v1/inference.pb.h"

namespace llmcore {

/// Normalized, transport-neutral error. Mirrors llmcore.v1.LlmcoreError.
class BridgeError : public std::runtime_error {
 public:
  std::string category;  ///< e.g. "ERROR_CATEGORY_PROVIDER"
  std::string code;      ///< e.g. "provider.rate_limited"
  std::string message;
  std::optional<std::int32_t> http_status;
  bool retryable = false;
  std::optional<double> retry_after_ms;
  std::optional<std::string> provider;
  std::optional<std::string> model;
  int grpc_code = 0;  ///< gRPC status code (0 == OK / not from gRPC)

  explicit BridgeError(std::string code_, std::string message_)
      : std::runtime_error(code_ + ": " + message_),
        code(std::move(code_)),
        message(std::move(message_)) {}
};

/// A cancellable server stream of ChatChunk frames.
class ChatStream {
 public:
  ~ChatStream();
  ChatStream(ChatStream&&) noexcept;
  ChatStream& operator=(ChatStream&&) noexcept;
  ChatStream(const ChatStream&) = delete;
  ChatStream& operator=(const ChatStream&) = delete;

  /// Reads the next chunk into *out. Returns true if a chunk was read, false at
  /// the clean end of the stream. Throws BridgeError on a non-OK terminal status.
  bool Read(llmcore::v1::ChatChunk* out);

  /// Cancels the stream (maps to gRPC CANCELLED).
  void Cancel();

 private:
  friend class Client;
  ChatStream();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// A cancellable bidi STT stream: Write AudioIn frames, Read
/// TranscriptionStreamEvents. (Tier 2 — audio.)
class TranscribeStream {
 public:
  ~TranscribeStream();
  TranscribeStream(TranscribeStream&&) noexcept;
  TranscribeStream& operator=(TranscribeStream&&) noexcept;
  TranscribeStream(const TranscribeStream&) = delete;
  TranscribeStream& operator=(const TranscribeStream&) = delete;

  /// Sends one AudioIn frame (open / audio / control). Returns false if the
  /// stream is already half-closed or broken.
  bool Write(const llmcore::v1::AudioIn& frame);
  /// Half-closes the request stream (no more Writes will be sent).
  void WritesDone();
  /// Reads the next event into *out. Returns true if a frame was read, false at
  /// the clean end. Throws BridgeError on a non-OK terminal status.
  bool Read(llmcore::v1::TranscriptionStreamEvent* out);
  /// Cancels the stream (maps to gRPC CANCELLED).
  void Cancel();

 private:
  friend class Client;
  TranscribeStream();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// A cancellable bidi TTS stream: Write SynthControl frames, Read AudioOut
/// chunks. (Tier 2 — audio.)
class SynthesizeStream {
 public:
  ~SynthesizeStream();
  SynthesizeStream(SynthesizeStream&&) noexcept;
  SynthesizeStream& operator=(SynthesizeStream&&) noexcept;
  SynthesizeStream(const SynthesizeStream&) = delete;
  SynthesizeStream& operator=(const SynthesizeStream&) = delete;

  /// Sends one SynthControl frame (open / text / control). Returns false if the
  /// stream is already half-closed or broken.
  bool Write(const llmcore::v1::SynthControl& frame);
  /// Half-closes the request stream.
  void WritesDone();
  /// Reads the next AudioOut into *out. Returns true if read, false at the clean
  /// end. Throws BridgeError on a non-OK terminal status.
  bool Read(llmcore::v1::AudioOut* out);
  /// Cancels the stream (maps to gRPC CANCELLED).
  void Cancel();

 private:
  friend class Client;
  SynthesizeStream();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// A cancellable bidi voice-agent stream: Write VoiceAgentClientEvents, Read
/// VoiceAgentEvents. (Tier 2 — audio.)
class VoiceAgentStream {
 public:
  ~VoiceAgentStream();
  VoiceAgentStream(VoiceAgentStream&&) noexcept;
  VoiceAgentStream& operator=(VoiceAgentStream&&) noexcept;
  VoiceAgentStream(const VoiceAgentStream&) = delete;
  VoiceAgentStream& operator=(const VoiceAgentStream&) = delete;

  /// Sends one VoiceAgentClientEvent (settings / audio / inject / ...). Returns
  /// false if the stream is already half-closed or broken.
  bool Write(const llmcore::v1::VoiceAgentClientEvent& event);
  /// Half-closes the request stream.
  void WritesDone();
  /// Reads the next VoiceAgentEvent into *out. Returns true if read, false at the
  /// clean end. Throws BridgeError on a non-OK terminal status.
  bool Read(llmcore::v1::VoiceAgentEvent* out);
  /// Cancels the stream (maps to gRPC CANCELLED).
  void Cancel();

 private:
  friend class Client;
  VoiceAgentStream();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// gRPC client. Construct with Create(); methods throw BridgeError on failure.
class Client {
 public:
  ~Client();

  /// Connect to a bridge at `target` ("host:port"). Insecure by default; pass a
  /// custom grpc::ChannelCredentials (e.g. grpc::SslCredentials) for TLS/mTLS.
  /// `creds` is type-erased to avoid leaking grpc headers into this public API;
  /// see the overload in client.cpp that takes a real credentials object.
  static std::unique_ptr<Client> Create(const std::string& target);

  // control
  llmcore::v1::ServerInfo GetInfo();
  llmcore::v1::HealthStatus Health();
  llmcore::v1::ReloadConfigResponse ReloadConfig(std::optional<std::string> path = std::nullopt);
  /// Verifies contract_version == "llmcore.v1" and that each capability is present.
  llmcore::v1::ServerInfo EnsureCompatible(const std::vector<std::string>& required_capabilities);

  // inference
  llmcore::v1::ChatResponse Chat(const llmcore::v1::ChatRequest& req);
  ChatStream ChatStreamCall(const llmcore::v1::ChatRequest& req);
  llmcore::v1::CountTokensResponse CountTokens(const llmcore::v1::CountTokensRequest& req);
  llmcore::v1::CostEstimate EstimateCost(const llmcore::v1::EstimateCostRequest& req);
  /// UNIMPLEMENTED in llmcore.v1 — throws BridgeError (UNSUPPORTED).
  llmcore::v1::EmbedResponse Embed(const llmcore::v1::EmbedRequest& req);

  // catalog
  llmcore::v1::ListProvidersResponse ListProviders();
  llmcore::v1::ListModelsResponse ListModels(const std::string& provider_name);
  llmcore::v1::ModelDetails GetProviderDetails(const std::string& provider_name);

  // audio (Tier 2) — available when the bridge advertises "tier2.audio".
  // One-shot:
  llmcore::v1::SpeechResult Synthesize(const llmcore::v1::SynthesizeRequest& req);
  llmcore::v1::TranscriptionResult Transcribe(const llmcore::v1::TranscribeRequest& req);
  llmcore::v1::ImageGenerationResult GenerateImage(const llmcore::v1::GenerateImageRequest& req);
  llmcore::v1::OCRResult Ocr(const llmcore::v1::OcrRequest& req);
  llmcore::v1::TextAnalysisResult AnalyzeText(const llmcore::v1::AnalyzeTextRequest& req);
  // Live duplex (Write frames, WritesDone, then Read until false):
  TranscribeStream TranscribeStreamCall();
  SynthesizeStream SynthesizeStreamCall();
  VoiceAgentStream VoiceAgentCall();

 private:
  Client();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace llmcore
