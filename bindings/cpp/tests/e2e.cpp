// End-to-end tests for the C++ client against a spawned bridge (FakeFacade).
//
// Set LLMCORE_BRIDGE_PYTHON to a python with llmcore[bridge] importable
// (default python3). Unix-only (fork/exec/kill).
#define _POSIX_C_SOURCE 200809L
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "llmcore/client.hpp"

namespace {

int FreePort() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  ::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  socklen_t len = sizeof(addr);
  ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
  int port = ntohs(addr.sin_port);
  ::close(fd);
  return port;
}

std::string EnvOr(const char* key, const char* def) {
  const char* v = std::getenv(key);
  return v ? v : def;
}

struct Bridge {
  pid_t pid = -1;
  std::string target;
  ~Bridge() {
    if (pid > 0) {
      ::kill(pid, SIGTERM);
      int status = 0;
      ::waitpid(pid, &status, 0);
    }
  }
};

std::unique_ptr<Bridge> StartBridge(bool with_audio = false) {
  auto b = std::make_unique<Bridge>();
  const std::string grpc_addr = "127.0.0.1:" + std::to_string(FreePort());
  const std::string http_addr = "127.0.0.1:" + std::to_string(FreePort());
  b->target = grpc_addr;
  const std::string py = EnvOr("LLMCORE_BRIDGE_PYTHON", "python3");

  pid_t pid = ::fork();
  if (pid == 0) {
    ::setenv("LLMCORE_BRIDGE_FAKE", "1", 1);
    if (with_audio) ::setenv("LLMCORE_BRIDGE_FAKE_AUDIO", "1", 1);
    std::vector<std::string> args = {py,
                                     "-m",
                                     "llmcore.bridge.cli",
                                     "serve",
                                     "--transport",
                                     "grpc,http",
                                     "--grpc-address",
                                     grpc_addr,
                                     "--http-address",
                                     http_addr,
                                     "--insecure",
                                     "--log-level",
                                     "WARNING"};
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    ::execvp(py.c_str(), argv.data());
    ::_exit(127);
  }
  b->pid = pid;

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(25);
  while (std::chrono::steady_clock::now() < deadline) {
    try {
      auto c = llmcore::Client::Create(b->target);
      if (c->Health().ok()) return b;
    } catch (...) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  std::cerr << "bridge not ready\n";
  std::exit(2);
}

int g_failures = 0;
#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::cerr << "FAIL: " << #cond << " @ line " << __LINE__ << "\n";   \
      ++g_failures;                                                       \
    }                                                                     \
  } while (0)

}  // namespace

int main() {
  auto bridge = StartBridge();
  auto client = llmcore::Client::Create(bridge->target);

  // ensure_compatible (accept + reject)
  {
    auto info = client->EnsureCompatible({"tier0"});
    CHECK(info.contract_version() == "llmcore.v1");
    bool threw = false;
    try {
      client->EnsureCompatible({"tier2.audio"});
    } catch (const llmcore::BridgeError& e) {
      threw = true;
      CHECK(e.code == "capability.missing");
    }
    CHECK(threw);
  }

  // chat + usage
  {
    llmcore::v1::ChatRequest req;
    req.set_message("hello world");
    auto res = client->Chat(req);
    CHECK(res.text() == "echo: hello world");
    CHECK(res.usage().prompt_tokens() == 2);
  }

  // stream concatenation
  {
    llmcore::v1::ChatRequest req;
    req.set_message("stream this please");
    auto s = client->ChatStreamCall(req);
    std::string parts;
    bool done = false;
    llmcore::v1::ChatChunk ch;
    while (s.Read(&ch)) {
      if (ch.done()) done = true;
      else parts += ch.text();
    }
    CHECK(done);
    CHECK(parts == "echo: stream this please");
  }

  // count + cost
  {
    llmcore::v1::CountTokensRequest ct;
    ct.set_text("one two three four");
    CHECK(client->CountTokens(ct).tokens() == 4);

    llmcore::v1::EstimateCostRequest ec;
    ec.set_provider_name("fake");
    ec.set_model_name("fake-1");
    ec.set_prompt_tokens(1000000);
    ec.set_completion_tokens(1000000);
    auto ce = client->EstimateCost(ec);
    CHECK(ce.currency() == "USD");
    CHECK(ce.total_cost() == 3.0);
  }

  // catalog
  {
    auto lp = client->ListProviders();
    CHECK(lp.providers_size() == 1 && lp.providers(0) == "fake");
    auto lm = client->ListModels("fake");
    CHECK(lm.models_size() == 2 && lm.models(0) == "fake-1" && lm.models(1) == "fake-2");
    auto d = client->GetProviderDetails("fake");
    CHECK(d.id() == "fake-1");
    CHECK(d.context_length() == 8192);
  }

  // embed unsupported
  {
    bool threw = false;
    llmcore::v1::EmbedRequest er;
    er.add_input("x");
    try {
      client->Embed(er);
    } catch (const llmcore::BridgeError& e) {
      threw = true;
      CHECK(e.category == "ERROR_CATEGORY_UNSUPPORTED");
    }
    CHECK(threw);
  }

  // provider rate-limit structured decode
  {
    bool threw = false;
    llmcore::v1::ChatRequest req;
    req.set_message("__error__:provider_rate_limited");
    try {
      client->Chat(req);
    } catch (const llmcore::BridgeError& e) {
      threw = true;
      CHECK(e.category == "ERROR_CATEGORY_PROVIDER");
      CHECK(e.code == "provider.rate_limited");
      CHECK(e.http_status.has_value() && *e.http_status == 429);
      CHECK(e.retryable);
      CHECK(e.retry_after_ms.has_value() && *e.retry_after_ms == 2000.0);
      CHECK(e.provider.has_value() && *e.provider == "fake");
    }
    CHECK(threw);
  }

  // stream cancellation (subsequent read returns false or throws CANCELLED)
  {
    llmcore::v1::ChatRequest req;
    req.set_message("__cancel__");
    auto s = client->ChatStreamCall(req);
    llmcore::v1::ChatChunk ch;
    CHECK(s.Read(&ch));
    s.Cancel();
    try {
      while (s.Read(&ch)) {
      }
    } catch (const llmcore::BridgeError&) {
    }
  }

  // ---- audio (Tier 2): a second, audio-enabled bridge ----
  {
    auto abridge = StartBridge(/*with_audio=*/true);
    auto ac = llmcore::Client::Create(abridge->target);

    // capabilities advertised
    {
      auto info = ac->GetInfo();
      const char* caps[] = {"tier2.audio",         "audio.transcribe_stream",
                            "audio.synthesize_stream", "audio.voice_agent",
                            "audio.synthesize",    "audio.transcribe",
                            "audio.generate_image", "audio.ocr",
                            "audio.analyze_text"};
      for (const char* want : caps) {
        bool found = false;
        for (const auto& have : info.capabilities()) {
          if (have == want) {
            found = true;
            break;
          }
        }
        CHECK(found);
      }
    }

    // transcribe stream: INTERIM, INTERIM, FINAL "hello world", UTTERANCE_END
    {
      auto s = ac->TranscribeStreamCall();
      llmcore::v1::AudioIn f1;
      f1.set_audio("hello");
      llmcore::v1::AudioIn f2;
      f2.set_audio("world");
      llmcore::v1::AudioIn f3;
      f3.set_control(llmcore::v1::STT_CONTROL_CLOSE);
      CHECK(s.Write(f1));
      CHECK(s.Write(f2));
      CHECK(s.Write(f3));
      s.WritesDone();
      std::vector<llmcore::v1::StreamEventType> types;
      std::string final_text;
      llmcore::v1::TranscriptionStreamEvent ev;
      while (s.Read(&ev)) {
        types.push_back(ev.type());
        if (ev.type() == llmcore::v1::STREAM_EVENT_TYPE_FINAL) final_text = ev.text();
      }
      CHECK(types.size() == 4);
      if (types.size() == 4) {
        CHECK(types[0] == llmcore::v1::STREAM_EVENT_TYPE_INTERIM);
        CHECK(types[1] == llmcore::v1::STREAM_EVENT_TYPE_INTERIM);
        CHECK(types[2] == llmcore::v1::STREAM_EVENT_TYPE_FINAL);
        CHECK(types[3] == llmcore::v1::STREAM_EVENT_TYPE_UTTERANCE_END);
      }
      CHECK(final_text == "hello world");
    }

    // synthesize stream: chunks foo/bar/baz with seq 0/1/2
    {
      auto s = ac->SynthesizeStreamCall();
      const char* pieces[] = {"foo", "bar", "baz"};
      for (const char* p : pieces) {
        llmcore::v1::SynthControl f;
        f.set_text(p);
        CHECK(s.Write(f));
      }
      llmcore::v1::SynthControl fc;
      fc.set_control(llmcore::v1::TTS_CONTROL_CLOSE);
      CHECK(s.Write(fc));
      s.WritesDone();
      std::vector<std::string> chunks;
      std::vector<long long> seqs;
      llmcore::v1::AudioOut out;
      while (s.Read(&out)) {
        chunks.push_back(out.audio());
        seqs.push_back(static_cast<long long>(out.seq()));
      }
      CHECK(chunks.size() == 3);
      if (chunks.size() == 3) {
        CHECK(chunks[0] == "foo" && chunks[1] == "bar" && chunks[2] == "baz");
        CHECK(seqs[0] == 0 && seqs[1] == 1 && seqs[2] == 2);
      }
    }

    // voice agent: non-settings leading frame -> default provider
    {
      auto s = ac->VoiceAgentCall();
      llmcore::v1::VoiceAgentClientEvent e1;
      e1.set_inject_user_message("hi there");
      llmcore::v1::VoiceAgentClientEvent e2;
      e2.set_audio(std::string("\x01\x02", 2));
      CHECK(s.Write(e1));
      CHECK(s.Write(e2));
      s.WritesDone();
      std::vector<llmcore::v1::VoiceAgentEvent> events;
      llmcore::v1::VoiceAgentEvent ev;
      while (s.Read(&ev)) events.push_back(ev);
      CHECK(!events.empty());
      if (!events.empty()) {
        CHECK(events.front().type() == llmcore::v1::VOICE_AGENT_EVENT_TYPE_WELCOME);
        CHECK(events.back().type() == llmcore::v1::VOICE_AGENT_EVENT_TYPE_CLOSE);
      }
      bool saw_conv = false, saw_audio = false;
      for (const auto& e : events) {
        if (e.type() == llmcore::v1::VOICE_AGENT_EVENT_TYPE_CONVERSATION_TEXT &&
            e.role() == "user" && e.content() == "hi there")
          saw_conv = true;
        if (e.type() == llmcore::v1::VOICE_AGENT_EVENT_TYPE_AUDIO &&
            e.audio() == std::string("agent:\x01\x02", 8))
          saw_audio = true;
      }
      CHECK(saw_conv);
      CHECK(saw_audio);
    }

    // unary: synthesize / transcribe / generate_image / ocr / analyze_text
    {
      llmcore::v1::SynthesizeRequest req;
      req.set_text("hello");
      auto r = ac->Synthesize(req);
      CHECK(r.audio_data() == "tts:hello");
      CHECK(r.model() == "fake-tts");
    }
    {
      llmcore::v1::TranscribeRequest req;
      req.set_audio_data("hello world");
      auto r = ac->Transcribe(req);
      CHECK(r.text() == "hello world");
      CHECK(r.language() == "en");
      CHECK(r.segments_size() == 1);
      if (r.segments_size() == 1) CHECK(r.segments(0).speaker() == "spk_0");
    }
    {
      llmcore::v1::GenerateImageRequest req;
      req.set_prompt("a cat");
      req.set_n(2);
      auto r = ac->GenerateImage(req);
      CHECK(r.images_size() == 2);
      if (r.images_size() >= 1) CHECK(r.images(0).data() == "aW1nOmEgY2F0");  // base64("img:a cat")
    }
    {
      llmcore::v1::OcrRequest req;
      req.set_data("PDFBYTES");
      auto r = ac->Ocr(req);
      CHECK(r.model() == "fake-ocr");
      CHECK(r.pages_processed() == 1);
      CHECK(r.doc_size_bytes() == 8);
      CHECK(r.pages_size() == 1);
    }
    {
      llmcore::v1::AnalyzeTextRequest req;
      req.set_text("some text");
      auto r = ac->AnalyzeText(req);  // no features
      CHECK(r.model() == "fake-analyze");
      CHECK(!r.has_summary());
      CHECK(r.topics_size() == 0);
    }
  }

  if (g_failures == 0) {
    std::cout << "ALL TESTS PASSED\n";
    return 0;
  }
  std::cout << g_failures << " FAILURES\n";
  return 1;
}
