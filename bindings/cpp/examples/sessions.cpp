// Tier-1 demo: sessions, context items, vector store, and context presets.
//
//   LLMCORE_BRIDGE_FAKE=1 LLMCORE_BRIDGE_FAKE_SESSIONS=1 LLMCORE_BRIDGE_FAKE_VECTOR=1 \
//     python -m llmcore.bridge.cli serve --transport grpc \
//     --grpc-address 127.0.0.1:50151 --insecure
//   LLMCORE_GRPC=127.0.0.1:50151 ./sessions
#include <cstdlib>
#include <iostream>

#include "llmcore/client.hpp"

int main() {
  const char* env = std::getenv("LLMCORE_GRPC");
  const std::string target = env ? env : "127.0.0.1:50151";

  try {
    auto client = llmcore::Client::Create(target);

    auto info = client->GetInfo();
    std::cout << "contract=" << info.contract_version() << " tiers=";
    for (const auto& t : info.tiers()) std::cout << t << " ";
    std::cout << "\n";

    // ---- sessions + context items ----
    llmcore::v1::CreateSessionRequest creq;
    creq.set_name("demo-session");
    creq.set_system_message("You are a terse assistant.");
    auto session = client->CreateSession(creq);
    std::cout << "created session " << session.id() << " (" << session.messages_size()
              << " message[s])\n";

    llmcore::v1::AddContextItemRequest areq;
    areq.set_session_id(session.id());
    areq.set_content("Remember: the launch date is June 30.");
    areq.set_type("user_text");
    auto added = client->AddContextItem(areq);

    llmcore::v1::GetContextItemRequest greq;
    greq.set_session_id(session.id());
    greq.set_item_id(added.item_id());
    auto item = client->GetContextItem(greq);
    std::cout << "context item [" << item.type() << "]: \"" << item.content() << "\"\n";

    // ---- vector store ----
    llmcore::v1::AddDocumentsRequest dreq;
    (*dreq.add_documents()->mutable_fields())["content"].set_string_value(
        "Paris is the capital of France.");
    client->AddDocuments(dreq);

    llmcore::v1::SearchVectorStoreRequest sreq;
    sreq.set_query("capital of France");
    sreq.set_k(3);
    auto hits = client->SearchVectorStore(sreq);
    for (const auto& d : hits.documents())
      std::cout << "  hit (score=" << d.score() << "): \"" << d.content() << "\"\n";

    // ---- context presets ----
    llmcore::v1::SaveContextPresetRequest preq;
    auto* preset = preq.mutable_preset();
    preset->set_name("preamble");
    preset->set_description("Standard preamble");
    auto* pi = preset->add_items();
    pi->set_type("preset_text_content");
    pi->set_content("Always cite sources.");
    client->SaveContextPreset(preq);

    llmcore::v1::GetContextPresetRequest gpreq;
    gpreq.set_preset_name("preamble");
    auto got = client->GetContextPreset(gpreq);
    std::cout << "preset \"" << got.name() << "\" has " << got.items_size() << " item[s]\n";

    // ---- cleanup ----
    llmcore::v1::DeleteSessionRequest delreq;
    delreq.set_session_id(session.id());
    client->DeleteSession(delreq);
    std::cout << "done.\n";
  } catch (const llmcore::BridgeError& e) {
    std::cerr << "error: " << e.category << " " << e.code << " — " << e.what() << "\n";
    return 1;
  }
  return 0;
}
