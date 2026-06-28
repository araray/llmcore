//! Tier-1 demo: sessions, context items, vector store, and context presets.
//!
//! Start a bridge with the Tier-1 fake stores enabled, then run the example:
//!
//! ```text
//! LLMCORE_BRIDGE_FAKE=1 LLMCORE_BRIDGE_FAKE_SESSIONS=1 LLMCORE_BRIDGE_FAKE_VECTOR=1 \
//!   python -m llmcore.bridge.cli serve --transport grpc \
//!   --grpc-address 127.0.0.1:50151 --insecure
//! LLMCORE_GRPC=http://127.0.0.1:50151 cargo run -p llmcore-client --example sessions
//! ```

use std::collections::BTreeMap;

use llmcore_client::{v1, LlmcoreClient};
use prost_types::{value::Kind, Struct, Value};

fn sval(s: &str) -> Value {
    Value { kind: Some(Kind::StringValue(s.to_string())) }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let endpoint = std::env::var("LLMCORE_GRPC").unwrap_or_else(|_| "http://127.0.0.1:50151".into());
    let mut c = LlmcoreClient::connect(endpoint).await?;

    let info = c.get_info().await?;
    println!("contract={} tiers={:?}", info.contract_version, info.tiers);

    // ---- sessions + context items ----
    let session = c
        .create_session(v1::CreateSessionRequest {
            name: Some("demo-session".into()),
            system_message: Some("You are a terse assistant.".into()),
            ..Default::default()
        })
        .await?;
    println!("created session {} ({} message[s])", session.id, session.messages.len());

    let added = c
        .add_context_item(v1::AddContextItemRequest {
            session_id: session.id.clone(),
            content: "Remember: the launch date is June 30.".into(),
            r#type: Some("user_text".into()),
            ..Default::default()
        })
        .await?;
    let item = c
        .get_context_item(v1::GetContextItemRequest {
            session_id: session.id.clone(),
            item_id: added.item_id,
        })
        .await?;
    println!("context item [{}]: {:?} ({} tokens)", item.r#type, item.content, item.tokens());

    // ---- vector store ----
    let mut fields = BTreeMap::new();
    fields.insert("content".to_string(), sval("Paris is the capital of France."));
    c.add_documents(v1::AddDocumentsRequest {
        documents: vec![Struct { fields }],
        ..Default::default()
    })
    .await?;
    let hits = c
        .search_vector_store(v1::SearchVectorStoreRequest {
            query: "capital of France".into(),
            k: 3,
            ..Default::default()
        })
        .await?;
    for d in &hits.documents {
        println!("  hit (score={:.3}): {:?}", d.score(), d.content);
    }

    // ---- context presets ----
    c.save_context_preset(v1::SaveContextPresetRequest {
        preset: Some(v1::ContextPreset {
            name: "preamble".into(),
            description: Some("Standard preamble".into()),
            items: vec![v1::ContextPresetItem {
                r#type: "preset_text_content".into(),
                content: Some("Always cite sources.".into()),
                ..Default::default()
            }],
            ..Default::default()
        }),
    })
    .await?;
    let preset = c
        .get_context_preset(v1::GetContextPresetRequest { preset_name: "preamble".into() })
        .await?;
    println!("preset {:?} has {} item[s]", preset.name, preset.items.len());

    // ---- cleanup ----
    c.delete_session(v1::DeleteSessionRequest { session_id: session.id }).await?;
    println!("done.");
    Ok(())
}
