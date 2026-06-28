//! End-to-end tests for the in-process binding against a **real** `LLMCore`.
//!
//! These embed CPython and drive `llmcore` directly. To stay offline and
//! deterministic they register `llmcore`'s shipped `FakeProvider` (an echo
//! provider) and build `LLMCore` with a config pointing at it — the same
//! approach as the bridge's real-LLMCore integration test.
//!
//! Run with the embedding interpreter selected + on the loader path:
//! ```text
//! PYO3_PYTHON=/path/python LD_LIBRARY_PATH=/path/python/lib \
//!   cargo test -p llmcore-embedded
//! ```

use futures::StreamExt;
use llmcore_embedded::{v1, Llmcore};
use pyo3::prelude::*;

/// Register the shipped offline echo provider into `PROVIDER_MAP["fake"]`.
fn register_fake_provider() {
    Python::with_gil(|py| {
        py.import_bound("llmcore.bridge._testing.fake_provider")
            .expect("import fake_provider")
            .call_method0("register_fake_provider")
            .expect("register_fake_provider()");
    });
}

/// Config that wires the `fake` provider with no vector store and a temp session
/// DB — fully offline.
fn overrides_json() -> String {
    let db = std::env::temp_dir()
        .join(format!("llmcore_embedded_{}.db", std::process::id()))
        .to_string_lossy()
        .into_owned();
    format!(
        r#"{{
            "llmcore": {{"default_provider": "fake"}},
            "providers": {{"fake": {{"type": "fake", "default_model": "fake-1"}}}},
            "storage": {{
                "vector": {{"type": ""}},
                "session": {{"type": "sqlite", "path": "{db}"}}
            }}
        }}"#
    )
}

async fn make_core() -> Llmcore {
    register_fake_provider();
    Llmcore::create_with_overrides(&overrides_json())
        .await
        .expect("LLMCore.create with fake provider")
}

#[tokio::test(flavor = "multi_thread")]
async fn info_and_catalog() {
    let core = make_core().await;

    let info = core.get_info();
    assert_eq!(info.contract_version, "llmcore.v1");
    assert_eq!(info.tiers, vec!["T0".to_string()]);
    assert!(info.capabilities.iter().any(|c| c == "tier0"));
    assert!(!info.llmcore_version.is_empty());

    let providers = core.list_providers().await.expect("list_providers").providers;
    assert!(providers.contains(&"fake".to_string()), "providers={providers:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_echoes() {
    let core = make_core().await;
    let res = core
        .chat(v1::ChatRequest { message: "hello world".into(), ..Default::default() })
        .await
        .expect("chat");
    // FakeProvider returns "echo: <last user text>".
    assert_eq!(res.text, "echo: hello world");
    let usage = res.usage.expect("usage present");
    assert!(usage.total_tokens.unwrap_or(0) > 0, "usage={usage:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_stream_reassembles() {
    let core = make_core().await;
    let mut stream = core
        .chat_stream(v1::ChatRequest { message: "hello world".into(), ..Default::default() })
        .await
        .expect("chat_stream");

    let mut text = String::new();
    let mut saw_done = false;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("chunk ok");
        if chunk.done {
            saw_done = true;
            break;
        }
        text.push_str(&chunk.text);
    }
    assert!(saw_done, "stream must end with a done=true chunk");
    assert_eq!(text, "echo: hello world");
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_with_tools_does_not_break() {
    let core = make_core().await;
    // The fake provider ignores tools but must accept them — proves tools/
    // tool_choice marshaling reaches llmcore without error.
    use llmcore_embedded::proto::prost_types;
    let mut params = prost_types::Struct::default();
    params.fields.insert(
        "type".into(),
        prost_types::Value { kind: Some(prost_types::value::Kind::StringValue("object".into())) },
    );
    let res = core
        .chat(v1::ChatRequest {
            message: "hi".into(),
            tools: vec![v1::Tool {
                name: "get_weather".into(),
                description: "Look up weather".into(),
                parameters: Some(params),
            }],
            tool_choice: Some("auto".into()),
            ..Default::default()
        })
        .await
        .expect("chat with tools");
    assert_eq!(res.text, "echo: hi");
}

#[tokio::test(flavor = "multi_thread")]
async fn sessions_crud() {
    let core = make_core().await;

    let session = core
        .create_session(v1::CreateSessionRequest {
            name: Some("demo".into()),
            system_message: Some("be terse".into()),
            ..Default::default()
        })
        .await
        .expect("create_session");
    assert!(!session.id.is_empty());
    assert_eq!(session.name.as_deref(), Some("demo"));

    let added = core
        .add_context_item(v1::AddContextItemRequest {
            session_id: session.id.clone(),
            content: "launch is June 30".into(),
            r#type: Some("user_text".into()),
            ..Default::default()
        })
        .await
        .expect("add_context_item");
    assert!(!added.item_id.is_empty());

    let item = core
        .get_context_item(v1::GetContextItemRequest {
            session_id: session.id.clone(),
            item_id: added.item_id.clone(),
        })
        .await
        .expect("get_context_item");
    assert_eq!(item.content, "launch is June 30");
    assert_eq!(item.r#type, "user_text");

    // Round-trip the session.
    let fetched = core
        .get_session(v1::GetSessionRequest { session_id: session.id.clone() })
        .await
        .expect("get_session");
    assert_eq!(fetched.id, session.id);
    assert_eq!(fetched.context_items.len(), 1);

    // NOT_FOUND for an unknown item.
    let missing = core
        .get_context_item(v1::GetContextItemRequest {
            session_id: session.id.clone(),
            item_id: "does-not-exist".into(),
        })
        .await;
    assert!(missing.is_err());
    assert_eq!(missing.unwrap_err().category, "ERROR_CATEGORY_NOT_FOUND");

    core.delete_session(v1::DeleteSessionRequest { session_id: session.id })
        .await
        .expect("delete_session");
}

#[tokio::test(flavor = "multi_thread")]
async fn count_tokens_and_cost() {
    let core = make_core().await;

    let n = core
        .count_tokens(v1::CountTokensRequest { text: "one two three".into(), ..Default::default() })
        .await
        .expect("count_tokens")
        .tokens;
    assert!(n > 0);

    // estimate_cost must not error even for an unknown (fake) model.
    let cost = core
        .estimate_cost(v1::EstimateCostRequest {
            provider_name: "fake".into(),
            model_name: "fake-1".into(),
            prompt_tokens: 10,
            completion_tokens: 20,
            ..Default::default()
        })
        .await
        .expect("estimate_cost");
    assert!(!cost.currency.is_empty() || !cost.pricing_source.is_empty());
}
