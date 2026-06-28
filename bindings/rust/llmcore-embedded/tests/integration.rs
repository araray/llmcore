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
