//! End-to-end tests against a spawned `llmcore-bridge` (FakeFacade).
//!
//! Set `LLMCORE_BRIDGE_PYTHON` to a python with `llmcore[bridge]` importable
//! (default `python3`). Unix-only harness (kills via the OS).

use std::net::TcpListener;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use llmcore_client::{v1, LlmcoreClient};

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// A spawned bridge process; killed on drop.
struct Bridge {
    child: Child,
    endpoint: String,
}

impl Bridge {
    async fn start() -> Bridge {
        let grpc_port = free_port();
        let http_port = free_port();
        let grpc_addr = format!("127.0.0.1:{}", grpc_port);
        let http_addr = format!("127.0.0.1:{}", http_port);
        let py = std::env::var("LLMCORE_BRIDGE_PYTHON").unwrap_or_else(|_| "python3".to_string());

        let child = Command::new(py)
            .args([
                "-m",
                "llmcore.bridge.cli",
                "serve",
                "--transport",
                "grpc,http",
                "--grpc-address",
                &grpc_addr,
                "--http-address",
                &http_addr,
                "--insecure",
                "--log-level",
                "WARNING",
            ])
            .env("LLMCORE_BRIDGE_FAKE", "1")
            .spawn()
            .expect("spawn bridge");

        let endpoint = format!("http://{}", grpc_addr);
        let deadline = Instant::now() + Duration::from_secs(25);
        loop {
            assert!(Instant::now() < deadline, "bridge not ready");
            if let Ok(mut c) = LlmcoreClient::connect(endpoint.clone()).await {
                if let Ok(h) = c.health().await {
                    if h.ok {
                        break;
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        Bridge { child, endpoint }
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

async fn connected() -> (LlmcoreClient, Bridge) {
    let b = Bridge::start().await;
    let c = LlmcoreClient::connect(b.endpoint.clone()).await.unwrap();
    (c, b)
}

#[tokio::test]
async fn ensure_compatible() {
    let (mut c, _b) = connected().await;
    let info = c.ensure_compatible(&["tier0"]).await.unwrap();
    assert_eq!(info.contract_version, "llmcore.v1");

    let err = c.ensure_compatible(&["tier2.audio"]).await.unwrap_err();
    assert_eq!(err.code, "capability.missing");
}

#[tokio::test]
async fn chat_and_usage() {
    let (mut c, _b) = connected().await;
    let res = c
        .chat(v1::ChatRequest {
            message: "hello world".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(res.text, "echo: hello world");
    assert_eq!(res.usage.unwrap().prompt_tokens, Some(2));
}

#[tokio::test]
async fn chat_stream_concat() {
    let (mut c, _b) = connected().await;
    let mut s = c
        .chat_stream(v1::ChatRequest {
            message: "stream this please".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    let mut parts = String::new();
    let mut done = false;
    while let Some(chunk) = s.message().await.unwrap() {
        if chunk.done {
            done = true;
        } else {
            parts.push_str(&chunk.text);
        }
    }
    assert!(done);
    assert_eq!(parts, "echo: stream this please");
}

#[tokio::test]
async fn count_and_cost() {
    let (mut c, _b) = connected().await;
    let ct = c
        .count_tokens(v1::CountTokensRequest {
            text: "one two three four".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(ct.tokens, 4);

    let ce = c
        .estimate_cost(v1::EstimateCostRequest {
            provider_name: "fake".into(),
            model_name: "fake-1".into(),
            prompt_tokens: 1_000_000,
            completion_tokens: 1_000_000,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(ce.currency, "USD");
    assert_eq!(ce.total_cost, 3.0);
}

#[tokio::test]
async fn catalog() {
    let (mut c, _b) = connected().await;
    assert_eq!(c.list_providers().await.unwrap().providers, vec!["fake"]);
    assert_eq!(
        c.list_models("fake").await.unwrap().models,
        vec!["fake-1", "fake-2"]
    );
    let d = c.get_provider_details(Some("fake".into())).await.unwrap();
    assert_eq!(d.id, "fake-1");
    assert_eq!(d.context_length, 8192);
}

#[tokio::test]
async fn embed_unsupported() {
    let (mut c, _b) = connected().await;
    let err = c
        .embed(v1::EmbedRequest {
            input: vec!["x".into()],
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(err.category, "ERROR_CATEGORY_UNSUPPORTED");
}

#[tokio::test]
async fn provider_rate_limit_decode() {
    let (mut c, _b) = connected().await;
    let err = c
        .chat(v1::ChatRequest {
            message: "__error__:provider_rate_limited".into(),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(err.category, "ERROR_CATEGORY_PROVIDER");
    assert_eq!(err.code, "provider.rate_limited");
    assert_eq!(err.http_status, Some(429));
    assert!(err.retryable);
    assert_eq!(err.retry_after_ms, Some(2000.0));
    assert_eq!(err.provider.as_deref(), Some("fake"));
}

#[tokio::test]
async fn stream_cancel() {
    let (mut c, _b) = connected().await;
    let mut s = c
        .chat_stream(v1::ChatRequest {
            message: "__cancel__".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    let first = s.message().await.unwrap();
    assert!(first.is_some());
    s.cancel();
    assert!(s.message().await.unwrap().is_none());
}
