//! Quickstart: drive the bridge over gRPC.
//!
//! ```text
//! LLMCORE_BRIDGE_FAKE=1 python -m llmcore.bridge.cli serve \
//!   --transport grpc --grpc-address 127.0.0.1:50151 --insecure
//! cargo run -p llmcore-client --example quickstart   # or set LLMCORE_GRPC
//! ```
use llmcore_client::{v1, LlmcoreClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let endpoint =
        std::env::var("LLMCORE_GRPC").unwrap_or_else(|_| "http://127.0.0.1:50151".to_string());
    let mut client = LlmcoreClient::connect(endpoint).await?;

    let info = client.ensure_compatible(&["tier0"]).await?;
    println!(
        "contract={} caps={:?}",
        info.contract_version, info.capabilities
    );

    let res = client
        .chat(v1::ChatRequest {
            message: "hello from rust".into(),
            ..Default::default()
        })
        .await?;
    println!("chat -> {:?} tokens={:?}", res.text, res.usage.and_then(|u| u.total_tokens));

    let mut stream = client
        .chat_stream(v1::ChatRequest {
            message: "stream me".into(),
            ..Default::default()
        })
        .await?;
    print!("stream -> ");
    while let Some(chunk) = stream.message().await? {
        if !chunk.done {
            print!("{}", chunk.text);
        }
    }
    println!();
    Ok(())
}
