//! In-process Tier-0 demo: no sidecar, no gRPC — CPython is embedded and
//! `llmcore` is called directly.
//!
//! By default this constructs an `LLMCore` from ambient config (so it needs a
//! configured provider + key, e.g. `OPENAI_API_KEY`, or a local Ollama). For a
//! fully offline run, point it at a provider that needs no network.
//!
//! ```text
//! PYO3_PYTHON=/path/to/python \
//! LD_LIBRARY_PATH=/path/to/python/lib \
//!   cargo run -p llmcore-embedded --example embedded_chat
//! ```
use futures::StreamExt;
use llmcore_embedded::{v1, Llmcore};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let core = Llmcore::create().await?;

    let info = core.get_info();
    println!(
        "in-process llmcore {} (contract={}, tiers={:?})",
        info.llmcore_version, info.contract_version, info.tiers
    );
    println!("providers: {:?}", core.list_providers().await?.providers);

    // Non-streaming chat with usage.
    let res = core
        .chat(v1::ChatRequest { message: "Say hello in five words.".into(), ..Default::default() })
        .await?;
    println!("chat -> {:?}", res.text);
    if let Some(u) = res.usage {
        println!("usage -> total_tokens={:?}", u.total_tokens);
    }

    // Streaming chat.
    print!("stream -> ");
    let mut stream = core
        .chat_stream(v1::ChatRequest { message: "Count to five.".into(), ..Default::default() })
        .await?;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if chunk.done {
            break;
        }
        print!("{}", chunk.text);
    }
    println!("\ndone.");
    Ok(())
}
