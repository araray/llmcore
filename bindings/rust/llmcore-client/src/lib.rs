//! Async Rust client for the **llmcore bridge** over gRPC, built on tonic and
//! generated from the frozen `llmcore.v1` contract. Depends only on the
//! contract, never on Python.
//!
//! ```no_run
//! use llmcore_client::{v1, LlmcoreClient};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let mut client = LlmcoreClient::connect("http://127.0.0.1:50151").await?;
//! client.ensure_compatible(&["tier0"]).await?;
//! let res = client.chat(v1::ChatRequest { message: "hello".into(), ..Default::default() }).await?;
//! println!("{}", res.text);
//!
//! let mut stream = client.chat_stream(v1::ChatRequest { message: "stream".into(), ..Default::default() }).await?;
//! while let Some(chunk) = stream.message().await? {
//!     if !chunk.done { print!("{}", chunk.text); }
//! }
//! # Ok(()) }
//! ```
//!
//! Surface (Tier 0): `chat`, `chat_stream` (cancellable), `count_tokens`,
//! `estimate_cost`, `list_providers`, `list_models`, `get_provider_details`,
//! `get_info`/`ensure_compatible`, `health`, `reload_config`. `embed` returns a
//! [`BridgeError`] (UNSUPPORTED) — Embed is UNIMPLEMENTED in `llmcore.v1`.

mod client;
mod error;

pub use client::{ChatStream, LlmcoreClient};
pub use error::{BridgeError, ERROR_METADATA_KEY};

/// The generated protobuf crate (messages, enums, tonic clients).
pub use llmcore_proto as proto;
/// Shortcut to the `llmcore.v1` generated module.
pub use llmcore_proto::v1;
