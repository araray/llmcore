//! Generated `llmcore.v1` protobuf types and tonic gRPC clients.
//!
//! This crate is codegen-only: the Rust sources are produced at build time by
//! `build.rs` (tonic-build) from `bindings/proto`. Do not edit generated items.
#![allow(clippy::all)]
#![allow(rustdoc::all)]

/// All `llmcore.v1` messages, enums, and service clients
/// (`inference_service_client`, `catalog_service_client`,
/// `control_service_client`, and the Tier-2 `audio_service_client`).
pub mod v1 {
    tonic::include_proto!("llmcore.v1");
}

// Re-exports so downstream crates can name the prost runtime/WKT consistently.
pub use prost;
pub use prost_types;
