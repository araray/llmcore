//! Transport-neutral error decoded from `llmcore.v1.LlmcoreError`.

use llmcore_proto::v1::{ErrorCategory, LlmcoreError};
use prost::Message;
use tonic::{Code, Status};

/// Binary trailing-metadata key carrying the structured error.
pub const ERROR_METADATA_KEY: &str = "llmcore-error-bin";

/// Normalized error surfaced by every client method.
///
/// Mirrors the fields of [`llmcore_proto::v1::LlmcoreError`].
#[derive(Debug, Clone, thiserror::Error)]
#[error("{code}: {message}")]
pub struct BridgeError {
    /// ErrorCategory enum name, e.g. `"ERROR_CATEGORY_PROVIDER"`.
    pub category: String,
    /// Stable dotted token, e.g. `"provider.rate_limited"`.
    pub code: String,
    /// Secret-redacted human-readable detail.
    pub message: String,
    /// Upstream HTTP status, if any.
    pub http_status: Option<i32>,
    /// Whether the caller may retry.
    pub retryable: bool,
    /// Suggested backoff in milliseconds, if any.
    pub retry_after_ms: Option<f64>,
    /// Upstream provider / model when applicable.
    pub provider: Option<String>,
    pub model: Option<String>,
    /// gRPC status code (as i32), when derived from a gRPC error.
    pub grpc_code: Option<i32>,
}

impl BridgeError {
    pub(crate) fn from_proto(pb: &LlmcoreError, grpc_code: Option<i32>) -> Self {
        let category = ErrorCategory::try_from(pb.category)
            .map(|e| e.as_str_name().to_string())
            .unwrap_or_else(|_| "ERROR_CATEGORY_UNSPECIFIED".to_string());
        BridgeError {
            category,
            code: pb.code.clone(),
            message: pb.message.clone(),
            http_status: pb.http_status,
            retryable: pb.retryable,
            retry_after_ms: pb.retry_after_ms,
            provider: pb.provider.clone(),
            model: pb.model.clone(),
            grpc_code,
        }
    }

    /// Decode a tonic [`Status`] into a `BridgeError`, preferring the structured
    /// payload from the `llmcore-error-bin` trailing metadata, then falling back
    /// to the gRPC status itself.
    pub(crate) fn from_status(status: Status) -> Self {
        if let Some(val) = status.metadata().get_bin(ERROR_METADATA_KEY) {
            if let Ok(bytes) = val.to_bytes() {
                if let Ok(pb) = LlmcoreError::decode(bytes) {
                    return Self::from_proto(&pb, Some(status.code() as i32));
                }
            }
        }
        BridgeError {
            category: "ERROR_CATEGORY_INTERNAL".to_string(),
            code: format!("grpc.{}", code_name(status.code())),
            message: status.message().to_string(),
            http_status: None,
            retryable: false,
            retry_after_ms: None,
            provider: None,
            model: None,
            grpc_code: Some(status.code() as i32),
        }
    }

    /// Construct a client-side error (no network round trip).
    pub(crate) fn local(category: &str, code: &str, message: String) -> Self {
        BridgeError {
            category: category.to_string(),
            code: code.to_string(),
            message,
            http_status: None,
            retryable: false,
            retry_after_ms: None,
            provider: None,
            model: None,
            grpc_code: None,
        }
    }
}

fn code_name(c: Code) -> String {
    format!("{:?}", c)
}
