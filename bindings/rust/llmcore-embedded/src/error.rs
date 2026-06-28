//! Error type for the in-process binding.
//!
//! Mirrors `llmcore-client`'s `BridgeError`: a Python exception raised by
//! `llmcore` is run through the bridge's own `to_llmcore_error`, which yields
//! the same structured `llmcore.v1.LlmcoreError` the gRPC path returns. We
//! serialize that proto in Python and decode it in Rust, so the category / code
//! / retry metadata are byte-identical across both transports.

use llmcore_proto::v1::{ErrorCategory, LlmcoreError};
use prost::Message;
use pyo3::prelude::*;

/// Normalized error surfaced by every `Llmcore` method.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{code}: {message}")]
pub struct EmbeddedError {
    /// `ErrorCategory` enum name, e.g. `"ERROR_CATEGORY_PROVIDER"`.
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
}

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, EmbeddedError>;

impl EmbeddedError {
    /// Sentinel for a Python `StopAsyncIteration` (a clean async-generator end),
    /// recognized by `code == "stop_async_iteration"`.
    pub(crate) fn stop_async() -> Self {
        EmbeddedError {
            category: "ERROR_CATEGORY_UNSPECIFIED".to_string(),
            code: "stop_async_iteration".to_string(),
            message: "StopAsyncIteration".to_string(),
            http_status: None,
            retryable: false,
            retry_after_ms: None,
            provider: None,
            model: None,
        }
    }

    /// A `NOT_FOUND` error (used when an in-process lookup returns `None`).
    pub(crate) fn not_found(message: impl Into<String>) -> Self {
        EmbeddedError {
            category: "ERROR_CATEGORY_NOT_FOUND".to_string(),
            code: "not_found".to_string(),
            message: message.into(),
            http_status: Some(404),
            retryable: false,
            retry_after_ms: None,
            provider: None,
            model: None,
        }
    }

    pub(crate) fn from_proto(pb: &LlmcoreError) -> Self {
        let category = ErrorCategory::try_from(pb.category)
            .map(|e| e.as_str_name().to_string())
            .unwrap_or_else(|_| "ERROR_CATEGORY_UNSPECIFIED".to_string());
        EmbeddedError {
            category,
            code: pb.code.clone(),
            message: pb.message.clone(),
            http_status: pb.http_status,
            retryable: pb.retryable,
            retry_after_ms: pb.retry_after_ms,
            provider: pb.provider.clone(),
            model: pb.model.clone(),
        }
    }

    /// Map a Python exception to a structured error via the bridge's mapper.
    pub(crate) fn from_pyerr(py: Python<'_>, err: &PyErr) -> Self {
        Self::try_from_pyerr(py, err).unwrap_or_else(|_| EmbeddedError {
            // Fallback if the mapper itself is unavailable / fails.
            category: "ERROR_CATEGORY_INTERNAL".to_string(),
            code: "internal".to_string(),
            message: err.to_string(),
            http_status: None,
            retryable: false,
            retry_after_ms: None,
            provider: None,
            model: None,
        })
    }

    fn try_from_pyerr(py: Python<'_>, err: &PyErr) -> PyResult<Self> {
        let exc = err.value_bound(py);
        let errors = py.import_bound("llmcore.bridge.errors")?;
        let pb_err = errors.call_method1("to_llmcore_error", (exc,))?;
        let bytes: Vec<u8> = pb_err.call_method0("SerializeToString")?.extract()?;
        let decoded = LlmcoreError::decode(bytes.as_slice())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self::from_proto(&decoded))
    }
}

impl From<PyErr> for EmbeddedError {
    fn from(err: PyErr) -> Self {
        Python::with_gil(|py| EmbeddedError::from_pyerr(py, &err))
    }
}
