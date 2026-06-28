//! In-process (PyO3) Rust binding for `llmcore` — the B5 "single-binary" track.
//!
//! This crate embeds CPython and calls the `llmcore` Python package **directly**,
//! instead of speaking gRPC to the out-of-process bridge ([`llmcore-client`]).
//! It reuses the `llmcore.v1` message types from [`llmcore_proto`], so its public
//! API mirrors the gRPC client and is a drop-in for the Tier-0 surface:
//!
//! ```no_run
//! use llmcore_embedded::{v1, Llmcore};
//! # async fn run() -> Result<(), llmcore_embedded::EmbeddedError> {
//! let core = Llmcore::create().await?;                 // embeds CPython + LLMCore.create()
//! let res = core.chat(v1::ChatRequest { message: "hello".into(), ..Default::default() }).await?;
//! println!("{}", res.text);
//! # Ok(()) }
//! ```
//!
//! Requirements: a shared libpython (`Py_ENABLE_SHARED`) and `llmcore`
//! importable by the embedded interpreter. Select the interpreter at build time
//! with `PYO3_PYTHON`. See `README.md`.
//!
//! [`llmcore-client`]: https://docs.rs/llmcore-client

mod convert;
mod error;
mod runtime;
mod tier1;

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

pub use error::{EmbeddedError, Result};
/// The `llmcore.v1` generated message types, shared with the gRPC client.
pub use llmcore_proto::v1;
pub use llmcore_proto as proto;

use runtime::PyRuntime;

/// Capabilities advertised by the in-process binding (Tier-0 + the embedded
/// "transport"). Mirrors the bridge's Tier-0 capability set.
const CAPABILITIES: &[&str] = &[
    "tier0",
    "inference.chat",
    "inference.chat_stream",
    "inference.count_tokens",
    "inference.estimate_cost",
    "catalog.providers",
    "catalog.models",
    "control.info",
    "transport.embedded",
];

/// An in-process `llmcore` instance.
///
/// Cheap to clone (the handle is reference-counted); all methods take `&self`.
#[derive(Clone)]
pub struct Llmcore {
    rt: Arc<PyRuntime>,
    /// The Python `LLMCore` instance.
    core: Arc<Py<PyAny>>,
}

/// A stream of response chunks from [`Llmcore::chat_stream`].
pub type ChatStream = ReceiverStream<Result<v1::ChatChunk>>;

impl Llmcore {
    /// Embed CPython and construct an `LLMCore` with ambient configuration
    /// (env / config file, `env_prefix = "LLMCORE"`).
    pub async fn create() -> Result<Self> {
        Self::build(None).await
    }

    /// Like [`create`](Self::create) but applies `config_overrides`, supplied as
    /// a JSON object string (parsed by Python's `json.loads` into the dict that
    /// `LLMCore.create(config_overrides=...)` expects).
    pub async fn create_with_overrides(overrides_json: &str) -> Result<Self> {
        Self::build(Some(overrides_json.to_string())).await
    }

    async fn build(overrides_json: Option<String>) -> Result<Self> {
        let rt = Arc::new(PyRuntime::new()?);
        let core = rt
            .run_coro(move |py| {
                let llmcore = py.import_bound("llmcore")?;
                let cls = llmcore.getattr("LLMCore")?;
                let kwargs = PyDict::new_bound(py);
                if let Some(json) = overrides_json {
                    let overrides = py
                        .import_bound("json")?
                        .call_method1("loads", (json,))?;
                    kwargs.set_item("config_overrides", overrides)?;
                }
                // LLMCore.create(...) is a coroutine returning the instance.
                Ok(cls.call_method("create", (), Some(&kwargs))?.unbind())
            })
            .await?;
        Ok(Self { rt, core: Arc::new(core) })
    }

    /// Non-streaming chat. Backed by `LLMCore.chat_with_usage`, so the returned
    /// [`v1::ChatResponse`] carries per-call token usage (RAG disabled, matching
    /// the contract). `tools` / `tool_choice` are not yet marshaled.
    pub async fn chat(&self, req: v1::ChatRequest) -> Result<v1::ChatResponse> {
        let core = self.core.clone();
        let result = self
            .rt
            .run_coro(move |py| {
                let kwargs = chat_kwargs(py, &req, /* stream */ false)?;
                Ok(core
                    .bind(py)
                    .call_method("chat_with_usage", (req.message.clone(),), Some(&kwargs))?
                    .unbind())
            })
            .await?;

        Python::with_gil(|py| {
            let tup = result.bind(py);
            let text: String = tup.get_item(0)?.extract()?;
            let usage = convert::usage_to_proto(&tup.get_item(1)?);
            Ok::<_, PyErr>(v1::ChatResponse {
                text,
                usage: Some(usage),
                ..Default::default()
            })
        })
        .map_err(EmbeddedError::from)
    }

    /// Streaming chat. Backed by `LLMCore.chat(..., stream=True)`; each yielded
    /// piece becomes a [`v1::ChatChunk`], terminated by a `done = true` chunk —
    /// the same framing as the gRPC `ChatStream`.
    pub async fn chat_stream(&self, req: v1::ChatRequest) -> Result<ChatStream> {
        // 1. await chat(stream=True) -> async generator object.
        let core = self.core.clone();
        let msg = req.message.clone();
        let gen = self
            .rt
            .run_coro(move |py| {
                let kwargs = chat_kwargs(py, &req, /* stream */ true)?;
                Ok(core.bind(py).call_method("chat", (msg,), Some(&kwargs))?.unbind())
            })
            .await?;

        // 2. Pump the async generator's __anext__ into a channel.
        let (tx, rx) = mpsc::channel::<Result<v1::ChatChunk>>(16);
        let rt = self.rt.clone();
        let gen = Arc::new(gen);
        tokio::spawn(async move {
            loop {
                let gen = gen.clone();
                let step = rt
                    .run_coro(move |py| {
                        Ok(gen.bind(py).call_method0("__anext__")?.unbind())
                    })
                    .await;
                match step {
                    Ok(piece) => {
                        let text = Python::with_gil(|py| {
                            piece.bind(py).extract::<String>().unwrap_or_default()
                        });
                        if tx.send(Ok(v1::ChatChunk { text, done: false })).await.is_err() {
                            return; // receiver dropped
                        }
                    }
                    Err(e) => {
                        // StopAsyncIteration is the normal terminator.
                        if e.code == "stop_async_iteration" {
                            let _ = tx.send(Ok(v1::ChatChunk { text: String::new(), done: true })).await;
                        } else {
                            let _ = tx.send(Err(e)).await;
                        }
                        return;
                    }
                }
            }
        });
        Ok(ReceiverStream::new(rx))
    }

    /// Count tokens in `text` (synchronous in `llmcore`; exposed `async` for
    /// API parity with the gRPC client).
    pub async fn count_tokens(&self, req: v1::CountTokensRequest) -> Result<v1::CountTokensResponse> {
        let core = self.core.clone();
        Python::with_gil(|py| {
            let llmcore = py.import_bound("llmcore")?;
            let kwargs = PyDict::new_bound(py);
            if let Some(m) = req.model_name.as_deref() {
                kwargs.set_item("model", m)?;
            }
            let _ = &core; // count_tokens is a module function, not on the instance
            let n: i32 = llmcore
                .call_method("count_tokens", (req.text,), Some(&kwargs))?
                .extract()?;
            Ok::<_, PyErr>(v1::CountTokensResponse { tokens: n })
        })
        .map_err(EmbeddedError::from)
    }

    /// Estimate cost for a token split (synchronous; `async` for parity).
    pub async fn estimate_cost(&self, req: v1::EstimateCostRequest) -> Result<v1::CostEstimate> {
        let core = self.core.clone();
        Python::with_gil(|py| {
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("cached_tokens", req.cached_tokens)?;
            kwargs.set_item("reasoning_tokens", req.reasoning_tokens)?;
            let ce = core.bind(py).call_method(
                "estimate_cost",
                (req.provider_name, req.model_name, req.prompt_tokens, req.completion_tokens),
                Some(&kwargs),
            )?;
            Ok::<_, PyErr>(convert::cost_to_proto(&ce))
        })
        .map_err(EmbeddedError::from)
    }

    /// List configured providers (synchronous; `async` for parity).
    pub async fn list_providers(&self) -> Result<v1::ListProvidersResponse> {
        let core = self.core.clone();
        Python::with_gil(|py| {
            let providers: Vec<String> =
                core.bind(py).call_method0("get_available_providers")?.extract()?;
            Ok::<_, PyErr>(v1::ListProvidersResponse { providers })
        })
        .map_err(EmbeddedError::from)
    }

    /// List models for a provider (synchronous; `async` for parity).
    pub async fn list_models(&self, req: v1::ListModelsRequest) -> Result<v1::ListModelsResponse> {
        let core = self.core.clone();
        Python::with_gil(|py| {
            let models: Vec<String> = core
                .bind(py)
                .call_method1("get_models_for_provider", (req.provider_name,))?
                .extract()?;
            Ok::<_, PyErr>(v1::ListModelsResponse { models })
        })
        .map_err(EmbeddedError::from)
    }

    /// Provider/model details (synchronous; `async` for parity).
    pub async fn get_provider_details(&self, req: v1::GetProviderRequest) -> Result<v1::ModelDetails> {
        let core = self.core.clone();
        Python::with_gil(|py| {
            let args = PyList::empty_bound(py);
            if let Some(p) = req.provider_name.as_deref() {
                args.append(p)?;
            }
            let md = core
                .bind(py)
                .call_method1("get_provider_details", args.to_tuple())?;
            Ok::<_, PyErr>(convert::model_details_to_proto(&md))
        })
        .map_err(EmbeddedError::from)
    }

    /// Reload `llmcore` configuration (async in `llmcore`).
    pub async fn reload_config(&self) -> Result<()> {
        let core = self.core.clone();
        self.rt
            .run_coro(move |py| Ok(core.bind(py).call_method0("reload_config")?.unbind()))
            .await
            .map(|_| ())
    }

    /// Assemble a [`v1::ServerInfo`] describing this in-process binding.
    pub fn get_info(&self) -> v1::ServerInfo {
        let llmcore_version = Python::with_gil(|py| {
            py.import_bound("llmcore")
                .ok()
                .and_then(|m| m.getattr("__version__").ok())
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_default()
        });
        v1::ServerInfo {
            llmcore_version,
            bridge_version: env!("CARGO_PKG_VERSION").to_string(),
            contract_version: "llmcore.v1".to_string(),
            transports: vec!["embedded".to_string()],
            providers: Vec::new(),
            capabilities: CAPABILITIES.iter().map(|s| s.to_string()).collect(),
            tiers: vec!["T0".to_string()],
        }
    }
}

/// Build the keyword arguments shared by `chat` / `chat_with_usage`.
fn chat_kwargs<'py>(
    py: Python<'py>,
    req: &v1::ChatRequest,
    stream: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let kwargs = PyDict::new_bound(py);
    macro_rules! set_opt {
        ($field:expr, $name:literal) => {
            if let Some(v) = $field.as_deref() {
                if !v.is_empty() {
                    kwargs.set_item($name, v)?;
                }
            }
        };
    }
    set_opt!(req.session_id, "session_id");
    set_opt!(req.system_message, "system_message");
    set_opt!(req.provider_name, "provider_name");
    set_opt!(req.model_name, "model_name");
    set_opt!(req.tool_choice, "tool_choice");
    kwargs.set_item("save_session", req.save_session)?;
    kwargs.set_item("enable_rag", false)?;
    if stream {
        kwargs.set_item("stream", true)?;
    }
    if !req.tools.is_empty() {
        // Mirror the bridge's `_tools`: build llmcore.models.Tool(name, description,
        // parameters=<JSON-schema dict>).
        let tool_cls = py.import_bound("llmcore.models")?.getattr("Tool")?;
        let tools = PyList::empty_bound(py);
        for t in &req.tools {
            let fields = PyDict::new_bound(py);
            fields.set_item("name", &t.name)?;
            fields.set_item("description", &t.description)?;
            let params = match t.parameters.as_ref() {
                Some(s) => convert::struct_to_pydict(py, s),
                None => PyDict::new_bound(py),
            };
            fields.set_item("parameters", params)?;
            tools.append(tool_cls.call((), Some(&fields))?)?;
        }
        kwargs.set_item("tools", tools)?;
    }
    if let Some(s) = req.provider_kwargs.as_ref() {
        let extra = convert::struct_to_pydict(py, s);
        for (k, v) in extra.iter() {
            kwargs.set_item(k, v)?;
        }
    }
    Ok(kwargs)
}
