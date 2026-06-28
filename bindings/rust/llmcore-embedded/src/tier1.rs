//! Tier-1 (sessions) surface for the in-process binding.
//!
//! These call `llmcore`'s public session API directly and marshal the returned
//! value objects through the bridge's own converters (`_chat_session_to_proto`,
//! `_context_item_to_proto`) for byte-identical parity with the gRPC path. The
//! vector-store and preset surfaces follow the identical pattern and are a
//! documented follow-up (they need a configured vector backend to exercise).

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{convert, v1, EmbeddedError, Llmcore, Result};

impl Llmcore {
    /// Create a session. Backed by `LLMCore.create_session`.
    pub async fn create_session(&self, req: v1::CreateSessionRequest) -> Result<v1::ChatSession> {
        let core = self.core.clone();
        let obj = self
            .rt
            .run_coro(move |py| {
                let kwargs = PyDict::new_bound(py);
                if let Some(v) = req.session_id.as_deref() {
                    if !v.is_empty() {
                        kwargs.set_item("session_id", v)?;
                    }
                }
                if let Some(v) = req.name.as_deref() {
                    kwargs.set_item("name", v)?;
                }
                if let Some(v) = req.system_message.as_deref() {
                    kwargs.set_item("system_message", v)?;
                }
                Ok(core.bind(py).call_method("create_session", (), Some(&kwargs))?.unbind())
            })
            .await?;
        Python::with_gil(|py| convert::via_converter(py, "_chat_session_to_proto", obj.bind(py)))
            .map_err(EmbeddedError::from)
    }

    /// Fetch a session by id. Backed by `LLMCore.get_session`.
    pub async fn get_session(&self, req: v1::GetSessionRequest) -> Result<v1::ChatSession> {
        let core = self.core.clone();
        let obj = self
            .rt
            .run_coro(move |py| {
                Ok(core.bind(py).call_method1("get_session", (req.session_id,))?.unbind())
            })
            .await?;
        Python::with_gil(|py| convert::via_converter(py, "_chat_session_to_proto", obj.bind(py)))
            .map_err(EmbeddedError::from)
    }

    /// Delete a session. Backed by `LLMCore.delete_session`.
    pub async fn delete_session(&self, req: v1::DeleteSessionRequest) -> Result<()> {
        let core = self.core.clone();
        self.rt
            .run_coro(move |py| {
                Ok(core.bind(py).call_method1("delete_session", (req.session_id,))?.unbind())
            })
            .await
            .map(|_| ())
    }

    /// Stage a context item. Backed by `LLMCore.add_context_item`; returns the
    /// new item id.
    pub async fn add_context_item(
        &self,
        req: v1::AddContextItemRequest,
    ) -> Result<v1::AddContextItemResponse> {
        let core = self.core.clone();
        let id_obj = self
            .rt
            .run_coro(move |py| {
                let kwargs = PyDict::new_bound(py);
                // proto `type` (string) -> llmcore.models.ContextItemType; default user_text.
                let type_str = req
                    .r#type
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .unwrap_or("user_text");
                let item_type = py
                    .import_bound("llmcore.models")?
                    .getattr("ContextItemType")?
                    .call1((type_str,))?;
                kwargs.set_item("item_type", item_type)?;
                if let Some(v) = req.source_id.as_deref() {
                    if !v.is_empty() {
                        kwargs.set_item("source_id", v)?;
                    }
                }
                if let Some(s) = req.metadata.as_ref() {
                    kwargs.set_item("metadata", convert::struct_to_pydict(py, s))?;
                }
                Ok(core
                    .bind(py)
                    .call_method("add_context_item", (req.session_id, req.content), Some(&kwargs))?
                    .unbind())
            })
            .await?;
        let item_id =
            Python::with_gil(|py| id_obj.bind(py).extract::<String>()).map_err(EmbeddedError::from)?;
        Ok(v1::AddContextItemResponse { item_id })
    }

    /// Fetch a staged context item. Backed by `LLMCore.get_context_item`
    /// (returns `NOT_FOUND` when the item does not exist).
    pub async fn get_context_item(
        &self,
        req: v1::GetContextItemRequest,
    ) -> Result<v1::ContextItem> {
        let core = self.core.clone();
        let (sid, iid) = (req.session_id.clone(), req.item_id.clone());
        let obj = self
            .rt
            .run_coro(move |py| {
                Ok(core
                    .bind(py)
                    .call_method1("get_context_item", (req.session_id, req.item_id))?
                    .unbind())
            })
            .await?;
        Python::with_gil(|py| {
            let bound = obj.bind(py);
            if bound.is_none() {
                return Err(EmbeddedError::not_found(format!(
                    "context item {iid} not found in session {sid}"
                )));
            }
            convert::via_converter(py, "_context_item_to_proto", bound).map_err(EmbeddedError::from)
        })
    }
}
