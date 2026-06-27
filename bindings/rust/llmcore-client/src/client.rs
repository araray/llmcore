//! Ergonomic async client wrapping the tonic-generated service clients.

use llmcore_proto::v1 as pb;
use llmcore_proto::v1::{
    catalog_service_client::CatalogServiceClient,
    control_service_client::ControlServiceClient,
    inference_service_client::InferenceServiceClient,
};
use tonic::transport::{Channel, Endpoint};
use tonic::Request;

use crate::error::BridgeError;

/// A connected client over a single gRPC channel.
#[derive(Clone)]
pub struct LlmcoreClient {
    inference: InferenceServiceClient<Channel>,
    catalog: CatalogServiceClient<Channel>,
    control: ControlServiceClient<Channel>,
}

/// A cancellable server stream of [`pb::ChatChunk`] frames.
pub struct ChatStream {
    inner: Option<tonic::Streaming<pb::ChatChunk>>,
}

impl ChatStream {
    /// Returns the next chunk, `Ok(None)` at the clean end of the stream (or
    /// after [`cancel`](Self::cancel)), or a [`BridgeError`].
    pub async fn message(&mut self) -> Result<Option<pb::ChatChunk>, BridgeError> {
        match self.inner.as_mut() {
            Some(s) => s.message().await.map_err(BridgeError::from_status),
            None => Ok(None),
        }
    }

    /// Cancel the stream. Dropping the inner stream resets the HTTP/2 stream,
    /// which the server observes as CANCELLED.
    pub fn cancel(&mut self) {
        self.inner = None;
    }
}

impl LlmcoreClient {
    /// Connect to a bridge at `endpoint`, e.g. `"http://127.0.0.1:50151"`
    /// (use `https://...` with [`with_channel`](Self::with_channel) for TLS).
    pub async fn connect(endpoint: impl Into<String>) -> Result<Self, BridgeError> {
        let ep = Endpoint::from_shared(endpoint.into()).map_err(|e| {
            BridgeError::local(
                "ERROR_CATEGORY_INVALID_ARGUMENT",
                "endpoint.invalid",
                e.to_string(),
            )
        })?;
        let channel = ep.connect().await.map_err(|e| {
            BridgeError::local("ERROR_CATEGORY_INTERNAL", "transport.connect", e.to_string())
        })?;
        Ok(Self::with_channel(channel))
    }

    /// Build a client over a pre-configured tonic [`Channel`] (custom TLS,
    /// timeouts, interceptors, ...).
    pub fn with_channel(channel: Channel) -> Self {
        Self {
            inference: InferenceServiceClient::new(channel.clone()),
            catalog: CatalogServiceClient::new(channel.clone()),
            control: ControlServiceClient::new(channel),
        }
    }

    // ---- control ------------------------------------------------------- //

    pub async fn get_info(&mut self) -> Result<pb::ServerInfo, BridgeError> {
        self.control
            .get_info(Request::new(pb::Empty {}))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn health(&mut self) -> Result<pb::HealthStatus, BridgeError> {
        self.control
            .health(Request::new(pb::Empty {}))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn reload_config(
        &mut self,
        path: Option<String>,
    ) -> Result<pb::ReloadConfigResponse, BridgeError> {
        self.control
            .reload_config(Request::new(pb::ReloadConfigRequest { path }))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    /// Verify the contract version and that every required capability is present.
    pub async fn ensure_compatible(
        &mut self,
        required: &[&str],
    ) -> Result<pb::ServerInfo, BridgeError> {
        let info = self.get_info().await?;
        if info.contract_version != "llmcore.v1" {
            return Err(BridgeError::local(
                "ERROR_CATEGORY_INVALID_ARGUMENT",
                "contract.mismatch",
                format!("server contract {} != llmcore.v1", info.contract_version),
            ));
        }
        let missing: Vec<&str> = required
            .iter()
            .copied()
            .filter(|c| !info.capabilities.iter().any(|x| x == c))
            .collect();
        if !missing.is_empty() {
            return Err(BridgeError::local(
                "ERROR_CATEGORY_UNSUPPORTED",
                "capability.missing",
                format!("server lacks required capabilities: {:?}", missing),
            ));
        }
        Ok(info)
    }

    // ---- inference ----------------------------------------------------- //

    pub async fn chat(&mut self, req: pb::ChatRequest) -> Result<pb::ChatResponse, BridgeError> {
        self.inference
            .chat(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn chat_stream(&mut self, req: pb::ChatRequest) -> Result<ChatStream, BridgeError> {
        let streaming = self
            .inference
            .chat_stream(Request::new(req))
            .await
            .map_err(BridgeError::from_status)?
            .into_inner();
        Ok(ChatStream {
            inner: Some(streaming),
        })
    }

    pub async fn count_tokens(
        &mut self,
        req: pb::CountTokensRequest,
    ) -> Result<pb::CountTokensResponse, BridgeError> {
        self.inference
            .count_tokens(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn estimate_cost(
        &mut self,
        req: pb::EstimateCostRequest,
    ) -> Result<pb::CostEstimate, BridgeError> {
        self.inference
            .estimate_cost(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    /// UNIMPLEMENTED in llmcore.v1 — always returns a [`BridgeError`]
    /// (UNSUPPORTED). Provided for surface completeness.
    pub async fn embed(&mut self, req: pb::EmbedRequest) -> Result<pb::EmbedResponse, BridgeError> {
        self.inference
            .embed(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    // ---- catalog ------------------------------------------------------- //

    pub async fn list_providers(&mut self) -> Result<pb::ListProvidersResponse, BridgeError> {
        self.catalog
            .list_providers(Request::new(pb::Empty {}))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn list_models(
        &mut self,
        provider_name: impl Into<String>,
    ) -> Result<pb::ListModelsResponse, BridgeError> {
        self.catalog
            .list_models(Request::new(pb::ListModelsRequest {
                provider_name: provider_name.into(),
            }))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn get_provider_details(
        &mut self,
        provider_name: Option<String>,
    ) -> Result<pb::ModelDetails, BridgeError> {
        self.catalog
            .get_provider_details(Request::new(pb::GetProviderRequest { provider_name }))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }
}
