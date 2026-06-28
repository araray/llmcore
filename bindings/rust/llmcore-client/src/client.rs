//! Ergonomic async client wrapping the tonic-generated service clients.

use llmcore_proto::v1 as pb;
use llmcore_proto::v1::{
    audio_service_client::AudioServiceClient,
    catalog_service_client::CatalogServiceClient,
    control_service_client::ControlServiceClient,
    inference_service_client::InferenceServiceClient,
    preset_service_client::PresetServiceClient,
    session_service_client::SessionServiceClient,
    vector_service_client::VectorServiceClient,
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
    audio: AudioServiceClient<Channel>,
    session: SessionServiceClient<Channel>,
    vector: VectorServiceClient<Channel>,
    preset: PresetServiceClient<Channel>,
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

/// A response stream of audio events (`message()` yields the next item,
/// `Ok(None)` at the clean end of the stream or after [`cancel`](Self::cancel)).
/// Used by the live duplex audio RPCs; the request side is the
/// `impl Stream<Item = ...>` passed to the opening call.
pub struct AudioStream<T> {
    inner: Option<tonic::Streaming<T>>,
}

impl<T> AudioStream<T> {
    /// Returns the next event, `Ok(None)` at the clean end, or a [`BridgeError`].
    pub async fn message(&mut self) -> Result<Option<T>, BridgeError> {
        match self.inner.as_mut() {
            Some(s) => s.message().await.map_err(BridgeError::from_status),
            None => Ok(None),
        }
    }

    /// Cancel the stream (drops the inner stream; the server observes CANCELLED).
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
            control: ControlServiceClient::new(channel.clone()),
            audio: AudioServiceClient::new(channel.clone()),
            session: SessionServiceClient::new(channel.clone()),
            vector: VectorServiceClient::new(channel.clone()),
            preset: PresetServiceClient::new(channel),
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

    // ---- audio: one-shot (Tier 2) -------------------------------------- //

    pub async fn synthesize(
        &mut self,
        req: pb::SynthesizeRequest,
    ) -> Result<pb::SpeechResult, BridgeError> {
        self.audio
            .synthesize(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn transcribe(
        &mut self,
        req: pb::TranscribeRequest,
    ) -> Result<pb::TranscriptionResult, BridgeError> {
        self.audio
            .transcribe(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn generate_image(
        &mut self,
        req: pb::GenerateImageRequest,
    ) -> Result<pb::ImageGenerationResult, BridgeError> {
        self.audio
            .generate_image(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn ocr(&mut self, req: pb::OcrRequest) -> Result<pb::OcrResult, BridgeError> {
        self.audio
            .ocr(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn analyze_text(
        &mut self,
        req: pb::AnalyzeTextRequest,
    ) -> Result<pb::TextAnalysisResult, BridgeError> {
        self.audio
            .analyze_text(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    // ---- sessions (Tier 1) --------------------------------------------- //

    pub async fn create_session(
        &mut self,
        req: pb::CreateSessionRequest,
    ) -> Result<pb::ChatSession, BridgeError> {
        self.session
            .create_session(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn get_session(
        &mut self,
        req: pb::GetSessionRequest,
    ) -> Result<pb::ChatSession, BridgeError> {
        self.session
            .get_session(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn list_sessions(
        &mut self,
        req: pb::ListSessionsRequest,
    ) -> Result<pb::ListSessionsResponse, BridgeError> {
        self.session
            .list_sessions(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn delete_session(
        &mut self,
        req: pb::DeleteSessionRequest,
    ) -> Result<pb::Empty, BridgeError> {
        self.session
            .delete_session(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn update_session_name(
        &mut self,
        req: pb::UpdateSessionNameRequest,
    ) -> Result<pb::Empty, BridgeError> {
        self.session
            .update_session_name(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn fork_session(
        &mut self,
        req: pb::ForkSessionRequest,
    ) -> Result<pb::ForkSessionResponse, BridgeError> {
        self.session
            .fork_session(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn clone_session(
        &mut self,
        req: pb::CloneSessionRequest,
    ) -> Result<pb::CloneSessionResponse, BridgeError> {
        self.session
            .clone_session(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn delete_messages(
        &mut self,
        req: pb::DeleteMessagesRequest,
    ) -> Result<pb::DeleteMessagesResponse, BridgeError> {
        self.session
            .delete_messages(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn get_messages_by_range(
        &mut self,
        req: pb::GetMessagesByRangeRequest,
    ) -> Result<pb::GetMessagesByRangeResponse, BridgeError> {
        self.session
            .get_messages_by_range(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn add_context_item(
        &mut self,
        req: pb::AddContextItemRequest,
    ) -> Result<pb::AddContextItemResponse, BridgeError> {
        self.session
            .add_context_item(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn get_context_item(
        &mut self,
        req: pb::GetContextItemRequest,
    ) -> Result<pb::ContextItem, BridgeError> {
        self.session
            .get_context_item(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn remove_context_item(
        &mut self,
        req: pb::RemoveContextItemRequest,
    ) -> Result<pb::RemoveContextItemResponse, BridgeError> {
        self.session
            .remove_context_item(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    // ---- vector store & RAG (Tier 1) ----------------------------------- //

    pub async fn add_documents(
        &mut self,
        req: pb::AddDocumentsRequest,
    ) -> Result<pb::AddDocumentsResponse, BridgeError> {
        self.vector
            .add_documents(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn search_vector_store(
        &mut self,
        req: pb::SearchVectorStoreRequest,
    ) -> Result<pb::SearchVectorStoreResponse, BridgeError> {
        self.vector
            .search_vector_store(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn list_vector_collections(
        &mut self,
    ) -> Result<pb::ListCollectionsResponse, BridgeError> {
        self.vector
            .list_vector_collections(Request::new(pb::Empty {}))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn list_rag_collections(
        &mut self,
    ) -> Result<pb::ListCollectionsResponse, BridgeError> {
        self.vector
            .list_rag_collections(Request::new(pb::Empty {}))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn get_rag_collection_info(
        &mut self,
        req: pb::GetRagCollectionInfoRequest,
    ) -> Result<pb::RagCollectionInfo, BridgeError> {
        self.vector
            .get_rag_collection_info(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn delete_rag_collection(
        &mut self,
        req: pb::DeleteRagCollectionRequest,
    ) -> Result<pb::DeleteRagCollectionResponse, BridgeError> {
        self.vector
            .delete_rag_collection(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    // ---- context presets (Tier 1) -------------------------------------- //

    pub async fn save_context_preset(
        &mut self,
        req: pb::SaveContextPresetRequest,
    ) -> Result<pb::Empty, BridgeError> {
        self.preset
            .save_context_preset(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn get_context_preset(
        &mut self,
        req: pb::GetContextPresetRequest,
    ) -> Result<pb::ContextPreset, BridgeError> {
        self.preset
            .get_context_preset(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn list_context_presets(
        &mut self,
    ) -> Result<pb::ListContextPresetsResponse, BridgeError> {
        self.preset
            .list_context_presets(Request::new(pb::Empty {}))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    pub async fn delete_context_preset(
        &mut self,
        req: pb::DeleteContextPresetRequest,
    ) -> Result<pb::DeleteContextPresetResponse, BridgeError> {
        self.preset
            .delete_context_preset(Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(BridgeError::from_status)
    }

    // ---- audio: live duplex (Tier 2) ----------------------------------- //
    //
    // Each method takes the request side as an `impl Stream<Item = ...>` (any
    // tokio/futures stream; e.g. `tokio_stream::iter(frames)` or a channel-backed
    // stream for dynamic sending) and returns an [`AudioStream`] of responses.
    // The request stream ending half-closes the call.

    pub async fn transcribe_stream(
        &mut self,
        requests: impl tokio_stream::Stream<Item = pb::AudioIn> + Send + 'static,
    ) -> Result<AudioStream<pb::TranscriptionStreamEvent>, BridgeError> {
        let streaming = self
            .audio
            .transcribe_stream(requests)
            .await
            .map_err(BridgeError::from_status)?
            .into_inner();
        Ok(AudioStream {
            inner: Some(streaming),
        })
    }

    pub async fn synthesize_stream(
        &mut self,
        requests: impl tokio_stream::Stream<Item = pb::SynthControl> + Send + 'static,
    ) -> Result<AudioStream<pb::AudioOut>, BridgeError> {
        let streaming = self
            .audio
            .synthesize_stream(requests)
            .await
            .map_err(BridgeError::from_status)?
            .into_inner();
        Ok(AudioStream {
            inner: Some(streaming),
        })
    }

    pub async fn voice_agent(
        &mut self,
        requests: impl tokio_stream::Stream<Item = pb::VoiceAgentClientEvent> + Send + 'static,
    ) -> Result<AudioStream<pb::VoiceAgentEvent>, BridgeError> {
        let streaming = self
            .audio
            .voice_agent(requests)
            .await
            .map_err(BridgeError::from_status)?
            .into_inner();
        Ok(AudioStream {
            inner: Some(streaming),
        })
    }
}
