/**
 * Typed gRPC client for the llmcore bridge (primary transport).
 *
 * Wraps the ts-proto/grpc-js generated stubs in a promise-based API, decodes
 * structured errors, exposes capability negotiation, and supports cancellation
 * of the server-streaming `ChatStream` RPC.
 */
import {
  ChannelCredentials,
  type ClientDuplexStream,
  type ClientReadableStream,
  type ServiceError,
} from "@grpc/grpc-js";

import { Empty } from "./gen/llmcore/v1/common";
import {
  CatalogServiceClient,
  GetProviderRequest,
  ListModelsRequest,
  type ListModelsResponse,
  type ListProvidersResponse,
  type ModelDetails,
} from "./gen/llmcore/v1/catalog";
import {
  ControlServiceClient,
  ReloadConfigRequest,
  type HealthStatus,
  type ReloadConfigResponse,
  type ServerInfo,
} from "./gen/llmcore/v1/control";
import {
  ChatRequest,
  type ChatChunk,
  type ChatResponse,
  CountTokensRequest,
  type CountTokensResponse,
  type CostEstimate,
  EmbedRequest,
  type EmbedResponse,
  EstimateCostRequest,
  InferenceServiceClient,
} from "./gen/llmcore/v1/inference";
import { BridgeError, bridgeErrorFromGrpc, ErrorCategory } from "./errors";
import { errorCategoryToJSON } from "./gen/llmcore/v1/errors";
import {
  AnalyzeTextRequest,
  AudioIn,
  type AudioOut,
  AudioServiceClient,
  type DeepPartial,
  GenerateImageRequest,
  type ImageGenerationResult,
  OcrRequest,
  type OCRResult,
  type SpeechResult,
  SynthControl,
  SynthesizeRequest,
  type TextAnalysisResult,
  TranscribeRequest,
  type TranscriptionResult,
  type TranscriptionStreamEvent,
  VoiceAgentClientEvent,
  type VoiceAgentEvent,
} from "./gen/llmcore/v1/audio";
import {
  AddContextItemRequest,
  type AddContextItemResponse,
  type ChatSession,
  CloneSessionRequest,
  type CloneSessionResponse,
  type ContextItem,
  CreateSessionRequest,
  DeleteMessagesRequest,
  type DeleteMessagesResponse,
  DeleteSessionRequest,
  ForkSessionRequest,
  type ForkSessionResponse,
  GetContextItemRequest,
  GetMessagesByRangeRequest,
  type GetMessagesByRangeResponse,
  GetSessionRequest,
  ListSessionsRequest,
  type ListSessionsResponse,
  RemoveContextItemRequest,
  type RemoveContextItemResponse,
  SessionServiceClient,
  UpdateSessionNameRequest,
} from "./gen/llmcore/v1/sessions";
import {
  AddDocumentsRequest,
  type AddDocumentsResponse,
  DeleteRagCollectionRequest,
  type DeleteRagCollectionResponse,
  GetRagCollectionInfoRequest,
  type ListCollectionsResponse,
  type RagCollectionInfo,
  SearchVectorStoreRequest,
  type SearchVectorStoreResponse,
  VectorServiceClient,
} from "./gen/llmcore/v1/vector";
import {
  type ContextPreset,
  DeleteContextPresetRequest,
  type DeleteContextPresetResponse,
  GetContextPresetRequest,
  type ListContextPresetsResponse,
  PresetServiceClient,
  SaveContextPresetRequest,
} from "./gen/llmcore/v1/presets";

export interface GrpcClientOptions {
  /** Channel credentials. Defaults to insecure (localhost/dev). */
  credentials?: ChannelCredentials;
}

/** A cancellable async-iterable wrapper over a `ChatStream` call. */
export class ChatStream implements AsyncIterable<ChatChunk> {
  constructor(private readonly call: ClientReadableStream<ChatChunk>) {}

  /** Cancel the in-flight stream (maps to gRPC CANCELLED). */
  cancel(): void {
    this.call.cancel();
  }

  async *[Symbol.asyncIterator](): AsyncIterator<ChatChunk> {
    try {
      for await (const chunk of this.call as AsyncIterable<ChatChunk>) {
        yield chunk;
      }
    } catch (err) {
      throw bridgeErrorFromGrpc(err as ServiceError);
    }
  }
}

/**
 * A bidirectional-streaming wrapper: write request frames with {@link write},
 * half-close with {@link end}, and async-iterate the server's response frames.
 * Errors surface as {@link BridgeError} when iterating.
 */
export class AudioDuplexStream<Req, Res> implements AsyncIterable<Res> {
  constructor(
    private readonly call: ClientDuplexStream<Req, Res>,
    private readonly encode: (req: DeepPartial<Req>) => Req,
  ) {}

  /** Send one request frame (partial; filled via the message's `fromPartial`). */
  write(req: DeepPartial<Req>): this {
    this.call.write(this.encode(req));
    return this;
  }

  /** Half-close the request stream (no more frames will be sent). */
  end(): void {
    this.call.end();
  }

  /** Cancel the in-flight stream (maps to gRPC CANCELLED). */
  cancel(): void {
    this.call.cancel();
  }

  async *[Symbol.asyncIterator](): AsyncIterator<Res> {
    try {
      for await (const msg of this.call as AsyncIterable<Res>) {
        yield msg;
      }
    } catch (err) {
      throw bridgeErrorFromGrpc(err as ServiceError);
    }
  }
}

export class LlmcoreGrpcClient {
  private readonly inference: InferenceServiceClient;
  private readonly catalog: CatalogServiceClient;
  private readonly control: ControlServiceClient;
  private readonly audio: AudioServiceClient;
  private readonly session: SessionServiceClient;
  private readonly vector: VectorServiceClient;
  private readonly preset: PresetServiceClient;

  constructor(address: string, options: GrpcClientOptions = {}) {
    const creds = options.credentials ?? ChannelCredentials.createInsecure();
    this.inference = new InferenceServiceClient(address, creds);
    this.catalog = new CatalogServiceClient(address, creds);
    this.control = new ControlServiceClient(address, creds);
    this.audio = new AudioServiceClient(address, creds);
    this.session = new SessionServiceClient(address, creds);
    this.vector = new VectorServiceClient(address, creds);
    this.preset = new PresetServiceClient(address, creds);
  }

  private call<Res>(
    invoke: (cb: (err: ServiceError | null, res: Res) => void) => unknown,
  ): Promise<Res> {
    return new Promise<Res>((resolve, reject) => {
      invoke((err, res) => (err ? reject(bridgeErrorFromGrpc(err)) : resolve(res)));
    });
  }

  // -- control --------------------------------------------------------- //
  getInfo(): Promise<ServerInfo> {
    return this.call<ServerInfo>((cb) => this.control.getInfo(Empty.fromPartial({}), cb));
  }

  health(): Promise<HealthStatus> {
    return this.call<HealthStatus>((cb) => this.control.health(Empty.fromPartial({}), cb));
  }

  reloadConfig(req: Partial<ReloadConfigRequest> = {}): Promise<ReloadConfigResponse> {
    return this.call<ReloadConfigResponse>((cb) =>
      this.control.reloadConfig(ReloadConfigRequest.fromPartial(req), cb),
    );
  }

  /**
   * Negotiate on connect: verify the contract version and that every required
   * capability is advertised. Throws {@link BridgeError} on mismatch.
   */
  async ensureCompatible(requiredCapabilities: string[] = ["tier0"]): Promise<ServerInfo> {
    const info = await this.getInfo();
    if (info.contractVersion !== "llmcore.v1") {
      throw new BridgeError({
        category: errorCategoryToJSON(ErrorCategory.ERROR_CATEGORY_INVALID_ARGUMENT),
        code: "contract.mismatch",
        message: `server contract ${info.contractVersion} != llmcore.v1`,
        retryable: false,
      });
    }
    const missing = requiredCapabilities.filter((c) => !info.capabilities.includes(c));
    if (missing.length > 0) {
      throw new BridgeError({
        category: errorCategoryToJSON(ErrorCategory.ERROR_CATEGORY_UNSUPPORTED),
        code: "capability.missing",
        message: `server lacks required capabilities: ${missing.join(", ")}`,
        retryable: false,
      });
    }
    return info;
  }

  // -- inference ------------------------------------------------------- //
  chat(req: Partial<ChatRequest>): Promise<ChatResponse> {
    const message = ChatRequest.fromPartial(req);
    return this.call<ChatResponse>((cb) => this.inference.chat(message, cb));
  }

  chatStream(req: Partial<ChatRequest>): ChatStream {
    const call = this.inference.chatStream(ChatRequest.fromPartial(req));
    return new ChatStream(call);
  }

  countTokens(req: Partial<CountTokensRequest>): Promise<CountTokensResponse> {
    return this.call<CountTokensResponse>((cb) =>
      this.inference.countTokens(CountTokensRequest.fromPartial(req), cb),
    );
  }

  estimateCost(req: Partial<EstimateCostRequest>): Promise<CostEstimate> {
    return this.call<CostEstimate>((cb) =>
      this.inference.estimateCost(EstimateCostRequest.fromPartial(req), cb),
    );
  }

  /** Embed is UNIMPLEMENTED in llmcore.v1 (throws a BridgeError UNSUPPORTED). */
  embed(req: Partial<EmbedRequest>): Promise<EmbedResponse> {
    return this.call<EmbedResponse>((cb) =>
      this.inference.embed(EmbedRequest.fromPartial(req), cb),
    );
  }

  // -- catalog --------------------------------------------------------- //
  listProviders(): Promise<ListProvidersResponse> {
    return this.call<ListProvidersResponse>((cb) =>
      this.catalog.listProviders(Empty.fromPartial({}), cb),
    );
  }

  listModels(providerName: string): Promise<ListModelsResponse> {
    return this.call<ListModelsResponse>((cb) =>
      this.catalog.listModels(ListModelsRequest.fromPartial({ providerName }), cb),
    );
  }

  getProviderDetails(providerName?: string): Promise<ModelDetails> {
    return this.call<ModelDetails>((cb) =>
      this.catalog.getProviderDetails(GetProviderRequest.fromPartial({ providerName }), cb),
    );
  }

  // -- audio: live duplex (Tier 2) ------------------------------------- //
  /** Bidi speech-to-text. Write `AudioIn` frames; iterate `TranscriptionStreamEvent`s. */
  transcribeStream(): AudioDuplexStream<AudioIn, TranscriptionStreamEvent> {
    return new AudioDuplexStream(this.audio.transcribeStream(), AudioIn.fromPartial);
  }

  /** Bidi text-to-speech. Write `SynthControl` frames; iterate `AudioOut` chunks. */
  synthesizeStream(): AudioDuplexStream<SynthControl, AudioOut> {
    return new AudioDuplexStream(this.audio.synthesizeStream(), SynthControl.fromPartial);
  }

  /** Bidi voice agent. Write `VoiceAgentClientEvent`s; iterate `VoiceAgentEvent`s. */
  voiceAgent(): AudioDuplexStream<VoiceAgentClientEvent, VoiceAgentEvent> {
    return new AudioDuplexStream(this.audio.voiceAgent(), VoiceAgentClientEvent.fromPartial);
  }

  // -- audio: one-shot (Tier 2) ---------------------------------------- //
  synthesize(req: DeepPartial<SynthesizeRequest>): Promise<SpeechResult> {
    return this.call<SpeechResult>((cb) =>
      this.audio.synthesize(SynthesizeRequest.fromPartial(req), cb),
    );
  }

  transcribe(req: DeepPartial<TranscribeRequest>): Promise<TranscriptionResult> {
    return this.call<TranscriptionResult>((cb) =>
      this.audio.transcribe(TranscribeRequest.fromPartial(req), cb),
    );
  }

  generateImage(req: DeepPartial<GenerateImageRequest>): Promise<ImageGenerationResult> {
    return this.call<ImageGenerationResult>((cb) =>
      this.audio.generateImage(GenerateImageRequest.fromPartial(req), cb),
    );
  }

  ocr(req: DeepPartial<OcrRequest>): Promise<OCRResult> {
    return this.call<OCRResult>((cb) => this.audio.ocr(OcrRequest.fromPartial(req), cb));
  }

  analyzeText(req: DeepPartial<AnalyzeTextRequest>): Promise<TextAnalysisResult> {
    return this.call<TextAnalysisResult>((cb) =>
      this.audio.analyzeText(AnalyzeTextRequest.fromPartial(req), cb),
    );
  }

  // -- sessions (Tier 1) ----------------------------------------------- //
  createSession(req: Partial<CreateSessionRequest> = {}): Promise<ChatSession> {
    return this.call<ChatSession>((cb) =>
      this.session.createSession(CreateSessionRequest.fromPartial(req), cb),
    );
  }

  getSession(req: Partial<GetSessionRequest>): Promise<ChatSession> {
    return this.call<ChatSession>((cb) =>
      this.session.getSession(GetSessionRequest.fromPartial(req), cb),
    );
  }

  listSessions(req: Partial<ListSessionsRequest> = {}): Promise<ListSessionsResponse> {
    return this.call<ListSessionsResponse>((cb) =>
      this.session.listSessions(ListSessionsRequest.fromPartial(req), cb),
    );
  }

  deleteSession(req: Partial<DeleteSessionRequest>): Promise<Empty> {
    return this.call<Empty>((cb) =>
      this.session.deleteSession(DeleteSessionRequest.fromPartial(req), cb),
    );
  }

  updateSessionName(req: Partial<UpdateSessionNameRequest>): Promise<Empty> {
    return this.call<Empty>((cb) =>
      this.session.updateSessionName(UpdateSessionNameRequest.fromPartial(req), cb),
    );
  }

  forkSession(req: Partial<ForkSessionRequest>): Promise<ForkSessionResponse> {
    return this.call<ForkSessionResponse>((cb) =>
      this.session.forkSession(ForkSessionRequest.fromPartial(req), cb),
    );
  }

  cloneSession(req: Partial<CloneSessionRequest>): Promise<CloneSessionResponse> {
    return this.call<CloneSessionResponse>((cb) =>
      this.session.cloneSession(CloneSessionRequest.fromPartial(req), cb),
    );
  }

  deleteMessages(req: Partial<DeleteMessagesRequest>): Promise<DeleteMessagesResponse> {
    return this.call<DeleteMessagesResponse>((cb) =>
      this.session.deleteMessages(DeleteMessagesRequest.fromPartial(req), cb),
    );
  }

  getMessagesByRange(
    req: Partial<GetMessagesByRangeRequest>,
  ): Promise<GetMessagesByRangeResponse> {
    return this.call<GetMessagesByRangeResponse>((cb) =>
      this.session.getMessagesByRange(GetMessagesByRangeRequest.fromPartial(req), cb),
    );
  }

  addContextItem(req: Partial<AddContextItemRequest>): Promise<AddContextItemResponse> {
    return this.call<AddContextItemResponse>((cb) =>
      this.session.addContextItem(AddContextItemRequest.fromPartial(req), cb),
    );
  }

  getContextItem(req: Partial<GetContextItemRequest>): Promise<ContextItem> {
    return this.call<ContextItem>((cb) =>
      this.session.getContextItem(GetContextItemRequest.fromPartial(req), cb),
    );
  }

  removeContextItem(
    req: Partial<RemoveContextItemRequest>,
  ): Promise<RemoveContextItemResponse> {
    return this.call<RemoveContextItemResponse>((cb) =>
      this.session.removeContextItem(RemoveContextItemRequest.fromPartial(req), cb),
    );
  }

  // -- vector store & RAG (Tier 1) ------------------------------------- //
  addDocuments(req: Partial<AddDocumentsRequest>): Promise<AddDocumentsResponse> {
    return this.call<AddDocumentsResponse>((cb) =>
      this.vector.addDocuments(AddDocumentsRequest.fromPartial(req), cb),
    );
  }

  searchVectorStore(
    req: Partial<SearchVectorStoreRequest>,
  ): Promise<SearchVectorStoreResponse> {
    return this.call<SearchVectorStoreResponse>((cb) =>
      this.vector.searchVectorStore(SearchVectorStoreRequest.fromPartial(req), cb),
    );
  }

  listVectorCollections(): Promise<ListCollectionsResponse> {
    return this.call<ListCollectionsResponse>((cb) =>
      this.vector.listVectorCollections(Empty.fromPartial({}), cb),
    );
  }

  listRagCollections(): Promise<ListCollectionsResponse> {
    return this.call<ListCollectionsResponse>((cb) =>
      this.vector.listRagCollections(Empty.fromPartial({}), cb),
    );
  }

  getRagCollectionInfo(
    req: Partial<GetRagCollectionInfoRequest>,
  ): Promise<RagCollectionInfo> {
    return this.call<RagCollectionInfo>((cb) =>
      this.vector.getRagCollectionInfo(GetRagCollectionInfoRequest.fromPartial(req), cb),
    );
  }

  deleteRagCollection(
    req: Partial<DeleteRagCollectionRequest>,
  ): Promise<DeleteRagCollectionResponse> {
    return this.call<DeleteRagCollectionResponse>((cb) =>
      this.vector.deleteRagCollection(DeleteRagCollectionRequest.fromPartial(req), cb),
    );
  }

  // -- context presets (Tier 1) ---------------------------------------- //
  saveContextPreset(req: Partial<SaveContextPresetRequest>): Promise<Empty> {
    return this.call<Empty>((cb) =>
      this.preset.saveContextPreset(SaveContextPresetRequest.fromPartial(req), cb),
    );
  }

  getContextPreset(req: Partial<GetContextPresetRequest>): Promise<ContextPreset> {
    return this.call<ContextPreset>((cb) =>
      this.preset.getContextPreset(GetContextPresetRequest.fromPartial(req), cb),
    );
  }

  listContextPresets(): Promise<ListContextPresetsResponse> {
    return this.call<ListContextPresetsResponse>((cb) =>
      this.preset.listContextPresets(Empty.fromPartial({}), cb),
    );
  }

  deleteContextPreset(
    req: Partial<DeleteContextPresetRequest>,
  ): Promise<DeleteContextPresetResponse> {
    return this.call<DeleteContextPresetResponse>((cb) =>
      this.preset.deleteContextPreset(DeleteContextPresetRequest.fromPartial(req), cb),
    );
  }

  /** Close all underlying channels. */
  close(): void {
    this.inference.close();
    this.catalog.close();
    this.control.close();
    this.audio.close();
    this.session.close();
    this.vector.close();
    this.preset.close();
  }
}
