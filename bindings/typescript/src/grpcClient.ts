/**
 * Typed gRPC client for the llmcore bridge (primary transport).
 *
 * Wraps the ts-proto/grpc-js generated stubs in a promise-based API, decodes
 * structured errors, exposes capability negotiation, and supports cancellation
 * of the server-streaming `ChatStream` RPC.
 */
import {
  ChannelCredentials,
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

export class LlmcoreGrpcClient {
  private readonly inference: InferenceServiceClient;
  private readonly catalog: CatalogServiceClient;
  private readonly control: ControlServiceClient;

  constructor(address: string, options: GrpcClientOptions = {}) {
    const creds = options.credentials ?? ChannelCredentials.createInsecure();
    this.inference = new InferenceServiceClient(address, creds);
    this.catalog = new CatalogServiceClient(address, creds);
    this.control = new ControlServiceClient(address, creds);
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

  /** Close all underlying channels. */
  close(): void {
    this.inference.close();
    this.catalog.close();
    this.control.close();
  }
}
