/**
 * @llmcore/client — TypeScript/Node client for the llmcore bridge.
 *
 * Two transports, one contract (`llmcore.v1`):
 *  - {@link LlmcoreGrpcClient} — primary, over gRPC (HTTP/2 + protobuf).
 *  - {@link LlmcoreHttpClient} — secondary, over HTTP + SSE (JSON).
 *
 * @example
 * ```ts
 * import { LlmcoreGrpcClient } from "@llmcore/client";
 * const client = new LlmcoreGrpcClient("127.0.0.1:50151");
 * await client.ensureCompatible(["tier0"]);
 * const res = await client.chat({ message: "hello" });
 * console.log(res.text);
 * for await (const chunk of client.chatStream({ message: "stream me" })) {
 *   if (!chunk.done) process.stdout.write(chunk.text ?? "");
 * }
 * client.close();
 * ```
 */
export {
  BridgeError,
  ERROR_METADATA_KEY,
  ErrorCategory,
  bridgeErrorFromGrpc,
  bridgeErrorFromHttp,
} from "./errors";
export type { LlmcoreErrorFields, HttpErrorBody } from "./errors";

export { LlmcoreGrpcClient, ChatStream } from "./grpcClient";
export type { GrpcClientOptions } from "./grpcClient";

export { LlmcoreHttpClient } from "./httpClient";
export type {
  ChatHttpRequest,
  ChatResponseJson,
  ChatChunkJson,
  CostEstimateJson,
  CountTokensResponseJson,
  HealthJson,
  ListModelsResponseJson,
  ListProvidersResponseJson,
  ModelDetailsJson,
  ProviderInfoJson,
  ServerInfoJson,
  UsageJson,
} from "./httpClient";

// Key generated protobuf message types (gRPC path).
export type {
  ChatRequest,
  ChatResponse,
  ChatChunk,
  CostEstimate,
  CountTokensRequest,
  CountTokensResponse,
  EstimateCostRequest,
} from "./gen/llmcore/v1/inference";
export type { ServerInfo, HealthStatus, ProviderInfo } from "./gen/llmcore/v1/control";
export type { ModelDetails, ListModelsResponse, ListProvidersResponse } from "./gen/llmcore/v1/catalog";
export type { Message, Tool, ToolCall, ToolResult, Usage, Role } from "./gen/llmcore/v1/common";
