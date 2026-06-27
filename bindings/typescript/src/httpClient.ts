/**
 * HTTP + SSE client for the llmcore bridge (secondary transport).
 *
 * Uses the global `fetch` (Node >= 18). Unary RPCs are JSON POSTs; `ChatStream`
 * is parsed as Server-Sent Events. Wire JSON is snake_case (the bridge uses
 * `preserving_proto_field_name=True`), so the request/response shapes below are
 * snake_case by design.
 */
import { BridgeError, bridgeErrorFromHttp, ErrorCategory } from "./errors";
import { errorCategoryToJSON } from "./gen/llmcore/v1/errors";

export interface UsageJson {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  provider?: string;
  model?: string;
}

export interface ChatResponseJson {
  text: string;
  usage?: UsageJson;
}

export interface ChatChunkJson {
  text?: string;
  done?: boolean;
}

export interface CountTokensResponseJson {
  tokens: number;
}

export interface CostEstimateJson {
  total_cost: number;
  currency: string;
  input_cost?: number;
  output_cost?: number;
  [key: string]: unknown;
}

export interface ProviderInfoJson {
  name: string;
  available: boolean;
}

export interface ServerInfoJson {
  llmcore_version: string;
  bridge_version: string;
  contract_version: string;
  transports: string[];
  capabilities: string[];
  providers?: ProviderInfoJson[];
  tiers?: string[];
}

export interface ModelDetailsJson {
  id: string;
  provider_name: string;
  context_length: number;
  [key: string]: unknown;
}

export interface ListProvidersResponseJson {
  providers: string[];
}

export interface ListModelsResponseJson {
  models: string[];
}

export interface HealthJson {
  ok: boolean;
  detail?: string;
}

export interface ChatHttpRequest {
  message: string;
  session_id?: string;
  system_message?: string;
  provider_name?: string;
  model_name?: string;
  save_session?: boolean;
  tools?: unknown[];
  tool_choice?: string;
  provider_kwargs?: Record<string, unknown>;
}

export class LlmcoreHttpClient {
  private readonly base: string;

  constructor(baseUrl: string) {
    this.base = baseUrl.replace(/\/+$/, "");
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(this.base + path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body ?? {}),
    });
    if (!res.ok) {
      let j: unknown = {};
      try {
        j = await res.json();
      } catch {
        // non-JSON error body
      }
      throw bridgeErrorFromHttp(res.status, j as { error?: Record<string, unknown> });
    }
    return (await res.json()) as T;
  }

  getInfo(): Promise<ServerInfoJson> {
    return this.post("/llmcore.v1/ControlService/GetInfo", {});
  }

  health(): Promise<HealthJson> {
    return this.post("/healthz", {});
  }

  /** Verify contract version and required capabilities; throws on mismatch. */
  async ensureCompatible(requiredCapabilities: string[] = ["tier0"]): Promise<ServerInfoJson> {
    const info = await this.getInfo();
    if (info.contract_version !== "llmcore.v1") {
      throw new BridgeError({
        category: errorCategoryToJSON(ErrorCategory.ERROR_CATEGORY_INVALID_ARGUMENT),
        code: "contract.mismatch",
        message: `server contract ${info.contract_version} != llmcore.v1`,
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

  chat(req: ChatHttpRequest): Promise<ChatResponseJson> {
    return this.post("/llmcore.v1/InferenceService/Chat", req);
  }

  countTokens(
    text: string,
    opts: { provider_name?: string; model_name?: string } = {},
  ): Promise<CountTokensResponseJson> {
    return this.post("/llmcore.v1/InferenceService/CountTokens", { text, ...opts });
  }

  estimateCost(req: {
    provider_name: string;
    model_name: string;
    prompt_tokens: number;
    completion_tokens: number;
    cached_tokens?: number;
    reasoning_tokens?: number;
  }): Promise<CostEstimateJson> {
    return this.post("/llmcore.v1/InferenceService/EstimateCost", req);
  }

  /** Embed is UNIMPLEMENTED in llmcore.v1 (HTTP 501 -> BridgeError UNSUPPORTED). */
  embed(req: { input: string[]; provider_name?: string; model_name?: string }): Promise<unknown> {
    return this.post("/llmcore.v1/InferenceService/Embed", req);
  }

  listProviders(): Promise<ListProvidersResponseJson> {
    return this.post("/llmcore.v1/CatalogService/ListProviders", {});
  }

  listModels(providerName: string): Promise<ListModelsResponseJson> {
    return this.post("/llmcore.v1/CatalogService/ListModels", { provider_name: providerName });
  }

  getProviderDetails(providerName?: string): Promise<ModelDetailsJson> {
    return this.post("/llmcore.v1/CatalogService/GetProviderDetails", {
      provider_name: providerName,
    });
  }

  /** Stream chat tokens over SSE. Throws {@link BridgeError} on an error event. */
  async *chatStream(req: ChatHttpRequest): AsyncGenerator<ChatChunkJson, void, unknown> {
    const res = await fetch(this.base + "/llmcore.v1/InferenceService/ChatStream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!res.ok || !res.body) {
      let j: unknown = {};
      try {
        j = await res.json();
      } catch {
        // ignore
      }
      throw bridgeErrorFromHttp(res.status, j as { error?: Record<string, unknown> });
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let sep: number;
        while ((sep = buffer.indexOf("\n\n")) >= 0) {
          const block = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          let event = "message";
          let data = "";
          for (const line of block.split("\n")) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            else if (line.startsWith("data:")) data += line.slice(5).trim();
          }
          if (!data) continue;
          const parsed = JSON.parse(data) as Record<string, unknown>;
          if (event === "error") {
            const status = (parsed.http_status as number | undefined) ?? 500;
            throw bridgeErrorFromHttp(status, { error: parsed });
          }
          if (event === "done") {
            yield { ...(parsed as ChatChunkJson), done: true };
            return;
          }
          yield parsed as ChatChunkJson;
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}
