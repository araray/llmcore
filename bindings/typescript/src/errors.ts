/**
 * Unified, transport-neutral error type for the llmcore bridge.
 *
 * Both transports surface the same `llmcore.v1.LlmcoreError`:
 *  - gRPC carries it as the binary trailing-metadata key `llmcore-error-bin`,
 *  - HTTP returns it as the JSON `{ "error": { ... } }` body.
 *
 * Both are normalized into {@link BridgeError}.
 */
import type { Metadata, ServiceError } from "@grpc/grpc-js";
import { ErrorCategory, LlmcoreError, errorCategoryToJSON } from "./gen/llmcore/v1/errors";

export { ErrorCategory } from "./gen/llmcore/v1/errors";

/** Binary trailing-metadata key the bridge uses for the structured error. */
export const ERROR_METADATA_KEY = "llmcore-error-bin";

export interface LlmcoreErrorFields {
  /** ErrorCategory name, e.g. "ERROR_CATEGORY_PROVIDER". */
  category: string;
  /** Stable dotted code, e.g. "provider.rate_limited". */
  code: string;
  message: string;
  httpStatus?: number;
  retryable: boolean;
  retryAfterMs?: number;
  provider?: string;
  model?: string;
  /** gRPC status code (only when the error came over gRPC). */
  grpcCode?: number;
}

/** A normalized bridge error carrying the full structured payload. */
export class BridgeError extends Error {
  readonly category: string;
  readonly code: string;
  readonly httpStatus?: number;
  readonly retryable: boolean;
  readonly retryAfterMs?: number;
  readonly provider?: string;
  readonly model?: string;
  readonly grpcCode?: number;

  constructor(fields: LlmcoreErrorFields) {
    super(fields.message || fields.code || "bridge error");
    this.name = "BridgeError";
    this.category = fields.category;
    this.code = fields.code;
    this.httpStatus = fields.httpStatus;
    this.retryable = fields.retryable;
    this.retryAfterMs = fields.retryAfterMs;
    this.provider = fields.provider;
    this.model = fields.model;
    this.grpcCode = fields.grpcCode;
  }
}

function fromProto(e: LlmcoreError, grpcCode?: number): BridgeError {
  return new BridgeError({
    category: errorCategoryToJSON(e.category),
    code: e.code,
    message: e.message,
    httpStatus: e.httpStatus,
    retryable: e.retryable,
    retryAfterMs: e.retryAfterMs,
    provider: e.provider,
    model: e.model,
    grpcCode,
  });
}

/** Decode a gRPC `ServiceError` into a {@link BridgeError}. */
export function bridgeErrorFromGrpc(err: ServiceError): BridgeError {
  const md: Metadata | undefined = err.metadata;
  if (md) {
    const values = md.get(ERROR_METADATA_KEY);
    if (values && values.length > 0) {
      const raw = values[0];
      const bytes = typeof raw === "string" ? Buffer.from(raw, "binary") : (raw as Buffer);
      try {
        return fromProto(LlmcoreError.decode(bytes), err.code);
      } catch {
        // fall through to status-based synthesis
      }
    }
  }
  return new BridgeError({
    category: errorCategoryToJSON(ErrorCategory.ERROR_CATEGORY_INTERNAL),
    code: `grpc.${err.code ?? "unknown"}`,
    message: err.details || err.message,
    retryable: false,
    grpcCode: err.code,
  });
}

/** Shape of the JSON error body the HTTP transport returns (snake_case). */
export interface HttpErrorBody {
  error?: {
    category?: string;
    code?: string;
    message?: string;
    http_status?: number;
    retryable?: boolean;
    retry_after_ms?: number;
    provider?: string;
    model?: string;
  };
}

/** Build a {@link BridgeError} from an HTTP status + JSON error body. */
export function bridgeErrorFromHttp(httpStatus: number, body: HttpErrorBody): BridgeError {
  const e = body?.error ?? {};
  return new BridgeError({
    category: e.category ?? errorCategoryToJSON(ErrorCategory.ERROR_CATEGORY_INTERNAL),
    code: e.code ?? `http.${httpStatus}`,
    message: e.message ?? `HTTP ${httpStatus}`,
    httpStatus: e.http_status ?? httpStatus,
    retryable: e.retryable ?? false,
    retryAfterMs: e.retry_after_ms,
    provider: e.provider,
    model: e.model,
  });
}
