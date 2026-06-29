# llmcore.v1 wire contract

The proto package `llmcore.v1` (`bindings/proto/llmcore/v1/`) is the **single
source of truth** for every binding. gRPC is the primary transport; HTTP+SSE is
a faithful secondary projection of the same messages.

## Services and methods

| Service | RPC | Kind | Backing facade symbol | Tier / phase |
|---|---|---|---|:--:|
| `InferenceService` | `Chat` | unary | `LLMCore.chat_with_usage` (api.py) | T0 ✅ |
| | `ChatStream` | server-stream | `LLMCore.chat(stream=True)` (api.py) | T0 ✅ |
| | `Embed` | unary | *(no public path)* | ⛔ UNIMPLEMENTED |
| | `CountTokens` | unary | `llmcore.count_tokens` (tokens.py) | T0 ✅ |
| | `EstimateCost` | unary | `LLMCore.estimate_cost` (api.py) | T0 ✅ |
| `CatalogService` | `ListProviders` | unary | `LLMCore.get_available_providers` (api.py) | T0 ✅ |
| | `ListModels` | unary | `LLMCore.get_models_for_provider` (api.py) | T0 ✅ |
| | `GetProviderDetails` | unary | `LLMCore.get_provider_details` (api.py) | T0 ✅ |
| `ControlService` | `GetInfo` | unary | bridge-assembled `ServerInfo` | T0 ✅ |
| | `Health` | unary | bridge | T0 ✅ |
| | `ReloadConfig` | unary | `LLMCore.reload_config` (api.py) | T0 ✅ |
| `AudioService` | (8 RPCs) | unary + bidi | multimodal/Deepgram surface | T2 ✅ (B3) |
| `SessionService` | `CreateSession`/`GetSession`/`ListSessions`/`DeleteSession`/`UpdateSessionName` | unary | `LLMCore.{create,get,list,delete,update_session_name}_session` (api.py) | T1 ✅ (B4) |
| | `ForkSession`/`CloneSession`/`DeleteMessages`/`GetMessagesByRange` | unary | `LLMCore.{fork,clone}_session` / `delete_messages` / `get_messages_by_range` (api.py) | T1 ✅ (B4) |
| | `AddContextItem`/`GetContextItem`/`RemoveContextItem` | unary | `LLMCore.{add,get,remove}_context_item` (api.py) | T1 ✅ (B4) |
| `VectorService` | `AddDocuments`/`SearchVectorStore` | unary | `LLMCore.{add_documents_to_vector_store,search_vector_store}` (api.py) | T1 ✅ (B4) |
| | `ListVectorCollections`/`ListRagCollections`/`GetRagCollectionInfo`/`DeleteRagCollection` | unary | `LLMCore.{list_vector_collections,list_rag_collections,get_rag_collection_info,delete_rag_collection}` (api.py) | T1 ✅ (B4) |
| `PresetService` | `SaveContextPreset`/`GetContextPreset`/`ListContextPresets`/`DeleteContextPreset` | unary | `LLMCore.{save,get,list,delete}_context_preset` (api.py) | T1 ✅ (B4) |

> **External-RAG invariant.** `VectorService` operates the vector store
> *directly* (add/search/manage). It never triggers chat-time retrieval — `Chat`
> keeps `enable_rag=false`. Callers stage context explicitly via `SessionService`
> / `VectorService`, matching llmcore's design.

## HTTP projection

* Unary: `POST /llmcore.v1/<Service>/<Method>` with a JSON body
  (`snake_case`, via `json_format` with `preserving_proto_field_name=True`).
  Response is the JSON-encoded proto.
* Server streaming (`ChatStream`): the same POST returns
  `Content-Type: text/event-stream`; one `data: {json-chunk}` per token, then a
  terminal `event: done` (or `event: error` with the structured error).
* Health: `GET /healthz` (also `POST /llmcore.v1/ControlService/Health`).
* Live duplex audio maps to **WebSocket** (phase B3); the one-shot Audio RPCs
  return HTTP `501` in v1 to mirror the gRPC `UNIMPLEMENTED`.

## Capability negotiation

Clients **must** call `ControlService.GetInfo` on connect and check:

* `contract_version == "llmcore.v1"`,
* every capability they require is present in `capabilities`.

B1 advertises:

```
tier0
inference.chat   inference.chat_stream   inference.count_tokens   inference.estimate_cost
catalog.providers   catalog.models   control.info
transport.grpc   transport.http        (per enabled transport)
```

Deliberately **absent** in v1 (so clients degrade gracefully):

* `chat.tool_calls` — `ChatResponse.tool_calls` / `finish_reason` are PROVISIONAL
  fields, not yet populated (pinned against the real provider response in a later
  phase).
* `tier2.*` — AudioService is UNIMPLEMENTED until B3.

## Error model

Every failure is a `llmcore.v1.LlmcoreError` with a coarse `ErrorCategory`, a
stable dotted `code`, a **secret-redacted** `message`, and provider metadata
(`provider`, `model`, `http_status`, `retryable`, `retry_after_ms`).

* **gRPC**: the RPC aborts with a mapped status code; the serialized
  `LlmcoreError` is attached as binary trailing metadata under
  **`llmcore-error-bin`**. (We avoid a hard `grpcio-status` dependency.)
* **HTTP**: `{"error": { ...LlmcoreError... }}` with the mapped HTTP status.

Category → status mapping (`src/llmcore/bridge/errors.py`):

| Category | gRPC status | HTTP | Source exception |
|---|---|---:|---|
| `PROVIDER` (429) | `RESOURCE_EXHAUSTED` | 429 | `ProviderError` |
| `PROVIDER` (401) | `UNAUTHENTICATED` | 401 | `ProviderError` |
| `PROVIDER` (403) | `PERMISSION_DENIED` | 403 | `ProviderError` |
| `PROVIDER` (408/504) | `DEADLINE_EXCEEDED` | (passthrough) | `ProviderError` |
| `PROVIDER` (other) | `UNAVAILABLE`/`INTERNAL`¹ | 502 | `ProviderError` |
| `CONFIG` | `FAILED_PRECONDITION` | 400 | `ConfigError` |
| `CONTEXT_LENGTH` | `INVALID_ARGUMENT` | 413 | `ContextLengthError` |
| `CONTEXT` | `INVALID_ARGUMENT` | 400 | `ContextError` |
| `NOT_FOUND` | `NOT_FOUND` | 404 | `SessionNotFoundError` |
| `STORAGE` | `UNAVAILABLE` | 503 | `StorageError` |
| `EMBEDDING` | `INTERNAL` | 500 | `EmbeddingError` |
| `SEARCH` | `INTERNAL` | 502 | `SearchProviderError` |
| `UNSUPPORTED` | `UNIMPLEMENTED` | 501 | `NotImplementedError` |
| `CANCELLED` | `CANCELLED` | 499 | `asyncio.CancelledError` |
| `INTERNAL` | `INTERNAL` | 500 | `LLMCoreError` / other |
| `INVALID_ARGUMENT` | `INVALID_ARGUMENT` | 400 | bad request at the edge |

¹ `UNAVAILABLE` when `retryable`, else `INTERNAL`.

## Cancellation

Cancellation is first-class. A gRPC client cancel propagates a
`CancelledError` into the servicer coroutine, which closes the underlying facade
generator; the RPC reports `CANCELLED`. On HTTP, a client disconnect cancels the
SSE generator the same way.

## Versioning & compatibility

* The proto **package** carries the major contract version (`llmcore.v1`). A
  breaking change means a new package (`llmcore.v2`), served alongside v1.
* Within `v1`, evolution is **additive only**: new fields, new messages, new
  RPCs, new enum values, new capability flags. Existing field numbers and enum
  values are immutable.
* `buf breaking` (config: `breaking.use: [FILE]`) gates every change in CI
  against the previous state.
* Clients gate behaviour on `capabilities`, never on version-string equality
  beyond the major check — this is what lets the server light up `tier2.*` or
  `chat.tool_calls` later without breaking older clients.

## buf lint exceptions (documented)

`buf.yaml` uses the `STANDARD` lint category with three explicit exceptions:

* `RPC_REQUEST_STANDARD_NAME`, `RPC_RESPONSE_STANDARD_NAME`,
  `RPC_REQUEST_RESPONSE_UNIQUE`.

Rationale: several RPCs intentionally take a shared `Empty` request and return
domain messages directly (`ModelDetails`, `CostEstimate`, `ServerInfo`, the
multimodal results) instead of single-field `Method{Request,Response}` wrappers.
Wrapping would add no semantic value and would bloat five generated clients. All
other STANDARD rules (enum value prefixes, `*_UNSPECIFIED` zero values,
package/directory match, field snake_case, etc.) are enforced. `buf lint` passes
clean.
