# Using the llmcore C client

## Prerequisites

A running bridge with the HTTP transport (dev / fake backend):

```bash
LLMCORE_BRIDGE_FAKE=1 python -m llmcore.bridge.cli serve \
  --transport http --http-address 127.0.0.1:50152 --insecure
```

## Connect

```c
#include "llmcore_client.h"
llmcore_client *c = llmcore_client_new("http://127.0.0.1:50152");
if (!c) { /* allocation failed */ }
```

### TLS / mTLS

Use an `https://` base URL. libcurl validates against the system CA bundle. For
client certificates (mTLS), add to `post_json` and `llmcore_chat_stream`:

```c
curl_easy_setopt(curl, CURLOPT_SSLCERT, "/path/client.pem");
curl_easy_setopt(curl, CURLOPT_SSLKEY,  "/path/client.key");
/* optionally CURLOPT_CAINFO for a private CA */
```

## Capability negotiation

```c
const char *caps[] = {"tier0"};
llmcore_error *e = llmcore_ensure_compatible(c, caps, 1);
if (e) {
  /* e->code is "contract.mismatch" or "capability.missing" */
  llmcore_error_free(e);
}
```

## Inference

```c
/* Unary */
llmcore_chat_result r;
llmcore_error *e = llmcore_chat(c, "Summarize the contract.", &r);
if (!e) {
  printf("%s\n", r.text);
  printf("tokens: %d/%d/%d\n", r.prompt_tokens, r.completion_tokens, r.total_tokens);
  llmcore_chat_result_free(&r);
} else {
  llmcore_error_free(e);
}

/* Streaming (+ cancellation) */
static int on_chunk(const char *text, int done, void *user) {
  if (!done) fputs(text, stdout);
  return 0;            /* return nonzero to cancel the stream */
}
e = llmcore_chat_stream(c, "stream this", on_chunk, NULL);
if (e) llmcore_error_free(e);
```

> The provided `llmcore_chat`/`llmcore_chat_stream` send only `message`. To pass
> `provider_name`, `model_name`, `system_message`, tools, or `provider_kwargs`,
> add the corresponding `cJSON_Add*` calls where the request object is built in
> `src/llmcore_client.c` (the HTTP body is plain JSON).

## Tokens, cost, catalog

```c
int n = 0;
llmcore_count_tokens(c, "a b c", &n);                 /* n == 3 */

double cost = 0; char *currency = NULL;
llmcore_estimate_cost(c, "openai", "gpt-4o-mini", 1000, 500, &cost, &currency);
free(currency);

char **items = NULL; size_t count = 0;
llmcore_list_providers(c, &items, &count);
llmcore_string_array_free(items, count);

llmcore_list_models(c, "openai", &items, &count);
llmcore_string_array_free(items, count);
```

## Sessions, vector store & presets (Tier 1)

Available when the bridge advertises `tier1.sessions` / `tier1.vector`; negotiate
with `llmcore_ensure_compatible(c, (const char*[]){"tier1.sessions",
"tier1.vector"}, 2)`. Driven over the same HTTP transport. For a dev bridge, gate
the fakes on with `LLMCORE_BRIDGE_FAKE_SESSIONS=1 LLMCORE_BRIDGE_FAKE_VECTOR=1`.

```c
/* sessions + context items */
llmcore_session s;
llmcore_error *e = llmcore_create_session(c, "demo", "You are terse.", &s);
char *item_id = NULL;
llmcore_add_context_item(c, s.id, "Launch date is June 30.", "user_text", &item_id);
llmcore_context_item it;
if (!llmcore_get_context_item(c, s.id, item_id, &it)) {
  printf("[%s] %s\n", it.type, it.content);
  llmcore_context_item_free(&it);
}
free(item_id);

/* vector store */
const char *docs[] = {"Paris is the capital of France."};
char **ids = NULL; size_t n_ids = 0;
llmcore_add_documents(c, docs, 1, /*collection=*/NULL, &ids, &n_ids);
llmcore_string_array_free(ids, n_ids);
llmcore_search_result *res = NULL; size_t n_res = 0;
if (!llmcore_search_vector_store(c, "capital of France", 3, NULL, &res, &n_res)) {
  for (size_t i = 0; i < n_res; i++) printf("%.3f %s\n", res[i].score, res[i].content);
  llmcore_search_results_free(res, n_res);
}

/* context presets */
llmcore_preset_item pi[] = {{"preset_text_content", "Always cite sources."}};
llmcore_save_context_preset(c, "preamble", "Standard preamble", pi, 1);
llmcore_preset p;
if (!llmcore_get_context_preset(c, "preamble", &p)) {
  printf("%s: %zu item(s)\n", p.name, p.item_count);
  llmcore_preset_free(&p);
}

llmcore_delete_session(c, s.id);
llmcore_session_free(&s);
```

The full surface also includes `llmcore_get_session`, `llmcore_list_sessions`,
`llmcore_update_session_name`, `llmcore_fork_session`, `llmcore_clone_session`,
`llmcore_delete_messages`, `llmcore_remove_context_item`,
`llmcore_list_vector_collections`, `llmcore_list_rag_collections`,
`llmcore_get_rag_collection_info`, `llmcore_delete_rag_collection`,
`llmcore_list_context_presets`, and `llmcore_delete_context_preset`. See
`examples/sessions.c` for a complete run.

## Audio (Tier 2)

Available when the bridge advertises `tier2.audio`; negotiate with
`llmcore_ensure_compatible(c, (const char*[]){"tier2.audio"}, 1)`. Each call fills
an `*_result` struct freed with its `*_result_free`. Proto `bytes` cross the wire
base64-encoded; the client handles that for you. The duplex RPCs are WebSocket
(out of scope here — see **D6**).

```c
/* text-to-speech: sp.audio holds the decoded bytes */
llmcore_speech_result sp;
llmcore_error *e = llmcore_synthesize(c, "hello", &sp);
if (!e) { fwrite(sp.audio, 1, sp.audio_len, out); llmcore_speech_result_free(&sp); }
else llmcore_error_free(e);

/* speech-to-text: pass the raw audio bytes */
llmcore_transcription_result tr;
e = llmcore_transcribe(c, pcm, pcm_len, &tr);
if (!e) { puts(tr.text); llmcore_transcription_result_free(&tr); } else llmcore_error_free(e);

/* image generation: images[i] are base64 (b64_json) strings */
llmcore_image_result img;
e = llmcore_generate_image(c, "a cat", 2, &img);
if (!e) { for (size_t i = 0; i < img.n_images; i++) puts(img.images[i]); llmcore_image_result_free(&img); }
else llmcore_error_free(e);

/* OCR: pass bytes (data,len) or a URL (NULL bytes); pages_json is raw JSON */
llmcore_ocr_result doc;
e = llmcore_ocr(c, pdf, pdf_len, /*url=*/NULL, &doc);
if (!e) { printf("%d pages, model=%s\n", doc.pages_processed, doc.model); llmcore_ocr_result_free(&doc); }
else llmcore_error_free(e);

/* text analysis: summary valid only if has_summary; topics_json is raw JSON */
llmcore_text_analysis_result an;
e = llmcore_analyze_text(c, "…", &an);
if (!e) { if (an.has_summary) puts(an.summary); llmcore_text_analysis_result_free(&an); }
else llmcore_error_free(e);
```

## Error handling & retries

```c
llmcore_error *e = llmcore_chat(c, msg, &r);
if (e) {
  if (e->retryable && e->retry_after_ms > 0) {
    /* sleep e->retry_after_ms, then retry */
  }
  fprintf(stderr, "[%s] %s (http=%d provider=%s)\n",
          e->code, e->message, e->http_status, e->provider ? e->provider : "-");
  llmcore_error_free(e);
}
```

Common `code`s: `provider.rate_limited` (HTTP 429, retryable, `retry_after_ms`
set), `provider.unauthenticated` (401), `context.too_long` (413),
`not_found.session` (404), `unsupported.capability` (UNSUPPORTED → from
`llmcore_embed`), `invalid_argument` (400), `internal` (500).

## Build commands

libcurl is the only required system dev package. cJSON is resolved automatically:
an installed cJSON (CMake config or pkg-config `libcjson`) is used when present,
otherwise it is fetched and built from source via `FetchContent` (a single small
TU — adds only seconds, no system package needed). Force the fetch with
`-DLLMCORE_FETCH_CJSON=ON`, or pin a tag with `-DLLMCORE_CJSON_TAG=vX.Y.Z`. Plain
`cmake -B build && cmake --build build` works out of the box.

```bash
# CMake (out-of-the-box: cJSON fetched if not installed)
cmake -B build && cmake --build build -j
export LLMCORE_BRIDGE_PYTHON=/path/to/venv/bin/python
ctest --test-dir build --output-on-failure

# or plain make (expects an installed cJSON)
make
LLMCORE_BRIDGE_PYTHON=/path/to/venv/bin/python make test
```

### Running the Tier-1 example

```bash
LLMCORE_BRIDGE_FAKE=1 LLMCORE_BRIDGE_FAKE_SESSIONS=1 LLMCORE_BRIDGE_FAKE_VECTOR=1 \
  python -m llmcore.bridge.cli serve --transport http \
  --http-address 127.0.0.1:50152 --insecure

cmake -B build && cmake --build build -j
LLMCORE_HTTP=http://127.0.0.1:50152 ./build/sessions
```
