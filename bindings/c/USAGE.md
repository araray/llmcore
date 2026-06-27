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

```bash
# CMake
cmake -S . -B build && cmake --build build -j
export LLMCORE_BRIDGE_PYTHON=/path/to/venv/bin/python
ctest --test-dir build --output-on-failure

# or plain make
make
LLMCORE_BRIDGE_PYTHON=/path/to/venv/bin/python make test
```
