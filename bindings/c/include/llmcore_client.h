/*
 * llmcore C client - public API.
 *
 * Talks to the llmcore bridge over its HTTP + SSE transport (JSON wire), built
 * on libcurl + cJSON. Tier 0 only (per decision D6, the gRPC path in C is
 * reserved for duplex audio in a later phase). Depends only on the contract.
 *
 * Memory: every function that yields heap data documents who frees it. Functions
 * return an `llmcore_error*` (NULL on success); the caller frees a non-NULL error
 * with llmcore_error_free().
 */
#ifndef LLMCORE_CLIENT_H
#define LLMCORE_CLIENT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque client handle. */
typedef struct llmcore_client llmcore_client;

/* Normalized error; mirrors llmcore.v1.LlmcoreError. Free with llmcore_error_free(). */
typedef struct {
  char *category;        /* e.g. "ERROR_CATEGORY_PROVIDER" (never NULL) */
  char *code;            /* e.g. "provider.rate_limited"   (never NULL) */
  char *message;         /* human-readable (never NULL)                 */
  int http_status;       /* 0 if unset                                  */
  int retryable;         /* 0/1                                         */
  double retry_after_ms; /* 0 if unset                                  */
  char *provider;        /* NULL if unset                               */
  char *model;           /* NULL if unset                               */
} llmcore_error;

void llmcore_error_free(llmcore_error *err);

/* Create a client for `base_url`, e.g. "http://127.0.0.1:50152".
 * Calls curl_global_init on first use. Returns NULL on allocation failure. */
llmcore_client *llmcore_client_new(const char *base_url);
void llmcore_client_free(llmcore_client *c);

/* Verify contract_version == "llmcore.v1" and that each capability is present. */
llmcore_error *llmcore_ensure_compatible(llmcore_client *c,
                                         const char *const *required_caps,
                                         size_t n_caps);

/* Unary chat. On success fills *out (caller frees with llmcore_chat_result_free).
 * Token fields are -1 when the server omits them. */
typedef struct {
  char *text;            /* malloc'd (caller frees) */
  int prompt_tokens;     /* -1 if unknown */
  int completion_tokens; /* -1 if unknown */
  int total_tokens;      /* -1 if unknown */
} llmcore_chat_result;

void llmcore_chat_result_free(llmcore_chat_result *r);

llmcore_error *llmcore_chat(llmcore_client *c, const char *message,
                            llmcore_chat_result *out);

/* Streaming chat over SSE. `cb` is invoked per frame; the terminal frame has
 * done=1. Returning nonzero from `cb` cancels the stream (treated as success). */
typedef int (*llmcore_chunk_cb)(const char *text, int done, void *user);

llmcore_error *llmcore_chat_stream(llmcore_client *c, const char *message,
                                   llmcore_chunk_cb cb, void *user);

llmcore_error *llmcore_count_tokens(llmcore_client *c, const char *text,
                                    int *out_tokens);

/* On success *out_total_cost is set and *out_currency is malloc'd (caller frees). */
llmcore_error *llmcore_estimate_cost(llmcore_client *c, const char *provider,
                                     const char *model, int prompt_tokens,
                                     int completion_tokens, double *out_total_cost,
                                     char **out_currency);

/* Always returns an UNSUPPORTED error - Embed is UNIMPLEMENTED in llmcore.v1. */
llmcore_error *llmcore_embed(llmcore_client *c, const char *const *inputs,
                             size_t n_inputs);

/* Catalog. On success *out_items is a malloc'd array of malloc'd strings of
 * length *out_n; free with llmcore_string_array_free. */
llmcore_error *llmcore_list_providers(llmcore_client *c, char ***out_items,
                                      size_t *out_n);
llmcore_error *llmcore_list_models(llmcore_client *c, const char *provider,
                                   char ***out_items, size_t *out_n);

void llmcore_string_array_free(char **items, size_t n);

llmcore_error *llmcore_health(llmcore_client *c, int *out_ok);

#ifdef __cplusplus
}
#endif

#endif /* LLMCORE_CLIENT_H */
