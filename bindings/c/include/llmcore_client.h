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

/* ===================== Audio (Tier 2) — unary RPCs ========================
 * Available when the bridge advertises "tier2.audio"; otherwise these return an
 * UNSUPPORTED error (HTTP 501). The live duplex RPCs (transcribe_stream,
 * synthesize_stream, voice_agent) are WebSocket-based and intentionally out of
 * scope for this HTTP/SSE client (per D6, C's duplex audio path is gRPC, a later
 * phase). Proto "bytes" fields cross the JSON wire base64-encoded; this client
 * handles that transparently (decoding audio out, encoding audio/data in). */

/* Text-to-speech result. Free with llmcore_speech_result_free.
 * `audio` holds the decoded audio bytes (`audio_len` long; may be NULL/0). */
typedef struct {
  unsigned char *audio; /* malloc'd, base64-decoded (caller frees) */
  size_t audio_len;
  char *format;         /* malloc'd ("" if absent) */
  char *model;          /* malloc'd ("" if absent) */
  char *voice;          /* malloc'd ("" if absent) */
} llmcore_speech_result;
void llmcore_speech_result_free(llmcore_speech_result *r);

llmcore_error *llmcore_synthesize(llmcore_client *c, const char *text,
                                  llmcore_speech_result *out);

/* Speech-to-text. `audio`/`audio_len` is the raw audio to transcribe. */
typedef struct {
  char *text;           /* malloc'd */
  char *language;       /* malloc'd ("" if absent) */
  char *model;          /* malloc'd ("" if absent) */
} llmcore_transcription_result;
void llmcore_transcription_result_free(llmcore_transcription_result *r);

llmcore_error *llmcore_transcribe(llmcore_client *c, const unsigned char *audio,
                                  size_t audio_len, llmcore_transcription_result *out);

/* Image generation. `images` is an array of `n_images` base64 (b64_json) strings. */
typedef struct {
  char **images;        /* malloc'd array of malloc'd base64 strings */
  size_t n_images;
  char *model;          /* malloc'd ("" if absent) */
} llmcore_image_result;
void llmcore_image_result_free(llmcore_image_result *r);

llmcore_error *llmcore_generate_image(llmcore_client *c, const char *prompt, int n,
                                      llmcore_image_result *out);

/* Document OCR. Provide either `data`/`data_len` (bytes) or `url` (pass NULL for
 * url to use the bytes). `pages_json` is the raw JSON array of pages. */
typedef struct {
  char *model;             /* malloc'd ("" if absent) */
  int pages_processed;
  long doc_size_bytes;     /* valid only when has_doc_size_bytes */
  int has_doc_size_bytes;
  char *pages_json;        /* malloc'd raw JSON array (e.g. "[]") */
} llmcore_ocr_result;
void llmcore_ocr_result_free(llmcore_ocr_result *r);

llmcore_error *llmcore_ocr(llmcore_client *c, const unsigned char *data, size_t data_len,
                           const char *url, llmcore_ocr_result *out);

/* Text analysis. `summary` is valid only when has_summary; `topics_json` is the
 * raw JSON array of topic objects. (Analysis feature flags are not exposed by
 * this convenience wrapper; the no-features result is returned.) */
typedef struct {
  char *summary;        /* malloc'd; valid only when has_summary */
  int has_summary;
  char *language;       /* malloc'd ("" if absent) */
  char *model;          /* malloc'd ("" if absent) */
  char *topics_json;    /* malloc'd raw JSON array (e.g. "[]") */
} llmcore_text_analysis_result;
void llmcore_text_analysis_result_free(llmcore_text_analysis_result *r);

llmcore_error *llmcore_analyze_text(llmcore_client *c, const char *text,
                                    llmcore_text_analysis_result *out);

/* ===================== Tier-1: sessions & context items ==================== */

/* A session summary. Free with llmcore_session_free. */
typedef struct {
  char *id;                 /* malloc'd */
  char *name;               /* malloc'd ("" if unset) */
  size_t message_count;
  size_t context_item_count;
} llmcore_session;
void llmcore_session_free(llmcore_session *s);

/* name/system_message may be NULL. On success fills *out. */
llmcore_error *llmcore_create_session(llmcore_client *c, const char *name,
                                      const char *system_message, llmcore_session *out);
llmcore_error *llmcore_get_session(llmcore_client *c, const char *session_id,
                                   llmcore_session *out);
/* *out_ids is a malloc'd array of malloc'd session ids; free with
 * llmcore_string_array_free. */
llmcore_error *llmcore_list_sessions(llmcore_client *c, char ***out_ids, size_t *out_n);
llmcore_error *llmcore_delete_session(llmcore_client *c, const char *session_id);
llmcore_error *llmcore_update_session_name(llmcore_client *c, const char *session_id,
                                           const char *new_name);
/* *out_new_id is malloc'd (caller frees). */
llmcore_error *llmcore_fork_session(llmcore_client *c, const char *session_id,
                                    char **out_new_id);
llmcore_error *llmcore_clone_session(llmcore_client *c, const char *session_id,
                                     char **out_new_id);
/* message_ids: array of length n. *out_deleted set to the removed count. */
llmcore_error *llmcore_delete_messages(llmcore_client *c, const char *session_id,
                                       const char *const *message_ids, size_t n,
                                       int *out_deleted);

/* A context item. Free with llmcore_context_item_free. */
typedef struct {
  char *id;      /* malloc'd */
  char *type;    /* malloc'd (ContextItemType value) */
  char *content; /* malloc'd */
} llmcore_context_item;
void llmcore_context_item_free(llmcore_context_item *it);

/* type may be NULL (defaults to "user_text"). *out_item_id is malloc'd. */
llmcore_error *llmcore_add_context_item(llmcore_client *c, const char *session_id,
                                        const char *content, const char *type,
                                        char **out_item_id);
llmcore_error *llmcore_get_context_item(llmcore_client *c, const char *session_id,
                                        const char *item_id, llmcore_context_item *out);
llmcore_error *llmcore_remove_context_item(llmcore_client *c, const char *session_id,
                                           const char *item_id, int *out_removed);

/* ===================== Tier-1: vector store & RAG ========================== */

/* A retrieved document. Free arrays with llmcore_search_results_free. */
typedef struct {
  char *id;      /* malloc'd */
  char *content; /* malloc'd */
  double score;  /* valid only when has_score */
  int has_score;
} llmcore_search_result;
void llmcore_search_results_free(llmcore_search_result *r, size_t n);

/* Each document is sent as {"content": contents[i]}. collection may be NULL.
 * *out_ids is a malloc'd string array (free with llmcore_string_array_free). */
llmcore_error *llmcore_add_documents(llmcore_client *c, const char *const *contents,
                                     size_t n_contents, const char *collection,
                                     char ***out_ids, size_t *out_n);
/* k<=0 lets the bridge default (5). collection may be NULL. */
llmcore_error *llmcore_search_vector_store(llmcore_client *c, const char *query, int k,
                                           const char *collection,
                                           llmcore_search_result **out, size_t *out_n);
llmcore_error *llmcore_list_vector_collections(llmcore_client *c, char ***out, size_t *out_n);
llmcore_error *llmcore_list_rag_collections(llmcore_client *c, char ***out, size_t *out_n);
/* *out_info_json is the raw JSON object string (malloc'd; caller frees). */
llmcore_error *llmcore_get_rag_collection_info(llmcore_client *c, const char *collection,
                                               char **out_info_json);
llmcore_error *llmcore_delete_rag_collection(llmcore_client *c, const char *collection,
                                             int force, int *out_deleted);

/* ===================== Tier-1: context presets ============================= */

/* An input preset entry. `type` is a ContextItemType value; `content` may be NULL. */
typedef struct {
  const char *type;
  const char *content;
} llmcore_preset_item;

/* description may be NULL; items may be NULL when n_items==0. */
llmcore_error *llmcore_save_context_preset(llmcore_client *c, const char *name,
                                           const char *description,
                                           const llmcore_preset_item *items, size_t n_items);

/* A fetched preset summary. Free with llmcore_preset_free. */
typedef struct {
  char *name;        /* malloc'd */
  char *description; /* malloc'd ("" if unset) */
  size_t item_count;
} llmcore_preset;
void llmcore_preset_free(llmcore_preset *p);

llmcore_error *llmcore_get_context_preset(llmcore_client *c, const char *name,
                                          llmcore_preset *out);
/* *out_names is a malloc'd string array of preset names (llmcore_string_array_free). */
llmcore_error *llmcore_list_context_presets(llmcore_client *c, char ***out_names, size_t *out_n);
llmcore_error *llmcore_delete_context_preset(llmcore_client *c, const char *name,
                                             int *out_deleted);

#ifdef __cplusplus
}
#endif

#endif /* LLMCORE_CLIENT_H */
