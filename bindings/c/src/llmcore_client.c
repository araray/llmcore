/* llmcore C client implementation (libcurl + cJSON, HTTP/SSE). */
#include "llmcore_client.h"

#include <stdlib.h>
#include <string.h>

#include <curl/curl.h>

#if defined(__has_include)
#  if __has_include(<cjson/cJSON.h>)
#    include <cjson/cJSON.h>
#  else
#    include <cJSON.h>
#  endif
#else
#  include <cjson/cJSON.h>
#endif

/* ------------------------- small utilities ------------------------- */

struct buf {
  char *p;
  size_t n;
  size_t cap;
};

static int buf_append(struct buf *b, const char *data, size_t n) {
  if (b->n + n > b->cap) {
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < b->n + n) cap *= 2;
    char *q = (char *)realloc(b->p, cap);
    if (!q) return -1;
    b->p = q;
    b->cap = cap;
  }
  memcpy(b->p + b->n, data, n);
  b->n += n;
  return 0;
}

static char *jstrdup(const char *s) {
  if (!s) return NULL;
  size_t n = strlen(s);
  char *q = (char *)malloc(n + 1);
  if (q) memcpy(q, s, n + 1);
  return q;
}

/* Read an integer that protobuf's canonical JSON mapping may encode as a number
 * OR a string: int64/uint64/fixed64 fields are emitted as JSON *strings* (e.g.
 * "8"), so a plain cJSON_IsNumber check misses them. Returns 1 and sets *out on
 * success, 0 otherwise. */
static int json_to_long(const cJSON *node, long *out) {
  if (cJSON_IsNumber(node)) {
    *out = (long)node->valuedouble;
    return 1;
  }
  if (cJSON_IsString(node) && node->valuestring && node->valuestring[0]) {
    char *end = NULL;
    long v = strtol(node->valuestring, &end, 10);
    if (end && *end == '\0') {
      *out = v;
      return 1;
    }
  }
  return 0;
}

static char *join_url(const char *base, const char *path) {
  size_t a = strlen(base), b = strlen(path);
  char *u = (char *)malloc(a + b + 1);
  if (!u) return NULL;
  memcpy(u, base, a);
  memcpy(u + a, path, b + 1);
  return u;
}

struct llmcore_client {
  char *base;
  CURL *curl;
};

static int g_curl_inited = 0;

/* ------------------------- error helpers --------------------------- */

void llmcore_error_free(llmcore_error *err) {
  if (!err) return;
  free(err->category);
  free(err->code);
  free(err->message);
  free(err->provider);
  free(err->model);
  free(err);
}

static llmcore_error *err_local(const char *category, const char *code, const char *message) {
  llmcore_error *e = (llmcore_error *)calloc(1, sizeof(*e));
  if (!e) return NULL;
  e->category = jstrdup(category);
  e->code = jstrdup(code);
  e->message = jstrdup(message ? message : "");
  return e;
}

static llmcore_error *parse_error_object(const cJSON *eo, long fallback_code) {
  llmcore_error *e = (llmcore_error *)calloc(1, sizeof(*e));
  if (!e) return NULL;
  if (!eo) {
    e->category = jstrdup("ERROR_CATEGORY_INTERNAL");
    e->code = jstrdup("http.error");
    e->message = jstrdup("request failed");
    e->http_status = (int)fallback_code;
    return e;
  }
  const cJSON *c;
  c = cJSON_GetObjectItemCaseSensitive(eo, "category");
  e->category = jstrdup(cJSON_IsString(c) ? c->valuestring : "ERROR_CATEGORY_INTERNAL");
  c = cJSON_GetObjectItemCaseSensitive(eo, "code");
  e->code = jstrdup(cJSON_IsString(c) ? c->valuestring : "internal");
  c = cJSON_GetObjectItemCaseSensitive(eo, "message");
  e->message = jstrdup(cJSON_IsString(c) ? c->valuestring : "");
  c = cJSON_GetObjectItemCaseSensitive(eo, "http_status");
  if (cJSON_IsNumber(c)) e->http_status = c->valueint;
  c = cJSON_GetObjectItemCaseSensitive(eo, "retryable");
  e->retryable = cJSON_IsTrue(c) ? 1 : 0;
  c = cJSON_GetObjectItemCaseSensitive(eo, "retry_after_ms");
  if (cJSON_IsNumber(c)) e->retry_after_ms = c->valuedouble;
  c = cJSON_GetObjectItemCaseSensitive(eo, "provider");
  if (cJSON_IsString(c)) e->provider = jstrdup(c->valuestring);
  c = cJSON_GetObjectItemCaseSensitive(eo, "model");
  if (cJSON_IsString(c)) e->model = jstrdup(c->valuestring);
  if (!e->http_status) e->http_status = (int)fallback_code;
  return e;
}

/* Build an error from an HTTP error body of the form {"error": {...}}. */
static llmcore_error *http_error(const char *body, long code) {
  cJSON *root = body ? cJSON_Parse(body) : NULL;
  const cJSON *eo = root ? cJSON_GetObjectItemCaseSensitive(root, "error") : NULL;
  llmcore_error *e = parse_error_object(eo, code);
  cJSON_Delete(root);
  return e;
}

/* ------------------------- HTTP plumbing --------------------------- */

static size_t write_buf(char *ptr, size_t size, size_t nmemb, void *userdata) {
  struct buf *b = (struct buf *)userdata;
  size_t n = size * nmemb;
  if (buf_append(b, ptr, n) != 0) return 0;
  return n;
}

/* POST `body` (JSON) to base+path. On transport failure returns an error;
 * otherwise sets *out_body (NUL-terminated, caller frees) and *out_code. */
static llmcore_error *post_json(llmcore_client *c, const char *path, const char *body,
                                char **out_body, long *out_code) {
  CURL *curl = c->curl;
  curl_easy_reset(curl);
  char *url = join_url(c->base, path);
  if (!url) return err_local("ERROR_CATEGORY_INTERNAL", "oom", "out of memory");
  const char *payload = body ? body : "{}";

  struct buf b;
  memset(&b, 0, sizeof(b));
  struct curl_slist *hdr = NULL;
  hdr = curl_slist_append(hdr, "Content-Type: application/json");

  curl_easy_setopt(curl, CURLOPT_URL, url);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(payload));
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdr);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_buf);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &b);

  CURLcode rc = curl_easy_perform(curl);
  curl_slist_free_all(hdr);
  free(url);

  if (rc != CURLE_OK) {
    free(b.p);
    return err_local("ERROR_CATEGORY_INTERNAL", "transport.error", curl_easy_strerror(rc));
  }
  long code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
  if (buf_append(&b, "", 1) != 0) { /* NUL-terminate */
    free(b.p);
    return err_local("ERROR_CATEGORY_INTERNAL", "oom", "out of memory");
  }
  *out_body = b.p;
  *out_code = code;
  return NULL;
}

/* --------------------------- lifecycle ----------------------------- */

llmcore_client *llmcore_client_new(const char *base_url) {
  if (!base_url) return NULL;
  if (!g_curl_inited) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    g_curl_inited = 1;
  }
  llmcore_client *c = (llmcore_client *)calloc(1, sizeof(*c));
  if (!c) return NULL;
  size_t L = strlen(base_url);
  while (L > 0 && base_url[L - 1] == '/') L--;
  c->base = (char *)malloc(L + 1);
  if (!c->base) {
    free(c);
    return NULL;
  }
  memcpy(c->base, base_url, L);
  c->base[L] = '\0';
  c->curl = curl_easy_init();
  if (!c->curl) {
    free(c->base);
    free(c);
    return NULL;
  }
  return c;
}

void llmcore_client_free(llmcore_client *c) {
  if (!c) return;
  if (c->curl) curl_easy_cleanup(c->curl);
  free(c->base);
  free(c);
}

void llmcore_chat_result_free(llmcore_chat_result *r) {
  if (!r) return;
  free(r->text);
  r->text = NULL;
}

void llmcore_string_array_free(char **items, size_t n) {
  if (!items) return;
  for (size_t i = 0; i < n; i++) free(items[i]);
  free(items);
}

/* --------------------------- endpoints ----------------------------- */

llmcore_error *llmcore_ensure_compatible(llmcore_client *c, const char *const *required_caps,
                                         size_t n_caps) {
  char *resp = NULL;
  long code = 0;
  llmcore_error *e = post_json(c, "/llmcore.v1/ControlService/GetInfo", "{}", &resp, &code);
  if (e) return e;
  if (code >= 400) {
    e = http_error(resp, code);
    free(resp);
    return e;
  }
  cJSON *root = cJSON_Parse(resp);
  free(resp);
  if (!root) return err_local("ERROR_CATEGORY_INTERNAL", "parse.error", "invalid JSON");

  const cJSON *cv = cJSON_GetObjectItemCaseSensitive(root, "contract_version");
  if (!cJSON_IsString(cv) || strcmp(cv->valuestring, "llmcore.v1") != 0) {
    e = err_local("ERROR_CATEGORY_INVALID_ARGUMENT", "contract.mismatch",
                  "server contract != llmcore.v1");
    cJSON_Delete(root);
    return e;
  }
  const cJSON *caps = cJSON_GetObjectItemCaseSensitive(root, "capabilities");
  for (size_t i = 0; i < n_caps; i++) {
    int found = 0;
    if (cJSON_IsArray(caps)) {
      const cJSON *it = NULL;
      cJSON_ArrayForEach(it, caps) {
        if (cJSON_IsString(it) && strcmp(it->valuestring, required_caps[i]) == 0) {
          found = 1;
          break;
        }
      }
    }
    if (!found) {
      e = err_local("ERROR_CATEGORY_UNSUPPORTED", "capability.missing", required_caps[i]);
      cJSON_Delete(root);
      return e;
    }
  }
  cJSON_Delete(root);
  return NULL;
}

llmcore_error *llmcore_chat(llmcore_client *c, const char *message, llmcore_chat_result *out) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "message", message);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);

  char *resp = NULL;
  long code = 0;
  llmcore_error *e = post_json(c, "/llmcore.v1/InferenceService/Chat", body, &resp, &code);
  free(body);
  if (e) return e;
  if (code >= 400) {
    e = http_error(resp, code);
    free(resp);
    return e;
  }
  cJSON *root = cJSON_Parse(resp);
  free(resp);
  if (!root) return err_local("ERROR_CATEGORY_INTERNAL", "parse.error", "invalid JSON");

  memset(out, 0, sizeof(*out));
  out->prompt_tokens = out->completion_tokens = out->total_tokens = -1;
  const cJSON *t = cJSON_GetObjectItemCaseSensitive(root, "text");
  out->text = jstrdup(cJSON_IsString(t) ? t->valuestring : "");
  const cJSON *u = cJSON_GetObjectItemCaseSensitive(root, "usage");
  if (cJSON_IsObject(u)) {
    const cJSON *x;
    x = cJSON_GetObjectItemCaseSensitive(u, "prompt_tokens");
    if (cJSON_IsNumber(x)) out->prompt_tokens = x->valueint;
    x = cJSON_GetObjectItemCaseSensitive(u, "completion_tokens");
    if (cJSON_IsNumber(x)) out->completion_tokens = x->valueint;
    x = cJSON_GetObjectItemCaseSensitive(u, "total_tokens");
    if (cJSON_IsNumber(x)) out->total_tokens = x->valueint;
  }
  cJSON_Delete(root);
  return NULL;
}

llmcore_error *llmcore_count_tokens(llmcore_client *c, const char *text, int *out_tokens) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "text", text);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);

  char *resp = NULL;
  long code = 0;
  llmcore_error *e = post_json(c, "/llmcore.v1/InferenceService/CountTokens", body, &resp, &code);
  free(body);
  if (e) return e;
  if (code >= 400) {
    e = http_error(resp, code);
    free(resp);
    return e;
  }
  cJSON *root = cJSON_Parse(resp);
  free(resp);
  const cJSON *t = root ? cJSON_GetObjectItemCaseSensitive(root, "tokens") : NULL;
  *out_tokens = cJSON_IsNumber(t) ? t->valueint : 0;
  cJSON_Delete(root);
  return NULL;
}

llmcore_error *llmcore_estimate_cost(llmcore_client *c, const char *provider, const char *model,
                                     int prompt_tokens, int completion_tokens,
                                     double *out_total_cost, char **out_currency) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "provider_name", provider);
  cJSON_AddStringToObject(req, "model_name", model);
  cJSON_AddNumberToObject(req, "prompt_tokens", prompt_tokens);
  cJSON_AddNumberToObject(req, "completion_tokens", completion_tokens);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);

  char *resp = NULL;
  long code = 0;
  llmcore_error *e = post_json(c, "/llmcore.v1/InferenceService/EstimateCost", body, &resp, &code);
  free(body);
  if (e) return e;
  if (code >= 400) {
    e = http_error(resp, code);
    free(resp);
    return e;
  }
  cJSON *root = cJSON_Parse(resp);
  free(resp);
  const cJSON *tc = root ? cJSON_GetObjectItemCaseSensitive(root, "total_cost") : NULL;
  const cJSON *cu = root ? cJSON_GetObjectItemCaseSensitive(root, "currency") : NULL;
  if (out_total_cost) *out_total_cost = cJSON_IsNumber(tc) ? tc->valuedouble : 0.0;
  if (out_currency) *out_currency = jstrdup(cJSON_IsString(cu) ? cu->valuestring : "");
  cJSON_Delete(root);
  return NULL;
}

llmcore_error *llmcore_embed(llmcore_client *c, const char *const *inputs, size_t n_inputs) {
  cJSON *req = cJSON_CreateObject();
  cJSON *arr = cJSON_AddArrayToObject(req, "input");
  for (size_t i = 0; i < n_inputs; i++)
    cJSON_AddItemToArray(arr, cJSON_CreateString(inputs[i]));
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);

  char *resp = NULL;
  long code = 0;
  llmcore_error *e = post_json(c, "/llmcore.v1/InferenceService/Embed", body, &resp, &code);
  free(body);
  if (e) return e;
  if (code >= 400) { /* expected: 501 UNSUPPORTED */
    e = http_error(resp, code);
    free(resp);
    return e;
  }
  free(resp);
  return err_local("ERROR_CATEGORY_INTERNAL", "unexpected", "embed unexpectedly succeeded");
}

static llmcore_error *get_string_array(llmcore_client *c, const char *path, const char *body,
                                       const char *field, char ***out_items, size_t *out_n) {
  char *resp = NULL;
  long code = 0;
  llmcore_error *e = post_json(c, path, body, &resp, &code);
  if (e) return e;
  if (code >= 400) {
    e = http_error(resp, code);
    free(resp);
    return e;
  }
  cJSON *root = cJSON_Parse(resp);
  free(resp);
  const cJSON *arr = root ? cJSON_GetObjectItemCaseSensitive(root, field) : NULL;
  size_t n = cJSON_IsArray(arr) ? (size_t)cJSON_GetArraySize(arr) : 0;
  char **items = n ? (char **)calloc(n, sizeof(char *)) : NULL;
  for (size_t i = 0; i < n; i++) {
    const cJSON *it = cJSON_GetArrayItem(arr, (int)i);
    items[i] = jstrdup(cJSON_IsString(it) ? it->valuestring : "");
  }
  *out_items = items;
  *out_n = n;
  cJSON_Delete(root);
  return NULL;
}

llmcore_error *llmcore_list_providers(llmcore_client *c, char ***out_items, size_t *out_n) {
  return get_string_array(c, "/llmcore.v1/CatalogService/ListProviders", "{}", "providers",
                          out_items, out_n);
}

llmcore_error *llmcore_list_models(llmcore_client *c, const char *provider, char ***out_items,
                                   size_t *out_n) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "provider_name", provider);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);
  llmcore_error *e =
      get_string_array(c, "/llmcore.v1/CatalogService/ListModels", body, "models", out_items, out_n);
  free(body);
  return e;
}

llmcore_error *llmcore_health(llmcore_client *c, int *out_ok) {
  char *resp = NULL;
  long code = 0;
  llmcore_error *e = post_json(c, "/healthz", "{}", &resp, &code);
  if (e) return e;
  if (code >= 400) {
    e = http_error(resp, code);
    free(resp);
    return e;
  }
  cJSON *root = cJSON_Parse(resp);
  free(resp);
  const cJSON *ok = root ? cJSON_GetObjectItemCaseSensitive(root, "ok") : NULL;
  if (out_ok) *out_ok = cJSON_IsTrue(ok) ? 1 : 0;
  cJSON_Delete(root);
  return NULL;
}

/* --------------------------- SSE stream ---------------------------- */

struct sse_ctx {
  struct buf accum;
  llmcore_chunk_cb cb;
  void *user;
  int cancelled;
  llmcore_error *err;
};

/* Process one SSE block [0,blen) (no trailing "\n\n"). Returns 1 to abort. */
static int sse_handle_block(struct sse_ctx *s, const char *base, size_t blen) {
  char event[32] = "message";
  struct buf data;
  memset(&data, 0, sizeof(data));

  size_t p = 0;
  while (p < blen) {
    size_t q = p;
    while (q < blen && base[q] != '\n') q++;
    size_t linelen = q - p;
    if (linelen >= 6 && strncmp(base + p, "event:", 6) == 0) {
      size_t s0 = p + 6;
      while (s0 < q && base[s0] == ' ') s0++;
      size_t el = q - s0;
      if (el > sizeof(event) - 1) el = sizeof(event) - 1;
      memcpy(event, base + s0, el);
      event[el] = '\0';
    } else if (linelen >= 5 && strncmp(base + p, "data:", 5) == 0) {
      size_t s0 = p + 5;
      while (s0 < q && base[s0] == ' ') s0++;
      buf_append(&data, base + s0, q - s0);
    }
    p = q + 1;
  }

  int abort_now = 0;
  if (data.n > 0) {
    buf_append(&data, "", 1); /* NUL */
    cJSON *obj = cJSON_Parse(data.p);
    if (strcmp(event, "error") == 0) {
      long fb = 500;
      const cJSON *hs = obj ? cJSON_GetObjectItemCaseSensitive(obj, "http_status") : NULL;
      if (cJSON_IsNumber(hs)) fb = hs->valueint;
      s->err = parse_error_object(obj, fb);
      abort_now = 1;
    } else {
      const cJSON *t = obj ? cJSON_GetObjectItemCaseSensitive(obj, "text") : NULL;
      int done = (strcmp(event, "done") == 0) ? 1 : 0;
      int rc = s->cb(cJSON_IsString(t) ? t->valuestring : "", done, s->user);
      if (rc != 0) {
        s->cancelled = 1;
        abort_now = 1;
      }
    }
    cJSON_Delete(obj);
  }
  free(data.p);
  return abort_now;
}

static size_t write_sse(char *ptr, size_t size, size_t nmemb, void *userdata) {
  struct sse_ctx *s = (struct sse_ctx *)userdata;
  size_t n = size * nmemb;
  if (buf_append(&s->accum, ptr, n) != 0) return 0;

  for (;;) {
    char *base = s->accum.p;
    size_t L = s->accum.n;
    long sep = -1;
    for (size_t i = 0; i + 1 < L; i++) {
      if (base[i] == '\n' && base[i + 1] == '\n') {
        sep = (long)i;
        break;
      }
    }
    if (sep < 0) break;
    size_t blen = (size_t)sep;
    int do_abort = sse_handle_block(s, base, blen);
    size_t consume = blen + 2;
    memmove(s->accum.p, s->accum.p + consume, s->accum.n - consume);
    s->accum.n -= consume;
    if (do_abort) return 0; /* signal CURLE_WRITE_ERROR */
  }
  return n;
}

llmcore_error *llmcore_chat_stream(llmcore_client *c, const char *message,
                                   llmcore_chunk_cb cb, void *user) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "message", message);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);

  CURL *curl = c->curl;
  curl_easy_reset(curl);
  char *url = join_url(c->base, "/llmcore.v1/InferenceService/ChatStream");
  struct curl_slist *hdr = curl_slist_append(NULL, "Content-Type: application/json");

  struct sse_ctx s;
  memset(&s, 0, sizeof(s));
  s.cb = cb;
  s.user = user;

  curl_easy_setopt(curl, CURLOPT_URL, url);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(body));
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdr);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_sse);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &s);

  CURLcode rc = curl_easy_perform(curl);
  curl_slist_free_all(hdr);
  free(url);
  free(body);

  long code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);

  llmcore_error *err = NULL;
  if (s.err) {
    err = s.err; /* mid-stream error event */
  } else if (rc == CURLE_WRITE_ERROR && s.cancelled) {
    err = NULL; /* user-requested cancel -> success */
  } else if (code >= 400) {
    buf_append(&s.accum, "", 1);
    err = http_error(s.accum.p, code);
  } else if (rc == CURLE_WRITE_ERROR) {
    err = err_local("ERROR_CATEGORY_INTERNAL", "stream.parse", "SSE parse/write error");
  } else if (rc != CURLE_OK) {
    err = err_local("ERROR_CATEGORY_INTERNAL", "transport.error", curl_easy_strerror(rc));
  }
  free(s.accum.p);
  return err;
}

/* ===================== Audio (Tier 2) — unary RPCs ===================== */

static const char B64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/* base64-encode `n` bytes; returns a malloc'd NUL-terminated string. */
static char *b64_encode(const unsigned char *data, size_t n) {
  size_t out_len = 4 * ((n + 2) / 3);
  char *out = (char *)malloc(out_len + 1);
  if (!out) return NULL;
  size_t i = 0, j = 0;
  for (; i + 2 < n; i += 3) {
    unsigned int v = ((unsigned int)data[i] << 16) | ((unsigned int)data[i + 1] << 8) |
                     (unsigned int)data[i + 2];
    out[j++] = B64[(v >> 18) & 0x3F];
    out[j++] = B64[(v >> 12) & 0x3F];
    out[j++] = B64[(v >> 6) & 0x3F];
    out[j++] = B64[v & 0x3F];
  }
  if (i < n) {
    int rem = (int)(n - i); /* 1 or 2 */
    unsigned int v = (unsigned int)data[i] << 16;
    if (rem == 2) v |= (unsigned int)data[i + 1] << 8;
    out[j++] = B64[(v >> 18) & 0x3F];
    out[j++] = B64[(v >> 12) & 0x3F];
    out[j++] = (rem == 2) ? B64[(v >> 6) & 0x3F] : '=';
    out[j++] = '=';
  }
  out[j] = '\0';
  return out;
}

static int b64val(char ch) {
  if (ch == '\0' || ch == '=') return -1;
  const char *p = strchr(B64, ch);
  return p ? (int)(p - B64) : -1;
}

/* base64-decode a NUL-terminated string; sets *out_len. Returns malloc'd bytes
 * (NULL on allocation failure). Non-base64 chars (e.g. newlines) are skipped. */
static unsigned char *b64_decode(const char *s, size_t *out_len) {
  size_t slen = strlen(s);
  unsigned char *out = (unsigned char *)malloc(slen / 4 * 3 + 3);
  if (!out) {
    *out_len = 0;
    return NULL;
  }
  size_t o = 0;
  int buf = 0, bits = 0;
  for (size_t i = 0; i < slen; i++) {
    if (s[i] == '=') break;
    int v = b64val(s[i]);
    if (v < 0) continue;
    buf = (buf << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out[o++] = (unsigned char)((buf >> bits) & 0xFF);
    }
  }
  *out_len = o;
  return out;
}

/* Parse the leading status checks shared by the audio unary calls: post, map an
 * HTTP error, then parse the JSON body. On success returns NULL and sets *root
 * (caller cJSON_Delete). On failure returns the error (and *root is NULL). */
static llmcore_error *audio_post(llmcore_client *c, const char *path, char *body,
                                 cJSON **root) {
  *root = NULL;
  char *resp = NULL;
  long code = 0;
  llmcore_error *e = post_json(c, path, body, &resp, &code);
  free(body);
  if (e) return e;
  if (code >= 400) {
    e = http_error(resp, code);
    free(resp);
    return e;
  }
  *root = cJSON_Parse(resp);
  free(resp);
  if (!*root) return err_local("ERROR_CATEGORY_INTERNAL", "parse.error", "invalid JSON");
  return NULL;
}

void llmcore_speech_result_free(llmcore_speech_result *r) {
  if (!r) return;
  free(r->audio);
  free(r->format);
  free(r->model);
  free(r->voice);
}

llmcore_error *llmcore_synthesize(llmcore_client *c, const char *text,
                                  llmcore_speech_result *out) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "text", text);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);

  cJSON *root = NULL;
  llmcore_error *e =
      audio_post(c, "/llmcore.v1/AudioService/Synthesize", body, &root);
  if (e) return e;

  memset(out, 0, sizeof(*out));
  const cJSON *a = cJSON_GetObjectItemCaseSensitive(root, "audio_data");
  if (cJSON_IsString(a)) out->audio = b64_decode(a->valuestring, &out->audio_len);
  const cJSON *f = cJSON_GetObjectItemCaseSensitive(root, "format");
  out->format = jstrdup(cJSON_IsString(f) ? f->valuestring : "");
  const cJSON *m = cJSON_GetObjectItemCaseSensitive(root, "model");
  out->model = jstrdup(cJSON_IsString(m) ? m->valuestring : "");
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(root, "voice");
  out->voice = jstrdup(cJSON_IsString(v) ? v->valuestring : "");
  cJSON_Delete(root);
  return NULL;
}

void llmcore_transcription_result_free(llmcore_transcription_result *r) {
  if (!r) return;
  free(r->text);
  free(r->language);
  free(r->model);
}

llmcore_error *llmcore_transcribe(llmcore_client *c, const unsigned char *audio,
                                  size_t audio_len, llmcore_transcription_result *out) {
  char *b64 = b64_encode(audio, audio_len);
  if (!b64) return err_local("ERROR_CATEGORY_INTERNAL", "oom", "out of memory");
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "audio_data", b64);
  free(b64);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);

  cJSON *root = NULL;
  llmcore_error *e =
      audio_post(c, "/llmcore.v1/AudioService/Transcribe", body, &root);
  if (e) return e;

  memset(out, 0, sizeof(*out));
  const cJSON *t = cJSON_GetObjectItemCaseSensitive(root, "text");
  out->text = jstrdup(cJSON_IsString(t) ? t->valuestring : "");
  const cJSON *l = cJSON_GetObjectItemCaseSensitive(root, "language");
  out->language = jstrdup(cJSON_IsString(l) ? l->valuestring : "");
  const cJSON *m = cJSON_GetObjectItemCaseSensitive(root, "model");
  out->model = jstrdup(cJSON_IsString(m) ? m->valuestring : "");
  cJSON_Delete(root);
  return NULL;
}

void llmcore_image_result_free(llmcore_image_result *r) {
  if (!r) return;
  if (r->images) {
    for (size_t i = 0; i < r->n_images; i++) free(r->images[i]);
    free(r->images);
  }
  free(r->model);
}

llmcore_error *llmcore_generate_image(llmcore_client *c, const char *prompt, int n,
                                      llmcore_image_result *out) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "prompt", prompt);
  cJSON_AddNumberToObject(req, "n", n);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);

  cJSON *root = NULL;
  llmcore_error *e =
      audio_post(c, "/llmcore.v1/AudioService/GenerateImage", body, &root);
  if (e) return e;

  memset(out, 0, sizeof(*out));
  const cJSON *imgs = cJSON_GetObjectItemCaseSensitive(root, "images");
  if (cJSON_IsArray(imgs)) {
    size_t n_imgs = (size_t)cJSON_GetArraySize(imgs);
    if (n_imgs > 0) {
      out->images = (char **)calloc(n_imgs, sizeof(char *));
      if (!out->images) {
        cJSON_Delete(root);
        return err_local("ERROR_CATEGORY_INTERNAL", "oom", "out of memory");
      }
      size_t k = 0;
      const cJSON *it;
      cJSON_ArrayForEach(it, imgs) {
        const cJSON *d = cJSON_GetObjectItemCaseSensitive(it, "data");
        out->images[k++] = jstrdup(cJSON_IsString(d) ? d->valuestring : "");
      }
      out->n_images = k;
    }
  }
  const cJSON *m = cJSON_GetObjectItemCaseSensitive(root, "model");
  out->model = jstrdup(cJSON_IsString(m) ? m->valuestring : "");
  cJSON_Delete(root);
  return NULL;
}

void llmcore_ocr_result_free(llmcore_ocr_result *r) {
  if (!r) return;
  free(r->model);
  free(r->pages_json);
}

llmcore_error *llmcore_ocr(llmcore_client *c, const unsigned char *data, size_t data_len,
                           const char *url, llmcore_ocr_result *out) {
  cJSON *req = cJSON_CreateObject();
  if (url) {
    cJSON_AddStringToObject(req, "url", url);
  } else {
    char *b64 = b64_encode(data, data_len);
    if (!b64) {
      cJSON_Delete(req);
      return err_local("ERROR_CATEGORY_INTERNAL", "oom", "out of memory");
    }
    cJSON_AddStringToObject(req, "data", b64);
    free(b64);
  }
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);

  cJSON *root = NULL;
  llmcore_error *e = audio_post(c, "/llmcore.v1/AudioService/Ocr", body, &root);
  if (e) return e;

  memset(out, 0, sizeof(*out));
  const cJSON *m = cJSON_GetObjectItemCaseSensitive(root, "model");
  out->model = jstrdup(cJSON_IsString(m) ? m->valuestring : "");
  const cJSON *pp = cJSON_GetObjectItemCaseSensitive(root, "pages_processed");
  if (cJSON_IsNumber(pp)) out->pages_processed = pp->valueint;
  const cJSON *ds = cJSON_GetObjectItemCaseSensitive(root, "doc_size_bytes");
  /* doc_size_bytes is int64 → protobuf JSON encodes it as a string. */
  if (json_to_long(ds, &out->doc_size_bytes)) {
    out->has_doc_size_bytes = 1;
  }
  const cJSON *pages = cJSON_GetObjectItemCaseSensitive(root, "pages");
  out->pages_json = cJSON_IsArray(pages) ? cJSON_PrintUnformatted(pages) : jstrdup("[]");
  cJSON_Delete(root);
  return NULL;
}

void llmcore_text_analysis_result_free(llmcore_text_analysis_result *r) {
  if (!r) return;
  free(r->summary);
  free(r->language);
  free(r->model);
  free(r->topics_json);
}

llmcore_error *llmcore_analyze_text(llmcore_client *c, const char *text,
                                    llmcore_text_analysis_result *out) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "text", text);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);

  cJSON *root = NULL;
  llmcore_error *e =
      audio_post(c, "/llmcore.v1/AudioService/AnalyzeText", body, &root);
  if (e) return e;

  memset(out, 0, sizeof(*out));
  const cJSON *s = cJSON_GetObjectItemCaseSensitive(root, "summary");
  if (cJSON_IsString(s)) {
    out->summary = jstrdup(s->valuestring);
    out->has_summary = 1;
  }
  const cJSON *l = cJSON_GetObjectItemCaseSensitive(root, "language");
  out->language = jstrdup(cJSON_IsString(l) ? l->valuestring : "");
  const cJSON *m = cJSON_GetObjectItemCaseSensitive(root, "model");
  out->model = jstrdup(cJSON_IsString(m) ? m->valuestring : "");
  const cJSON *topics = cJSON_GetObjectItemCaseSensitive(root, "topics");
  out->topics_json = cJSON_IsArray(topics) ? cJSON_PrintUnformatted(topics) : jstrdup("[]");
  cJSON_Delete(root);
  return NULL;
}

/* ===================== Tier-1: sessions, vector, presets =================== */

/* POST `body` to `path`; on HTTP success parse the JSON body into *out_root
 * (caller cJSON_Delete). HTTP>=400 becomes a structured error. */
static llmcore_error *post_parse(llmcore_client *c, const char *path, const char *body,
                                 cJSON **out_root) {
  char *resp = NULL;
  long code = 0;
  llmcore_error *e = post_json(c, path, body, &resp, &code);
  if (e) return e;
  if (code >= 400) {
    e = http_error(resp, code);
    free(resp);
    return e;
  }
  cJSON *root = cJSON_Parse(resp);
  free(resp);
  if (!root) return err_local("ERROR_CATEGORY_INTERNAL", "decode", "invalid JSON response");
  *out_root = root;
  return NULL;
}

/* POST and discard the (empty) JSON body; used by void-returning RPCs. */
static llmcore_error *post_discard(llmcore_client *c, const char *path, const char *body) {
  cJSON *root = NULL;
  llmcore_error *e = post_parse(c, path, body, &root);
  if (e) return e;
  cJSON_Delete(root);
  return NULL;
}

/* Extract a malloc'd string field (returns jstrdup("") when absent). */
static char *get_str(const cJSON *o, const char *key) {
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(o, key);
  return jstrdup(cJSON_IsString(v) ? v->valuestring : "");
}

static size_t count_array(const cJSON *o, const char *key) {
  const cJSON *a = cJSON_GetObjectItemCaseSensitive(o, key);
  return cJSON_IsArray(a) ? (size_t)cJSON_GetArraySize(a) : 0;
}

static void parse_session(const cJSON *o, llmcore_session *out) {
  out->id = get_str(o, "id");
  out->name = get_str(o, "name");
  out->message_count = count_array(o, "messages");
  out->context_item_count = count_array(o, "context_items");
}

void llmcore_session_free(llmcore_session *s) {
  if (!s) return;
  free(s->id);
  free(s->name);
  s->id = s->name = NULL;
  s->message_count = s->context_item_count = 0;
}

void llmcore_context_item_free(llmcore_context_item *it) {
  if (!it) return;
  free(it->id);
  free(it->type);
  free(it->content);
  it->id = it->type = it->content = NULL;
}

void llmcore_search_results_free(llmcore_search_result *r, size_t n) {
  if (!r) return;
  for (size_t i = 0; i < n; i++) {
    free(r[i].id);
    free(r[i].content);
  }
  free(r);
}

void llmcore_preset_free(llmcore_preset *p) {
  if (!p) return;
  free(p->name);
  free(p->description);
  p->name = p->description = NULL;
  p->item_count = 0;
}

/* -- sessions ------------------------------------------------------------- */

llmcore_error *llmcore_create_session(llmcore_client *c, const char *name,
                                      const char *system_message, llmcore_session *out) {
  cJSON *req = cJSON_CreateObject();
  if (name) cJSON_AddStringToObject(req, "name", name);
  if (system_message) cJSON_AddStringToObject(req, "system_message", system_message);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);
  cJSON *root = NULL;
  llmcore_error *e = post_parse(c, "/llmcore.v1/SessionService/CreateSession", body, &root);
  free(body);
  if (e) return e;
  parse_session(root, out);
  cJSON_Delete(root);
  return NULL;
}

llmcore_error *llmcore_get_session(llmcore_client *c, const char *session_id,
                                   llmcore_session *out) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "session_id", session_id);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);
  cJSON *root = NULL;
  llmcore_error *e = post_parse(c, "/llmcore.v1/SessionService/GetSession", body, &root);
  free(body);
  if (e) return e;
  parse_session(root, out);
  cJSON_Delete(root);
  return NULL;
}

llmcore_error *llmcore_list_sessions(llmcore_client *c, char ***out_ids, size_t *out_n) {
  cJSON *root = NULL;
  llmcore_error *e = post_parse(c, "/llmcore.v1/SessionService/ListSessions", "{}", &root);
  if (e) return e;
  const cJSON *arr = cJSON_GetObjectItemCaseSensitive(root, "sessions");
  size_t n = cJSON_IsArray(arr) ? (size_t)cJSON_GetArraySize(arr) : 0;
  char **items = n ? (char **)calloc(n, sizeof(char *)) : NULL;
  for (size_t i = 0; i < n; i++) {
    items[i] = get_str(cJSON_GetArrayItem(arr, (int)i), "id");
  }
  *out_ids = items;
  *out_n = n;
  cJSON_Delete(root);
  return NULL;
}

llmcore_error *llmcore_delete_session(llmcore_client *c, const char *session_id) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "session_id", session_id);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);
  llmcore_error *e = post_discard(c, "/llmcore.v1/SessionService/DeleteSession", body);
  free(body);
  return e;
}

llmcore_error *llmcore_update_session_name(llmcore_client *c, const char *session_id,
                                           const char *new_name) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "session_id", session_id);
  cJSON_AddStringToObject(req, "new_name", new_name);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);
  llmcore_error *e = post_discard(c, "/llmcore.v1/SessionService/UpdateSessionName", body);
  free(body);
  return e;
}

/* Shared by Fork/Clone: both return {"session_id": "<new id>"}. */
static llmcore_error *session_returning_id(llmcore_client *c, const char *path,
                                           const char *session_id, char **out_new_id) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "session_id", session_id);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);
  cJSON *root = NULL;
  llmcore_error *e = post_parse(c, path, body, &root);
  free(body);
  if (e) return e;
  *out_new_id = get_str(root, "session_id");
  cJSON_Delete(root);
  return NULL;
}

llmcore_error *llmcore_fork_session(llmcore_client *c, const char *session_id,
                                    char **out_new_id) {
  return session_returning_id(c, "/llmcore.v1/SessionService/ForkSession", session_id,
                              out_new_id);
}

llmcore_error *llmcore_clone_session(llmcore_client *c, const char *session_id,
                                     char **out_new_id) {
  return session_returning_id(c, "/llmcore.v1/SessionService/CloneSession", session_id,
                              out_new_id);
}

llmcore_error *llmcore_delete_messages(llmcore_client *c, const char *session_id,
                                       const char *const *message_ids, size_t n,
                                       int *out_deleted) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "session_id", session_id);
  cJSON *ids = cJSON_AddArrayToObject(req, "message_ids");
  for (size_t i = 0; i < n; i++) cJSON_AddItemToArray(ids, cJSON_CreateString(message_ids[i]));
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);
  cJSON *root = NULL;
  llmcore_error *e = post_parse(c, "/llmcore.v1/SessionService/DeleteMessages", body, &root);
  free(body);
  if (e) return e;
  const cJSON *dc = cJSON_GetObjectItemCaseSensitive(root, "deleted_count");
  *out_deleted = cJSON_IsNumber(dc) ? dc->valueint : 0;
  cJSON_Delete(root);
  return NULL;
}

llmcore_error *llmcore_add_context_item(llmcore_client *c, const char *session_id,
                                        const char *content, const char *type,
                                        char **out_item_id) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "session_id", session_id);
  cJSON_AddStringToObject(req, "content", content);
  if (type) cJSON_AddStringToObject(req, "type", type);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);
  cJSON *root = NULL;
  llmcore_error *e = post_parse(c, "/llmcore.v1/SessionService/AddContextItem", body, &root);
  free(body);
  if (e) return e;
  *out_item_id = get_str(root, "item_id");
  cJSON_Delete(root);
  return NULL;
}

llmcore_error *llmcore_get_context_item(llmcore_client *c, const char *session_id,
                                        const char *item_id, llmcore_context_item *out) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "session_id", session_id);
  cJSON_AddStringToObject(req, "item_id", item_id);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);
  cJSON *root = NULL;
  llmcore_error *e = post_parse(c, "/llmcore.v1/SessionService/GetContextItem", body, &root);
  free(body);
  if (e) return e;
  out->id = get_str(root, "id");
  out->type = get_str(root, "type");
  out->content = get_str(root, "content");
  cJSON_Delete(root);
  return NULL;
}

llmcore_error *llmcore_remove_context_item(llmcore_client *c, const char *session_id,
                                           const char *item_id, int *out_removed) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "session_id", session_id);
  cJSON_AddStringToObject(req, "item_id", item_id);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);
  cJSON *root = NULL;
  llmcore_error *e = post_parse(c, "/llmcore.v1/SessionService/RemoveContextItem", body, &root);
  free(body);
  if (e) return e;
  const cJSON *r = cJSON_GetObjectItemCaseSensitive(root, "removed");
  *out_removed = cJSON_IsTrue(r) ? 1 : 0;
  cJSON_Delete(root);
  return NULL;
}

/* -- vector store & RAG --------------------------------------------------- */

llmcore_error *llmcore_add_documents(llmcore_client *c, const char *const *contents,
                                     size_t n_contents, const char *collection,
                                     char ***out_ids, size_t *out_n) {
  cJSON *req = cJSON_CreateObject();
  cJSON *docs = cJSON_AddArrayToObject(req, "documents");
  for (size_t i = 0; i < n_contents; i++) {
    cJSON *d = cJSON_CreateObject();
    cJSON_AddStringToObject(d, "content", contents[i]);
    cJSON_AddItemToArray(docs, d);
  }
  if (collection) cJSON_AddStringToObject(req, "collection_name", collection);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);
  llmcore_error *e =
      get_string_array(c, "/llmcore.v1/VectorService/AddDocuments", body, "ids", out_ids, out_n);
  free(body);
  return e;
}

llmcore_error *llmcore_search_vector_store(llmcore_client *c, const char *query, int k,
                                           const char *collection,
                                           llmcore_search_result **out, size_t *out_n) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "query", query);
  if (k > 0) cJSON_AddNumberToObject(req, "k", k);
  if (collection) cJSON_AddStringToObject(req, "collection_name", collection);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);
  cJSON *root = NULL;
  llmcore_error *e = post_parse(c, "/llmcore.v1/VectorService/SearchVectorStore", body, &root);
  free(body);
  if (e) return e;
  const cJSON *arr = cJSON_GetObjectItemCaseSensitive(root, "documents");
  size_t n = cJSON_IsArray(arr) ? (size_t)cJSON_GetArraySize(arr) : 0;
  llmcore_search_result *res = n ? (llmcore_search_result *)calloc(n, sizeof(*res)) : NULL;
  for (size_t i = 0; i < n; i++) {
    const cJSON *d = cJSON_GetArrayItem(arr, (int)i);
    res[i].id = get_str(d, "id");
    res[i].content = get_str(d, "content");
    const cJSON *sc = cJSON_GetObjectItemCaseSensitive(d, "score");
    if (cJSON_IsNumber(sc)) {
      res[i].score = sc->valuedouble;
      res[i].has_score = 1;
    }
  }
  *out = res;
  *out_n = n;
  cJSON_Delete(root);
  return NULL;
}

llmcore_error *llmcore_list_vector_collections(llmcore_client *c, char ***out, size_t *out_n) {
  return get_string_array(c, "/llmcore.v1/VectorService/ListVectorCollections", "{}",
                          "collections", out, out_n);
}

llmcore_error *llmcore_list_rag_collections(llmcore_client *c, char ***out, size_t *out_n) {
  return get_string_array(c, "/llmcore.v1/VectorService/ListRagCollections", "{}",
                          "collections", out, out_n);
}

llmcore_error *llmcore_get_rag_collection_info(llmcore_client *c, const char *collection,
                                               char **out_info_json) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "collection_name", collection);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);
  cJSON *root = NULL;
  llmcore_error *e = post_parse(c, "/llmcore.v1/VectorService/GetRagCollectionInfo", body, &root);
  free(body);
  if (e) return e;
  const cJSON *info = cJSON_GetObjectItemCaseSensitive(root, "info");
  *out_info_json = info ? cJSON_PrintUnformatted(info) : jstrdup("{}");
  cJSON_Delete(root);
  return NULL;
}

llmcore_error *llmcore_delete_rag_collection(llmcore_client *c, const char *collection,
                                             int force, int *out_deleted) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "collection_name", collection);
  if (force) cJSON_AddBoolToObject(req, "force", 1);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);
  cJSON *root = NULL;
  llmcore_error *e = post_parse(c, "/llmcore.v1/VectorService/DeleteRagCollection", body, &root);
  free(body);
  if (e) return e;
  const cJSON *d = cJSON_GetObjectItemCaseSensitive(root, "deleted");
  *out_deleted = cJSON_IsTrue(d) ? 1 : 0;
  cJSON_Delete(root);
  return NULL;
}

/* -- context presets ------------------------------------------------------ */

llmcore_error *llmcore_save_context_preset(llmcore_client *c, const char *name,
                                           const char *description,
                                           const llmcore_preset_item *items, size_t n_items) {
  cJSON *req = cJSON_CreateObject();
  cJSON *preset = cJSON_AddObjectToObject(req, "preset");
  cJSON_AddStringToObject(preset, "name", name);
  if (description) cJSON_AddStringToObject(preset, "description", description);
  cJSON *arr = cJSON_AddArrayToObject(preset, "items");
  for (size_t i = 0; i < n_items; i++) {
    cJSON *it = cJSON_CreateObject();
    cJSON_AddStringToObject(it, "type", items[i].type ? items[i].type : "preset_text_content");
    if (items[i].content) cJSON_AddStringToObject(it, "content", items[i].content);
    cJSON_AddItemToArray(arr, it);
  }
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);
  llmcore_error *e = post_discard(c, "/llmcore.v1/PresetService/SaveContextPreset", body);
  free(body);
  return e;
}

llmcore_error *llmcore_get_context_preset(llmcore_client *c, const char *name,
                                          llmcore_preset *out) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "preset_name", name);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);
  cJSON *root = NULL;
  llmcore_error *e = post_parse(c, "/llmcore.v1/PresetService/GetContextPreset", body, &root);
  free(body);
  if (e) return e;
  out->name = get_str(root, "name");
  out->description = get_str(root, "description");
  out->item_count = count_array(root, "items");
  cJSON_Delete(root);
  return NULL;
}

llmcore_error *llmcore_list_context_presets(llmcore_client *c, char ***out_names, size_t *out_n) {
  cJSON *root = NULL;
  llmcore_error *e = post_parse(c, "/llmcore.v1/PresetService/ListContextPresets", "{}", &root);
  if (e) return e;
  const cJSON *arr = cJSON_GetObjectItemCaseSensitive(root, "presets");
  size_t n = cJSON_IsArray(arr) ? (size_t)cJSON_GetArraySize(arr) : 0;
  char **items = n ? (char **)calloc(n, sizeof(char *)) : NULL;
  for (size_t i = 0; i < n; i++) {
    items[i] = get_str(cJSON_GetArrayItem(arr, (int)i), "name");
  }
  *out_names = items;
  *out_n = n;
  cJSON_Delete(root);
  return NULL;
}

llmcore_error *llmcore_delete_context_preset(llmcore_client *c, const char *name,
                                             int *out_deleted) {
  cJSON *req = cJSON_CreateObject();
  cJSON_AddStringToObject(req, "preset_name", name);
  char *body = cJSON_PrintUnformatted(req);
  cJSON_Delete(req);
  cJSON *root = NULL;
  llmcore_error *e = post_parse(c, "/llmcore.v1/PresetService/DeleteContextPreset", body, &root);
  free(body);
  if (e) return e;
  const cJSON *d = cJSON_GetObjectItemCaseSensitive(root, "deleted");
  *out_deleted = cJSON_IsTrue(d) ? 1 : 0;
  cJSON_Delete(root);
  return NULL;
}
