/* Tier-1 demo: sessions, context items, vector store, and context presets,
 * driven over the bridge's HTTP/JSON transport.
 *
 *   LLMCORE_BRIDGE_FAKE=1 LLMCORE_BRIDGE_FAKE_SESSIONS=1 LLMCORE_BRIDGE_FAKE_VECTOR=1 \
 *     python -m llmcore.bridge.cli serve --transport http \
 *     --http-address 127.0.0.1:50152 --insecure
 *   ./sessions            # or set LLMCORE_HTTP=http://host:port
 */
#include <stdio.h>
#include <stdlib.h>

#include "llmcore_client.h"

/* Print + free an error, returning 1 so callers can `if (fail(e)) goto done;`. */
static int fail(llmcore_error *e) {
  if (!e) return 0;
  fprintf(stderr, "error: %s %s — %s\n", e->category, e->code, e->message);
  llmcore_error_free(e);
  return 1;
}

int main(void) {
  const char *base = getenv("LLMCORE_HTTP");
  if (!base) base = "http://127.0.0.1:50152";

  llmcore_client *c = llmcore_client_new(base);
  if (!c) {
    fprintf(stderr, "client alloc failed\n");
    return 1;
  }
  int rc = 1;

  const char *caps[] = {"tier1.sessions", "tier1.vector"};
  if (fail(llmcore_ensure_compatible(c, caps, 2))) goto done;

  /* ---- sessions + context items ---- */
  llmcore_session s;
  if (fail(llmcore_create_session(c, "demo-session", "You are a terse assistant.", &s)))
    goto done;
  printf("created session %s (%zu message[s])\n", s.id, s.message_count);

  char *item_id = NULL;
  if (fail(llmcore_add_context_item(c, s.id, "Remember: the launch date is June 30.",
                                    "user_text", &item_id))) {
    llmcore_session_free(&s);
    goto done;
  }
  llmcore_context_item it;
  if (!fail(llmcore_get_context_item(c, s.id, item_id, &it))) {
    printf("context item [%s]: \"%s\"\n", it.type, it.content);
    llmcore_context_item_free(&it);
  }
  free(item_id);

  /* ---- vector store ---- */
  const char *docs[] = {"Paris is the capital of France."};
  char **ids = NULL;
  size_t n_ids = 0;
  if (!fail(llmcore_add_documents(c, docs, 1, NULL, &ids, &n_ids))) {
    printf("indexed %zu document[s]\n", n_ids);
    llmcore_string_array_free(ids, n_ids);
  }
  llmcore_search_result *res = NULL;
  size_t n_res = 0;
  if (!fail(llmcore_search_vector_store(c, "capital of France", 3, NULL, &res, &n_res))) {
    for (size_t i = 0; i < n_res; i++)
      printf("  hit (score=%.3f): \"%s\"\n", res[i].score, res[i].content);
    llmcore_search_results_free(res, n_res);
  }

  /* ---- context presets ---- */
  llmcore_preset_item pitems[] = {{"preset_text_content", "Always cite sources."}};
  if (!fail(llmcore_save_context_preset(c, "preamble", "Standard preamble", pitems, 1))) {
    llmcore_preset p;
    if (!fail(llmcore_get_context_preset(c, "preamble", &p))) {
      printf("preset \"%s\" has %zu item[s]\n", p.name, p.item_count);
      llmcore_preset_free(&p);
    }
  }

  /* ---- cleanup ---- */
  if (!fail(llmcore_delete_session(c, s.id))) printf("done.\n");
  llmcore_session_free(&s);
  rc = 0;

done:
  llmcore_client_free(c);
  return rc;
}
