/* Quickstart: drive the bridge over HTTP/SSE.
 *
 *   LLMCORE_BRIDGE_FAKE=1 python -m llmcore.bridge.cli serve \
 *     --transport http --http-address 127.0.0.1:50152 --insecure
 *   ./quickstart            # or set LLMCORE_HTTP=http://host:port
 */
#include <stdio.h>
#include <stdlib.h>

#include "llmcore_client.h"

static int print_chunk(const char *text, int done, void *user) {
  (void)user;
  if (!done) fputs(text, stdout);
  return 0; /* return nonzero to cancel */
}

int main(void) {
  const char *base = getenv("LLMCORE_HTTP");
  if (!base) base = "http://127.0.0.1:50152";

  llmcore_client *c = llmcore_client_new(base);
  if (!c) {
    fprintf(stderr, "client alloc failed\n");
    return 1;
  }

  const char *caps[] = {"tier0"};
  llmcore_error *e = llmcore_ensure_compatible(c, caps, 1);
  if (e) {
    fprintf(stderr, "incompatible: %s %s\n", e->category, e->message);
    llmcore_error_free(e);
    llmcore_client_free(c);
    return 1;
  }

  llmcore_chat_result r;
  e = llmcore_chat(c, "hello from c", &r);
  if (e) {
    fprintf(stderr, "chat error: %s %s\n", e->category, e->message);
    llmcore_error_free(e);
  } else {
    printf("chat -> %s (tokens=%d)\n", r.text, r.total_tokens);
    llmcore_chat_result_free(&r);
  }

  printf("stream -> ");
  fflush(stdout);
  e = llmcore_chat_stream(c, "stream me", print_chunk, NULL);
  printf("\n");
  if (e) {
    fprintf(stderr, "stream error: %s %s\n", e->category, e->message);
    llmcore_error_free(e);
  }

  llmcore_client_free(c);
  return 0;
}
