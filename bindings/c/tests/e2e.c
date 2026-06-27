/* End-to-end tests for the C client against a spawned bridge (FakeFacade).
 *
 * Set LLMCORE_BRIDGE_PYTHON to a python with llmcore[bridge] importable
 * (default python3). Unix-only (fork/exec/kill). */
#define _POSIX_C_SOURCE 200809L
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "llmcore_client.h"

static int g_failures = 0;
#define CHECK(cond)                                                  \
  do {                                                               \
    if (!(cond)) {                                                   \
      fprintf(stderr, "FAIL: %s @ line %d\n", #cond, __LINE__);      \
      ++g_failures;                                                  \
    }                                                                \
  } while (0)

static int free_port(void) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  struct sockaddr_in a;
  memset(&a, 0, sizeof(a));
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  a.sin_port = 0;
  bind(fd, (struct sockaddr *)&a, sizeof(a));
  socklen_t L = sizeof(a);
  getsockname(fd, (struct sockaddr *)&a, &L);
  int port = ntohs(a.sin_port);
  close(fd);
  return port;
}

static void msleep(int ms) {
  struct timespec ts;
  ts.tv_sec = ms / 1000;
  ts.tv_nsec = (long)(ms % 1000) * 1000000L;
  nanosleep(&ts, NULL);
}

struct concat {
  char buf[512];
  int done;
};

static int concat_cb(const char *text, int done, void *user) {
  struct concat *a = (struct concat *)user;
  if (done) {
    a->done = 1;
  } else {
    size_t used = strlen(a->buf);
    strncat(a->buf, text, sizeof(a->buf) - used - 1);
  }
  return 0;
}

static int cancel_cb(const char *text, int done, void *user) {
  (void)text;
  (void)done;
  int *n = (int *)user;
  (*n)++;
  return 1; /* cancel on first frame */
}

struct bridge {
  pid_t pid;
  char base[64];
};

static struct bridge start_bridge(int with_audio) {
  struct bridge b;
  b.pid = -1;
  int hport = free_port();
  snprintf(b.base, sizeof(b.base), "http://127.0.0.1:%d", hport);
  char http_addr[64];
  snprintf(http_addr, sizeof(http_addr), "127.0.0.1:%d", hport);

  const char *py = getenv("LLMCORE_BRIDGE_PYTHON");
  if (!py) py = "python3";

  pid_t pid = fork();
  if (pid == 0) {
    setenv("LLMCORE_BRIDGE_FAKE", "1", 1);
    if (with_audio) setenv("LLMCORE_BRIDGE_FAKE_AUDIO", "1", 1);
    char *argv[] = {(char *)py,
                    "-m",
                    "llmcore.bridge.cli",
                    "serve",
                    "--transport",
                    "http",
                    "--http-address",
                    http_addr,
                    "--insecure",
                    "--log-level",
                    "WARNING",
                    NULL};
    execvp(py, argv);
    _exit(127);
  }
  b.pid = pid;

  llmcore_client *c = llmcore_client_new(b.base);
  for (int i = 0; i < 125; i++) { /* up to ~25s */
    int ok = 0;
    llmcore_error *e = llmcore_health(c, &ok);
    if (e) {
      llmcore_error_free(e);
    } else if (ok) {
      llmcore_client_free(c);
      return b;
    }
    msleep(200);
  }
  llmcore_client_free(c);
  fprintf(stderr, "bridge not ready\n");
  kill(pid, SIGTERM);
  exit(2);
}

int main(void) {
  struct bridge b = start_bridge(0);
  llmcore_client *c = llmcore_client_new(b.base);

  /* ensure_compatible accept */
  {
    const char *caps[] = {"tier0"};
    llmcore_error *e = llmcore_ensure_compatible(c, caps, 1);
    CHECK(e == NULL);
    if (e) llmcore_error_free(e);
  }
  /* ensure_compatible reject */
  {
    const char *caps[] = {"tier2.audio"};
    llmcore_error *e = llmcore_ensure_compatible(c, caps, 1);
    CHECK(e != NULL);
    if (e) {
      CHECK(strcmp(e->code, "capability.missing") == 0);
      llmcore_error_free(e);
    }
  }
  /* chat + usage */
  {
    llmcore_chat_result r;
    llmcore_error *e = llmcore_chat(c, "hello world", &r);
    CHECK(e == NULL);
    if (!e) {
      CHECK(strcmp(r.text, "echo: hello world") == 0);
      CHECK(r.prompt_tokens == 2);
      llmcore_chat_result_free(&r);
    } else {
      llmcore_error_free(e);
    }
  }
  /* stream concatenation */
  {
    struct concat a;
    memset(&a, 0, sizeof(a));
    llmcore_error *e = llmcore_chat_stream(c, "stream this please", concat_cb, &a);
    CHECK(e == NULL);
    if (e) llmcore_error_free(e);
    CHECK(a.done == 1);
    CHECK(strcmp(a.buf, "echo: stream this please") == 0);
  }
  /* count */
  {
    int n = 0;
    llmcore_error *e = llmcore_count_tokens(c, "one two three four", &n);
    CHECK(e == NULL);
    if (e) llmcore_error_free(e);
    CHECK(n == 4);
  }
  /* cost */
  {
    double tc = 0.0;
    char *cur = NULL;
    llmcore_error *e = llmcore_estimate_cost(c, "fake", "fake-1", 1000000, 1000000, &tc, &cur);
    CHECK(e == NULL);
    if (e) llmcore_error_free(e);
    CHECK(cur && strcmp(cur, "USD") == 0);
    CHECK(tc == 3.0);
    free(cur);
  }
  /* catalog: providers */
  {
    char **it = NULL;
    size_t n = 0;
    llmcore_error *e = llmcore_list_providers(c, &it, &n);
    CHECK(e == NULL);
    if (e) llmcore_error_free(e);
    CHECK(n == 1 && it && strcmp(it[0], "fake") == 0);
    llmcore_string_array_free(it, n);
  }
  /* catalog: models */
  {
    char **it = NULL;
    size_t n = 0;
    llmcore_error *e = llmcore_list_models(c, "fake", &it, &n);
    CHECK(e == NULL);
    if (e) llmcore_error_free(e);
    CHECK(n == 2 && strcmp(it[0], "fake-1") == 0 && strcmp(it[1], "fake-2") == 0);
    llmcore_string_array_free(it, n);
  }
  /* embed unsupported */
  {
    const char *in[] = {"x"};
    llmcore_error *e = llmcore_embed(c, in, 1);
    CHECK(e != NULL);
    if (e) {
      CHECK(strcmp(e->category, "ERROR_CATEGORY_UNSUPPORTED") == 0);
      llmcore_error_free(e);
    }
  }
  /* provider rate-limit structured decode */
  {
    llmcore_chat_result r;
    llmcore_error *e = llmcore_chat(c, "__error__:provider_rate_limited", &r);
    CHECK(e != NULL);
    if (e) {
      CHECK(strcmp(e->category, "ERROR_CATEGORY_PROVIDER") == 0);
      CHECK(strcmp(e->code, "provider.rate_limited") == 0);
      CHECK(e->http_status == 429);
      CHECK(e->retryable == 1);
      CHECK(e->retry_after_ms == 2000.0);
      CHECK(e->provider && strcmp(e->provider, "fake") == 0);
      llmcore_error_free(e);
    } else {
      llmcore_chat_result_free(&r);
    }
  }
  /* stream cancellation */
  {
    int cnt = 0;
    llmcore_error *e = llmcore_chat_stream(c, "__cancel__", cancel_cb, &cnt);
    CHECK(e == NULL); /* user cancel reported as success */
    if (e) llmcore_error_free(e);
    CHECK(cnt >= 1);
  }

  /* ---- audio (Tier 2): a second, audio-enabled bridge ---- */
  {
    struct bridge ab = start_bridge(1);
    llmcore_client *ac = llmcore_client_new(ab.base);

    /* synthesize: audio_data decodes to "tts:hello" */
    {
      llmcore_speech_result r;
      llmcore_error *e = llmcore_synthesize(ac, "hello", &r);
      CHECK(e == NULL);
      if (e) {
        llmcore_error_free(e);
      } else {
        CHECK(r.audio_len == 9 && memcmp(r.audio, "tts:hello", 9) == 0);
        CHECK(strcmp(r.model, "fake-tts") == 0);
        llmcore_speech_result_free(&r);
      }
    }
    /* transcribe: audio bytes round-trip to text */
    {
      llmcore_transcription_result r;
      llmcore_error *e =
          llmcore_transcribe(ac, (const unsigned char *)"hello world", 11, &r);
      CHECK(e == NULL);
      if (e) {
        llmcore_error_free(e);
      } else {
        CHECK(strcmp(r.text, "hello world") == 0);
        CHECK(strcmp(r.language, "en") == 0);
        llmcore_transcription_result_free(&r);
      }
    }
    /* generate_image: n=2, data[0] == base64("img:a cat") */
    {
      llmcore_image_result r;
      llmcore_error *e = llmcore_generate_image(ac, "a cat", 2, &r);
      CHECK(e == NULL);
      if (e) {
        llmcore_error_free(e);
      } else {
        CHECK(r.n_images == 2);
        if (r.n_images >= 1) CHECK(strcmp(r.images[0], "aW1nOmEgY2F0") == 0);
        CHECK(strcmp(r.model, "fake-img") == 0);
        llmcore_image_result_free(&r);
      }
    }
    /* ocr: bytes -> model/pages_processed/doc_size_bytes */
    {
      llmcore_ocr_result r;
      llmcore_error *e =
          llmcore_ocr(ac, (const unsigned char *)"PDFBYTES", 8, NULL, &r);
      CHECK(e == NULL);
      if (e) {
        llmcore_error_free(e);
      } else {
        CHECK(strcmp(r.model, "fake-ocr") == 0);
        CHECK(r.pages_processed == 1);
        CHECK(r.has_doc_size_bytes && r.doc_size_bytes == 8);
        llmcore_ocr_result_free(&r);
      }
    }
    /* analyze_text (no features): model set, no summary */
    {
      llmcore_text_analysis_result r;
      llmcore_error *e = llmcore_analyze_text(ac, "some text", &r);
      CHECK(e == NULL);
      if (e) {
        llmcore_error_free(e);
      } else {
        CHECK(r.has_summary == 0);
        CHECK(strcmp(r.model, "fake-analyze") == 0);
        llmcore_text_analysis_result_free(&r);
      }
    }

    llmcore_client_free(ac);
    kill(ab.pid, SIGTERM);
    int ast;
    waitpid(ab.pid, &ast, 0);
  }

  llmcore_client_free(c);
  kill(b.pid, SIGTERM);
  int st;
  waitpid(b.pid, &st, 0);

  if (g_failures == 0) {
    printf("ALL TESTS PASSED\n");
    return 0;
  }
  printf("%d FAILURES\n", g_failures);
  return 1;
}
