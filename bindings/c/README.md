# llmcore C client

C client for the **llmcore bridge** over its **HTTP + SSE** transport (JSON wire),
built on **libcurl** + **cJSON**. Tier 0 plus the **Tier-2 audio one-shot (unary)
RPCs** — per decision **D6**, the duplex audio path in C (WebSocket/gRPC) is
reserved for a later phase. Depends only on the contract.

> **Build status / sandbox note.** The C source is complete and passes
> `gcc -std=c11 -Wall -Wextra -fsyntax-only` against minimal stub headers, but it
> was **not linked or run** in the authoring sandbox: libcurl/cJSON development
> packages were unavailable and unfetchable there. Build + test it as below in an
> environment with those libraries; syntax-checking proves well-formedness, not
> linkage or runtime behavior.

## Prerequisites

```bash
apt install libcurl4-openssl-dev libcjson-dev   # Debian/Ubuntu
# Fedora: dnf install libcurl-devel cjson-devel
# macOS:  brew install curl cjson
```
Plus a C11 compiler and either CMake ≥ 3.16 or GNU make + pkg-config.

## Build & test

CMake:
```bash
cd bindings/c
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
export LLMCORE_BRIDGE_PYTHON=/path/to/venv/bin/python   # python with llmcore[bridge]
ctest --test-dir build --output-on-failure
./build/quickstart
```

Plain make:
```bash
cd bindings/c
make
LLMCORE_BRIDGE_PYTHON=/path/to/venv/bin/python make test
```

## API sketch

```c
#include "llmcore_client.h"

llmcore_client *c = llmcore_client_new("http://127.0.0.1:50152");

const char *caps[] = {"tier0"};
llmcore_error *e = llmcore_ensure_compatible(c, caps, 1);   /* NULL on success */

llmcore_chat_result r;
e = llmcore_chat(c, "hello", &r);                            /* r.text, r.total_tokens */
llmcore_chat_result_free(&r);

/* streaming: callback per frame; return nonzero to cancel */
e = llmcore_chat_stream(c, "stream me", on_chunk, user);

llmcore_client_free(c);
```

Every call returns `llmcore_error*` (NULL = success). A non-NULL error is owned by
the caller — free it with `llmcore_error_free`. See **USAGE.md** for the full API,
ownership rules, error fields, and TLS guidance.

## Surface (Tier 0)

`llmcore_chat`, `llmcore_chat_stream` (cancellable), `llmcore_count_tokens`,
`llmcore_estimate_cost`, `llmcore_list_providers`, `llmcore_list_models`,
`llmcore_ensure_compatible`, `llmcore_health`. `llmcore_embed` always returns an
UNSUPPORTED error (Embed is UNIMPLEMENTED in `llmcore.v1`).

## Surface (Tier 2 — audio, unary)

Available when the bridge advertises `tier2.audio`: `llmcore_synthesize`,
`llmcore_transcribe`, `llmcore_generate_image`, `llmcore_ocr`,
`llmcore_analyze_text` (each fills an `*_result` struct freed with its
`*_result_free`). Proto `bytes` fields cross the JSON wire base64-encoded; the
client decodes/encodes transparently. The **live duplex** RPCs
(`transcribe_stream`, `synthesize_stream`, `voice_agent`) are WebSocket-based and
intentionally **not** wrapped here — per **D6**, C's duplex-audio path is gRPC, a
later phase.

```c
llmcore_speech_result sp;
llmcore_error *e = llmcore_synthesize(c, "hello", &sp);   /* sp.audio (bytes), sp.model */
if (!e) llmcore_speech_result_free(&sp);

llmcore_transcription_result tr;
e = llmcore_transcribe(c, pcm, pcm_len, &tr);             /* tr.text, tr.language */
if (!e) llmcore_transcription_result_free(&tr);
```

## Errors

`llmcore_error` carries `category`, `code`, `message`, `http_status`, `retryable`,
`retry_after_ms`, `provider`, `model`, decoded from the HTTP error body
`{"error": {...}}` (unary) or the SSE `event: error` frame (streaming).

## Memory ownership

* Errors → `llmcore_error_free`.
* `llmcore_chat_result` → `llmcore_chat_result_free` (frees `text`).
* `*out_currency` from `llmcore_estimate_cost` → `free`.
* String arrays from the catalog calls → `llmcore_string_array_free(items, n)`.

## Notes

* Single libcurl easy handle per client, reset per call (not thread-safe across
  concurrent calls on one client; use one client per thread).
* SSE cancellation: returning nonzero from the chunk callback aborts the transfer
  (libcurl `CURLE_WRITE_ERROR`), reported back as success.
* TLS: pass an `https://` base URL; libcurl uses the system CA store. For mTLS,
  extend the client to set `CURLOPT_SSLCERT` / `CURLOPT_SSLKEY` (one-line additions
  in `post_json`/`llmcore_chat_stream`).
* The e2e harness is Unix-only (fork/exec/kill).
