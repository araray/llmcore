# llmcore C client

C client for the **llmcore bridge** over its **HTTP + SSE** transport (JSON wire),
built on **libcurl** + **cJSON**. Tier 0, the **Tier-1 sessions / vector store /
presets** surface, plus the **Tier-2 audio one-shot (unary) RPCs** — per decision
**D6**, the duplex audio path in C (WebSocket/gRPC) is reserved for a later phase.
Depends only on the contract (see **../CONTRACT.md**).

> **Build status / sandbox note.** The C source is complete and passes
> `gcc -std=c11 -Wall -Wextra -fsyntax-only` against minimal stub headers, but it
> was **not linked or run** in the authoring sandbox: libcurl/cJSON development
> packages were unavailable and unfetchable there. Build + test it as below in an
> environment with those libraries; syntax-checking proves well-formedness, not
> linkage or runtime behavior.

## Prerequisites

**libcurl** is the only required system dev package:

```bash
apt install libcurl4-openssl-dev   # Debian/Ubuntu
# Fedora: dnf install libcurl-devel
# macOS:  brew install curl
```

**cJSON is resolved automatically** — no system package is needed. An installed
cJSON (CMake config, or pkg-config `libcjson`: `apt install libcjson-dev`) is used
when present; otherwise cJSON is fetched and built from source via CMake
`FetchContent`. It is a single small translation unit, so this adds only seconds to
the build. Force the fetch with `-DLLMCORE_FETCH_CJSON=ON`, or pin a tag with
`-DLLMCORE_CJSON_TAG=vX.Y.Z`. Plain `cmake -B build && cmake --build build` works
out of the box.

Plus a C11 compiler and either CMake ≥ 3.16 or GNU make + pkg-config (the plain
`make` build still expects an installed cJSON).

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

## Surface (Tier 1 — sessions, vector store & presets)

Available when the bridge advertises `tier1.sessions` and/or `tier1.vector`;
negotiate with `llmcore_ensure_compatible(c, (const char*[]){"tier1.sessions",
"tier1.vector"}, 2)`. The C client drives these over the same HTTP transport.

* **Sessions & context items** — `llmcore_create_session`, `llmcore_get_session`,
  `llmcore_list_sessions`, `llmcore_delete_session`, `llmcore_update_session_name`,
  `llmcore_fork_session`, `llmcore_clone_session`, `llmcore_delete_messages`;
  `llmcore_add_context_item`, `llmcore_get_context_item`,
  `llmcore_remove_context_item`. Results fill `llmcore_session` /
  `llmcore_context_item` (freed with `llmcore_session_free` /
  `llmcore_context_item_free`).
* **Vector store & RAG** — `llmcore_add_documents`, `llmcore_search_vector_store`
  (fills a `llmcore_search_result[]`, freed with `llmcore_search_results_free`),
  `llmcore_list_vector_collections`, `llmcore_list_rag_collections`,
  `llmcore_get_rag_collection_info`, `llmcore_delete_rag_collection`.
* **Context presets** — `llmcore_save_context_preset`,
  `llmcore_get_context_preset` (fills `llmcore_preset`, freed with
  `llmcore_preset_free`), `llmcore_list_context_presets`,
  `llmcore_delete_context_preset`.

See `examples/sessions.c` for an end-to-end walkthrough and **USAGE.md** for the
calling and ownership details.

### Running the Tier-1 example

Start a bridge over HTTP with the Tier-1 fakes gated on:

```bash
LLMCORE_BRIDGE_FAKE=1 LLMCORE_BRIDGE_FAKE_SESSIONS=1 LLMCORE_BRIDGE_FAKE_VECTOR=1 \
  python -m llmcore.bridge.cli serve --transport http \
  --http-address 127.0.0.1:50152 --insecure
```

Then build and run the example:

```bash
cmake -B build && cmake --build build -j
LLMCORE_HTTP=http://127.0.0.1:50152 ./build/sessions
```

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
