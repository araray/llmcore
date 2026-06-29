# llmcore-embedded

In-process (PyO3) Rust binding for `llmcore` — the **B5 "single-binary" track**.

Where [`llmcore-client`](../llmcore-client) speaks gRPC to the out-of-process
[bridge](../../README.md), this crate **embeds CPython** and calls the `llmcore`
Python package directly, driving its `asyncio` coroutines from Rust. It reuses
the `llmcore.v1` message types from `llmcore-proto`, so the public API mirrors
the gRPC client and is a drop-in for the **Tier-0** surface — you can switch a
program between "sidecar" and "in-process" by changing only the type it
constructs.

```rust
use llmcore_embedded::{v1, Llmcore};

# async fn run() -> Result<(), llmcore_embedded::EmbeddedError> {
let core = Llmcore::create().await?;                       // embeds CPython + LLMCore.create()
let res  = core.chat(v1::ChatRequest { message: "hello".into(), ..Default::default() }).await?;
println!("{} (tokens={:?})", res.text, res.usage.and_then(|u| u.total_tokens));
# Ok(()) }
```

## When to use this vs. the gRPC client

| | `llmcore-client` (gRPC) | `llmcore-embedded` (PyO3) |
|---|---|---|
| Deployment | thin client + separate bridge process | **single binary** (CPython embedded) |
| Isolation / fault containment | strong (separate process) | shared process |
| Languages | all five bindings | Rust only |
| Startup | connect to a socket | initialize an interpreter |

The out-of-process sidecar remains the default architecture (D1); this is the
one place embedding is ergonomic — single-binary Rust deployments (D8).

## Surface

All methods take `&self` and are `async` (the synchronous `llmcore` calls are
exposed `async` too, for parity with the gRPC client):

**Tier 0**

- **Inference:** `chat` (→ `ChatResponse` with token usage; `tools` /
  `tool_choice` are marshaled to `llmcore.models.Tool`), `chat_stream`
  (→ a `Stream` of `ChatChunk`, terminated by `done = true`), `count_tokens`,
  `estimate_cost`.
- **Catalog:** `list_providers`, `list_models`, `get_provider_details`.
- **Control:** `get_info` (locally-assembled `ServerInfo`), `reload_config`.

**Tier 1 (sessions)**

- `create_session`, `get_session`, `delete_session`, `add_context_item`,
  `get_context_item`.

Session results are marshaled through the bridge's own converters
(`_chat_session_to_proto` / `_context_item_to_proto`), so they are byte-identical
to the gRPC path. The vector-store and preset surfaces follow the identical
pattern and are a documented follow-up (they need a configured vector backend to
exercise).

Construct with `Llmcore::create()` (ambient config) or
`Llmcore::create_with_overrides(json)` (a JSON object passed to
`LLMCore.create(config_overrides=...)`).

Not marshaled: `ModelDetails.metadata`. `Embed` is UNIMPLEMENTED in the contract.
Tier-2 (audio) is out of scope for this track.

### Errors

Failures surface as `EmbeddedError`, built by running the Python exception
through the bridge's own `to_llmcore_error` and decoding the resulting
`llmcore.v1.LlmcoreError` — so `category` / `code` / `retryable` match the gRPC
path byte-for-byte.

## Building

This crate links a **shared** libpython and needs `llmcore` importable by that
interpreter. It is **excluded from the workspace default build** (`cargo build`
stays python-free); build it explicitly:

```bash
# Pick the interpreter (must be built with --enable-shared, i.e. Py_ENABLE_SHARED=1)
export PYO3_PYTHON=/path/to/python            # e.g. a pyenv/venv python with llmcore installed
export LD_LIBRARY_PATH="$($PYO3_PYTHON -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))'):$LD_LIBRARY_PATH"

cargo build   -p llmcore-embedded
cargo test    -p llmcore-embedded -- --test-threads=1
cargo run     -p llmcore-embedded --example embedded_chat
```

The tests are fully offline: they register `llmcore`'s shipped `FakeProvider`
(an echo provider) and build `LLMCore` against it, then exercise chat (incl.
tools), streaming, token counting, cost estimation, the catalog, and the Tier-1
sessions CRUD (a temp sqlite session store). The session tests need the bridge's
Tier-1 converters in the importable `llmcore`; until B4 is in your installed
build, point the interpreter at this tree with `PYTHONPATH=<repo>/src`.

## How the async bridge works

`llmcore` is async-first, so the binding runs a single `asyncio` event loop on a
daemon thread. Each call builds a coroutine under the GIL, hands it to
`asyncio.run_coroutine_threadsafe`, and awaits the resulting
`concurrent.futures.Future` on a tokio blocking thread — which releases the GIL
while it waits, so the loop thread is free to run the coroutine. Streaming pumps
the async generator's `__anext__` through the same path into a channel.
