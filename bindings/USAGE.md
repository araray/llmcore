# Using the llmcore bridge

## Install

```bash
pip install "llmcore[bridge]"        # server + runtime deps
pip install "llmcore[bridge-dev]"    # + codegen/guard tooling (buf installed separately)
```

## Run the server

The server is the `llmcore-bridge` console command (equivalently
`python -m llmcore.bridge.cli`).

### Local development (plaintext TCP)

```bash
llmcore-bridge serve \
  --transport grpc,http \
  --grpc-address 127.0.0.1:50151 \
  --http-address 127.0.0.1:50152 \
  --insecure
```

`--insecure` is required to serve plaintext over TCP and is intended for
localhost/dev only.

### Unix-domain sockets (local, no TLS needed)

UDS endpoints rely on filesystem permissions for trust:

```bash
llmcore-bridge serve \
  --transport grpc,http \
  --grpc-address unix:/run/llmcore/grpc.sock \
  --http-address unix:/run/llmcore/http.sock
```

### Production over TCP (TLS / mTLS + auth)

TCP endpoints refuse to start without TLS *and* an auth mode (unless
`--insecure`):

```bash
llmcore-bridge serve \
  --transport grpc \
  --grpc-address 0.0.0.0:50151 \
  --tls-cert /etc/llmcore/server.crt \
  --tls-key  /etc/llmcore/server.key \
  --tls-client-ca /etc/llmcore/client-ca.crt \   # enables mTLS
  --auth authflow --authflow-url https://authflow.internal
```

> The `authflow` mode installs a credential-presence interceptor in B1; full
> AuthFlow token introspection + assurance-level enforcement land in the
> security phase. mTLS is wired (server creds + optional client-CA).

### Configuring the underlying LLMCore

Without `LLMCORE_BRIDGE_FAKE`, the bridge constructs a real `LLMCore` via
`LLMCore.create()`. Point it at your config and environment:

```bash
llmcore-bridge serve --config /etc/llmcore/llmcore.toml --env-prefix LLMCORE ...
```

### Deterministic fake backend (demos / CI)

```bash
LLMCORE_BRIDGE_FAKE=1 llmcore-bridge serve \
  --transport grpc,http \
  --grpc-address 127.0.0.1:50151 --http-address 127.0.0.1:50152 --insecure
```

The fake echoes prompts (`"echo: <message>"`), computes deterministic token
usage, and returns real `CostEstimate`/`ModelDetails` objects — no provider keys
or network required.

Higher tiers are off by default even under the fake (so they stay UNIMPLEMENTED
unless asked for). Opt into the in-memory Tier-1 / Tier-2 stores with:

| Env var | Enables | Advertised capabilities |
|---|---|---|
| `LLMCORE_BRIDGE_FAKE_SESSIONS=1` | in-memory session store | `tier1`, `tier1.sessions` |
| `LLMCORE_BRIDGE_FAKE_VECTOR=1` | in-memory vector store + presets | `tier1`, `tier1.vector` |
| `LLMCORE_BRIDGE_FAKE_AUDIO=1` | deterministic audio surface | `tier2`, `tier2.*` |

```bash
# A bridge that serves Tier-0 inference + the full Tier-1 surface:
LLMCORE_BRIDGE_FAKE=1 LLMCORE_BRIDGE_FAKE_SESSIONS=1 LLMCORE_BRIDGE_FAKE_VECTOR=1 \
  llmcore-bridge serve --transport grpc,http \
  --grpc-address 127.0.0.1:50151 --http-address 127.0.0.1:50152 --insecure
```

This is exactly what the `examples/sessions.*` demos in each client expect.
Against a **real** `LLMCore`, the same capabilities are advertised automatically
when the configured backend supports sessions / a vector store.

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--transport` | `grpc,http` | comma list of transports to serve |
| `--grpc-address` | `127.0.0.1:50151` | `host:port` or `unix:/path` |
| `--http-address` | `127.0.0.1:50152` | `host:port` or `unix:/path` |
| `--config` | – | TOML config path (confy) |
| `--env-prefix` | `LLMCORE` | env-var prefix for config |
| `--tls-cert` / `--tls-key` | – | server TLS material (PEM) |
| `--tls-client-ca` | – | client CA bundle → enables mTLS |
| `--auth` | `none` | `none` \| `authflow` |
| `--authflow-url` | – | AuthFlow introspection base URL |
| `--insecure` | off | permit plaintext TCP (dev only) |
| `--log-level` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |

Lifecycle: on Linux the process sets `PR_SET_PDEATHSIG` (dies with its parent);
`SIGINT`/`SIGTERM` trigger a graceful drain of both servers.

## Call it

### gRPC (Python reference client)

```bash
python bindings/examples/python/client.py --transport grpc \
  --grpc-address 127.0.0.1:50151
```

### HTTP with curl

```bash
# unary
curl -s localhost:50152/llmcore.v1/InferenceService/Chat \
  -H 'content-type: application/json' \
  -d '{"message":"hello"}'
# {"text":"echo: hello","usage":{"prompt_tokens":1,...}}

# streaming (SSE)
curl -N localhost:50152/llmcore.v1/InferenceService/ChatStream \
  -H 'content-type: application/json' \
  -d '{"message":"stream me"}'
# data: {"text":"echo: st"}
# data: {"text":"ream me"}
# event: done
# data: {"done":true}

# capability handshake
curl -s localhost:50152/llmcore.v1/ControlService/GetInfo -d '{}'

# health
curl -s localhost:50152/healthz
```

### Error shape (HTTP)

```bash
curl -s -o /dev/null -w '%{http_code}\n' \
  localhost:50152/llmcore.v1/InferenceService/Embed -d '{"input":["x"]}'
# 501
```

```json
{"error":{"category":"ERROR_CATEGORY_UNSUPPORTED","code":"unsupported.capability",
          "message":"Embed is not available in llmcore.v1: ...","http_status":501}}
```

## Tier-1: sessions, vector store, and presets (HTTP)

With a bridge started using the Tier-1 fake gates above (`curl` against the HTTP
projection; gRPC clients call the same `SessionService` / `VectorService` /
`PresetService` RPCs):

```bash
H=http://127.0.0.1:50152

# create a session with a system message
SID=$(curl -s -X POST $H/llmcore.v1/SessionService/CreateSession \
  -d '{"name":"demo","system_message":"You are terse."}' | jq -r .id)

# stage a context item, then read it back
curl -s -X POST $H/llmcore.v1/SessionService/AddContextItem \
  -d "{\"session_id\":\"$SID\",\"content\":\"launch is June 30\",\"type\":\"user_text\"}"

# index a document and search the vector store directly (no chat-time RAG)
curl -s -X POST $H/llmcore.v1/VectorService/AddDocuments \
  -d '{"documents":[{"content":"Paris is the capital of France."}]}'
curl -s -X POST $H/llmcore.v1/VectorService/SearchVectorStore \
  -d '{"query":"capital of France","k":3}'

# save + fetch a reusable context preset
curl -s -X POST $H/llmcore.v1/PresetService/SaveContextPreset \
  -d '{"preset":{"name":"preamble","items":[{"type":"preset_text_content","content":"Cite sources."}]}}'
curl -s -X POST $H/llmcore.v1/PresetService/GetContextPreset -d '{"preset_name":"preamble"}'
```

Each client ships an idiomatic version of this flow at `examples/sessions.*`
(Go/Rust/TS/C++ over gRPC, C over HTTP).

## Common workflows

**Regenerate stubs after editing the contract**

```bash
( cd bindings && buf lint )      # validate first
sh bindings/scripts/gen_proto.sh # regenerate Python stubs
pytest tests/bridge -q           # re-run conformance
```

**Guard against llmcore drift (CI)**

```bash
python bindings/scripts/contract_guard.py   # exits non-zero on drift
```

**Golden end-to-end (real subprocess, both transports)**

```bash
python bindings/scripts/e2e_smoke.py         # prints "E2E SMOKE: PASS"
```

**Run the real-LLMCore integration test** (needs the sovereign `confy>=0.4.2`,
so run it in the llmcore dev environment, not generic CI):

```bash
pip install -e ".[bridge]"
pytest tests/bridge/test_integration_real_llmcore.py -q
```
