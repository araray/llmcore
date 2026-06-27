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
