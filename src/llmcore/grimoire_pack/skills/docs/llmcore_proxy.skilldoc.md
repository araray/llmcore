---
id: skills/llmcore/proxy
name: llmcore as an LLM proxy for agent harnesses
version: 1.0.0
description: >
  Run llmcore as an OpenAI-compatible endpoint so an unmodified agent harness
  gets routing, failover, cheap-model selection and cost accounting without
  knowing llmcore exists.
tags: [llmcore, routing, proxy, agents, cost]
sections:
  - id: when_to_use
    tags: [routing, proxy]
  - id: start
    tags: [proxy, cli]
  - id: harness_config
    tags: [proxy, agents]
  - id: model_names
    tags: [proxy, routing]
  - id: saving_money
    tags: [cost, proxy]
  - id: security
    tags: [proxy, security]
  - id: troubleshooting
    tags: [proxy, debugging]
---

## when_to_use

Use proxy mode when something **other than your own code** makes the LLM call —
an agent harness, an IDE plugin, a CLI tool, anything you cannot add a keyword
argument to. If you are writing the call yourself, skip this and use
`llm.chat(pool=..., lane=...)` directly; the proxy adds a network hop for
nothing.

The proxy earns its place when:

- a harness is burning expensive-model calls on trivial turns, and you want
  cheap turns routed to a cheap model;
- you want failover across vendors without the harness knowing;
- you want one place that records what every tool in your setup actually spent;
- a prompt must never reach a remote vendor, and you want that enforced at the
  boundary rather than trusted to each tool.

## start

Requires the bridge extra: `pip install "llmcore[bridge]"`.

```bash
llmcore-bridge proxy                      # 127.0.0.1:8900, no auth needed
llmcore-bridge proxy --port 9000
llmcore-bridge proxy --config ./llmcore.toml
```

Configure it in `[routing.proxy]` instead of flags when you want it persistent:

```toml
[routing.proxy]
enabled = true
host = "127.0.0.1"
port = 8900
default_model = "auto"
```

Check it is alive:

```bash
curl -s localhost:8900/healthz
curl -s localhost:8900/v1/models | jq '.data[].id'
```

## harness_config

The whole interface is two environment variables, because that is all a harness
reliably exposes:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8900/v1
export OPENAI_API_KEY=llmcore-local        # most harnesses insist on something
```

Then set the harness's model to one of the names in `model_names` below.
Anything that speaks the OpenAI chat-completions API works unchanged: the
official SDKs, LangChain, LlamaIndex, Aider, Continue, OpenWebUI, and most
agent frameworks.

`llmcore-bridge proxy` prints exactly these variables at startup, so they can be
pasted.

## model_names

A harness can only send a **model name**, so the model name carries the routing
intent:

| `model` | effect |
|---|---|
| `auto` | run the classifier chain and let it decide |
| `lane:deep` | route to the `deep` lane |
| `pool:main` | route through the `main` pool, with failover |
| `profile:frugal` | apply a parameter profile |
| `openai:gpt-5.4?effort=high` | an explicit target, parameters and all |
| `gpt-4o-mini` | a plain model name, as before |

Lanes and pools appear in `GET /v1/models`, so a harness with a model picker
becomes a way to choose routing policy from its own UI.

If the harness lets you send extra fields (`extra_body` in the OpenAI SDKs),
routing can be set there instead:

```python
client.chat.completions.create(
    model="auto",
    messages=[...],
    extra_body={"llmcore": {"lane": "deep", "routing": {"max_attempts": 5}}},
)
```

## saving_money

The reason most people reach for this. Three steps, in order of payoff:

1. **Route trivial turns to a cheap model.** Define lanes, give each one a
   description, and let the classifier chain sort turns into them. Start with
   the free classifiers — they cost nothing and handle the obvious cases:

   ```toml
   [routing.classifier]
   chain = ["hint", "magic_string", "heuristic"]

   [routing.lanes]
   trivial = "pool:cheap"
   standard = "pool:main"
   deep = "anthropic:claude-opus-5-5?effort=max"
   ```

2. **Let the agent route itself.** A harness's model can write
   `[[lane:trivial]]` into its own prompt, and llmcore strips the marker before
   it reaches a provider. This is often more accurate than any classifier,
   because the model knows what it is about to do.

3. **Add a cascade** for batch and agent work: answer cheaply, verify, escalate
   only when the cheap answer fell short. Off by default because it costs
   latency and an extra call, which is the wrong trade interactively.

Check what it actually spent with the `skills/llmcore/cost` skilldoc.

Before trusting a lane assignment, look at it:

```bash
curl -s --get localhost:8900/v1/routing/explain \
  --data-urlencode 'prompt=rename this variable' | jq
```

## security

**This process holds every provider credential in your config.** That is the
whole security model in one sentence, and the defaults follow from it:

- It binds to `127.0.0.1`. A non-loopback bind **without** a bearer token is
  refused at startup — not warned about, refused.
- For any other bind, set a token:

  ```bash
  export LLMCORE_ROUTING__PROXY__API_KEY="$(openssl rand -hex 24)"
  llmcore-bridge proxy --host 0.0.0.0
  ```

  Clients then send `Authorization: Bearer <token>`.
- Put the PII transform in front of it if prompts from other tools may contain
  personal data. `on_detect = "constrain"` routes those turns to a local-only
  pool, which is a guarantee; redaction alone is not.

## troubleshooting

**Every request 401s.** A token is configured; send
`Authorization: Bearer <token>`. `OPENAI_API_KEY` in the harness becomes that
header.

**"Refusing to bind the routing proxy".** A non-loopback host with no token.
Set `routing.proxy.api_key` or bind to loopback.

**The wrong model answers.** `GET /v1/routing/explain?prompt=...` shows the
lane, the classifier that chose it, and why each candidate was or was not
used. Under a pool, the `llmcore` block in the response names the target that
actually served the request.

**A harness reports model `gpt-4o-mini` when you asked for `pool:main`.** That
is correct and deliberate: the `model` field reports what answered, not what
was requested, so cost logs are not wrong.

**Requests 503 with "exhausted every candidate".** Every pool member was
unusable. `GET /v1/routing/health` lists each target's cooldown and last
failure.

**Streaming never fails over.** By design: once bytes have reached the client
the call cannot be retracted. A stream that fails *before* its first chunk does
fail over.
