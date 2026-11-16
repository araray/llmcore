# Command Code provider: usage guide

[Command Code](https://commandcode.ai) is a coding agent with a paid model
gateway, the **Provider API**. One key reaches 98+ models: Claude, GPT, Gemini,
Grok, DeepSeek, Kimi, GLM, Qwen, MiniMax, Muse and more. It also serves
TypeSafe's Jev decision model.

The gateway speaks **three protocols**, and each model answers on only one of
them:

| Models | Route | Wire shape |
|---|---|---|
| Claude | `/messages` | Anthropic Messages |
| everything else | `/chat/completions` | OpenAI Chat Completions |
| `typesafe/jev` | `/systemone` | TypeSafe System One |

Sending a model to the wrong route returns `400 unsupported_model`. llmcore
reads each model's route from the live `/models` catalog and sends the call
there, so you never pick the route yourself. You call `chat()` as with any
provider.

---

## 1. Install and configure

```bash
pip install "llmcore[commandcode]"
export CMD_API_KEY="..."        # from https://commandcode.ai/studio (API keys)
```

The key is looked up in this order: `api_key`, then `api_key_env_var`, then
`CMD_API_KEY`, `COMMAND_CODE_API_KEY` and `COMMANDCODE_API_KEY`. The first two
variable names are the ones Command Code's own CLI reads, so one key serves
both.

> **Plan:** API access needs a GOAT plan or higher. The $1 Go plan is
> CLI-only, and its key gets `403 upgrade_required`, which llmcore reports in
> those words.

`[providers.commandcode]` (all keys optional):

```toml
[providers.commandcode]
# api_key_env_var = "CMD_API_KEY"
default_model = "claude-sonnet-5-5"
transport = "httpx"          # direct HTTP (default) | "sdk": the OpenAI / Anthropic SDKs
zdr = false                  # zero data retention on every request (or env CMD_ZDR=1)
catalog_ttl = 3600           # seconds the live /models catalog is trusted
timeout = 120
max_retries = 2
# base_url = "https://api.commandcode.ai/provider"   # env CMD_BASE_URL
```

No section is needed for one-off calls. With a key in the environment, a
dynamic target works on its own:

```python
await llm.chat("hi", target="commandcode:kimi-k3")
await llm.chat("hi", target="commandcode:claude-opus-5-5?effort=high")
```

`command_code` and `command-code` are accepted as provider-name aliases.

---

## 2. Chat

```python
from llmcore import LLMCore

llm = await LLMCore.create()
await llm.chat("Explain CRDTs in two lines", provider_name="commandcode",
               model_name="claude-sonnet-5-5")
await llm.chat("Same, terser", provider_name="commandcode", model_name="kimi-k3")
await llm.chat("Same, in Portuguese", provider_name="commandcode",
               model_name="deepseek/deepseek-v4.1-flash", stream=True)
```

### Model names

A name resolves the way the Command Code CLI resolves `--model`:

1. the exact id (`moonshotai/Kimi-K3`)
2. the same id, ignoring case
3. the part after `/` (`kimi-k3`, `glm-5.3`, `deepseek-v4-flash`)

A short name that matches more than one model raises `ConfigError` and lists
the candidates. It never picks one silently.

```python
cc = llm.get_provider("commandcode")
cc.resolve_model("kimi-k3")          # "moonshotai/Kimi-K3"
cc.route_for("claude-haiku-5-5")     # "messages"
cc.route_for("gpt-6-sol")            # "chat"
await cc.get_models_details()        # every model, with route + context window
await cc.refresh_catalog()           # re-read /models now
```

### Reasoning effort

`effort=` is llmcore's canonical knob, and it is translated per route:

- **Claude** (`/messages`): the Anthropic provider's own rules apply.
  `output_config.effort` with adaptive thinking on Claude 4.6+, and a thinking
  budget on older models.
- **Everything else** (`/chat/completions`): `effort` is sent as
  `reasoning_effort` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`,
  `max`). Upstream-specific `reasoning={...}` and `thinking={...}` are also
  accepted and passed through.

Reasoning text is available either way:

```python
resp = await cc.chat_completion(messages, "deepseek/deepseek-v4-pro")
cc.extract_reasoning_content(resp)   # open models' `reasoning`, or Claude's thinking
cc.extract_reasoning_details(resp)   # structured reasoning to send back on the next turn
```

For multi-turn tool use on reasoning models, put
`extract_reasoning_details(resp)` in that assistant message's
`metadata["reasoning_details"]`. llmcore sends it back on the next request.

### Tools, vision, caching

These work as they do on the underlying protocol, because each route is served
by llmcore's own Anthropic or OpenAI implementation. That covers `tools=` and
`tool_choice=`, images through `metadata["inline_images"]`, and Claude prompt
caching through `metadata["cache_control"]`. Request bodies are capped at
32 MiB.

---

## 3. Zero data retention

```python
await llm.chat("...", provider_name="commandcode", zdr=True)   # this call only
```

`zdr = true` in config (or `CMD_ZDR=1`) turns it on for every call. It sends
`x-cmd-zdr: 1`, so only upstreams that retain nothing serve the request. If a
model has no such upstream, the call fails with `422 cmd_zdr_no_providers`
rather than quietly using one that retains data. The free models and Jev have
none. Claude's `/messages` route is zero-retention at account level already.

---

## 4. Decision model (Jev)

```python
from llmcore.providers.typesafe_provider import Choice, Noul, Score

result = await cc.system_one(
    state="The user said: the build is green and all tests pass.",
    questions={
        "done": Noul(instructions="Is the work finished?"),
        "mood": Choice(instructions="Tone?", criteria={"calm": "...", "urgent": "..."}),
    },
)
result.answers_dict()   # {"done": {"type": "noul", "noul": 0.59}, ...}
```

The arguments and result are those of the
[TypeSafe provider](TypeSafe_provider_usage.md). Jev is billed at $0.042 per
1M input tokens, and output is free. Calling `chat()` with `typesafe/jev`
raises an error that points here.

---

## 5. Errors and tracing

Every error is a `ProviderError` (or a `ContextLengthError`) with these
attributes:

| attribute | meaning |
|---|---|
| `status_code` | HTTP status, or the upstream's code for in-band failures |
| `error_code` | Command Code's code, e.g. `cmd_zdr_no_providers`, `upgrade_required` |
| `failure_kind` | what routing should do: `auth`, `rate_limit`, `server`, `model_not_found`, `refusal`, `insufficient_credit`, … |
| `retryable` | whether the same call may succeed later |
| `trace_id` | Command Code's `x-trace-id`; quote it when reporting a problem |
| `gateway_routing` | the gateway's upstream attempts, when it sent them |

Command Code's 403s are not all the same, and `failure_kind` tells them apart.
`model_region_unavailable` lets routing fail over to a peer, while
`upgrade_required` marks the target unusable.

Gateway behaviour handled for you:

- **Upstream failures sent as HTTP 200.** An upstream can fail mid-response
  and the gateway still answers 200, with `finish_reason: "error"` and partial
  content. llmcore raises a retryable `ProviderError` and keeps the fragment on
  `err.partial_content`. It never returns the fragment as an answer. In a
  stream, the error is raised after the chunks already delivered.
- **Nested error messages.** Some errors carry the upstream error as JSON
  inside `error.message`, with the gateway's routing attempts. These are
  unwrapped into a readable message.
- **Stale catalog.** If the gateway says a model is served on a different
  route, the catalog is re-read and the call retried once on the right one.
- **Stream failures.** A stream that fails before its first chunk raises at
  call time, where routing can still fail over.

`cc.last_call_info()` returns the trace id, upstream and route of the last
call in the current task. Non-streaming responses carry the same data under
`response["commandcode"]`, and `extract_usage_details()` includes it.

---

## 6. Model cards and pricing

`default_cards/commandcode/` holds a card for every served model, plus Jev.
`cardctl generate commandcode` rebuilds them from the public catalog and needs
no key.

- **Prices** are transcribed from Command Code's
  [pricing page](https://commandcode.ai/docs/resources/pricing-limits) into
  `tools/cardctl/enrichments/commandcode.toml`, with input, output, cache-read
  and cache-write rates. They are Command Code's prices, which differ from the
  vendors' own.
- **Peak/off-peak** (DeepSeek V4.x): the card holds the *peak* rate, so cost
  estimates never come in low. The off-peak rate and schedule are in
  `provider_extension.off_peak`.
- **Deals** (MiniMax M3, -50%): the card holds the list price, with the deal in
  `provider_extension.deal`.
- **Capabilities and max output** come from llmcore's card for the same model
  at its original vendor (`provider_extension.capabilities_from`), unless that
  card is a placeholder. Then they come from model-family defaults.
- **Free models** (`glyph-cluster:free`, `inkling-2-small:free`) carry their
  daily request caps.

---

## 7. Limits

- `/responses` is not used. Every model served there also answers on
  `/chat/completions`.
- Command Code publishes no rate-limit headers and no usage or balance API.
  Spend is visible in Studio.
- Token counts are estimates (`o200k_base`). The gateway serves a dozen
  tokenizer families and exposes no count endpoint.
