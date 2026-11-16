# Command Code provider: usage guide

[Command Code](https://commandcode.ai) is a coding agent with a paid model
gateway, the **Provider API**. One key reaches 98+ models: Claude, GPT, Gemini,
Grok, DeepSeek, Kimi, GLM, Qwen, MiniMax, Muse and more. It also serves
TypeSafe's Jev decision model.

The gateway speaks **four protocols**, and each model answers on only some of
them:

| Models | Route | Wire shape |
|---|---|---|
| Claude | `/messages` | Anthropic Messages |
| GPT (default policy) | `/responses` | OpenAI Responses |
| everything else | `/chat/completions` | OpenAI Chat Completions |
| `typesafe/jev` | `/systemone` | TypeSafe System One |

Sending a model to a route that does not serve it returns
`400 unsupported_model`. llmcore reads each model's routes from the live
`/models` catalog and sends the call there, so you never pick the route
yourself. You call `chat()` as with any provider. Every route's answer comes
back in the same Chat Completions shape.

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
# zdr = true                 # zero data retention on every request (or env CMD_ZDR=1;
                             # setting zdr = false here would override that variable)
responses_api = "openai"     # GPT -> /responses | "all" | "never"
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
- **GPT** (`/responses`, the default for GPT): `effort` is sent as
  `reasoning.effort`. Pass `reasoning_summary="auto"` (or
  `reasoning={"effort": ..., "summary": "auto"}`) to also get a readable
  summary, when the model offers one.
- **Everything else** (`/chat/completions`): `effort` is sent as
  `reasoning_effort` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`,
  `max`). Upstream-specific `reasoning={...}` and `thinking={...}` are also
  accepted and passed through.

Reasoning text is available the same way on every route:

```python
resp = await cc.chat_completion(messages, "deepseek/deepseek-v4-pro")
cc.extract_reasoning_content(resp)   # open models' `reasoning`, or Claude's thinking
cc.extract_reasoning_details(resp)   # structured reasoning to send back on the next turn
```

For multi-turn tool use on reasoning models, send the model's reasoning back
on the next request. On `/chat/completions`, put `extract_reasoning_details(resp)`
in that assistant message's `metadata["reasoning_details"]`. On `/responses`,
put `extract_reasoning_items(resp)` in `metadata["reasoning_items"]` (see the
Responses section below).

### Tools, vision, caching

These work as they do on the underlying protocol, because each route is served
by llmcore's own Anthropic or OpenAI implementation. That covers `tools=` and
`tool_choice=`, images through `metadata["inline_images"]`, and Claude prompt
caching through `metadata["cache_control"]`. Request bodies are capped at
32 MiB.

### OpenAI Responses API

Most non-Claude models answer on both `/chat/completions` and `/responses`.
`responses_api` decides which one is used:

| `responses_api` | GPT models | other models |
|---|---|---|
| `"openai"` (default) | `/responses` | `/chat/completions` |
| `"all"` | `/responses` | `/responses` where offered |
| `"never"` | `/chat/completions` | `/chat/completions` |

The default is the same split Command Code's official pi provider makes.
Override it per call with `responses=True` or `responses=False`. A model listed
on only one of the two routes uses that route whatever the setting.

What the Responses route adds:

- **Reasoning that survives a tool loop.** Requests are stateless
  (`store: false`) and ask for encrypted reasoning. Take
  `cc.extract_reasoning_items(resp)` and put it in that assistant turn's
  `metadata["reasoning_items"]`. llmcore sends it back ahead of the turn, so
  the model keeps its chain of thought without anything stored server-side:

  ```python
  first = await cc.chat_completion(history, "gpt-6-sol", tools=[weather])
  history.append(Message(
      role=Role.ASSISTANT, content="",
      tool_calls=first["choices"][0]["message"]["tool_calls"],
      metadata={"reasoning_items": cc.extract_reasoning_items(first) or []},
  ))
  history.append(Message(role=Role.TOOL, content="22C", tool_call_id="call_..."))
  second = await cc.chat_completion(history, "gpt-6-sol", tools=[weather])
  ```

- **Parameters in either spelling.** `max_tokens`, `response_format`,
  `reasoning_effort` and `effort` are translated into `max_output_tokens`,
  `text.format` and `reasoning.effort`. The Responses fields themselves
  (`reasoning={"effort": ..., "summary": "auto"}`, `text`, `include`,
  `previous_response_id`, `store`, `truncation`, …) are accepted and win when
  both are given.
- **Raw tool declarations** through `extra_tools=[...]` (`custom`,
  `local_shell`), next to llmcore `Tool` objects. Command Code forwards built-in
  tools such as `web_search` upstream, but in testing (2026-11-15) a
  `web_search` declaration produced no search call and no citations. So
  `native_search` is not advertised.
- **Status mapping.** `status: "incomplete"` reports `finish_reason: "length"`
  (or `content_filter`). `status: "failed"` raises a `ProviderError` with the
  gateway mapping, retryable where the upstream says so.

The request/response conversion is a standalone module,
`llmcore.providers.openai_responses`, so other providers can reuse it.

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

- Built-in Responses tools (`web_search`) are forwarded by the gateway, but
  have not been seen to run. See the Responses section.
- Command Code publishes no rate-limit headers and no usage or balance API.
  Spend is visible in Studio.
- Token counts are estimates (`o200k_base`). The gateway serves a dozen
  tokenizer families and exposes no count endpoint.
