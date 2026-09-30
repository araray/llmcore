# FriendliAI provider — usage guide

`llmcore` ships a first-class provider for [FriendliAI](https://friendli.ai/docs),
covering all three of its inference surfaces through one configuration section:

| `endpoint_type` | Surface | Base URL | `model` field |
|---|---|---|---|
| `serverless` (default) | **Friendli Model APIs** — the hosted, pay-per-token catalog | `https://api.friendli.ai/serverless/v1` | catalog model ID (`zai-org/GLM-5.3`) |
| `dedicated` | **Friendli Dedicated Endpoints** — your own GPU deployments | `https://api.friendli.ai/dedicated/v1` | **endpoint ID** (or `ENDPOINT_ID:ADAPTER_ROUTE` for Multi-LoRA) |
| `container` | **Friendli Container** — self-hosted Friendli Engine | your URL (required) | whatever the container serves |

The chat endpoint is OpenAI-compatible, so everything llmcore already does —
streaming, tool calling, structured output, sessions, RAG, agents — works
unchanged. On top of that the provider exposes Friendli's own extensions:
reasoning controls, chat-template switches, Friendli Engine sampling,
regex-constrained output, an exact tokenizer endpoint, and the Suite billing API.

---

## 1. Install & configure

```bash
pip install "llmcore[friendli]"        # openai + httpx (+ the optional vendor SDK)
export FRIENDLI_TOKEN="flp_..."        # https://friendli.ai/suite/~/setting/keys
export FRIENDLI_TEAM_ID="..."          # optional; scopes requests + billing reads
```

### Environment variables

The provider accepts every spelling in circulation, checked in this order:

| Purpose | Variables (in order) |
|---|---|
| API key | `api_key` → `api_key_env_var` → `FRIENDLI_TOKEN` → `FRIENDLIAI_API_KEY` → `FRIENDLI_API_KEY` |
| Team ID | `team_id` → `team_id_env_var` → `FRIENDLI_TEAM_ID` → `FRIENDLIAI_TEAM_ID` |

`FRIENDLI_TOKEN` is the variable the official SDK reads; `FRIENDLIAI_API_KEY` is
the spelling used throughout friendli.ai's own documentation examples. Both work,
so no renaming is required if you already have one of them set.

### `[providers.friendli]`

```toml
[providers.friendli]
# api_key = "flp_..."              # prefer FRIENDLI_TOKEN / FRIENDLIAI_API_KEY
# team_id = "..."                  # prefer FRIENDLI_TEAM_ID / FRIENDLIAI_TEAM_ID
endpoint_type = "serverless"       # "serverless" | "dedicated" | "container"
# backend = "openai"               # "openai" (default) | "httpx" | "sdk"
default_model = "zai-org/GLM-5.3"
timeout = 300

# --- Reasoning ---
# reasoning_effort = "high"        # minimal|low|medium|high|xhigh|max|ultracode
# reasoning_budget = 10000         # hard cap on chain-of-thought tokens
parse_reasoning = true             # split reasoning out of `content`
# include_reasoning = true
# enable_thinking = true           # chat_template_kwargs.enable_thinking

# --- Token counting ---
native_token_count = false         # true => exact, but 1 API request per count
fallback_context_length = 131072
```

`friendliai` and `friendli_ai` are accepted as aliases for the provider name
(`get_provider("friendliai")`).

---

## 2. Transport backends — and why the vendor SDK is not the default

`backend` selects how requests reach Friendli:

| Backend | Library | Notes |
|---|---|---|
| `openai` | `openai` (`AsyncOpenAI`) | **Default.** Native async, battle-tested SSE, Friendli extensions travel in `extra_body`. |
| `httpx` | `httpx` | Direct REST. Same wire format, no SDK in the path. |
| `sdk` | `friendli` (`AsyncFriendli`) | The official SDK. Fully supported, but lossy — see below. |

With `backend` unset (or `"auto"`) the provider resolves **openai → httpx → sdk**
based on what is installed.

The vendor SDK is last on purpose. Its generated response models are strict
(`extra` is ignored), so any field Friendli returns outside the published schema
is silently dropped — including `reasoning_content` and `reasoning` on assistant
messages — and it offers no `extra_body` escape hatch for request fields it does
not declare. Verified directly:

```python
from friendli.models import ServerlessChatCompleteSuccess
ServerlessChatCompleteSuccess.model_validate({
    ..., "choices": [{"index": 0, "finish_reason": "stop",
                      "message": {"role": "assistant", "content": "hi",
                                  "reasoning_content": "THINK"}}],
}).model_dump()
# -> message == {"role": "assistant", "content": "hi"}     # reasoning_content gone
```

The `openai` and `httpx` backends both preserve it. If you set `backend = "sdk"`
while `parse_reasoning` is on, the provider logs a warning at startup.

---

## 3. Chat

```python
from llmcore import LLMCore

CONFIG = {
    "llmcore": {"default_provider": "friendli"},
    "providers": {"friendli": {"default_model": "zai-org/GLM-5.3-Flash"}},
}

async with await LLMCore.create(config_overrides=CONFIG) as llm:
    print(await llm.chat("What makes the Friendli Engine fast?"))

    # Streaming
    async for chunk in await llm.chat("Explain speculative decoding.", stream=True):
        print(chunk, end="", flush=True)
```

Per-request Friendli parameters go straight through `chat()` / `chat_completion()`:

```python
await llm.chat(
    "Plan a migration.",
    provider_name="friendli",
    reasoning_effort="max",       # minimal|low|medium|high|xhigh|max|ultracode
    reasoning_budget=8000,        # cap the chain of thought
    top_k=40, min_p=0.05,         # Friendli Engine sampling
    repetition_penalty=1.05,
)
```

---

## 4. Reasoning

Friendli splits reasoning control across four body fields plus the chat template.
The provider applies your configured defaults and lets any call override them.

| Parameter | Type | Meaning |
|---|---|---|
| `reasoning_effort` | str | How hard the model thinks. Accepted tiers vary per model — see `reasoning_options` in the model's catalog entry. |
| `reasoning_budget` | int | Hard cap (tokens) on the chain of thought. |
| `parse_reasoning` | bool | Split the chain of thought out of `content` into `reasoning_content`. |
| `include_reasoning` | bool | With parsing on, include the parsed reasoning in the response. |
| `enable_thinking` | bool | Chat-template switch for *controllable* reasoning models (e.g. `zai-org/GLM-5.2`). Folded into `chat_template_kwargs`. |
| `clear_thinking` | bool | Chat-template switch; drop prior reasoning from the context window. |

Reading the parsed reasoning back:

```python
provider = llm._provider_manager.get_provider("friendli")

resp = await provider.chat_completion(
    [Message(role=Role.USER, content="Is 8191 prime? Think it through.")],
    reasoning_effort="high",
)
print(provider.extract_response_content(resp))     # the answer
print(provider.extract_reasoning_content(resp))    # the chain of thought

# Streaming: reasoning arrives as its own delta field
async for chunk in await provider.chat_completion(msgs, stream=True):
    text = provider.extract_delta_content(chunk)
    think = provider.extract_delta_reasoning_content(chunk)
```

A model's supported tiers are discoverable — `get_models_details()` puts the raw
`reasoning_options` in `ModelDetails.metadata`, and the generated model cards
carry `provider_extension.reasoning_effort_levels`.

---

## 5. Tool calling and structured output

Tool calling is OpenAI-shaped and works through llmcore's unified `Tool` /
`ToolCall` types, including the full assistant-`tool_calls` → `role="tool"`
round trip.

```python
tool = Tool(name="get_weather", description="Get the weather for a city.",
            parameters={"type": "object", "properties": {"city": {"type": "string"}},
                        "required": ["city"]})

resp = await provider.chat_completion(msgs, tools=[tool], tool_choice="required")
calls = provider.extract_tool_calls(resp)     # [ToolCall(name='get_weather', ...)]
```

`response_format` accepts `json_schema`, `json_object`, `text`, and Friendli's
**`regex`** variant:

```python
await provider.chat_completion(
    msgs,
    response_format={"type": "regex", "schema": r"[A-Z][a-z]+, [A-Z]{2}"},
)
```

Friendli rejects some combinations outright — `min_tokens` and `response_format`
alongside `tools`, and `min_tokens` alongside `response_format`. The provider
drops the offending field with a warning rather than letting the request 422.

---

## 6. Multimodal input

Vision/audio/video models take content parts. Supply them through message
metadata and the provider assembles the payload:

```python
Message(
    role=Role.USER,
    content="What is in this image?",
    metadata={"inline_images": ["https://example.com/photo.png"]},
)
# also: inline_audio, inline_videos, or a ready-made content_parts list
```

Entries may be HTTPS URLs or base64 data URIs. Which modalities a model accepts
is in its catalog entry (`input_modalities`) and on its model card
(`capabilities.vision` / `audio_input` / `video_input`).

---

## 7. Model discovery and cost data

Model APIs exposes an unusually rich catalog, which the provider maps onto
`ModelDetails` and `cardctl` turns into model cards:

```python
for d in await provider.get_models_details():
    print(d.id, d.context_length, d.supports_reasoning, d.metadata["pricing"])
```

Regenerate the bundled cards after Friendli changes the catalog:

```bash
python -m tools.cardctl generate friendli     # context, pricing, capabilities, reasoning options
python -m tools.cardctl diff friendli         # read-only comparison vs the live API
python -m tools.cardctl validate friendli
```

Pricing comes straight from the API (per-token USD, converted to per-million),
so cards stay accurate without manual curation. The overlay in
`tools/cardctl/enrichments/friendli.toml` only carries what the API cannot know:
architecture family/type, display names, aliases.

Dedicated Endpoints and Containers serve a single deployment each and have no
catalog, so `get_models_details()` describes the configured model from its card.

---

## 8. Auxiliary endpoints

```python
provider = llm._provider_manager.get_provider("friendli")

# Exact tokenization with the model's own tokenizer
tokens = await provider.tokenize("What is generative AI?")   # [3838, 374, ...]

# Raw (non-chat) completions — the chat template is NOT applied
resp = await provider.text_completion("Once upon a time", max_tokens=64)

# Audio transcription (all three endpoint types)
result = await provider.transcribe_audio(
    "/path/to/audio.mp3", model="openai/whisper-large-v3", language="en"
)

# Embeddings and image generation (Dedicated Endpoints / Container only)
vectors = await provider.create_embeddings(["hello"])
image = await provider.generate_image("an orange Lamborghini", num_inference_steps=10)

# Friendli Suite billing (uses the team ID / X-Friendli-Team header)
cost = await provider.get_team_cost("2026-09-01T00:00:00Z", "2026-09-20T00:00:00Z")
usage = await provider.get_team_usage("2026-09-01T00:00:00Z", "2026-09-20T00:00:00Z")
```

`detokenize()` and `render_chat()` are implemented against the documented
`/detokenize` and `/chat/render` routes, but those currently return **404 on
Model APIs** (verified 2026-09-20); use them on Dedicated Endpoints or a
Container. Everything that depends on them degrades gracefully.

---

## 9. Token counting

`count_tokens()` and `count_message_tokens()` are **local by default**
(tiktoken `cl100k_base`, then a character-ratio estimate). Set
`native_token_count = true` to route them through Friendli's `/tokenize`
endpoint for exact, per-model counts.

The trade-off is rate limit, not accuracy: llmcore counts tokens on every turn
for context budgeting, and each native count is one API request. Model APIs
limits scale with your usage tier — tier 0 is only a couple of requests per
minute — so native counting is opt-in. `tokenize()` / `detokenize()` remain
available regardless of the setting, and a failed native count falls back
locally rather than raising.

For an exact, template-accurate prompt count on a surface that serves
`/chat/render`:

```python
exact = len(await provider.tokenize(await provider.render_chat(messages)))
```

---

## 10. Dedicated Endpoints and Container

```toml
[providers.friendli_dedicated]
type = "friendli"
endpoint_type = "dedicated"
default_model = "YOUR_ENDPOINT_ID"        # not a model name
# default_model = "YOUR_ENDPOINT_ID:adapter-route"   # Multi-LoRA

[providers.friendli_local]
type = "friendli"
endpoint_type = "container"
base_url = "http://localhost:8000/v1"     # REQUIRED
# api_key is optional when the container has no auth
default_model = "meta-llama/Llama-3.1-8B-Instruct"
```

Both sections coexist with `[providers.friendli]`; pick one per call with
`provider_name=`.

---

## 11. Errors and rate limits

| Status | Mapped to | Message hints |
|---|---|---|
| 400/422 with overflow wording | `ContextLengthError` | carries the model's context limit |
| 401 | `ProviderError` | which env vars to check; keys start with `flp_` |
| 403 | `ProviderError` | names the active `X-Friendli-Team` scope |
| 404 | `ProviderError` | endpoint-ID vs model-name confusion, and the routes that 404 on Model APIs |
| 429 | `ProviderError` (`retryable=True`) | links the rate-limit tiers |
| other | `ProviderError` | status + body |

Model APIs rate limits are tier-based and rise with lifetime spend; tier 0 gets
"adaptive" limits that in practice are a couple of requests per minute, and the
responses carry `x-ratelimit-limit-requests` / `x-ratelimit-remaining-requests` /
`x-ratelimit-reset-requests`. Since 429 is marked retryable, llmcore's
`chat_completion_with_retry` policy applies on top.

---

## 12. References

- Documentation index: <https://friendli.ai/docs/llms.txt>
- OpenAI compatibility: <https://friendli.ai/docs/guides/openai-compatibility>
- Chat completions API: <https://friendli.ai/docs/openapi/model-apis/chat-completions>
- Reasoning: <https://friendli.ai/docs/guides/capabilities/reasoning>
- Structured outputs: <https://friendli.ai/docs/guides/structured-outputs>
- Models & pricing: <https://friendli.ai/docs/guides/model-apis/pricing>
- Rate limits: <https://friendli.ai/docs/guides/model-apis/rate-limits>
