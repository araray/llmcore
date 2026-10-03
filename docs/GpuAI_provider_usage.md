# gpu.ai provider

[gpu.ai](https://gpu.ai) sells serverless inference behind an
OpenAI-compatible API, media generation (image and video), and GPU rental.
llmcore's provider covers the **text surface**: 14 chat models and one
embedding model.

The other two surfaces are not served yet. The 20 image and 14 video models
need media adapters, and GPU rental needs a `runtimes` backend — llmcore has
only ever had one of those (Colab), which is also why DeepInfra's rental
half is unsupported.

## Setup

```toml
[providers.gpuai]
# default_model = "gpuai/qwen3.8-flash"
# base_url = "https://api.gpu.ai/v1"
```

```bash
export GPUAI_API_KEY="..."        # or GPU_AI_API_KEY
```

```python
answer = await llm.chat(
    "Summarise this changelog",
    provider_name="gpuai",
    model_name="gpuai/qwen3.8-flash",
)

vectors = await provider.create_embeddings(
    ["first", "second"], model="gpuai/qwen3-embedding-8b"
)
```

Model ids already carry the `gpuai/` prefix, so pass them as the vendor
publishes them. Many models also have an alias (`qwen-flash`), and aliases
resolve to the same card.

## Two things to know before you trust a number

### `max_tokens` includes reasoning, so a small budget returns nothing

On reasoning models the budget is consumed by the reasoning pass before any
content is emitted. Observed on `gpuai/gpt-oss-120b`:

| `max_tokens` | content | completion tokens |
|---|---|---|
| 8 | `""` | 8 |
| 256 | `"OK"` | 45 |

So a layer that trims `max_tokens` to save money produces **empty
responses**, not shorter ones. `MIN_REASONING_MAX_TOKENS` (256) documents a
floor worth respecting, and an empty result with a truncation finish reason
should be treated as a failure worth retrying elsewhere rather than a valid
answer.

### Reasoning tokens are billed as output, with no breakdown

Responses carry no `completion_tokens_details`, so the reasoning share
cannot be separated from visible output — the 45 tokens above were almost
all reasoning. Cost estimates therefore **assume** reasoning is charged at
the output rate, which the catalogue's single output price supports. The
assumption is recorded on each card under
`provider_metadata.reasoning_billing` rather than left implicit.

## Pricing

Model cards for gpu.ai are generated from its `/v1/models` catalogue, which
is unusually complete — it states `context_length`, `pricing`,
`supported_parameters` and `aliases` inline. Rates are published in cents
per million tokens and converted to dollars for the card.

Media models are priced per image, per megapixel, per video or per second of
video, which `per_unit` carries:

```python
card = registry.get("gpuai", "gpuai/happyhorse-1.0-t2v")
cost = card.pricing.get_cost(0, 0, video_seconds=5)   # 24¢/s -> $1.20
```

> **Prices here are a reference, not an invoice.** They come from gpu.ai's
> published rate card. What a call actually costs can differ — serverless
> tiers change, promotional rates lapse, and the reasoning-token opacity
> above means the billed output volume is not fully visible from a response.
> Use these figures to compare and plan, and check real numbers against your
> own gpu.ai billing before relying on them.

## Behaviour notes

- **Usage follows the OpenAI inclusive convention**: `prompt_tokens`
  *includes* `prompt_tokens_details.cached_tokens`. llmcore's usage
  extraction decides this from the provider's own key names, so cached
  tokens are priced correctly without configuration.
- **`encoding_format` is rejected** by gpu.ai — the parameter itself, not a
  particular value. The OpenAI SDK injects it, so this provider builds the
  embedding payload directly and sends only `model` and `input`.
- **Image and video models are filtered out of discovery.** They exist in
  the catalogue and have cards, but listing them as chat models would make
  them selectable by a router that cannot call them.
- **Streaming support is read from the catalogue.** It enumerates accepted
  parameters, so a model without `stream` genuinely does not accept it.
