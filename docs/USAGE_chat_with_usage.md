# `chat_with_usage` — per-call token usage

`LLMCore.chat_with_usage(...)` is the **usage-returning** companion to
`chat()`. It lets a *caller* meter token consumption per call **without**
turning on session persistence, by returning the prompt/completion token
counts that llmcore already computes internally.

It is **additive and opt-in**: `chat() -> str` is unchanged, and nothing in
llmcore depends on this surface. Callers who don't need usage keep using
`chat()`.

## When to use it

- You want to **count tokens per call** (e.g. to drive a usage/quota or
  cost system) but you do **not** want to persist a session.
- You are integrating an external metering layer (such as Convergence's
  metering bridge) that prefers an explicit `(text, usage)` return over a
  stateful "last call" accessor.

If you only need the text, use `chat()`. If you want aggregate per-session
totals after persisting a session, use `get_session_usage_stats(session_id)`.

## API

```python
async def chat_with_usage(
    message: str,
    *,
    session_id: str | None = None,
    system_message: str | None = None,
    provider_name: str | None = None,
    model_name: str | None = None,
    save_session: bool = True,
    enable_rag: bool = False,
    # ... (same RAG / staging / tools keywords as chat())
    **provider_kwargs,
) -> tuple[str, ChatUsage]
```

`ChatUsage` is a frozen dataclass:

| Field | Type | Notes |
|---|---|---|
| `prompt_tokens` | `int \| None` | Prompt/context tokens (alias: `tokens_in`) |
| `completion_tokens` | `int \| None` | Response tokens (alias: `tokens_out`) |
| `total_tokens` | `int \| None` | `prompt + completion` as computed by llmcore |
| `provider` | `str \| None` | Provider that **served** the call |
| `model` | `str \| None` | Model that **served** the call |
| `is_available` | `bool` (property) | `True` when at least one count is present |

When usage cannot be determined, every count is `None` (and
`is_available is False`) so a downstream meter can no-op instead of recording a
zero-token event.

> **Non-streaming only.** Passing `stream=True` raises `ValueError`. For
> streaming, use `chat(stream=True)` and read totals afterwards with
> `get_session_usage_stats(session_id)`.

## Examples

### Meter a transient call (no session saved)

```python
from llmcore import LLMCore

llm = await LLMCore.create()
text, usage = await llm.chat_with_usage(
    message="Summarise the attached notes.",
    provider_name="openai",
    save_session=False,          # nothing is persisted
)
print(text)
if usage.is_available:
    print(usage.tokens_in, usage.tokens_out, usage.total_tokens)
    print(usage.provider, usage.model)
```

### Compute a usage unit (the metering pattern)

```python
# units = multiplier * (tokens_in * w_in + tokens_out * w_out)
text, usage = await llm.chat_with_usage(message=prompt, save_session=False)
if usage.is_available:
    units = multiplier * (usage.tokens_in * w_in + usage.tokens_out * w_out)
```

### Keep a real session (and its introspection)

```python
text, usage = await llm.chat_with_usage(
    message="Continue our discussion.",
    session_id="chat_123",       # caller-owned: NOT cleaned up
    save_session=True,
)
# Introspection for the caller-owned session still works afterwards:
details = llm.get_last_interaction_context_info("chat_123")
assert details.prompt_tokens == usage.tokens_in
```

## Behavioural notes

- **As-served counts.** `provider` / `model` and the token counts reflect the
  model that actually produced the response (post-fallback), not a
  requested-but-unavailable one.
- **Call failures are not swallowed.** Only *missing usage* degrades to
  all-`None`; provider/context/storage errors propagate exactly as from
  `chat()`.
- **Concurrency-safe.** Usage is read back under a call-local session id, so
  concurrent `chat_with_usage` calls never read each other's counts.
- **No residue.** When `session_id` is omitted, an ephemeral session id is
  synthesised and its transient caches are dropped on return.

## Drop-in for `tokens_in` / `tokens_out` consumers

Consumers that read `.tokens_in` / `.tokens_out` off the returned object work
unchanged, because `ChatUsage` exposes those as aliases of `prompt_tokens` /
`completion_tokens`:

```python
_, usage = await llm.chat_with_usage(message="hi", save_session=False)
tokens_in, tokens_out = usage.tokens_in, usage.tokens_out
```
