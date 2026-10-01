---
id: skills/llmcore/routing
name: Routing in llmcore — pools, lanes and failover
version: 1.0.0
description: >
  Configure and debug llmcore routing: reach any provider+model on the fly,
  fail over across a pool, classify requests into lanes, and keep sensitive
  prompts on your own hardware.
tags: [llmcore, routing, pools, lanes, failover, privacy]
sections:
  - id: mental_model
    tags: [routing]
  - id: dynamic_targets
    tags: [routing, targets]
  - id: pools
    tags: [routing, pools, failover]
  - id: strategies
    tags: [routing, pools, cost]
  - id: lanes
    tags: [routing, lanes, classifiers]
  - id: privacy
    tags: [routing, privacy, pii]
  - id: effort
    tags: [routing, effort, parameters]
  - id: overrides
    tags: [routing, config]
  - id: debugging
    tags: [routing, debugging]
---

## mental_model

Five layers that compose. Adopt any one without the others:

| Layer | Answers |
|---|---|
| **Target** | which provider, model and parameters? |
| **Pool** | if that one is broken, slow or broke, what else? |
| **Lane** | what *kind* of request is this? |
| **Cascade** | was the cheap answer good enough? |
| **Transform** | what must not leave this machine? |

Two words that are deliberately different things. A **pool** is a set of
*interchangeable* targets — it exists for failover. A **lane** is a *named
destination* a classifier picks — it exists to express intent (`cheap`, `deep`,
`private`). A lane points at a pool, or at one target.

Nothing is on by default. With no `[routing]` configuration, every call resolves
exactly as it did before routing existed.

## dynamic_targets

**Config is a set of presets, not an allow-list.** Any provider+model is
reachable by spec string as long as its credential is in the environment, with
or without a `[providers.*]` section:

```python
await llm.chat("hi", target="xai:grok-4.1-20251117?effort=high")
await llm.chat("hi", target="ollama:llama3.3:70b")        # colons in the model are fine
await llm.chat("hi", target="vllm:Qwen/Qwen3-30B#my-box")  # '#' pins an instance
```

Grammar: `provider:[model][#instance][?params]`. The provider is everything
before the **first** colon, because model names contain colons and slashes.

llmcore builds the instance on demand and marks it ephemeral. To forbid that —
a shared service that must not reach an unbudgeted vendor — set
`routing.autoprovision = false`.

A pinned `target=` bypasses classification and pools entirely, which makes it
the way to reproduce a result exactly.

## pools

```toml
[routing.pools.main]
targets = [
  "anthropic:claude-opus-5-5?effort=high",
  "openai:gpt-5.4",
  "gemini:gemini-3.8-flash",
]
strategy = "priority"
max_attempts = 3
```

```python
await llm.chat("hi", pool="main")
```

**Failures are not treated alike**, and that is the part that matters:

| Condition | This target | Another? |
|---|---|---|
| 429 rate limit | short cooldown, honours `Retry-After` | yes |
| insufficient credit / 402 | **long** cooldown (minutes), balance recorded as zero | yes |
| timeout | short cooldown | yes |
| 5xx / connection | retried **once in place**, then moved | yes |
| 401 / 403 auth | unusable for the process — a bad key does not heal | yes |
| 404 / model not found | long cooldown; deterministic here, but a peer has a different model | yes |
| prompt too long | **no** cooldown — routed to a larger window instead | yes |
| 400 bad request | fails everywhere | **no** |
| content refusal | configurable, default **no** | no |

Two of these are worth knowing about:

- **An over-long prompt is a routing signal, not an error.** llmcore knows every
  model's context window from its card, so a prompt that overflows a 200k model
  goes to a 1M model instead of failing.
- **Refusals do not fail over by default.** Retrying a refusal on another vendor
  is "shop until someone says yes". Opt in with
  `routing.on_refusal = "failover"` if that is what you want.

Order tiers express "my hardware first, pay only if it is down":

```toml
[routing.pools.local_first]
targets = ["ollama:llama3.3:70b", "openai:gpt-5.4?order=1"]
```

Tier 1 is held back entirely while tier 0 has a usable member.

## strategies

| Strategy | Picks | Needs |
|---|---|---|
| `priority` | first healthy target in declared order | nothing |
| `round_robin` | next healthy target | nothing |
| `weighted` | weighted random (`?weight=3`; `0` = spare only) | weights |
| `lowest_latency` | lowest smoothed latency; unmeasured explored first | observations |
| `lowest_cost` | cheapest for *this* prompt, from card pricing | model cards |
| `least_busy` | fewest in-flight requests | nothing |
| `most_credits` | largest known balance | a balance endpoint |

`lowest_cost` is the one most likely to pay for itself: pricing comes from the
bundled model cards and the token count from the prompt, so a long prompt and a
short one can legitimately choose differently. Self-hosted providers are priced
at zero, so a local model wins outright.

Set the default once:

```toml
[routing]
default_strategy = "lowest_cost"
```

Three honest limits:

- An **unpriced** target ranks *after* every priced one. Treating unknown as
  zero would make a model with no card beat one known to be free.
- An **unknown balance** ranks after a known one but *ahead of* a known-empty
  one. Most vendors expose no balance at all.
- `lowest_latency` tries an unmeasured target first, because it cannot prefer
  low latency without a measurement.

## lanes

A classifier names a lane; the lane→model binding is separate config. So
swapping models never touches the classifier.

```toml
[routing.lanes]
trivial = "pool:cheap"
standard = "pool:main"

[routing.lanes.deep]
target = "anthropic:claude-opus-5-5"
params = { effort = "max" }
description = "Multi-step reasoning, proofs, architecture review, hard debugging"

[routing.classifier]
chain = ["hint", "magic_string", "heuristic"]
min_confidence = 0.55
bias = "quality"
```

`description` is **not decoration** — the zero-shot classifiers score the prompt
against these descriptions, so this text is what makes them work.

Chain members, cheapest first:

| Classifier | How | Cost |
|---|---|---|
| `hint` | `lane=` / `complexity=` / `effort=` on the call | free |
| `script` | your own function: `"my_module:classify"` | free |
| `magic_string` | `[[lane:deep]]` in the prompt, stripped before egress | free |
| `heuristic` | length, code fences, simple-task verbs | free |
| `local_encoder` | a 350M encoder on CPU, zero-shot over your lanes | ~200–300 ms |
| `typesafe_jev` | a TypeSafe `choice` question | one cheap call |
| `llm` | ask a cheap llmcore target | one cheap call |

llmcore reorders the chain: cheapest first, and within a cost band instructions
before guesses. So a `lane=` argument always beats a length heuristic, and a
marker found in *content* never overrides either the caller or your own script —
in a RAG path that content may not be yours.

`bias = "quality"` leans to the stronger lane on a borderline call, because the
two mistakes are not symmetric: too cheap gives a bad answer, too expensive only
costs money.

## privacy

The strong primitive is **routing, not redaction**:

```toml
[routing.lanes.private]
pool = "local_only"

[routing.pools.local_only]
targets = ["ollama:llama3.3:70b", "vllm:Qwen/Qwen3-30B#my-box"]

[routing.transforms]
chain = ["pii"]

[routing.transforms.pii]
detector = "regex"        # or "vela_pii", a local 307M token classifier
on_detect = "constrain"   # route it somewhere it cannot leak
pool = "local_only"
redact = true             # and redact anyway, as defence in depth
```

**Say this out loud before relying on it:** a detector that misses one
identifier has leaked it, and no detector catches everything. `constrain` is a
guarantee because a miss stays on your own hardware. `redact` is a mitigation.
Do not describe redaction to anyone as a guarantee.

Findings are recorded as a hash, never the matched value. `on_detect = "block"`
refuses to send at all.

## effort

llmcore has one effort vocabulary — `none`, `minimal`, `low`, `medium`, `high`,
`xhigh`, `max` — mapped per provider:

```python
await llm.chat("prove this", effort="max")
await llm.chat("rename it", target="openai:gpt-5.4?effort=minimal")
```

Precedence, lowest first: model-card defaults → `[providers.*]` → target params
→ lane params → profile → the per-call keyword. The per-call value always wins.

Profiles name a bundle once:

```toml
[routing.profiles.frugal]
effort = "minimal"
max_tokens = 512
```

```python
await llm.chat("summarise", profile="frugal")
```

A target that does not support a parameter does not fail the request: under a
pool it is dropped with a warning, because members genuinely have different
parameter surfaces. Outside a pool a bogus parameter still raises, since you
named one provider and one parameter that does not exist.

## overrides

**Config is a warm-up, not a cage.** Every `[routing]` setting resolves
config → environment → request, and the request wins:

```bash
export LLMCORE_ROUTING__MAX_ATTEMPTS=5
export LLMCORE_ROUTING__ON_REFUSAL=failover
```

```python
await llm.chat("hi", pool="main", routing={"max_attempts": 5, "on_refusal": "failover"})
```

A misspelled override raises rather than being ignored — one that silently did
nothing would be a routing bug nobody could see.

## debugging

```python
plan = await llm.routing.explain("summarise this file")
print(plan.summary())
print(await llm.routing.why("summarise this file"))   # multi-line, with reasons

llm.routing.health()        # per-target cooldowns, latency, failures, balance
llm.routing.pools()
llm.routing.lanes()
llm.routing.classifiers()   # the chain in the order it will actually run
llm.routing.settings()      # resolved, after config and env

await llm.routing.clear_health("openai:gpt-5.4")   # after topping up or fixing a key
await llm.routing.probe_balances()
```

`explain()` runs the classifier chain for real but never calls the target model,
so a chain with a paid classifier makes that one cheap call.

Every decision also emits a structured event on llmcore's event spine —
`routing.classified`, `routing.selected`, `routing.attempt`,
`routing.failover`, `routing.exhausted` — carrying the candidates considered
and why each was skipped.

Common surprises:

- **"It picked the slow one."** `explain()` will show the fast one in a cooldown
  with the seconds remaining.
- **"Lane X is not configured."** The classifier named a lane you have not
  defined; llmcore logs it and falls back to the default pool.
- **An auth failure benches a target for the whole process.** That is deliberate.
  `clear_health()` undoes it without a restart.
