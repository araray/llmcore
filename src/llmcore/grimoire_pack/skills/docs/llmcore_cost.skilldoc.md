---
id: skills/llmcore/cost
name: Knowing and reducing what llmcore spends
version: 1.0.0
description: >
  Estimate a call before making it, see what was actually spent by provider,
  model, lane and pool, and apply the levers that reduce it without guessing.
tags: [llmcore, cost, routing, observability]
sections:
  - id: before_you_optimise
    tags: [cost]
  - id: estimate
    tags: [cost, estimate]
  - id: measure
    tags: [cost, observability]
  - id: levers
    tags: [cost, routing]
  - id: cascade
    tags: [cost, cascade]
  - id: limits
    tags: [cost]
---

## before_you_optimise

Measure first. The two numbers that decide everything are **which model served
each request** and **how many tokens it used**, and llmcore has both. Guessing
which part of a workload is expensive is usually wrong — in agent workloads the
cost is normally a small number of very long contexts, not the number of calls.

Order of payoff, in practice:

1. shorten the context (usually the biggest single win)
2. route cheap turns to cheap models
3. lower effort/thinking where it is not needed
4. cascade (cheap first, escalate on failure) for batch and agent work

## estimate

Free, no call made:

```python
plan = await llm.routing.explain("summarise this file")
print(plan.estimated_cost_usd, plan.chosen.spec())
```

The estimate uses the bundled model cards' pricing and the prompt's own token
count, so it reflects *this* prompt rather than a static ranking. It returns
`None` when the model has no card, which means **unknown** — not free. A
self-hosted provider (`ollama`, `vllm`) estimates at `0.0`, because there is no
per-token vendor charge; GPU time is real but is not something llmcore can
price.

Compare targets directly:

```python
for spec in ("openai:gpt-5.4", "anthropic:claude-opus-5-5", "gemini:gemini-3.8-flash"):
    plan = await llm.routing.explain(prompt, target=spec)
    print(f"{spec:34} ${plan.estimated_cost_usd:.6f}")
```

## measure

Per-turn, after a call:

```python
answer = await llm.chat("...", session_id="s1")
info = llm.get_last_interaction_context_info("s1")
print(info.provider, info.model, info.prompt_tokens, info.completion_tokens)
```

Under a pool, `info.provider`/`info.model` name the target that **actually**
answered, which is not necessarily the one requested. The proxy reports the same
thing in an `llmcore` block on the response, so a harness's own cost log stays
correct.

Across calls, attach a sink to the event spine and keep the routing events:

```python
from llmcore import shared_events

def sink(event):
    if event.type == "routing.attempt" and event.payload["ok"]:
        record(event.payload["target"], event.payload.get("cost_usd"),
               event.payload["latency_seconds"])

shared_events.register_sink(sink)
```

Useful event types: `routing.classified` (which lane, on what evidence),
`routing.selected` (chosen target plus every rejected candidate and why),
`routing.attempt` (outcome, latency, cost), `routing.failover`,
`routing.exhausted`.

Health carries the running picture:

```python
llm.routing.health()    # per-target successes, failures, smoothed latency, balance
```

## levers

**1. Route cheap turns to cheap models.** The biggest structural win, and the
free classifiers cost nothing:

```toml
[routing.lanes]
trivial = "pool:cheap"
standard = "pool:main"
deep = "anthropic:claude-opus-5-5?effort=max"

[routing.classifier]
chain = ["hint", "magic_string", "heuristic"]
```

**2. Let the caller or the model say so.** Free and more accurate than any
classifier, because the caller knows the intent:

```python
await llm.chat("rename this variable", lane="trivial")
```

In a harness, the model can write `[[lane:trivial]]` into its own prompt;
llmcore strips the marker before it reaches a provider.

**3. Pick by price automatically.**

```toml
[routing.pools.main]
targets = ["openai:gpt-5.4", "anthropic:claude-opus-5-5", "gemini:gemini-3.8-flash"]
strategy = "lowest_cost"
```

**4. Lower effort where it is wasted.** Reasoning tokens are billed as output:

```toml
[routing.profiles.frugal]
effort = "minimal"
max_tokens = 512
```

```python
await llm.chat("reformat this JSON", profile="frugal")
```

**5. Prefer your own hardware.** Order tiers make a local model the default and
a paid API the fallback, with no code change:

```toml
[routing.pools.local_first]
targets = ["ollama:llama3.3:70b", "openai:gpt-5.4?order=1"]
```

**6. Cap the blast radius.** `routing.max_attempts` is a spend cap: on a
vendor-wide outage an unbounded pool walk turns one failed request into several
paid attempts.

## cascade

Answer cheaply, judge the answer, escalate only if it fell short:

```toml
[routing.cascade]
enabled = true
max_rungs = 2
threshold = 0.7
on_unknown = "accept"

[routing.cascade.rungs.default]
rungs = ["pool:cheap", "pool:main"]
verifier = "script"
script = "my_checks:answer_is_usable"
```

Off by default: it trades latency and an extra call for cost, which is the wrong
trade interactively and the right one for batch and agents.

**Use a `script` verifier if you possibly can.** "Does this compile", "does this
match the schema", "do the tests pass" are exact and cheap, where a model
judging another model is a guess that costs money. `max_rungs` caps how much a
cheap answer can end up costing.

`on_unknown = "accept"` is the default because escalating on every unjudgeable
answer inverts the saving the cascade exists for.

## limits

Stated plainly, because a cost feature that overstates itself is worse than none:

- **Estimates are estimates.** Output length is unknown before the call, so the
  estimate assumes a fixed output. It is good for *comparing* targets, not for
  billing.
- **No accuracy claim is made for any classifier.** None of them has been
  validated on your traffic. Measure the lane assignments on your own prompts
  before trusting them with quality-sensitive work.
- **A classifier wrong in the cheap direction costs you an answer**, not just
  money. That asymmetry is why `bias` defaults to `quality`.
- **Unpriced models are unknown, not free.** Cards do not cover every model, and
  llmcore ranks unknown prices after known ones rather than guessing.
- **A local classifier is not free in latency** — ~200–300 ms of CPU per call,
  measured. Worth it when it chooses between multi-second calls; not worth it to
  shave a cheap interactive turn.
