# Routing — usage guide

Reach any provider+model on the fly, fail over across a pool, route by request
kind, keep sensitive prompts on your own hardware, and put llmcore in front of
an agent harness as an OpenAI-compatible proxy.

For the design and the reasoning behind each decision, see
the routing subsystem design spec. This document is how
to use it.

**Nothing here is on by default.** With no `[routing]` section, every call
resolves exactly as it did before routing existed.

---

## Contents

- [Five layers](#five-layers)
- [1. Any model, on the fly](#1-any-model-on-the-fly)
- [2. Pools: failover](#2-pools-failover)
- [3. Lanes: routing by request kind](#3-lanes-routing-by-request-kind)
- [4. Cascades: cheap first, escalate on failure](#4-cascades-cheap-first-escalate-on-failure)
- [5. Transforms: the privacy path](#5-transforms-the-privacy-path)
- [Effort, thinking and parameters](#effort-thinking-and-parameters)
- [Proxy mode for agent harnesses](#proxy-mode-for-agent-harnesses)
- [A per-turn step budget](#a-per-turn-step-budget)
- [Inspecting what routing did](#inspecting-what-routing-did)
- [Measuring a classifier](#measuring-a-classifier)
- [Overriding anything, anywhere](#overriding-anything-anywhere)
- [What this does not claim](#what-this-does-not-claim)

---

## Five layers

| Layer | Answers | Useful alone? |
|---|---|---|
| **Target** | which provider, model and parameters? | yes — fixes on-the-fly targets |
| **Pool** | if that one is broken, slow or broke, what else? | yes — failover |
| **Lane** | what *kind* of request is this? | yes — classification |
| **Cascade** | was the cheap answer good enough? | yes — a quality floor |
| **Transform** | what must not leave this machine? | yes — privacy |

A **pool** is a set of *interchangeable* targets; it exists for failover. A
**lane** is a *named destination* a classifier picks; it exists to express
intent. A lane points at a pool, or at one target. They are kept separate
because a failover set needs no classifier and a one-target destination needs no
failover set.

---

## 1. Any model, on the fly

Config is a set of **presets, not an allow-list**. Any provider+model is
reachable by spec string as long as its credential is in the environment:

```python
await llm.chat("hi", target="xai:grok-4.1-20251117?effort=high")
await llm.chat("hi", target="ollama:llama3.3:70b")
await llm.chat("hi", target="vllm:Qwen/Qwen3-30B#my-box")
await llm.chat("hi", target="replicate:black-forest-labs/flux-schnell")
```

Grammar:

```
provider ":" [model] ["#" instance] ["?" params]
```

The provider is everything before the **first** colon, because model names
routinely contain colons (`llama3.3:70b`) and slashes (`Qwen/Qwen3-30B`).
`#instance` pins a configured instance and is never autoprovisioned — naming
something that does not exist is an error, not a request to build something
similar.

llmcore builds a missing instance on demand, marks it ephemeral, and tears it
down with `close()`. To forbid that:

```toml
[routing]
autoprovision = false
```

A pinned `target=` bypasses classification and pools entirely, which makes it
the way to reproduce a result exactly.

---

## 2. Pools: failover

```toml
[routing.pools.main]
targets = [
  "anthropic:claude-opus-5-5?effort=high",
  "openai:gpt-5.4",
  "gemini:gemini-3.8-flash",
]
strategy = "priority"
max_attempts = 3
affinity = "session"
```

```python
await llm.chat("hi", pool="main")
```

### Failures are not treated alike

Conflating them is how naive failover burns money or loops.

| Condition | This target | Try another? |
|---|---|---|
| 429 rate limit | short cooldown, honours `Retry-After` | yes |
| insufficient credit / 402 | **long** cooldown (10 min default); balance recorded as zero | yes |
| timeout | short cooldown | yes |
| 5xx / connection | retried **once in place** first | yes |
| 401 / 403 auth | unusable for the process | yes |
| 404 / model not found | long cooldown — deterministic here, but a peer has a different model | yes |
| prompt too long | **no** cooldown — routed to a larger window | yes |
| 400 bad request | — | **no**, it fails everywhere |
| content refusal | — | no, unless `on_refusal = "failover"` |

Two of these are worth calling out:

- **An over-long prompt is a routing signal, not an error.** llmcore knows every
  model's context window from its card, so a prompt that overflows a 128k model
  goes to a 1M model instead of failing. It also skips a target *before* calling
  it when the card already proves the prompt will not fit.
- **Refusals do not fail over by default.** Retrying a refusal on another vendor
  is "shop until someone says yes" — a decision to opt into, not inherit.

### Strategies

| Strategy | Picks | Needs |
|---|---|---|
| `priority` | first healthy target in declared order | nothing |
| `round_robin` | next healthy target | nothing |
| `weighted` | weighted random (`?weight=3`; `0` = spare only) | weights |
| `lowest_latency` | lowest smoothed latency; unmeasured explored first | observations |
| `lowest_cost` | cheapest for *this* prompt, from card pricing | model cards |
| `least_busy` | fewest in-flight requests | nothing |
| `most_credits` | largest known balance | a balance endpoint |

```toml
[routing]
default_strategy = "lowest_cost"     # applies to any pool that declares none
```

`lowest_cost` prices the prompt against the bundled model cards, so a long
prompt and a short one can legitimately choose different members. Self-hosted
providers (`ollama`, `vllm`) price at zero, so your own hardware wins outright.

Three deliberate refusals to guess:

- an **unpriced** target ranks after every priced one (unknown is not free);
- an **unknown balance** ranks after a known one but *ahead of* a known-empty
  one (most vendors expose no balance at all);
- `lowest_latency` tries an unmeasured target first, because it cannot prefer
  low latency without a measurement.

### Order tiers

"My own GPU, and only pay a vendor if it is down":

```toml
[routing.pools.local_first]
targets = ["ollama:llama3.3:70b", "openai:gpt-5.4?order=1"]
```

Tier 1 is held back entirely while tier 0 has a usable member.

### Session affinity

`affinity = "session"` (the default) pins a conversation to its first target,
because changing model mid-conversation throws away the cached prompt prefix,
shifts output style under few-shot expectations, and invalidates preserved
reasoning blocks. A single 429 moves *that turn* while the pin stays; the pin
moves only when the target is persistently unusable. `affinity = "none"`
distributes every request.

---

## 3. Lanes: routing by request kind

A classifier names a **lane**, never a model, so swapping models never touches
the classifier.

```toml
[routing.lanes]
trivial = "pool:cheap"
standard = "pool:main"

[routing.lanes.deep]
target = "anthropic:claude-opus-5-5"
params = { effort = "max" }
description = "Multi-step reasoning, proofs, architecture review, hard debugging"

[routing.lanes.private]
pool = "local_only"
description = "Contains personal data, credentials or other sensitive information"

[routing.classifier]
chain = ["hint", "magic_string", "heuristic"]
min_confidence = 0.55
bias = "quality"
```

Lane names are yours. Complexity tiers, speed tiers, domains and a privacy class
are all the same mechanism — "focus groups per complexity level" are simply
lanes named after complexity levels.

`description` is **not decoration**: the zero-shot classifiers score the prompt
against these descriptions, so this text is what makes them work.

### The chain

| Classifier | Mechanism | Cost | Authority |
|---|---|---|---|
| `hint` | `lane=` / `complexity=` / `effort=` on the call | free | caller |
| `script` | your own function (`"my_module:classify"`) | free | policy |
| `magic_string` | `[[lane:deep]]` in the prompt, stripped before egress | free | prompt |
| `heuristic` | length, code fences, simple-task verbs | free | inferred |
| `local_encoder` | a 350M encoder on CPU, zero-shot over your lanes | ~200–300 ms | inferred |
| `typesafe_jev` | a TypeSafe `choice` question | one cheap call | inferred |
| `llm` | ask a cheap llmcore target | one cheap call | inferred |

llmcore reorders the chain itself: **cheapest first, and within a cost band
instructions before guesses.** Both halves matter. Paying for a classifier before
reading an explicit hint is always wrong, and letting a length heuristic override
`lane="deep"` is always wrong.

Authority has four levels because a marker found in *content* is not as
trustworthy as an argument on the call: in any RAG or tool-output path that text
may have come from a retrieved document, where `[[lane:deep]]` would be a
one-line prompt injection — or a way out of the private lane. So a caller's
`lane=` always wins over a marker, and your own `script` wins over both markers
and guesses.

### Magic strings: letting the model route itself

```python
await llm.chat("[[lane:trivial]] rename this variable")
```

This is the one channel that always exists inside an agent harness: the harness
owns the API call, but the model's text passes through. llmcore strips the
marker before it reaches a provider — always, acted on or not.

### Bias

```toml
[routing.classifier]
bias = "quality"      # "cost" | "neutral"
```

Only moves a *borderline* decision. The two mistakes are not symmetric: routing
too cheap produces a bad answer, routing too expensive only costs money.

### The local encoder

```toml
[routing.classifier]
chain = ["hint", "magic_string", "heuristic", "local_encoder"]

[routing.classifier.local_encoder]
model_id = "LiquidAI/LFM2.5-Encoder-350M-Prompt-Router"
device = "cpu"
revision = "<commit-sha>"       # strongly recommended, see below
```

Needs `pip install "llmcore[local]"`. Zero-shot: your lane descriptions are the
labels, so adding a lane needs no training.

Two honest caveats:

- **It costs latency, not money.** Measured on 8 CPU threads: p50 191 ms for 2
  lanes, 246 ms for 5, 314 ms for 9, plus ~40 s once to load. Worth it when it
  chooses between multi-second calls; not worth it to shave a cheap interactive
  turn. The forward pass runs in a worker thread, so it cannot stall the event
  loop.
- **It loads remote code** (`trust_remote_code=True` is required by the model).
  Pin `revision` to a commit sha so that code cannot change under you; an
  unpinned load logs a warning saying so.

---

## 4. Cascades: cheap first, escalate on failure

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
trade interactively and the right one for batch and agent work.

Verifiers: `script`, `json`, `non_empty`, `typesafe_jev`, `llm`.

**Prefer `script`.** "Does this compile", "does this match the schema", "do the
tests pass" are exact and cheap; a model judging another model is a guess that
costs money. A cascade whose verifier is a compiler is not a heuristic at all.

A verdict is **three-valued**: sufficient, insufficient, or *could not judge*.
The third is not folded into either, because reading it as a fail escalates
every unjudgeable answer and inverts the saving, while reading it as a pass
silently disables the quality floor the moment the judge breaks.
`on_unknown = "accept"` is the default.

---

## 5. Transforms: the privacy path

```toml
[routing.pools.local_only]
targets = ["ollama:llama3.3:70b", "vllm:Qwen/Qwen3-30B#my-box"]

[routing.transforms]
chain = ["pii"]
fail_closed = true

[routing.transforms.pii]
detector = "regex"          # or "vela_pii" — a local 307M token classifier
on_detect = "constrain"
pool = "local_only"
redact = true
hash_salt = "something-local"
```

### Redaction is not a guarantee. Routing is.

A detector that misses one identifier has leaked it, and no detector catches
everything. `on_detect = "constrain"` changes the *destination*: a prompt with
personal data in it goes to a pool that never leaves the machine, so a miss
stays on your own hardware. `redact` rewrites the text and is worth stacking,
but it is defence in depth — **do not describe it to anyone as a guarantee.**

Actions: `allow`, `redact`, `constrain`, `block`.

Three ways this refuses to degrade quietly:

- `constrain` with no `pool` configured **blocks** rather than sending;
- a constrain pool that does not exist **blocks**;
- a detector that raises becomes a **block** (`fail_closed = true`), because a
  detector that crashed has not cleared the prompt.

The `regex` detector covers identifiers with a checkable shape: email,
Luhn-valid card numbers, IBAN, US SSN, phone, IPv4, and common API-key prefixes.
It deliberately does not look for names or addresses — patterns cannot find
those without flagging ordinary English. Use `vela_pii` for those.

Findings are recorded as a 16-character hash, never the matched value — in logs,
in events, and in the exception message.

---

## Effort, thinking and parameters

One vocabulary across providers: `none`, `minimal`, `low`, `medium`, `high`,
`xhigh`, `max` (plus Friendli's `ultracode`), mapped to each vendor's wire
format.

```python
await llm.chat("prove this", effort="max")
await llm.chat("rename it", target="openai:gpt-5.4?effort=minimal")
```

Precedence, lowest first:

1. model-card defaults
2. `[providers.<name>]`
3. target params (`?effort=high`)
4. lane params
5. a named profile
6. the per-call keyword argument

Profiles name a bundle once:

```toml
[routing.profiles.frugal]
effort = "minimal"
max_tokens = 512

[routing.profiles.thorough]
effort = "max"
temperature = 0.2
```

```python
await llm.chat("summarise", profile="frugal")
```

A target that does not support a parameter does not fail the request. Under a
pool it is dropped with a warning, because members genuinely have different
parameter surfaces and the caller cannot know which one will serve. Outside a
pool a bogus parameter still raises — you named one provider and one parameter
that does not exist. `routing.on_unsupported_param = "error"` makes the pool
case strict too.

---

## Proxy mode for agent harnesses

```bash
pip install "llmcore[bridge]"
llmcore-bridge proxy
```

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8900/v1
export OPENAI_API_KEY=llmcore-local
```

Any tool speaking the OpenAI chat-completions API now goes through llmcore's
routing. The **model name** carries the routing intent, because that is the only
field a harness reliably exposes:

| `model` | effect |
|---|---|
| `auto` | run the classifier chain |
| `lane:deep` | that lane |
| `pool:main` | that pool, with failover |
| `profile:frugal` | that parameter profile |
| `openai:gpt-5.4?effort=high` | an explicit target |
| `gpt-4o-mini` | a plain model name, as before |

Lanes and pools appear in `GET /v1/models`, so a harness's model picker becomes
a way to choose routing policy.

Extra endpoints: `GET /v1/routing/health`,
`GET /v1/routing/explain?prompt=...`.

**Usage reports the target that actually answered**, in an `llmcore` block
beside the standard `usage` numbers. Under a pool that is genuinely not the model
requested, and a harness logging spend per model should not be lied to.

**Security.** This process holds every provider credential in your config. It
binds to loopback, and a non-loopback bind without a bearer token is *refused at
startup*:

```bash
export LLMCORE_ROUTING__PROXY__API_KEY="$(openssl rand -hex 24)"
llmcore-bridge proxy --host 0.0.0.0
```

**Streaming never fails over**: once bytes have reached the client the call
cannot be retracted. A stream that fails before its first chunk does fail over.

---

## A per-turn step budget

### Why steps, and not prompt complexity

Measured on this project's own traffic:

| | share of spend |
|---|---|
| routing by predicted prompt complexity | **~2.2%** |
| turns longer than 200 steps | **62%** |

Cost is roughly `steps × context × cached_rate`, and context per step is close
to flat. So the quantity worth bounding is the one that is observable
**directly, cheaply and exactly** — not the one a classifier has to guess. A
prompt's eventual turn length is not predictable from its text.

The agent circuit breaker already bounds an agent *run*. This bounds a *turn*
on the chat path, which is what the proxy exposes to a harness — and the proxy
is where the expensive turns in the measurements actually came from.

### Configuring it

```toml
[routing.budget]
max_steps = 200          # the unit the measurements are in
max_cost_usd = 5.00      # what an operator actually budgets
warn_at_fraction = 0.8   # also applied to the projection
action = "warn"          # "report" | "warn" | "stop"
```

**There are no defaults.** An unset dial is unbounded. The agent circuit
breaker shipped with a `$1.00` default that would have cut off **50.7% of this
project's normal turns**, and a number nobody chose is worse than no number.

Start on `report`, read the numbers it emits, then choose a limit and switch to
`stop`. That is how a limit gets picked rather than guessed.

### What the proxy does with it

On `stop`, a request past the ceiling is refused with **429** and
`type: "budget_exceeded"` — 429 rather than 400 because the request is
well-formed and the same request in a new conversation would succeed, which is
what a harness's retry logic reads that status as. Every response carries the
state:

```json
"llmcore": {
  "target": "openai:gpt-4o-mini",
  "budget": {
    "state": "warning",
    "steps": 168,
    "cost_usd": 4.21,
    "projected_cost_usd": 5.01,
    "cost_is_partial": false,
    "reason": "168 of 200 steps used; on course for $5.01 by step 200, over the $5.00 ceiling"
  }
}
```

Streaming is counted too. Otherwise `stream=true` would be a way to spend
without being seen.

### It can only count a turn the harness identifies

A step budget needs a stable key across the calls of one turn, and the proxy
only has one when the harness says which conversation a call belongs to —
through the standard `user` field or `llmcore.session_id` in `extra_body`:

```python
client.chat.completions.create(
    model="lane:standard",
    messages=messages,
    user="my-session-7",          # this is what makes the budget countable
)
```

**Without it, nothing is tracked.** Every request would get a fresh synthetic
session, so "steps so far" would always be zero and enforcement would be
theatre. The proxy declines to track the turn rather than enforce something
meaningless, and omits the `budget` block so you can tell the difference.

### Unknown cost is not zero

A step against a target with no price records its cost as **unknown**, not
`0.00`. The step still counts, and `cost_is_partial` goes true:

```
"reason": "3 step(s) could not be priced, so the spend ceiling cannot be enforced"
```

A budget that quietly treated unpriced calls as free would never trip — which
is the failure this subsystem's cost model has had at several layers. A step
ceiling still works regardless, which is part of why it is the better dial.

### There is no mid-turn model switch

The design considered a `constrain` action that would drop to a cheaper target
mid-turn. It is **not** implemented, because it is not safe: changing model
inside a conversation changes behaviour, and with preserved-thinking models it
invalidates reasoning blocks already in the history. The actions stop at
`stop`. A caller that wants a cheaper target for the *next* turn can choose one
itself, using the state above.

### Using it directly

The budget is a plain object, so anything can own one — not just the proxy:

```python
from llmcore.routing.budget import BudgetAction, BudgetPolicy, TurnBudget

budget = TurnBudget(BudgetPolicy(max_steps=200, action=BudgetAction.STOP))

while working:
    verdict = budget.check()
    if verdict.should_stop:
        raise RuntimeError(verdict.reason)

    answer = await llm.chat(..., session_id=session)
    info = llm.get_last_interaction_context_info(session)
    budget.record(
        cost_usd=priced_or_none,
        input_tokens=info.prompt_tokens,
        output_tokens=info.completion_tokens,
    )
```

It is fed explicitly rather than hooked onto routing's event stream on
purpose: those events are guarded by `has_sinks()`, so a budget built on them
would silently stop counting whenever nobody was listening. A spend guard must
not depend on observability being switched on.

---

## Inspecting what routing did

```python
plan = await llm.routing.explain("summarise this file")
print(plan.summary())
# lane=trivial via heuristic(0.65) pool=cheap -> gemini:gemini-3.8-flash ~$0.0004

print(await llm.routing.why("summarise this file"))
#   lane=trivial pool=cheap strategy=priority chosen=gemini:gemini-3.8-flash
#     classifier: heuristic — ~7 tokens and a simple-task verb
#     -> gemini:gemini-3.8-flash est=0.0004
#        openai:gpt-5.4 — skipped: cooling down for 12s after rate_limit

llm.routing.health()
llm.routing.pools()
llm.routing.lanes()
llm.routing.classifiers()    # the chain in the order it will actually run
llm.routing.settings()       # resolved, after config and environment

await llm.routing.clear_health("openai:gpt-5.4")
await llm.routing.probe_balances()
```

`explain()` runs the classifier chain for real but never calls the target model.

Every decision also emits a structured event on llmcore's event spine:
`routing.classified`, `routing.transformed`, `routing.selected`,
`routing.attempt`, `routing.failover`, `routing.exhausted`. `routing.selected`
carries every candidate considered and why each was skipped, which is what makes
"why did this answer look different today?" answerable at all.

---

## Measuring a classifier

llmcore ships no accuracy numbers. Here is how to get your own, and the one
thing the report insists on.

**What you need to supply:** your prompts, each labelled with the lane it
*should* go to. Nothing substitutes for it — a classifier that scores well on
someone else's benchmark tells you nothing about your lanes, your prompts and
your models. Fifty cases catch an obviously wrong chain; two hundred let you
compare classifiers with some confidence.

```jsonl
{"prompt": "rename this variable", "expected": "trivial"}
{"prompt": "prove this lemma", "expected": "deep", "note": "maths"}
```

[`docs/examples/lane_eval_starter.jsonl`](examples/lane_eval_starter.jsonl) is a
29-case starting point so this is an edit rather than a blank page. It is not a
benchmark and its README says so.

```bash
llmcore-routing eval my_traffic.jsonl \
  --lane-order trivial,code,standard,deep \
  --chain --show-misroutes 5
```

```
heuristic
  answered   29/29 (100% coverage, 0 abstentions)
  agreement  10/29 (34%)
  too cheap  11  <- these produce bad answers
  too dear   8   <- these only cost money
  latency    p50 0 ms, p95 0 ms
```

**Read the last two lines, not the percentage.** The two error directions are
not interchangeable: routing too cheap produces a bad answer, routing too
expensive only costs money. A classifier at 67% that never routes too cheap is
usable; the same 67% erring downward may not be. `--show-misroutes` prints the
individual prompts that went too cheap, which is where the information is.

Two deliberate choices in the scoring:

- **An abstention is not an error.** A classifier that declines is passing the
  turn to the next one in the chain, which is the designed behaviour; counting
  it as wrong would make the most honest classifier look like the worst.
  Coverage is reported separately from accuracy.
- **Without `--lane-order`, no direction is reported at all** rather than a
  guessed one.

You can also point it at unlabelled traffic (a `.txt` file, one prompt per
line). That measures coverage and latency without anyone labelling anything,
which is a reasonable way to decide whether labelling is worth it.

### A worked example of why the direction matters

Running the starter set found that the free `heuristic` routes

> "Here is my patient record: John Doe, DOB 1971-03-02, diagnosed with
> hypertension. Summarise it."

to the **trivial** lane, because "summarise" is a simple-task verb and the
heuristic is documented as not looking for personal data.

That is survivable only because **the privacy guarantee does not depend on the
classifier**: transforms run after target selection and change the destination,
so the prompt is still constrained to a local-only pool before anything is
sent. llmcore asserts this rather than assuming it — one test proves the
classifier really does get it wrong, so the premise cannot rot silently, and
another proves the prompt still cannot reach a remote target.

---

## Overriding anything, anywhere

Config is a **warm-up, not a cage**. Every setting resolves config →
environment → request, and the request wins:

```bash
export LLMCORE_ROUTING__MAX_ATTEMPTS=5
export LLMCORE_ROUTING__ON_REFUSAL=failover
export LLMCORE_ROUTING__DEFAULT_STRATEGY=lowest_cost
```

```python
await llm.chat("hi", pool="main", routing={"max_attempts": 5, "on_refusal": "failover"})
```

A misspelled override raises rather than being silently dropped.

---

## What this does not claim

- **No accuracy figure is given for any classifier.** None has been validated on
  your traffic, and any number here would be invented. Measure it — see
  [Measuring a classifier](#measuring-a-classifier) below, which exists
  precisely so this gap can be closed with a number rather than an assurance.
- **Cost estimates are estimates.** Output length is unknown before a call, so
  the estimate assumes a fixed output. Good for comparing targets, not for
  billing.
- **PII redaction is not a leak guarantee.** Routing (`constrain`) is; redaction
  is defence in depth.
- **A local classifier is not free** — ~200–300 ms of CPU per call, measured.
- **Routing state is per process.** Cooldowns and latency are not shared between
  processes. A `RoutingStateStore` protocol exists so a shared backend can be
  added without a rewrite, but none ships.
- **`most_credits` only works where the vendor exposes a balance**, which most do
  not. Where none is available, the strategy still helps, because an observed
  insufficient-credit failure records a balance of zero.
