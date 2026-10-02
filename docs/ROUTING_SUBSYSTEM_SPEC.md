# Routing Subsystem — Design & Specification

Target resolution, failover pools, classifier-driven lanes, response cascades,
prompt transforms, and proxy mode for agent harnesses.

- **Status:** design + specification. **Nothing implemented.** Awaiting approval.
- **Written:** 2026-10-01
- **Requested by:** Araray, 2026-10-01 — provider/model groups with failover and
  prioritisation; unrestricted on-the-fly targets; complexity-based routing with
  pluggable estimators; effort/parameter control; proxy mode for agent harnesses.
- **Related:** [`PROVIDER_SUPPORT_MATRIX.md`](PROVIDER_SUPPORT_MATRIX.md),
  [`model_cards.md`](model_cards.md),
  [`COLAB_RUNTIME_SPEC.md`](COLAB_RUNTIME_SPEC.md)

---

## 1. What was asked, and what this proposes

The request bundled six capabilities. They share machinery, but they are **not
one feature**, and specifying them as one would produce something that does
every job badly. This document separates them into five layers that compose:

| Layer | Answers | Independently useful? |
|---|---|---|
| **1. Target** | *"Which provider, model and parameters?"* | Yes — fixes on-the-fly targets alone |
| **2. Pool** | *"If that one is broken, slow or broke, what else?"* | Yes — failover alone |
| **3. Lane + Classifier** | *"What kind of request is this?"* | Yes — classification alone |
| **4. Cascade** | *"Was the cheap answer good enough?"* | Yes — quality floor alone |
| **5. Transform** | *"What must not leave this machine?"* | Yes — privacy alone |

A user can adopt any one layer without the others. That is the main structural
decision here, and §2 argues for it.

### 1.1 On the name

"Group" was asked about directly. The proposal is **not** one name, because the
request contains two different ideas that were called the same thing:

- A **Pool** is a set of *interchangeable* targets. Membership means "any of
  these can serve this request." It exists for failover and load distribution.
- A **Lane** is a *named destination* chosen by a classifier. It exists to
  express intent — `cheap`, `deep`, `fast`, `private`, `code`. A lane points at
  a pool, or at a single target.

Collapsing them into "group" forces every group to be both a failover set and a
routing destination, which is wrong in both directions: a failover pool needs no
classifier, and a lane with one target needs no pool. Keeping them separate also
matches how the prior art converged — see §7.

---

## 2. Layer 1 — Targets, and why config must stop being a whitelist

### 2.1 The problem, precisely

> *"I have issues when I need, on the fly, to move to a different provider+model
> which is not in my config... this should never be an issue."*

This is not a missing feature; it is a design flaw with an exact location.
`ProviderManager.get_provider()` resolves only against `self._providers`, which
is populated **once** during `__init__` from `[providers.*]` sections. A provider
class registered in `PROVIDER_MAP`, with its API key sitting in the environment
and its base URL already in `_OPENAI_COMPATIBLE_DEFAULTS`, is still unreachable:

```python
await llm.chat("hi", provider_name="xai")
# ConfigError: Provider instance 'xai' not configured or failed to load.
```

Config is acting as an allow-list. It should be a **preset layer**.

> Found during this work: `xai`, `groq` and `together` had **no**
> `[providers.*]` section at all, so all three were unreachable despite being
> registered and having default base URLs. Sections have been added as a
> stop-gap, but that fix only moves the wall — it does not remove it.

### 2.2 `Target` — the unit of routing

```python
@dataclass(frozen=True, slots=True)
class Target:
    provider: str                     # PROVIDER_MAP type, or a configured instance name
    model: str | None = None          # None → that provider's default_model
    params: Mapping[str, Any] = ...   # effort, temperature, max_tokens, ...
    instance: str | None = None       # pin a specific configured instance
    label: str | None = None          # for metrics and logs
```

### 2.3 Target spec strings

Targets are addressable as strings, so they work in config, in CLI flags, in
environment variables and over the bridge — anywhere a dataclass cannot go:

```
openai:gpt-5.4
anthropic:claude-opus-5-5?effort=high
gemini:gemini-3.8-flash?thinking=low&temperature=0.2
vllm:Qwen/Qwen3-30B#my-box          # '#' pins the configured instance
ollama:llama3.3:70b                  # model may contain ':' — split on the FIRST ':'
```

Grammar: `provider ":" [model] ["#" instance] ["?" params]`. Model names
routinely contain `:` and `/`, so the provider is everything before the **first**
colon and the model is the remainder, with `#`/`?` stripped from the end. Params
parse as a query string; values are coerced using the provider's
`get_supported_parameters()` where it declares a type, and left as strings
otherwise.

### 2.4 Resolution order

`ProviderManager.resolve(target) -> BaseProvider`:

1. **A configured instance wins.** If `target.instance` is given, or
   `target.provider` names a configured instance, use it. Presets stay
   authoritative — this is what makes the change safe for existing users.
2. **Otherwise autoprovision.** If `target.provider` is a key in `PROVIDER_MAP`
   and a credential is discoverable (provider's own env var chain), build the
   instance, register it as **ephemeral**, and cache it for the process.
3. **Otherwise fail with an actionable error** naming the env var that was
   missing and the providers that *are* reachable.

Autoprovisioned instances reuse the existing `register_instance(..., ephemeral=True)`
machinery added for the runtimes subsystem, so `close_all()` already tears them
down.

**Config knobs** (`[routing]`):

```toml
autoprovision = true        # false restores today's allow-list behaviour
autoprovision_cache = true  # keep autoprovisioned instances for the process
```

`autoprovision = false` exists because some deployments genuinely want a closed
set — a shared service that must not reach an unbudgeted vendor. The default is
`true`, because the complaint above is the common case.

---

## 3. Layer 2 — Pools: failover and prioritisation

### 3.1 Shape

```toml
[routing.pools.main]
# Ordered unless the strategy says otherwise.
targets = [
  "anthropic:claude-opus-5-5?effort=high",
  "openai:gpt-5.4",
  "gemini:gemini-3.8-flash",
]
strategy = "priority"        # see §3.2
affinity = "session"         # see §3.5
max_attempts = 3             # distinct targets tried per request
```

### 3.2 Selection strategies

| Strategy | Picks | Needs |
|---|---|---|
| `priority` | first healthy target in declared order | nothing |
| `round_robin` | next healthy target | nothing |
| `weighted` | weighted random over healthy targets | per-target `weight` |
| `lowest_latency` | lowest EWMA latency among healthy | observed latency |
| `lowest_cost` | cheapest per estimated token spend | **model cards** |
| `least_busy` | fewest in-flight requests | in-flight counters |
| `most_credits` | most remaining balance | a **balance probe** (§3.4) |

`lowest_cost` is free to implement correctly because pricing is already in the
2,319 bundled model cards, and the prompt's token count is already computable
per provider. This is the strategy most likely to pay for the whole subsystem.

### 3.3 Failure taxonomy — the part that matters

The request named three failure conditions ("429, or not enough credits, or
times out") and they must **not** be treated alike. Conflating them is how
naive failover burns money or loops.

| Condition | Action on this target | Try another? | Cooldown |
|---|---|---|---|
| **429 rate limit** | cooldown, honour `Retry-After` | yes | from header, else short (5–30s) |
| **Insufficient credit / 402** | cooldown **long** | yes | minutes–hours; it will not fix itself in seconds |
| **Timeout** | cooldown short | yes | short |
| **5xx / connection** | retry *same* target once, then move | yes | short |
| **401 / 403 auth** | mark **unusable for the process** | yes | until restart; a bad key does not heal |
| **`ContextLengthError`** | **do not** cooldown | yes — but only to a target with a *larger* window | none |
| **400 bad request** | fail the request | **no** | none — it will fail everywhere |
| **Content policy refusal** | configurable; default **no** failover | no | none |

Two of these are design contributions rather than transcription:

- **`ContextLengthError` is a routing signal, not an error.** llmcore knows every
  model's context window from its card. A prompt that overflows a 200k model can
  be routed to a 1M model instead of failing. This is pure upside and costs
  nothing to add once pools exist.
- **Content-policy refusals default to no failover.** Retrying a refusal on
  another vendor is "shop until someone says yes", which is a decision a user
  must opt into explicitly (`on_refusal = "failover"`), not a default.

### 3.4 Balance probing — honest limits

> *"llmcore can prioritize the one provider+model from the group with more
> credits"*

This is only possible where the vendor exposes a balance, and **most do not**.
Rather than guess, define a protocol and implement it only where real:

```python
@runtime_checkable
class BalanceProbe(Protocol):
    async def remaining_balance(self) -> Balance | None: ...
```

`Balance` carries `amount`, `currency`, `unit` (`"usd" | "credits" | "tokens"`),
and `as_of`. Returning `None` means **unknown**, which is distinct from zero.

| Provider | Balance available? |
|---|---|
| FriendliAI | yes — `get_team_cost()` / `get_team_usage()` already implemented |
| OpenRouter | yes — `/api/v1/key` reports limit and usage |
| Replicate | partial — account endpoint |
| Hugging Face | partial — `canPay`, plus observed `402` |
| OpenAI, Anthropic, Google, most others | **no public balance endpoint** |

So `most_credits` is specified as: rank targets whose balance is *known*, place
*unknown* ones after them in declared order, and treat an observed
insufficient-credit error as a balance of zero with a long cooldown. That last
part is what makes the strategy useful even where no endpoint exists — the signal
comes from failure rather than from polling.

### 3.5 Session affinity — a risk the request did not raise

Failing over **mid-conversation** changes the model. Three consequences:

1. **Prompt caching is lost.** The new target has no cached prefix, so the
   "cheap" failover can cost more than waiting.
2. **Style and format shift**, which breaks few-shot and structured-output
   expectations that were tuned on one model.
3. **Preserved reasoning blocks become invalid** across models — relevant to any
   harness replaying thinking.

So: `affinity = "session"` is the **default** when a `session_id` is present. The
pin moves only after a target is persistently unhealthy, not on a single 429
(which is retried on a peer for *that turn* while the pin stays). `affinity =
"none"` opts into per-request distribution for stateless workloads.

### 3.6 Observability

Every routing decision emits a structured event: pool, candidates considered,
why each was skipped, chosen target, attempt index, latency, outcome, and cost.
This is non-negotiable for a feature that silently changes which vendor served a
request — without it, "why did this answer look different today?" is unanswerable.

---

## 4. Layer 3 — Lanes and classifiers

### 4.1 Route to lanes, not to models

The single most useful finding from the prior art (§7): **Arch-Router routes to
user-defined lanes, and the lane→model binding is separate config.** The
classifier never learns model names, so swapping models requires no retraining
and no classifier change.

```toml
[routing.lanes]
# lane -> pool or target. Lane names are yours.
trivial  = "pool:cheap"
standard = "pool:main"
deep     = "anthropic:claude-opus-5-5?effort=max"
code     = "pool:code"
private  = "pool:local_only"
```

This generalises every variant in the request — complexity tiers, speed tiers,
domain routing, privacy classes — into **one** mechanism. "Focus groups for
complexity levels" are lanes whose names happen to be complexity levels.

### 4.2 The classifier protocol

```python
@dataclass(frozen=True, slots=True)
class Classification:
    lane: str | None = None            # a lane name
    effort: str | None = None          # canonical effort vocabulary (§6)
    target: Target | None = None       # an outright suggestion
    confidence: float | None = None
    rationale: str | None = None       # for logs and audit
    source: str = ""                   # which classifier decided

@runtime_checkable
class RequestClassifier(Protocol):
    name: str
    cost_hint: str                     # "free" | "local" | "api"
    async def classify(self, request: RoutingRequest) -> Classification | None: ...
```

Returning `None` means **no opinion**, which is how a chain stays composable.

### 4.3 Chain of classifiers, cheapest first

```toml
[routing.classifier]
chain = ["hint", "magic_string", "heuristic", "local_encoder"]
on_low_confidence = "standard"     # fallback lane
min_confidence = 0.55
```

Evaluated in order; the first non-`None` result above `min_confidence` wins. The
ordering rule is **cheapest first**, because a classifier that costs an API call
to save an API call is only worth it when the cheap signals are absent.

### 4.4 Implementations

| Classifier | Mechanism | Cost | Notes |
|---|---|---|---|
| `hint` | explicit `lane=` / `complexity=` / `effort=` kwarg on `chat()` | free | always wins when present |
| `magic_string` | configurable pattern in the prompt, e.g. `[[lane:deep]]` | free | **stripped before egress**; this is how a *model* in a harness can route itself |
| `heuristic` | token count, code fences, question count, language | free | crude but surprisingly effective as a first filter |
| `script` | user callable / entry point | free | the escape hatch; full access to the request |
| `typesafe_jev` | TypeSafe `choice` question over lane names | 1 cheap API call | **natural fit** — `choice` returns a pick *plus* per-option probabilities *and* a confidence, which is exactly `Classification` |
| `local_encoder` | `LiquidAI/LFM2.5-Encoder-350M-Prompt-Router` | local CPU | **zero-shot over free-text lanes** — no training, lanes from config verbatim |
| `local_router` | `katanemo/Arch-Router-1.5B` | local GPU/CPU | domain+action preference routing, 1.5B |
| `vela` | `llm-semantic-router/Vela-1.0-Encoder-307M-{Domain,Guard,PII}` | local CPU | task-specific 307M encoders |
| `llm` | ask any cheap llmcore target to classify | 1 cheap call | dogfoods the library; good default when a local model is unwanted |

The two local options deserve emphasis because they make the feature *free at
the margin*: a 307–350M encoder scoring a prompt against lane descriptions runs
on CPU in milliseconds, so classification does not become a tax on every call.

Local models are fetched through the **existing** Hugging Face provider and
cached by `huggingface_hub`; no new download machinery is needed.

### 4.5 Honest limits of classification

- A classifier that is wrong in the *cheap* direction produces a bad answer; one
  that is wrong in the *expensive* direction just costs money. These are not
  symmetric, and the default `min_confidence` should bias toward the stronger
  lane. The config exposes `bias = "quality" | "cost" | "neutral"`.
- No classifier here is validated on llmcore's own traffic. **Any quality claim
  would be invented.** The spec therefore ships an evaluation harness (§9) and
  no accuracy numbers.
- RouteLLM's finding that routers generalise across model pairs is encouraging
  but was measured on preference data, not on this library's workloads.

---

## 5. Layer 4 — Cascade: verify, then escalate

Classification guesses difficulty **before** seeing the answer. The literature's
other approach checks **after**: answer cheaply, judge the answer, escalate only
if it falls short. FrugalGPT reports up to 98% cost reduction at GPT-4-level
quality this way; the mechanism is a scoring function plus a threshold.

This was not in the request and is proposed as an addition, because llmcore is
unusually well placed to do it:

```toml
[routing.cascade.default]
rungs = ["pool:cheap", "pool:main", "anthropic:claude-opus-5-5?effort=max"]
verifier = "typesafe_jev"
threshold = 0.7
max_rungs = 2            # cap the escalation spend
```

```python
@runtime_checkable
class ResponseVerifier(Protocol):
    async def verify(self, request: RoutingRequest, response: str) -> Verdict: ...
```

`Verdict` carries `sufficient: bool | None`, `score: float | None` and a
rationale. `None` means *could not judge* — which must **not** silently count as
pass or fail; it is configurable (`on_unknown = "accept" | "escalate"`,
default `accept`, because escalating on every unjudgeable answer inverts the
cost saving).

**Verifier implementations:**

- `typesafe_jev` — a `noul` question: *"Does this answer fully address the
  request?"* returns a calibrated probability. TypeSafe is already a provider;
  this is the single neatest reuse in the whole design.
- `vela_factcheck` / `vela_halu` — local 307M encoders for factuality and
  hallucination signals.
- `llm` — any cheap target as a judge.
- `script` — user callable (regex, schema validation, a compiler, a test run).

The `script` verifier is the strongest in practice for agent harnesses: "does
this code compile", "does this JSON match the schema" are cheap, exact and
better than any model judgment.

**Cascade is opt-in and off by default.** It trades latency and an extra call
for cost, and that trade is wrong for interactive use.

---

## 6. Effort, thinking and arbitrary parameters

llmcore already has a **canonical reasoning-effort vocabulary** —
`none | minimal | low | medium | high | xhigh | max`, plus Friendli's
out-of-band `ultracode` — with per-provider wire mapping documented in
[`model_cards.md`](model_cards.md#reasoning-effort-vocabulary). This layer
extends it rather than inventing anything.

### 6.1 Where parameters come from

Precedence, lowest to highest:

1. model card defaults
2. `[providers.<name>]` config
3. **target params** (`?effort=high`)
4. **lane params** (a lane may impose `effort`)
5. **param profile** (§6.2)
6. per-call `chat(**kwargs)`

### 6.2 Param profiles

Named bundles, so intent is expressed once:

```toml
[routing.profiles.frugal]
effort = "minimal"
max_tokens = 512

[routing.profiles.thorough]
effort = "max"
temperature = 0.2
```

`chat("...", profile="frugal")`. Profiles are the mechanism the request asked
for by implication: *"this feature can be used to save money when using llmcore
in agent harnesses as a proxy."*

### 6.3 Capability-aware degradation

A target that does not support a requested parameter must not fail the request.
Policy per parameter, from the model card:

- **`effort`** — fold to the nearest supported rung (the per-provider maps
  already do this), or drop with a debug log where the model has no reasoning
  mode.
- **unknown parameter** — `on_unsupported = "drop" | "error"`, default `drop`
  with a warning. A pool whose members have different parameter surfaces is the
  normal case, and erroring would make pools unusable.

### 6.4 Known defect to fix in this work

`PROVIDER_MODERNIZATION_PLAN.md` records an Anthropic `thinking_budget_tokens`
400. Current Claude models take `thinking: {type: "adaptive"}` and **reject**
`budget_tokens`; pre-4.6 models require `budget_tokens`. The effort mapper must
branch on the model generation, which is exactly the kind of fact a model card
should carry rather than code guessing.

---

## 7. Prior art, and what is borrowed

| Source | Borrowed |
|---|---|
| **LiteLLM Router** | model-group-by-logical-name, strategy set (shuffle/latency/usage/least-busy/cost), `order` priority tiers, cooldowns, error-specific retry policy, pre-call context-window checks, session affinity |
| **Arch-Router** (arXiv:2506.16655) | **route to named lanes, not models**; domain+action framing; no retraining when models change; the "models-native proxy for agents" posture |
| **RouteLLM** (arXiv:2406.18665) | strong/weak routing framing; router architectures; the finding that routers generalise across model pairs |
| **FrugalGPT** (arXiv:2305.05176) | the cascade: scoring function + threshold, escalate on insufficiency |
| **LiquidAI LFM2.5-Encoder-350M-Prompt-Router** | zero-shot scoring of a prompt against free-text lanes, on CPU |
| **llm-semantic-router / Vela + Decision** | task-specific small encoders: Domain, PII, Guard, FactCheck, Halu |

**Deliberately not borrowed:** LiteLLM's Redis-backed cross-process state.
llmcore is a library, not a service; cross-process coordination is specified as a
pluggable `RoutingStateStore` (§8.3) with an in-process default, so a single
process needs no infrastructure.

---

## 8. Layer 5 — Prompt transforms and the privacy path

### 8.1 Protocol

```python
@runtime_checkable
class PromptTransform(Protocol):
    name: str
    async def apply(self, request: RoutingRequest, target: Target) -> TransformResult: ...
```

`TransformResult` carries the possibly-rewritten messages, a list of
`Finding`s (what was detected, where, and a **hash** — never the value), and an
`action` of `allow | redact | block | constrain`.

Transforms receive the **target**, so policy can depend on where the prompt is
about to go. That is the key to making this meaningful.

### 8.2 Why constraining the pool beats obfuscation

> *"making something able to be more privacy oriented and like preprocessing the
> prompts to obfuscate, ensure no PII or sensitive data is leaked"*

Redaction is useful but it is not a guarantee — a detector that misses one
identifier has leaked it. The stronger primitive is **routing**: if a prompt
contains PII, send it to a target that never leaves the machine.

```toml
[routing.lanes]
private = "pool:local_only"

[routing.pools.local_only]
targets = ["ollama:llama3.3:70b", "vllm:Qwen/Qwen3-30B#my-box"]

[routing.transforms.pii]
detector = "vela_pii"          # local 307M token-classifier
on_detect = "constrain"        # route to a local-only pool
redact = true                  # and redact anyway, as defence in depth
audit = true                   # record findings (hashed) in observability
```

`constrain` is the contribution here: detection changes the *destination*, not
just the text. Redaction remains available and can be stacked, but the spec is
explicit that **redaction alone should not be presented as a leak guarantee**,
and the documentation must say so.

A `local_only` pool is also where the Colab runtimes subsystem earns its place:
a runtime you provisioned is a target that is yours.

### 8.3 Routing state

```python
@runtime_checkable
class RoutingStateStore(Protocol):
    async def get_health(self, target_key: str) -> TargetHealth: ...
    async def record(self, target_key: str, outcome: Outcome) -> None: ...
```

In-process default. A Redis/SQLite implementation can follow if someone runs
llmcore in several processes against shared quota; it is explicitly out of scope
for the first implementation.

---

## 9. Proxy mode for agent harnesses

> *"add skills and proper ways to configure llmcore as the 'proxy' for calls for
> agent harnesses"*

llmcore already ships `llmcore.bridge` — a gRPC + HTTP/SSE server whose own
docstring says *"No retrieval/RAG/routing logic lives here — that remains in
llmcore proper."* Proxy mode is therefore **an OpenAI-compatible surface on the
existing bridge**, not a new server.

```
POST /v1/chat/completions      model: "lane:deep" | "pool:main" | "openai:gpt-5.4"
GET  /v1/models                lists lanes, pools and reachable targets
```

A harness configured with `base_url=http://localhost:PORT/v1` and
`model="lane:standard"` gets llmcore's routing, cascades, transforms and cost
accounting without knowing llmcore exists. That is the money-saving use case in
the request, and it requires no change to the harness.

**Design points:**

- Lanes and pools appear in `/v1/models`, so a harness's model picker shows
  them. This is how a user selects routing policy from an existing tool.
- Usage reporting returns the **actual** target's tokens and cost, plus an
  `llmcore` extension block naming the target that served the request — a
  harness that logs usage should not be lied to about which model answered.
- Streaming maps to SSE, which the bridge already does.
- **Auth:** the proxy binds to loopback by default and requires a bearer token
  for any non-loopback bind. A routing proxy holds every provider credential;
  binding it to `0.0.0.0` without auth would be handing them out.

### 9.1 Skills

Three skills, since the request asked for them:

- `llmcore-proxy` — start/stop the proxy, print the harness config to paste.
- `llmcore-routing` — inspect and explain routing: `why <prompt>` shows which
  lane a classifier picks and why; `health` shows per-target cooldowns.
- `llmcore-cost` — report spend by lane, pool and target over a window.

---

## 10. Public API

```python
# Dynamic targets — no config needed
await llm.chat("hi", target="xai:grok-4.1-20251117?effort=high")

# Pools and lanes
await llm.chat("hi", pool="main")
await llm.chat("hi", lane="deep")
await llm.chat("hi", profile="frugal")

# Explicit hint for a classifier chain
await llm.chat("refactor this", complexity="high")

# Introspection — no call made
plan = await llm.routing.explain("summarise this file")
# RoutingPlan(lane='trivial', classifier='local_encoder', confidence=0.81,
#             pool='cheap', candidates=[...], chosen=Target(...),
#             estimated_cost_usd=0.0004)

llm.routing.health()           # per-target cooldowns, latency, last error
llm.routing.pools()            # configured pools and membership
await llm.routing.probe_balances()   # where providers expose it
```

`llm.routing.explain()` is deliberately part of the first implementation.
Opaque routing is a support burden: the first question anyone asks is "why did
it pick that?", and the answer must not require reading logs.

---

## 11. Implementation plan

| Phase | Scope | Gate |
|---|---|---|
| **T1** | `Target`, spec-string parse/format, `ProviderManager.resolve()` with autoprovision, `[routing]` config, `llm.chat(target=...)` | A provider absent from config is reachable by spec string; presets still win; `autoprovision=false` restores today's behaviour |
| **T2** | `Pool`, health/cooldown state, failure taxonomy, `priority`/`round_robin`/`weighted`, `max_attempts`, routing events | A 429 fails over; a 400 does not; auth failure is not retried; every decision is observable |
| **T3** | Card-driven strategies: `lowest_cost`, `lowest_latency`, `least_busy`; `ContextLengthError` → larger-window routing | Cost strategy picks the cheaper target on a mixed pool, verified against card pricing |
| **T4** | `BalanceProbe` for Friendli + OpenRouter; `most_credits` with honest unknowns; insufficient-credit → long cooldown | Unknown balance never ranks as zero |
| **T5** | Lanes, `RequestClassifier`, chain, and the free classifiers (`hint`, `magic_string`, `heuristic`, `script`) + `llm.routing.explain()` | Magic strings never reach a provider; `explain()` is accurate |
| **T6** | `typesafe_jev` classifier; `local_encoder` (LFM2.5) via the HF provider; optional `local_router` (Arch-Router) and `vela` | Local classification adds <50 ms p50 on CPU — **measured, not assumed** |
| **T7** | Cascade: `ResponseVerifier`, rungs, thresholds, `script`/`typesafe_jev`/`vela` verifiers, spend caps | A cheap-then-escalate run costs less than always-strong on an eval set |
| **T8** | `PromptTransform`, `vela_pii` detector, `constrain` action, local-only pools, hashed audit findings | PII in a prompt provably does not reach a remote target |
| **T9** | Proxy mode on the bridge: OpenAI-compatible endpoints, lanes in `/v1/models`, loopback-default auth, the three skills | An unmodified agent harness routes through llmcore |
| **T10** | Session affinity, `RoutingStateStore` protocol, docs, evaluation harness | Mid-conversation failover does not silently change model without a recorded reason |

T1 and T2 are the ones that pay immediately. T5 onward is where the value
compounds, and each needs measurement rather than assertion.

---

## 12. Risks and open questions

| Risk | Mitigation |
|---|---|
| Routing makes behaviour non-reproducible | Every decision is an event; `explain()` is first-class; a run can pin `target=` to bypass routing entirely |
| Classifier mistakes degrade answers invisibly | `bias` defaults toward quality; confidence floor; evaluation harness ships with T5 |
| Cascade doubles latency on escalation | Off by default; `max_rungs` caps it; documented as a batch/agent feature, not interactive |
| Failover hides a broken primary | Health is reported, cooldowns are visible, and persistent failover emits a warning rather than staying quiet |
| Autoprovision reaches an unbudgeted vendor | `autoprovision = false`; and autoprovision requires a credential to already be present |
| Proxy mode concentrates every credential | Loopback by default; bearer token required for any other bind |
| Local classifier adds a torch dependency | Optional extra; `local_encoder` is opt-in and the free classifiers cover the common case |
| PII redaction oversold as a guarantee | Documentation states plainly that `constrain` (routing) is the guarantee and redaction is defence in depth |

**Open questions for you:**

1. **Naming.** `Pool` + `Lane` as proposed, or do you prefer different words
   (`fleet`/`route`, `group`/`tier`)? The split matters more than the words.
2. **Default strategy** for a pool with no `strategy` set — `priority` (ordered,
   predictable) or `lowest_cost` (saves money immediately but reorders silently)?
   I lean `priority`.
3. **Cascade default**: off everywhere, or on for non-interactive paths
   (agents/batch) where latency matters less?
4. **Where should complexity hints live on the wire** in proxy mode — a magic
   string in the prompt, an OpenAI-style `extra_body`, or a model name suffix
   (`lane:deep`)? All three are implementable; the third needs no harness change.
5. **Scope of the first PR.** T1+T2 alone is already useful and reviewable; T1–T5
   is a coherent "routing works" milestone but a large diff.
