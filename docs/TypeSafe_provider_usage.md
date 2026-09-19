# TypeSafe.ai provider — usage guide

`llmcore` ships a first-class provider for [TypeSafe.ai](https://docs.typesafe.ai)'s
**System One** API (Jev models). It is **not a chat model**: you hand it a piece of
*state* and a map of *typed questions*, and it returns calibrated, structured answers
your code can act on directly — no prompt-and-parse.

| Question type | Ask | Answer |
|---|---|---|
| `noul` | a yes/no question | `noul` — probability of *yes* (0–1) |
| `choice` | pick one option from a set you define | `choice`, `probabilities` per option, `confidence` |
| `score` | rate the state along an ordered rubric | `score` (probability-weighted), `legend`, `probabilities` per level, `confidence` |

All questions in one request see the same state and are evaluated independently, in
parallel — batching many questions into one call is the intended (and cheapest) pattern.

---

## 1. Install & configure

```bash
pip install "llmcore[typesafe]"          # only pulls in httpx; no vendor SDK
export TYPESAFE_API_KEY="..."            # https://console.typesafe.ai/settings/keys
```

`[providers.typesafe]` in `default_config.toml` (all keys optional):

```toml
[providers.typesafe]
# api_key = "..."                 # prefer TYPESAFE_API_KEY / LLMCORE_PROVIDERS__TYPESAFE__API_KEY
# api_key_env_var = "TYPESAFE_API_KEY"
# base_url = "https://api.typesafe.ai"   # env TYPESAFE_BASE_URL honoured when unset
default_model = "jev-latest"      # env TYPESAFE_DEFAULT_MODEL honoured when unset
timeout = 30                      # seconds per HTTP operation
max_retries = 2                   # 408/429/5xx/529 + timeouts; honours Retry-After / retry-after-ms
retry_backoff_initial = 0.5
retry_backoff_max = 5.0
fallback_context_length = 65536   # 64k tokens/request; 32k for state + longest question
```

Precedence for `base_url` / `default_model`: config value → `TYPESAFE_*` env var → built-in
default. The provider is skipped with a warning (not an error) when no key is available, so
the section can stay in your config permanently. `jev` is accepted as an alias for the
provider name (`get_provider("jev")`).

---

## 2. Getting the provider

```python
from llmcore import LLMCore
from llmcore.providers.typesafe_provider import Choice, Noul, Score

llm = await LLMCore.create()                        # or your app's factory
provider = llm._provider_manager.get_provider("typesafe")
```

Standalone (no `LLMCore` facade):

```python
from llmcore.providers.typesafe_provider import TypeSafeProvider

provider = TypeSafeProvider({"default_model": "jev-latest"})   # key from TYPESAFE_API_KEY
...
await provider.close()
```

Always `await provider.close()` (or `await llm.close()`) when finished — it releases the
shared `httpx.AsyncClient`.

---

## 3. `system_one()` — the real surface

```python
system_one(
    state: str | dict | list,                      # the content to evaluate (text only)
    questions: Mapping[str, Noul | Choice | Score | dict],
    *,
    model: str | None = None,                      # default: provider.default_model
    timeout: float | None = None,                  # per-call override, seconds
    extra_headers: Mapping[str, str] | None = None,
    extra_body: Mapping[str, Any] | None = None,   # extra top-level request fields
) -> SystemOneResult
```

```python
ticket = {
    "subject": "Charged twice this month",
    "body": "I see two charges of $49. I only have one account. Please fix this ASAP.",
}

result = await provider.system_one(
    state=ticket,
    questions={
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            },
        ),
        "frustration": Score(
            instructions="How frustrated is the customer?",
            criteria=["Calm, just stating facts", "Frustrated but civil", "Very angry"],
        ),
        "is_urgent": Noul(
            instructions="Does this convey urgency?",
            criteria={"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
        ),
    },
)

result.model                                  # "jev-1.13.0" — the versioned id that answered
result.choices["department"].choice           # "billing"
result.choices["department"].probabilities    # {"billing": 0.66, "technical": 0.34, "sales": 0.0}
result.choices["department"].confidence       # 0.49
result.scores["frustration"].score            # 1.0   (probability-weighted; may fall between levels)
result.scores["frustration"].legend           # {0: "Calm, ...", 1: "Frustrated but civil", 2: "Very angry"}
result.scores["frustration"].probabilities    # {0: 0.0, 1: 1.0, 2: 0.0}
result.nouls["is_urgent"].noul                # 0.99
result.usage.input_tokens                     # 424 (output tokens are free)
result.request_id                             # "req_..." (x-typesafe-request-id)
result.raw                                    # the decoded JSON body
```

### Question builders

| Builder | Required | Optional | Notes |
|---|---|---|---|
| `Noul(instructions, criteria=None)` | — | `criteria={"true": ..., "false": ...}` | describe what a yes / no means |
| `Choice(criteria, instructions=None)` | `criteria`: `{option: description \| None}` | `instructions` | ≥1 option; add a "none of the above" option when nothing may fit |
| `Score(criteria, instructions=None)` | `criteria`: ordered list of level descriptions | `instructions` | position = score, starting at 0; use ≥2 levels |

`instructions` and every description accept a string, a JSON object, or an array
(see [Advanced: structure](https://docs.typesafe.ai/primitives/advanced)). Raw dicts with a
`type` key are accepted too and passed through untouched, so future API fields work without
an llmcore upgrade:

```python
{"q": {"type": "noul", "instructions": "Is this spam?"}}
```

Local validation (`normalize_questions()`) raises `ValueError` *before* any request for an
empty map, an unknown `type`, missing/empty `choice`/`score` criteria, or a wrong criteria
shape. Question ids are for your code only — they are not shown to the model, so put the full
meaning in `instructions`.

### Answers

* `SystemOneResult.answers` — every answer keyed by your ids; `.nouls` / `.choices` /
  `.scores` filter by type.
* `ScoreAnswer.legend` / `.probabilities` are keyed by **integer** level (the wire format
  uses strings); `answers_dict()` / `answers_json()` give the wire shape back.
* Answer kinds a future API adds are dropped from `answers` with a warning but stay in
  `raw`.

### Listing models

```python
for m in await provider.list_models():          # GET /v1/models
    print(m.name, m.release_date, m.description)  # jev-latest, jev-preview, ...
```

`get_models_details()` (no network) returns the builtin card's versioned id **and** its
aliases as `ModelDetails` with `model_type="decision"`, so they appear in
`LLMCore.get_available_models()`.

---

## 4. Using confidence

`choice` and `score` answers carry `confidence` (0–1), a summary of how concentrated the
probability distribution is; `noul` answers carry none (the probability *is* the signal —
≈0.5 means genuinely unsure, not "medium"). A solid starting pattern is three bands, with
thresholds that scale with the stakes of the action:

```python
action = result.choices["action"]

if action.confidence < 0.5:                 # model is genuinely unsure — don't guess
    route_to_human(ticket)
elif action.choice == "check_balance":      # low stakes: act
    show_balance(account_id)
elif action.choice == "approve_transfer":
    if action.confidence > 0.9:             # high stakes: act only when very sure
        confirm_then_execute(account_id)
    else:
        ask_user_to_confirm(account_id)
```

Keep the questions and the thresholds together in one module so they are easy to review,
start conservative, and tune on your own data. If you only need the best option, take the
`choice`/argmax and skip thresholds; if you have a specific statistic in mind, use
`probabilities` directly. See [Confidence](https://docs.typesafe.ai/confidence) and the
[patterns](https://docs.typesafe.ai/patterns) (speculative fan-out, confidence-gated routing,
composite scoring, intent routing).

---

## 5. The chat bridge (`LLMCore.chat`)

Generic llmcore surfaces reach TypeSafe through `chat_completion`, which requires a
`questions` kwarg and returns the answers **as a JSON string**:

```python
import json

answer = await llm.chat(
    "I see two charges of $49 this month. Please fix this ASAP.",
    provider_name="typesafe",
    questions={
        "billing": Noul(instructions="Is this about billing?"),
        "tone": Choice(instructions="Tone?", criteria={"calm": None, "frustrated": None}),
    },
    save_session=False,
)
answers = json.loads(answer)
answers["tone"]["choice"]           # "frustrated"
answers["billing"]["noul"]          # 0.98
```

* The conversation (system/user/assistant messages, in order) is sent as the `state` —
  an array of `{"role", "content"}` entries — unless you pass an explicit `state=`.
  Tool-role messages are flattened to text first.
* Accepted kwargs: `questions` (required), `state`, `timeout`, `extra_body`,
  `extra_headers` (`get_supported_parameters()`); anything else is a `ValueError`.
* `stream=True`, `tools=` or `tool_choice=` raise a non-retryable `ProviderError` (400) —
  TypeSafe returns one structured response and has no tool protocol. Model the decision as
  questions instead (see the
  [function-calling cookbook](https://docs.typesafe.ai/cookbooks/function_calling)).
* Usage flows into llmcore's cost tracking as `prompt_tokens` = input tokens,
  `completion_tokens` = output tokens (free), and the raw result rides along under the
  `"typesafe"` key of the provider response dict.

Calling `chat_completion` / `chat()` **without** `questions` raises a `ProviderError` (400)
that points you to `system_one()`.

---

## 6. Errors & retries

| Status | Meaning | `ProviderError.retryable` |
|---|---|---|
| `401` | missing/invalid key — message names `TYPESAFE_API_KEY` | `False` |
| `403` / `404` | permission denied / unknown route | `False` |
| `422` | request failed validation — message carries the server's field path + reason (e.g. `questions.q.choice.criteria: Field required`) | `False` |
| `429` | rate limit (250k tokens/s, 1200 requests/min) | `True` (retried in-provider) |
| `529` | TypeSafe temporarily overloaded | `True` (retried in-provider) |
| `408` / `5xx` | timeout / server error | `True` (retried in-provider) |
| timeout / connection error | no response | `True` (retried in-provider) |

Every `ProviderError` carries `status_code`, `retryable`, `retry_after_seconds` (from
`retry-after-ms` or `Retry-After`, seconds or HTTP-date), the response headers and the
request id in its message. Retries happen inside the provider (`max_retries`, default 2)
with exponential backoff (`retry_backoff_initial` → `retry_backoff_max`, with jitter),
always preferring the server's requested wait (capped at 60 s). A retry attempt sends
`X-TypeSafe-Retry-Count`. Set `max_retries = 0` to disable.

Invalid inputs (`state=None`, bad questions) raise `ValueError` locally; an unexpected
response body raises a non-retryable `ProviderError` naming the offending field.

---

## 7. Limits, pricing, tokens

* **Model**: Jev 1.13 (`jev-1.13.0`). Aliases `jev-latest` (stable) and `jev-preview`
  (newest build) currently both resolve to it; responses echo the versioned id. Pin the
  version if you tuned thresholds.
* **Budget**: 64k tokens per request (state + all questions); 32k for the state plus the
  single longest question. `get_max_context_length()` reports 64k from the model card.
* **Price**: $0.042 per 1M input tokens ($42 per billion); output tokens are free.
* **Input**: text only (string / JSON object / array of text). Pre-process images, audio
  and binaries into text or structured fields first. English is the primary training
  language; other languages work with lower accuracy — watch `confidence`.
* **Tokens**: `count_tokens()` uses tiktoken `cl100k_base` when available (else `len/4`) —
  an approximation for budgeting; TypeSafe does not publish its tokenizer.
* **Model card**: `src/llmcore/model_cards/default_cards/typesafe/jev-1.13.0.json`
  (`model_type: "decision"`, aliases, context, pricing, rate limits, question types).
  Regenerate with `python -m tools.cardctl generate typesafe` (adapter collapses the
  listed aliases onto the versioned id; `tools/cardctl/enrichments/typesafe.toml` pins
  the static facts).

Known model quirks (counting, arithmetic, date comparison, indirection, very large states)
are documented at [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

---

## 8. Reference

| Item | Value |
|---|---|
| Endpoints | `POST /v1/systemone`, `GET /v1/models` |
| Auth | `Authorization: Bearer $TYPESAFE_API_KEY` |
| Env vars | `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL` (same names as the official SDK) |
| Provider keys | `typesafe` (alias `jev`) in `PROVIDER_MAP` |
| Extra | `llmcore[typesafe]` (httpx) |
| Public types | `llmcore.providers.typesafe_provider`: `TypeSafeProvider`, `Noul`, `Choice`, `Score`, `normalize_questions`, `NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer`, `Answer`, `SystemOneResult`, `SystemOneUsage`, `TypeSafeModelInfo` |
| Example | `examples/typesafe_example.py` |
| Tests | `tests/providers/test_typesafe_provider.py` (offline), `tests/integration/test_typesafe_live.py` (needs the key) |
| Upstream docs | https://docs.typesafe.ai (index: https://docs.typesafe.ai/llms.txt) |
