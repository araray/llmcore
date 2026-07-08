# Darwin — Deep Dive: Architecture, Evaluation, and Improvement Brief

**Audience:** the llmcore/wairu engineering team, for brainstorming how to make
the Darwin cognitive engine better and more useful.
**Status:** living document. **Created:** 2026-07-08.
**Scope:** what Darwin and Lite are, how Darwin is built (the 8-phase cognitive
cycle in llmcore), how wairu wires it, the prompt/tool/memory/Grimoire control
plane, the head-to-head evaluation and its numbers, the bugs it surfaced, and a
structured set of improvement directions. Every architectural claim here is
grounded in the current source (file:line references throughout) as of this
date; verify against HEAD before acting on a specific line number.

> **How to read this.** Parts I–IV are *understanding* (what exists and why).
> Part V is the *evidence* (the eval). Part VI is *what's broken*. Part VII is
> the *brainstorm material* (improvement directions, ranked). Part VIII will
> hold the external-research synthesis (how the wider field solves these
> problems) once that work lands. If you only read one thing, read the
> Executive Summary then Part VII.

---

## Executive Summary

Wairu (the agentic CLI) can run a turn through one of two **execution
engines**:

- **Lite** (default) — wairu owns the loop: a single `ToolDispatcher` runs
  `chat → extract tool_calls → approve → execute → feed results back → chat`
  against `LLMCore.chat()`. One feedback round per tool batch. Fast, cheap,
  simple.
- **Darwin** (opt-in) — wairu delegates the loop to **llmcore's cognitive
  cycle**, an 8-phase deliberation: **PERCEIVE → PLAN → THINK → VALIDATE → ACT
  → OBSERVE → REFLECT → UPDATE**, iterated up to a budget. Deliberate, stateful,
  self-reflective.

We ran the first head-to-head evaluation (5 modern models × 7 deterministic
tasks × both engines = 70 live runs, 2026-07-08). **Headline:**

| Engine | Accuracy | Mean latency/turn | Native convergence |
|--------|:--------:|:-----------------:|:------------------:|
| **Darwin** | **97.1%** | **87s** | 74% (26% rescued by a fallback) |
| **Lite** | **91.4%** | **12s** | n/a |

Darwin is **more accurate but ~7× slower**, and its **entire** accuracy edge is
**one task category: error-recovery (multi-round tool use), 100% vs 40%.** On
the other six categories the engines tie. Darwin's cost is real: it does ~7 LLM
calls per iteration, over-tools simple tasks (it once ran 5 tool calls and 206s
to answer a *git-knowledge* question), and only self-terminates 74% of the time
— the rest are rescued by wairu's grounded "grace" fallback.

**Verdict:** keep Lite the default; make Darwin the opt-in for its niche
(multi-step, recover-and-continue turns) via `--engine auto`. The architecture
is validated *for that niche* — but its default prompts and convergence logic
are the ceiling, and it runs on **hardcoded f-string prompts, not the Grimoire
control plane** everyone assumed was managing it.

The eval also surfaced **four bugs** the mock test suite could never catch —
three now fixed in llmcore/wairu, one open:

1. **Darwin episodic memory was 100% dead** (two stacked bugs — wrong enum, then
   wrong field names — both swallowed by a broad `except`). **Fixed.**
2. **kimi 400'd on every tool turn** (empty assistant content). **Fixed.**
3. **deepseek-v4-pro leaks tool-call markup as its answer** on multi-round
   turns. **Open** (needs a provider-side parser).

---

# Part I — The Two Engines

## 1. Lite — wairu owns the loop

`ToolDispatcher` (`wairu/sessions/tool_dispatch.py:710`) is a reusable ReAct
loop shared by the `fly` (autonomous), REPL (interactive), and headless (`-p`)
surfaces. One instance per session. `run_turn` (`tool_dispatch.py:1086`):

1. Pre-flight input moderation (SF-1); a block returns immediately with no LLM
   call.
2. Aggregate tools off the event loop (an MCP source may lazily spawn on first
   listing).
3. Filter the *advertised* tool set by the active persona's allowlist (F7).
4. First `chat()` with native tool schemas.
5. **Tool rounds loop** — `for _round in range(max_tool_rounds)` (default **5**):
   extract tool calls from the raw response; none → break (final text);
   otherwise moderate the output, execute the round (per-call: unknown-tool
   check → persona allowlist → approval if `requires_approval` → pre_tool hook
   → undo snapshot → execute → post_tool hook), then **feed results back** with
   `tools=None` (either the R-2 native tool-role protocol — an assistant message
   carrying `tool_calls` + one `role="tool"` message per result — or a legacy
   user-role text block).

The feedback message carries a `continuation_instruction`. For `fly` it's the
`GOAL_COMPLETE`/`NEEDS_ESCALATION` scaffolding parsed later; interactive/
headless callers pass a plainer "answer directly" instruction.

**Key property:** Lite's loop is shallow and reactive. It gets `max_tool_rounds`
(5) chances to call tools, but each round is a single model turn with no
explicit planning, self-critique, or progress tracking. When a model *stalls*
mid-recovery ("I need one more tool call…") the loop has already spent its turn.

## 2. Darwin — llmcore owns the loop

For Darwin, wairu prepares an `EnhancedAgentState`, bridges its plugin tools
into llmcore's `ToolManager`, and drives **`CognitiveCycle.run_streaming()`**,
consuming normalized per-iteration updates. The actual deliberation — plan, act,
reflect, decide-to-continue — happens inside llmcore. Wairu's job shrinks to:
state prep, tool bridging (with its security boundary re-implemented at each
bridged handler), update streaming, and answer rescue. Parts II–III detail both
halves.

---

# Part II — The Darwin Cognitive Cycle (llmcore)

Source: `llmcore/agents/cognitive/phases/` (all 8 phases + the orchestrator
`cycle.py`), models in `cognitive/models.py`, wrapped by `SingleAgentMode` in
`agents/single_agent.py`. Canonical order (`models.py:174-181`, `CognitivePhase`
enum): **PERCEIVE → PLAN → THINK → VALIDATE → ACT → OBSERVE → REFLECT → UPDATE**.

**Only 4 of the 8 phases make an LLM call: PLAN, THINK, VALIDATE, REFLECT.**
PERCEIVE, ACT, OBSERVE, UPDATE are deterministic. That's why a single Darwin
iteration is "~7 LLM calls" in the loose sense (4 phase calls + tool/activity
calls) and why latency is dominated by the LLM phases.

## 3. The eight phases

| # | Phase | LLM? | Purpose | Sets `final_answer`? |
|---|-------|:----:|---------|:--------------------:|
| 1 | PERCEIVE | no | Retrieve context, snapshot working memory, capture env | no |
| 2 | PLAN | **yes** | Decompose goal → ordered steps + reasoning + risks (conditional) | no |
| 3 | THINK | **yes** | ReAct decision: a thought + (a tool call **or** a final answer) | **yes** (primary) |
| 4 | VALIDATE | **yes** | Safety gate before execution; can reject / escalate to human | no (sets `awaiting_human_approval`) |
| 5 | ACT | no | Execute the tool via `ToolManager` | (activity path can) |
| 6 | OBSERVE | no | Turn the raw `ToolResult` into a structured observation | no |
| 7 | REFLECT | **yes** | Self-evaluate, estimate progress, decide plan updates | no |
| 8 | UPDATE | no | Commit reflection to state/memory, decide whether to continue | (via `_should_continue`) |

Selected details that matter for improvement work:

- **PLAN** (`plan.py`) only runs when `should_plan` (first iteration, empty
  plan, or a `plan_needs_update` flag) (`cycle.py:302-306`). System prompt:
  *"You are a strategic planning agent…"*, `temperature=0.7`. It parses the
  model's free text with regexes keyed to `PLAN:` / `REASONING:` / `RISKS:`
  headers. Steps become `PlanStepSpec`s that can carry a structured
  `tool_name` — a plan step *can pre-commit a tool call*.

- **THINK** (`think.py`) is the ReAct core. System prompt: *"You are an
  autonomous AI agent using the ReAct framework…"*, and it passes the native
  tool schemas. It has **three routes**: (1) a **structured-plan-step shortcut**
  that emits a tool call with *no LLM call* when the current plan step carries a
  `tool_name`; (2) an **activity fallback** (XML-based tool emission) for
  providers without native tool support; (3) the **native path**. It parses a
  `Final Answer:` regex → sets `is_finished=True` and
  `agent_state.final_answer` (`think.py:273-275`) — **this is the main
  convergence trigger.**

- **VALIDATE** (`validate.py`) runs only when THINK proposed an action *and*
  `skip_validation` is False. Two deterministic pre-checks (a `DANGEROUS_PATTERNS`
  regex list — `rm -rf /`, `DROP DATABASE`, `sudo`, `eval(`… — and a
  tool-registry existence check) run *before* the LLM. When `skip_validation`
  is True the orchestrator fabricates an `APPROVED` result and **skips the phase
  entirely — including the deterministic guards** (`cycle.py:383-392`; see §6).

- **ACT** (`act.py`) is the only executing phase. Core call:
  `tool_result = await tool_manager.execute_tool(tool_call=...)` (`act.py:151`),
  a single attempt by default (`max_retries=0`). It honors VALIDATE's verdict
  (rejected/approval-required short-circuit to a non-success result).

- **OBSERVE** (`observe.py`) formats the result (truncating tool output at
  **4000 chars** — raised from 500 because short truncation starved the model).
  Note: the orchestrator always passes `expected_outcome=None`
  (`cycle.py:433`, with a TODO), so OBSERVE's expectation-matching logic is dead
  code.

- **REFLECT** (`reflect.py`) always runs (not gated on an action). It produces a
  `progress_estimate ∈ [0,1]` via a **5-strategy parse cascade** ending in a
  ~200-line keyword bucket estimator (`_estimate_progress_from_content`,
  `reflect.py:449-655`) that quantizes progress to fixed constants
  {0.05, 0.15, 0.35, 0.45, 0.65, 0.85}. This estimate feeds termination.

- **UPDATE** (`update.py`) commits plan/step/progress changes to state, appends
  insights to working memory, records an episode (this path was **doubly
  broken** — see Part VI), and computes `should_continue`.

## 4. The orchestrator and data flow

`run_iteration` (`cycle.py:233-513`) runs one `CycleIteration`. The important
control-flow facts:

- **Early exit on final answer:** if THINK returns `is_final_answer`, the
  orchestrator finalizes and returns immediately — VALIDATE/ACT/OBSERVE/REFLECT/
  UPDATE are **all skipped** (`cycle.py:373-377`). This is the cheap, healthy
  path: a one-iteration answer.
- VALIDATE → ACT → OBSERVE are all nested under `if think_output.proposed_action`.
  If THINK proposes neither an action nor a final answer, all three are skipped
  and a warning is logged — a wasted iteration.
- REFLECT and UPDATE always run (when reached).

## 5. Convergence — how the loop actually stops

There is **no single stop condition**; convergence is a union of signals
funneled through `agent_state.is_finished` and `UpdateOutput.should_continue`:

- `is_finished` (`models.py:778-804`) is True if an explicit override was set,
  **or** all (non-empty) plan steps are `"completed"`, **or**
  `metadata["goal_achieved"]` is truthy.
- **Who sets `final_answer`:** THINK's native/parse path (main), THINK's
  activity fallback, ACT's activity fallback, and — for `is_finished` only,
  *without* a `final_answer` — `_should_continue` when `progress ≥ 1.0` or all
  steps completed.
- The loop (`run_until_complete` / `run_streaming`) checks `is_finished` at the
  top of each iteration, breaks when UPDATE says `should_continue=False`, and is
  bounded by `max_iterations` and an `AgentCircuitBreaker` (max iterations,
  repeated identical errors, wall-clock, cost, progress stall).

**The load-bearing weakness:** `final_answer` is set primarily by a
`Final Answer:` **regex** on free text (`think.py:847-854`), and secondary
termination leans on the **heuristic `progress_estimate`**. When a model keeps
proposing tool calls and never emits the exact "Final Answer:" contract, the
cycle *acts until the budget runs out* without ever synthesizing a reply — which
is exactly the situation wairu's "grace" fallback (Part III §9) exists to
rescue. In the eval, **26% of Darwin runs** ended this way.

## 6. `skip_validation` (default True) — what it turns off

Threaded from wairu into `run_streaming`. When True, the orchestrator
**does not call the VALIDATE phase at all** and fabricates an `APPROVED` result
(`cycle.py:383-392`). Rationale (sound): wairu already governs every tool call
at its bridged handler (approval, moderation, persona allowlist, hooks), so
llmcore's *LLM* safety-validator is redundant — and worse, it was
non-deterministically **rejecting legitimate tool calls**, starving the cycle of
observations so it never converged. **But** the fabrication also skips the
*deterministic* `DANGEROUS_PATTERNS` and tool-registry checks — those are lost
too, not just the LLM judgment (an improvement opportunity in Part VII).

## 7. Streaming

`run_streaming` (`cycle.py:726-999`) mirrors the bounded loop but `yield`s a
`StreamingIterationResult` after each iteration (phase, action, observation,
progress, `is_final`, `stop_reason`). `SingleAgentMode.run_streaming` re-maps it
to the public `IterationUpdate` DTO. Wairu normalizes both spellings into one
dict (`darwin_stream.normalize_streaming_update`).

---

# Part III — How wairu Integrates Darwin

## 8. Tool bridging and the security boundary

Because the cognitive cycle drives tools *directly through llmcore's
ToolManager*, it **never flows through Lite's `ToolDispatcher`** — so wairu's
approval/moderation/hook/persona boundary has to be re-implemented *inside each
bridged tool handler*. `register_plugin_tools` (`darwin_stream.py:344`) converts
each wairu tool to an llmcore `Tool`, registers an implementation, and upserts it
into the ToolManager. There are **three** handler implementations enforcing the
same boundary at three call sites:

| Path | Handler | Approval model |
|------|---------|----------------|
| **fly** | `manager.py:_make_darwin_tool_handler` | deferred (queue + `PermissionError`; fly UI approves then retries) |
| **REPL** | `repl/session.py:_make_repl_darwin_tool_handler` | interactive prompt + F3 diff preview |
| **headless** | `darwin_headless.py` local `make_handler` | declarative policy (no human) |

⚠️ The three handlers enforce the same *set* of checks but in **different
orders** (fly: persona→approval→pre_tool→moderation; REPL:
persona→pre_tool→moderation→approval; headless: moderation→pre_tool→approval).
That ordering has security-relevant differences and is a consolidation target
(Part VII).

## 9. The three entry paths + grounded "grace" synthesis

- **fly** (`fly.py` + `manager.run_darwin_streaming`) — the autonomous loop runs
  the goal in *stream segments*, converting llmcore terminal signals (pending
  approvals, `human_approval_required`, errors) into wairu escalations.
- **REPL** (`repl/session.py`) — `active_engine` is `lite` / `darwin` / `auto`.
  `auto` routes only turns the F5 heuristic classifier calls `complex`. Reads
  the answer from **`state.final_answer`** (never the streaming `message`, which
  for a tool-executing final iteration is a tool-activity *echo* — surfacing
  that was a historical bug).
- **headless `-p`** (`print_cmd.py` + `darwin_headless.py`) — added this
  session. `--engine lite|darwin|auto`; runs one cognitive-cycle turn with full
  tool-gating and reports `terminal_mode` in the JSON.

**Grounded grace synthesis** (`darwin_headless._grace_synthesize`; REPL twin
`_grace_synthesize_darwin`) is the safety net for the 26% of runs that exhaust
their budget without a `final_answer`. It reads the run's *actual* gathered tool
observations (`state.recent_history_summaries(...)`), JSON-encodes them into a
tool-less prompt — *"Give the user a direct, complete final answer using ONLY
these results. Do not call any tools."* — and runs a fresh `chat`. Grounding in
the real observations (not blind session memory) is what keeps it from
hallucinating. **This is a wairu-side crutch for an llmcore-side convergence
weakness** — it works, but it's triplicated and only exists because the cycle
doesn't reliably self-terminate.

## 10. Engine selection

`--engine` (validated, raises on unknown) → config default
(`[wairu.autonomous] execution_engine` for fly, `[wairu.darwin] engine` for
headless, `[wairu.darwin] repl_routing` → `auto` for REPL). `auto` uses the F5
`SmartRouter.classify` (a *heuristic* GoalClassifier — regex + length, **no
extra LLM call**); only `complex` turns route to Darwin. If Darwin can't
initialize, `auto` silently downgrades to `lite`; an explicit `darwin` stays and
warns.

---

# Part IV — Prompts, Tools, Skills & Grimoire (the control plane)

This is the part most likely to surprise the team: **the assumed control plane
is almost entirely inert by default.**

## 11. Prompts: 4 of 8 phases, f-strings by default

Only **PLAN, THINK, VALIDATE, REFLECT** consult a `prompt_registry`, each with a
single hardcoded template id (`planning_prompt`, `thinking_prompt`,
`validation_prompt`, `reflection_prompt`). Every `render()` call is wrapped in
`try/except` that logs *"Failed to use prompt registry… falling back"* and drops
to a **hardcoded f-string** (e.g. `think.py:721-751`, `plan.py:220-257`). With
stock config the registry is `None`, so **all phases run on f-strings.**

The f-string prompts are plain and generic. The THINK fallback is essentially
"GOAL / CURRENT STEP / TOOLS / use the ReAct format, respond now" — with **no
instruction to answer directly when no tool is needed**, and **no strong
convergence push**. This is a prime suspect for the over-tooling and weak
self-convergence the eval measured.

The parsers depend on **exact section labels** (`PLAN:`, `Final Answer:`,
`APPROVED:`, `PROGRESS:`, `STEP_COMPLETED:`). Any prompt (or Grimoire spell) that
reformats these silently degrades to weak fallbacks.

## 12. Grimoire: wired but inert

`GrimoirePromptRegistryAdapter` (`agents/prompts/grimoire_adapter.py`) maps the 4
cycle ids to Grimoire spell ids (`DEFAULT_TEMPLATE_MAP` is an **identity map**).
It's only constructed when `[wairu.grimoire] enabled=true` **and** a `repo_path`
is set, and only reaches the cycle via `_init_darwin_manager`. Three independent
reasons it does nothing today:

1. **Default off.** `[wairu.grimoire] enabled=false`.
2. **No matching spells ship.** The shipped pack (`wairu/prompts/`,
   `wairu-research-spells`) has ids like `research/query_decomposition` — **none**
   equal the cycle's 4 ids. With the identity map and no `prompt_map`,
   `render("thinking_prompt")` calls `get_spell("thinking_prompt")` → raises →
   f-string fallback. The shipped pack is a *research-workflow* pack, not a
   *cognitive-cycle* pack.
3. **The VALIDATE arm is dead anyway** because `skip_validation=true`.

So even with Grimoire enabled, **only THINK/PLAN/REFLECT** are overridable in
practice, and only if you author spells at the right ids (or a `prompt_map`) that
**reproduce the exact parser output contracts** and declare exactly the
variables each phase passes (with `strict=True` by default, a variable-name typo
→ silent fallback). A cognitive-cycle spell pack is a concrete, bounded piece of
work — see Part VII.

## 13. Tools: two parallel registries

There are **two independent** tool-registration paths:

- **(A) Direct bridge into llmcore's `ToolManager`** (`_register_darwin_plugin_tools`,
  `manager.py:1545`; `darwin_stream.register_plugin_tools`) — **this is what the
  cognitive cycle actually executes.** Wairu plugin tools → `register_implementations`
  → per-run ToolManager.
- **(B) Grimoire runes** (`_register_grimoire_plugin_tools`, `manager.py:1821`) —
  registers wairu tools *as Grimoire runes* (the **reverse** direction), used for
  Grimoire's own procedural selection. The `grimoire.bind.llmcore` bridge that
  would turn runes into ToolManager tools **is never imported by wairu.**

So Grimoire does **not** manage the cycle's executable tools; it's a parallel
registry. Two sources of truth invite drift (a rune not bridged into ToolManager
is invisible to THINK's native tool schemas).

## 14. Skills / rituals: not used by the cycle at all

F8 skill contracts and Grimoire rituals attach only to the *Lite* `ToolDispatcher`
and the REPL `/skills` command. The Darwin cycle never touches them.

**Bottom line for the team:** the belief that "Grimoire handles prompts/tools/
skills for llmcore's Darwin" is *architecturally supported but not the live
default*. Today Darwin = f-string prompts + directly-registered wairu tools + no
skills.

---

# Part V — The Evaluation

## 15. Methodology

Both engines run through the **same CLI path** — `wairu -p PROMPT --engine
{lite,darwin} --provider P --model M --output-format json --permission-mode
approve-all` — so the engine is the only variable. Darwin gets a reduced
iteration budget (`headless_max_iterations=5`; the default 10 makes trivial
turns run for minutes and time out). Harness: `wairu/benchmarks/darwin_vs_lite.py`
(resumable; deterministic Python checkers, **no LLM judge**). Raw data:
`wairu/benchmarks/results/results_2026-07-08.jsonl` (70 lines). Full methodology:
`wairu/docs/DARWIN_VS_LITE_EVAL_PLAN.md`.

**Task suite (7 deterministic tasks):** trivial_math (no-tool, *penalizes tool
use*), multihop_reason (no-tool), knowledge (no-tool), count_files (single-tool),
read_extract (read), largest_file (multi-step tool), error_recovery (fail →
recover). **Models (5):** deepseek-v4-pro, gpt-5.4, kimi-k2.6, qwen3-235b, glm-5.2
(gemini-3.1 and sonnet-4.6 were dropped — unreachable). 1 trial.

## 16. Results

**Overall:**

| Engine | Runs | Accuracy | Mean latency | Tokens |
|--------|:----:|:--------:|:------------:|:------:|
| darwin | 35 | 97.14% | 87.01s | 0 (uncaptured — see Part VI) |
| lite | 35 | 91.43% | 12.06s | ~1461 |

**By category** — the whole story is two rows:

| Category | Darwin | Lite |
|----------|:------:|:----:|
| **error-recovery** | **100%** (5/5) | **40%** (2/5) |
| trivial-notool | 80% (4/5) | 100% (5/5) |
| knowledge / reasoning / single-tool / read-extract / multi-step | 100% | 100% |

**By model** (accuracy · mean latency, Darwin/Lite):

| Model | Darwin acc | Lite acc | Darwin lat | Lite lat |
|-------|:----------:|:--------:|-----------:|---------:|
| deepseek-v4 | 86% | 86% | 97.6s | 10.4s |
| gpt-5.4 | **100%** | 86% | **24.8s** | 9.1s |
| glm-5.2 | **100%** | 86% | 43.3s | 13.8s |
| kimi-k2.6 | 100% | 100% | **219.1s** | 16.6s |
| qwen3 | 100% | 100% | 50.2s | 10.4s |

**Darwin health:** 74.29% converged natively (26/35); 25.71% (9/35) rescued by
grace; **0 fallbacks**. Mean iterations 2.29. The 4 runs that hit the 5-iteration
budget cap all relied on grace or over-tooling.

## 17. Where Darwin wins, loses, and why

- **Wins on error-recovery (the multi-round discriminator).** "Read a missing
  file; if absent, list the dir." Lite fails on 3/5 models: deepseek leaks DSML
  markup (bug 3), gpt-5.4 and glm-5.2 **stall** — they announce "I need one more
  tool call" but the single-shot loop already spent its turn. Darwin's iterate-
  then-grace loop recovers on **all 5**. This is Darwin's reason to exist.
- **Loses on trivial restraint.** deepseek Darwin ran a shell command to compute
  17×23 (right answer, but the checker penalizes tooling a trivial task). Only
  deepseek over-tooled *trivial_math*; but Darwin over-tools *knowledge* on 4/5
  models (deepseek 5 tools/206s, kimi 4 tools/422s) — invisible in the accuracy
  table only because the knowledge checker doesn't penalize tool use.
- **Model sensitivity is huge.** gpt-5.4 is the ideal Darwin driver (100% @ 25s,
  converges natively, doesn't over-tool). kimi is the worst fit (219s mean, two
  ~420s grace outliers). Darwin amplifies model quality differences that Lite
  smooths over.

---

# Part VI — Bugs the Eval Surfaced

Running against *real providers* (the whole test suite is mock-based) exposed
four defects immediately:

1. **Darwin episodic memory was 100% dead — two stacked bugs (FIXED).**
   `update.py` recorded each iteration as `EpisodeType.TOOL_USE` — a member that
   doesn't exist (`AttributeError`, swallowed). After fixing the enum
   (`→ ACTION`, `9d8c540`), it *still* failed: it constructed
   `Episode(episode_type=, content=, metadata=)` but the `Episode` model requires
   `event_type` + `data` (`ValidationError`, also swallowed). **Both** are now
   fixed (`80fef60`): `Episode(event_type=EpisodeType.ACTION, data={...})`. Net:
   before this session, *no Darwin iteration ever wrote an episode* — the
   "learning" purpose of UPDATE was inert. **A test that asserts an episode row
   is written is still needed** (both bugs were swallowed exceptions; only a live
   run or a persistence assertion catches this class).

2. **kimi 400 on every tool turn (FIXED, `ca0ce10`).** Moonshot rejects
   empty/null assistant content; a tool-calls-only turn plus a spurious
   empty-text assistant message both tripped it. All 4 kimi *lite* tool tasks
   400'd; now 7/7. Fixed by padding empty assistant content.

3. **deepseek-v4-pro DSML tool-call leak (OPEN).** On multi-round turns deepseek
   emits `<｜｜DSML｜｜tool_calls>…invoke name="list_dir"…>` **text** in
   `message.content` instead of the structured field; `extract_tool_calls` reads
   only the structured field, so the markup becomes the final answer. Fix: parse
   text-format DSML tool calls as a fallback and strip the markup. This bites any
   multi-round deepseek use, both engines.

4. **Darwin token capture is broken (OPEN, harness/llmcore).** All 35 Darwin
   runs report 0 tokens — the cognitive-cycle session's token stats aren't
   populated the way Lite's are, *and* `update_token_totals_from_phases` only
   sums 4 phases (tool/activity/PERCEIVE cost is invisible). So the cost axis of
   the comparison is missing and *understated* — the true token/$ gap is much
   larger than the 7× latency ratio.

---

# Part VII — Improvement Directions (brainstorm material)

Grouped by theme, roughly ranked within each. These come from reading the
current source; the external-research synthesis (Part VIII) will map field-proven
techniques onto them.

## A. Convergence & the "grace" crutch (highest leverage)

The single biggest structural weakness: **the cycle self-terminates only 74% of
the time and relies on a wairu-side grace fallback for the other 26%.** Directions:

- **A1. An explicit, first-class "finish" mechanism.** Today final-answer
  detection is a `Final Answer:` *regex* on free text. Give THINK a real
  **`finish`/`final_answer` tool** (llmcore already registers a `finish` builtin
  in `load_default_tools` — it isn't wired into the convergence path) so the
  model *calls* to terminate instead of emitting a magic string. This is what
  smolagents (`final_answer` tool), DSPy ReAct (`Finish` action), and LangGraph
  (`END`) all do. Structured termination >> regex termination.
- **A2. Replace heuristic `progress_estimate` with a structured signal.** The
  ~200-line keyword bucket estimator quantizes progress to 6 fixed values and
  *drives termination and the stall breaker*. A structured REFLECT output
  (JSON: `{progress: 0-1, done: bool, reason}`) or provider structured-output
  mode would be far more reliable.
- **A3. Make grace a signal, not a silent rescue.** When grace fires, that's a
  *convergence failure* worth logging/counting as a first-class metric so we can
  drive it toward zero, not paper over it. (And de-triplicate the three grace
  implementations into one — Part III §9.)

## B. Over-tooling & the "answer directly" gap

Darwin tools *no-tool* tasks (5 calls / 206s for a git-knowledge question). The
f-string THINK/PLAN prompts never say "answer directly when you already know."

- **B1. Prompt the direct-answer path.** Add explicit "if you can answer from
  your own knowledge without a tool, do so and finish" instructions to THINK (and
  PLAN's "don't plan tool steps for a question you can just answer"). This is the
  cheapest lever and directly targets the measured failure.
- **B2. Complexity-gate the whole cycle, not just the engine.** `--engine auto`
  already routes trivial turns to Lite via a heuristic classifier. Consider an
  *intra-Darwin* fast-path: a cheap PERCEIVE/PLAN check that emits a direct
  answer for trivial goals without spinning up the full loop (llmcore already has
  a `fast_path` IterationUpdate stub — `single_agent.py:790` — that's underused).
- **B3. Budget-aware planning.** PLAN estimates `iterations = len(steps)*2` but
  nothing feeds the remaining budget back into THINK's decision to keep tooling.

## C. Prompt & parser robustness (the Grimoire opportunity)

- **C1. Ship a real Darwin cognitive-cycle spell pack.** Author
  `thinking_prompt` / `planning_prompt` / `reflection_prompt` spells (VALIDATE is
  moot under `skip_validation`) that (a) encode the direct-answer heuristic (B1),
  (b) push convergence (A1), and (c) **reproduce the exact parser contracts**.
  Then re-run the eval as the planned `darwin+grimoire` arm to measure whether
  prompts were the ceiling. This is a bounded, high-value experiment.
- **C2. Move phases off regex-on-prose toward structured/JSON or native
  tool-calls.** Every phase parses free text with rigid header regexes; any
  reformat silently degrades. THINK already partly uses native tool-calls — extend
  that discipline to PLAN/REFLECT so a prompt change can't break parsing.
- **C3. Fail loudly on prompt-registry misconfig.** All four phases swallow
  `render()` errors to WARNING. A startup validation pass (render each mapped
  spell once with dummy vars) would surface a `strict=True` variable typo instead
  of silently reverting to f-strings while reporting `has_grimoire_prompt_registry=true`.

## D. Latency & cost

- **D1. Fix token capture (Part VI #4)** — prerequisite for any honest cost work.
- **D2. Collapse phases for simple goals.** 4 LLM calls/iteration is a lot when
  PERCEIVE retrieved nothing and PLAN produced one step. Consider merging
  PLAN+THINK for single-step goals, or skipping REFLECT on a converged iteration.
- **D3. Parallelize independent phase work** where safe (PERCEIVE retrieval can
  overlap PLAN for a cached plan).
- **D4. Model-tier the phases.** VALIDATE/REFLECT could run on a cheaper/faster
  model than THINK — the eval shows kimi (a slow thinking model) is punished
  ~13× by running *every* phase on it.

## E. Safety & correctness of the machinery

- **E1. Keep deterministic guards even when `skip_validation=True`.** Today
  fabricating APPROVED skips the `DANGEROUS_PATTERNS` + tool-registry checks along
  with the LLM judge. Separate the two: always run the cheap deterministic guards;
  only skip the *LLM* validation.
- **E2. Consolidate the three Darwin tool handlers** into one
  `build_darwin_tool_handler(...)` with a fixed enforcement order (they currently
  differ, with security-relevant ordering differences).
- **E3. Propagate `expected_outcome` into OBSERVE** (THINK has it; the
  orchestrator hardcodes `None`, making OBSERVE's expectation logic dead) so
  follow-up detection is real, not just "was there an error."
- **E4. Fix the deepseek DSML leak (Part VI #3).**

## F. Observability & memory

- **F1. Add the episode-persistence test** (Part VI #1) and verify episodic
  memory end-to-end on a live run — the "learning" loop was dead for its entire
  life and only a persistence assertion will keep it honest.
- **F2. Capture tool arguments on the Darwin path.** `_darwin_streaming_result`
  hardcodes `"arguments": {}`, so `/undo`, checkpoints, and usage attribution get
  no argument data on Darwin (unlike Lite).
- **F3. Persist prompt-optimization metrics.** The adapter's `record_use`
  metrics are in-memory only; if we want to A/B or optimize Darwin prompts, they
  need to survive the process.

## G. Evaluation methodology (make the next round sharper)

- **G1. Restraint-aware accuracy.** Add "correct AND tool-count ≤ expected" so
  over-tooling shows up in the tables, not just anecdotes.
- **G2. A lite `stalled`/`incomplete` signal** so single-shot breakage is
  measured, not eyeballed from response text.
- **G3. ≥2 trials** (per-model latencies are single-sample and outlier-dominated
  today) and **capture Darwin tokens** (G1 depends on D1).
- **G4. Separate "grace due to budget exhaustion" from "genuine early grace"** so
  the convergence rate isn't confounded by the reduced budget.

## The one-paragraph recommendation

Darwin's architecture is validated for multi-step / recover-and-continue work
but is bottlenecked by **weak default prompts and regex-based convergence**, not
by the 8-phase idea itself. The highest-leverage, lowest-risk sequence is:
**(1)** a Darwin cognitive-cycle prompt pack encoding direct-answer + explicit
finish (C1+B1+A1), **(2)** a first-class finish tool + structured progress
(A1+A2), **(3)** token capture + restraint-aware eval (D1+G1), then re-run this
exact 70-run matrix to measure the delta. That turns "Darwin is more accurate but
7× slower and over-tools" into a tunable, measurable engine.

---

# Part VIII — External Research: How the Field Solves This

*(Forthcoming — being gathered by a deep, multi-source research pass on: fixing
agent-loop failure modes, SOTA agentic reasoning architectures, production agent
frameworks, and prompt design for phase loops, plus a primary-source study of
LangGraph / OpenAI Agents SDK / smolagents / CrewAI / DSPy / Swarm agent loops.
This section will map field-proven techniques onto the Part VII directions.)*

---

## Appendix A — File map

| Concern | Location |
|---------|----------|
| Cognitive cycle orchestrator | `llmcore/agents/cognitive/phases/cycle.py` |
| The 8 phases | `llmcore/agents/cognitive/phases/{perceive,plan,think,validate,act,observe,reflect,update}.py` |
| State / models | `llmcore/agents/cognitive/models.py`, `llmcore/models.py` (Episode/EpisodeType) |
| Manager wrapper | `llmcore/agents/single_agent.py` |
| Grimoire prompt adapter | `llmcore/agents/prompts/grimoire_adapter.py` |
| Tool manager | `llmcore/agents/tools.py` |
| Lite engine | `wairu/sessions/tool_dispatch.py` |
| Darwin streaming adapter | `wairu/sessions/darwin_stream.py` |
| Headless Darwin runner | `wairu/sessions/darwin_headless.py` |
| Darwin wiring (fly) | `wairu/sessions/manager.py` |
| REPL Darwin routing | `wairu/repl/session.py` |
| Engine selection (headless) | `wairu/cli/commands/print_cmd.py` |
| Eval harness + data | `wairu/benchmarks/darwin_vs_lite.py`, `wairu/benchmarks/results/` |
| Eval methodology + results | `wairu/docs/DARWIN_VS_LITE_EVAL_PLAN.md` |

## Appendix B — Key config knobs

```toml
[wairu.darwin]
engine = "lite"                 # default engine for headless -p (lite|darwin|auto)
headless_max_iterations = 10    # Darwin budget for one -p turn (eval used 5)
repl_max_iterations = 10        # Darwin budget for one routed REPL turn
repl_routing = false            # true => REPL 'auto' engine (route complex turns)
streaming = true                # live phase updates for fly darwin
skip_validation = true          # bypass llmcore's VALIDATE phase (see Part II §6)

[wairu.grimoire]
enabled = false                 # master switch (inert cycle control plane when off)
repo_path = ""                  # a Grimoire repo with cognitive-cycle spells
register_prompt_registry = true
strict_prompts = true           # a variable-name typo -> silent f-string fallback
# prompt_map maps the 4 cycle ids -> your spell ids (see Part IV §12)

[wairu.autonomous]
execution_engine = "lite"       # default engine for fly
```

## Appendix C — Reproducing the eval

```bash
# from /av/data/repos/wairu, with provider keys loaded (never ollama):
python benchmarks/darwin_vs_lite.py run --engines lite,darwin --trials 1
python benchmarks/darwin_vs_lite.py summary \
  --out benchmarks/results/results_2026-07-08.jsonl
```
