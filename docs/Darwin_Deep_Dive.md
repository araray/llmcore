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
> the *brainstorm material* (improvement directions). Part VIII is *how the
> wider field solves these problems* — a six-framework source study (§18–19) and
> a peer-reviewed literature synthesis (§20), converging on one prescription
> (§21). If you only read one thing, read the Executive Summary, then **§21**
> (the unified, ranked action map). References in Appendix D.

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

This part maps field-proven techniques onto the Part VII directions. It draws on
a **primary-source study of six production agent frameworks** — their actual
source cloned to `/av/avalon/xrepos` and read (§18–§19) — and a
**multi-source academic + practitioner web-research pass** (§20, forthcoming).

## 18. The cross-framework consensus (the important part)

We read the agent-loop, termination, and tool-gating source of **LangGraph,
OpenAI Agents SDK, smolagents, CrewAI, DSPy, and Swarm.** The convergence is
striking — and it lands squarely on Darwin's two measured weaknesses
(74% self-convergence, over-tooling). The table below is the single most useful
artifact for the brainstorm: **every column-3 "✗/partial" is a concrete Darwin
gap that ≥3 mature frameworks close the same way.**

| Pattern | Who does it | Darwin today | The transfer |
|---------|-------------|--------------|--------------|
| **Explicit finish/submit tool** — the model *calls* to terminate; detected structurally by tool name/exception | smolagents (`final_answer`), DSPy (`finish`/`submit`), OpenAI (`StopAtTools`), CrewAI (`result_as_answer`) | **✗** — a `Final Answer:` **regex** on free text (`think.py:847`). llmcore *registers* a `finish` builtin but never wires it to convergence | **A1.** Wire the existing `finish` tool into the stop path; THINK/ACT recognize a `finish` call → done. Structural >> regex. *This is the #1 fix — 4 of 6 frameworks converge here.* |
| **Forced synthesis on cap-hit, inside the loop** — never return nothing; on budget exhaustion do one tool-less pass to synthesize a best-effort answer | smolagents (`provide_final_answer`), LangGraph (soft finish before the wall), OpenAI (`RunErrorHandler`), DSPy V2 (`_forced_submit`, pins `tool_choice=submit`), CrewAI (`handle_max_iterations_exceeded` — invariant: the loop *cannot* exit un-converged), Swarm (flagged as its #1 gap) | **partial** — exists but **wairu-side** (grace synthesis), triplicated, and only after the fact | **A3/A1.** Move grace *into the cognitive cycle* as a guaranteed forced-finalize; make "loop cannot exit un-converged" an invariant. All 6 frameworks do cap-synthesis in-loop. |
| **`remaining_steps` injected into the prompt** — the model self-paces toward the budget | LangGraph (computed `stop − step`, injected every step), smolagents (planning injects `remaining_steps`) | **✗** — budget is invisible to THINK/PLAN | **A2/B3.** Compute `budget − iteration` and pass it into THINK/PLAN so the model knows how much runway it has. |
| **Soft budget-aware forced finish** — intercept the model *before* it over-commits when `remaining < 2` and swap in a terminal answer | LangGraph (`_are_more_steps_needed`) | **✗** | **A1.** When `remaining_steps < 2`, skip ACT/OBSERVE and force a final answer. Directly attacks the 26%. |
| **Machine-readable `termination_reason`** — `converged`/`forced`/`max_steps`/`parse_error` | DSPy V2 (`termination_reason`), smolagents (`RunResult.state`) | **partial** — wairu records `terminal_mode` for the *whole turn*, but the cycle has no per-run stop reason | **A3/G4.** Have REFLECT/UPDATE emit *why* it stopped — essential telemetry to drive 74%→higher. |
| **Complexity routing / fast-path** — cheaply classify, then skip phases or shrink the toolset for simple tasks | CrewAI (`reasoning_effort` low/med/high gates REFLECT depth), OpenAI (`tool_use_behavior` callable), LangGraph (`pre_model_hook` router), Swarm (agent/toolset swap), DSPy (per-call `max_iters`) | **partial** — `--engine auto` routes *between engines*, but **inside** Darwin every task runs all phases at full depth | **B1/B2/D2.** Add an intra-cycle fast-path (PERCEIVE→answer for trivial goals) and gate PLAN/REFLECT depth by a cheap complexity signal. llmcore has an unused `fast_path` stub. |
| **Redundant / repeated-tool-call detection** — same tool+args as last step → reject/penalize | CrewAI (`_check_tool_repeated_usage`), OpenAI (`AgentToolUseTracker`) | **✗** (LangGraph, Swarm, Darwin all lack it) | **B/E.** A cheap deterministic loop-breaker: dedupe tool-call signatures in OBSERVE/REFLECT. Both frameworks that *have* it and both that *lack* it flag it as the missing guard. |
| **`reset_tool_choice` after first tool** — stop *forcing* tool calls once one has run | OpenAI (one-line, docstring: "so the agent doesn't enter an infinite loop of tool usage") | n/a (Darwin doesn't force tool_choice, but the prompt effectively does) | **B1.** Ensure THINK isn't implicitly pushing a tool every iteration; add an explicit "you may answer now" affordance. |
| **Separate acting from answering** — the loop only gathers observations; a *dedicated extraction pass* produces the final typed answer | DSPy (`extract` runs unconditionally after the loop) | **partial** — this is *exactly* what grace does, but only as a fallback | **A/C.** Consider making a final synthesis pass the **normal** terminal step, not just the rescue — DSPy converges more reliably by never asking the last loop-turn to be both correct *and* terminal. |
| **Reflection → targeted, re-injected advice** — per-phase blame + concrete correction fed into the next attempt | DSPy (`Refine.OfferFeedback`, `hint_` injection) | **partial** — REFLECT produces `insights`/`next_focus` appended to working memory, but not phase-keyed corrective advice | **F/C.** Make REFLECT→UPDATE carry specific corrective guidance keyed to the phase that erred, so a retry is a *better* retry. |
| **Errors-as-observations** — a failed tool's error becomes an observation; the loop continues and self-repairs | all six | **✓** — OBSERVE does this (truncated 4000 chars) | keep; it's a strength. |
| **Structured output / typed done-contract** — `submit` validates all required output fields exist before accepting convergence | DSPy V2 (`submit` rejects missing fields), LangGraph (`generate_structured_response` node), OpenAI (`output_type` schema) | **✗** — every phase parses free-text with header regexes | **C2.** Move PLAN/THINK/REFLECT toward structured/JSON or native tool-calls so a prompt reword can't break parsing *and* so "done" is a typed assertion. |
| **Gate phase depth / don't run every phase every loop** — planning on an interval, single-tool-then-reflect | smolagents (`planning_interval`), CrewAI (one tool → forced `post_tool_reasoning`) | **partial** — PLAN is conditional, but THINK/REFLECT/UPDATE run every iteration | **D2.** Most iterations should be ACT/OBSERVE; replan/reflect on a cadence, not always. |
| **Bounded replanning** — a hard replan budget that keeps completed results | CrewAI (`max_replans=3`) | **✗** — the UPDATE→PLAN back-edge (`plan_needs_update`) has no replan cap | **A/E.** Cap replans so the cycle can't thrash re-planning. |

## 19. Per-framework one-line highlights

- **smolagents** — *code-as-action*: one step composes many operations in a
  single LLM turn (collapsing N JSON round-trips), with an inner
  `MAX_OPERATIONS` guard. `final_answer` is a real tool; cap-hit runs
  `provide_final_answer`. The strongest anti-over-tooling idea we saw.
- **LangGraph** — the loop *is a cyclic graph*; termination = "no tool call →
  `END`" + a computed `remaining_steps` + a **soft forced-finish** one step
  before the recursion wall. Also `return_direct` tools (tool output *is* the
  answer, skip the final LLM round-trip).
- **OpenAI Agents SDK** — type-driven termination (`NextStepFinalOutput`),
  `reset_tool_choice`, `AgentToolUseTracker`, and a per-run `tool_use_behavior`
  policy that's the cleanest complexity-routing injection point.
- **DSPy** — `ReActV2` is the gold standard for convergence: explicit `submit`
  with a **typed completeness contract**, **forced finalize** on cap, a
  machine-readable `termination_reason`, and `Refine`'s per-module feedback loop.
- **CrewAI** — the richest guard set: `handle_max_iterations_exceeded` invariant,
  `reasoning_effort` depth routing, repeated-tool-call refusal, per-tool budgets,
  bounded replanning, early goal-achieved detection.
- **Swarm** — minimalist ("no tool call = done", agent handoffs); its *absences*
  (no cap-synthesis, no redundant-call guard) are themselves a lesson — it
  documents exactly the gaps Darwin also has.

## 20. Academic & production web research

Three multi-source, adversarially-verified research passes (each: fan-out
search → fetch → 3-vote refutation → synthesis) covered (a) agent-loop failure
modes & fixes, (b) SOTA agentic reasoning architectures, (c) production
frameworks (the doc layer complementing §18's source layer). The literature is
unusually decisive and lands on the *same* levers §18 surfaced from code — with
several findings that **directly implicate Darwin's design.**

### 20.1 Over-tooling is an intrinsic, measurable, model-specific bias

- **Agents systematically over-call tools; the hard part is deciding when NOT
  to.** On the When2Call benchmark, six models show high call-accuracy but weak
  *no-call* accuracy (55–70% overall); "To Call or Not to Call"
  ([arXiv 2605.18882]) formalizes an **Intrinsic Bias toward CALL** — the model
  favors calling even when call/no-call evidence is at parity. Prompting "use
  tools less" is the *wrong* lever: it cuts calls indiscriminately and hard
  tasks pay a disproportionate accuracy price.
- **The tool-vs-answer decision is already latent in the model.** "LLM Agents
  Already Know When to Call Tools" ([arXiv 2605.09252], UCSD/Amazon) shows
  tool-necessity is **linearly decodable from the pre-generation hidden state
  (AUROC 0.89–0.96)**. A tiny linear probe ("Probe&Prefill") cuts tool calls
  **~48% at 1.7% accuracy loss** — 8× better than the best prompt-only baseline
  (which cuts only 6%) — and on a real agentic benchmark cuts API calls 20–56%
  with *no* accuracy loss. (Caveat: needs white-box activations — applies to
  self-hosted models, not API-only.)
- **Necessity is model-specific** ([arXiv 2605.14038], UMD): routing must be
  *capability-calibrated per model* — there's a 26–54% "knowing-doing gap"
  where the model recognizes a tool is needed but fails to act. **This is
  precisely our eval finding that Darwin amplifies model quality differences**
  (gpt-5.4 100%@25s vs kimi 100%@219s) — the over-tooling/convergence gap is
  model-dependent, so any fix should be tuned per model, not globally.

### 20.2 Convergence: never delegate the stop decision to the model alone

- **The single most-cited cause of non-termination is an unbounded feedback
  path.** An empirical study of 68 real infinite-loop failures across 47
  projects ([arXiv 2607.01641]) finds every one shared a missing/mis-scoped
  bound, and **38.2% stem from delegating the stop decision to the model.** A
  stop criterion only works if it's a *verifiable* "strong finite bound sitting
  **on** the loop's feedback path"** — exactly the explicit finish-tool +
  in-loop forced-finalize pattern §18 found universal.
- **Running to budget exhaustion is actively harmful, not just wasteful** —
  correct answers get *discarded during over-refinement.* Adaptive early
  termination (an LLM-as-judge deciding each round whether to stop, with a
  minimum-2-round floor — TUMIX, [arXiv 2510.01279], Google) reaches near-full
  accuracy at **~49% of fixed-round cost.** Darwin's "act until the budget runs
  out" default is the anti-pattern here; the 4 budget-exhausting runs in our
  eval are the symptom.
- **Reflexion's bounded termination is the template**: stop on *max-trials* OR
  *no-improvement between two trials* OR *success* ([arXiv 2303.11366]).

### 20.3 Reflection only helps when it's grounded in an external check

This is the subtlest — and most important — nuance for Darwin's REFLECT phase:

- **Same-model Self-Refine works** (~20% absolute preference gain, no training;
  NeurIPS 2023 [arXiv 2303.17651]) **when the critique targets stylistic/format
  quality with a clear improvement signal.**
- **BUT intrinsic self-correction — a model re-judging its own reasoning with NO
  external signal — is unreliable and frequently NET-NEGATIVE** (ICLR 2024
  [arXiv 2310.01798]; TACL 2024): performance often *degrades* after
  self-correction on reasoning tasks.
- **Reflexion works because it retries against a task signal** (unit tests, a
  binary reward/heuristic), writing verbal critiques into episodic memory. The
  lesson: **REFLECT should be gated on a verifiable external check (a tool
  result, an oracle, a test), not on the model re-scoring itself.** Darwin's
  REFLECT currently self-evaluates and emits a heuristic `progress_estimate`
  with no external anchor — a documented anti-pattern.

### 20.4 The architecture taxonomy — three cost tiers keyed to LLM-calls/step

| Architecture | Structure | Termination | vs ReAct | Cost |
|---|---|---|---|---|
| **ReAct** ([2210.03629]) | interleave reason+act+observe, 1 call/step | no tool call → done | baseline; +34%/+10% ALFWorld/WebShop over non-reasoning | **1×** (cheapest) |
| **Reflexion** ([2303.11366]) | verbal critique → episodic memory → retry | max-trials / no-improve / success | 80→91% HumanEval; 75→97% AlfWorld ≤12 trials | +few/failed step |
| **ADaPT** ([2311.05772]) | recursively decompose **only on failure**, bounded depth | executor self-classifies done/failed | +28/27/33 on ALFWorld/WebShop/TextCraft | conditional |
| **ReWOO** ([2305.18323]) | Planner blueprints all tool calls upfront → Worker → Solver | plan consumed | +4% HotpotQA, **5× token efficiency** | low (decoupled) |
| **Self-Consistency** ([2203.11171]) | sample N CoT paths, majority vote | fixed N | +18% GSM8K | **N×** |
| **LATS** ([2310.04406]) | MCTS over trajectories + LM value + reflection | solved or compute budget | HotpotQA 0.61 vs 0.32; HumanEval 92.7% | **5–20×** |

**The accuracy-without-latency-explosion lever is architectural and
conditional:** reflect/decompose *only when a step fails* (ADaPT/Reflexion),
tune search width (LATS n=1 ≈ ReAct), or plan tool calls upfront to decouple
reasoning from observations (ReWOO). The verified synthesis recommendation for
an 8-phase cycle: **a ReAct backbone + Reflexion-style verbal REFLECT/UPDATE
into memory + ADaPT-style *bounded, failure-triggered* decomposition at PLAN +
a binary self-heuristic at VALIDATE, under a hard cap, reserving LATS-style
branching for genuinely hard sub-tasks only.**

### 20.5 Latency/cost: route by complexity, prune the trajectory

- **Complexity routing to both a cheaper model and a lighter reasoning
  strategy** (Route-To-Reason, [arXiv 2505.19435]): +2.5pp accuracy at **−60%
  tokens.** Heavy reasoning helps mainly on hard tasks and is *marginal or
  negative* on easy ones — the empirical basis for a fast-path.
- **Inference-time trajectory reduction** (AgentDiet, FSE 2026
  [arXiv 2509.23586]): removing redundant/expired info from the running
  trajectory cuts input tokens **40–60%** and cost **21–36%** with no
  performance loss. Mirrors CrewAI's isolated-context and DSPy's oldest-first
  truncation from §18.

### 20.6 Three findings that directly implicate Darwin's design

1. **Darwin PLANs before it observes — and the literature says that hurts
   knowledge tasks.** BOLAA ([arXiv 2308.05960]) shows PlanAct/PlanReAct
   *underperform* plain ReAct on knowledge-reasoning (HotpotQA): "plans
   generated before interaction lack contextualized information and induce more
   hallucination." Darwin's cycle runs **PLAN → THINK**, planning up front every
   time. This plausibly explains **why Darwin over-tools *knowledge* questions**
   (it plans tool steps for a question it could just answer) — the most
   surprising over-tooling result in our eval. **Directions:** make PLAN
   conditional/failure-triggered (ADaPT), or defer planning until after a first
   PERCEIVE/THINK observation, or skip PLAN entirely on no-tool goals.
2. **Darwin's REFLECT self-judges without an external anchor** (§20.3) — the
   documented net-negative pattern. Gate it on tool-result/oracle signals.
3. **Darwin runs all reflective phases every iteration** — the literature says
   reflect/decompose *conditionally on failure* is what buys accuracy without
   latency. Darwin's every-iteration REFLECT/UPDATE is pure overhead on
   succeeding steps.

### 20.7 Engineering ⋂ Academia — the consensus is one prescription

The six-framework source study (§18) and the peer-reviewed literature (§20)
independently converge:

| Lever | §18 frameworks | §20 literature |
|-------|----------------|----------------|
| Explicit finish/submit tool | 4/6 | non-termination study: bound must be *on* the feedback path |
| In-loop forced-finalize on cap | 6/6 | TUMIX adaptive stop; budget-exhaustion is harmful |
| Complexity routing / fast-path | CrewAI, OpenAI, LangGraph, Swarm | RTR (−60% tokens); Probe&Prefill (−48% calls) |
| Conditional (not every-iteration) reflect/plan | smolagents `planning_interval`, CrewAI | ADaPT (decompose on failure); BOLAA (plan-first hurts) |
| Reflection grounded in external check | errors-as-observations (all 6) | Self-Refine caveat; Reflexion retries vs a test |
| Structured/typed done-contract | DSPy `submit`, OpenAI `output_type` | — |
| Trajectory/context pruning | CrewAI isolated ctx, DSPy truncation | AgentDiet (−40–60% tokens) |

When the practitioners who *built* the frameworks and the researchers who
*measured* the failure modes agree this precisely, the path is clear.

## 21. Consensus → Darwin action map (ranked, unified)

Synthesizing Part VII + §18 (framework source) + §20 (literature). Each item
cites its dual evidence — *what mature frameworks build* and *what the research
measures*. This is the recommended sequence for the team.

**Tier 1 — convergence (attacks the 26% non-convergence + budget-exhaustion):**

1. **Wire a first-class `finish`/`submit` tool into the stop path** (A1) —
   replace the `Final Answer:` regex; llmcore already registers an unused
   `finish` builtin. *Frameworks: 4/6. Research: the stop bound must sit ON the
   feedback path ([2607.01641]); delegating stop to the model is the #1
   non-termination cause.*
2. **Move forced-finalize into the cycle as an invariant** (A1/A3) — on cap-hit
   and when `remaining_steps < 2`, do one tool-less synthesis pass; "cannot exit
   un-converged." Retire the triplicated wairu-side grace crutch. *Frameworks:
   6/6. Research: budget-exhaustion is actively harmful; TUMIX adaptive stop hits
   ~full accuracy at ~49% cost.*
3. **Inject `remaining_steps` + emit a structured `termination_reason`**
   (A2/A3/G4) — self-pacing + convergence telemetry. *Frameworks: LangGraph/DSPy/
   smolagents. Research: Reflexion's max-trials/no-improve/success template.*

**Tier 2 — over-tooling (attacks the 5-tools-for-a-knowledge-question problem):**

4. **A direct-answer heuristic in the THINK/PLAN prompt + an intra-cycle
   fast-path** (B1/B2) — "answer from your own knowledge if no tool is needed."
   *Frameworks: complexity routing in CrewAI/OpenAI/LangGraph/Swarm. Research:
   over-calling is an intrinsic bias; prompting "use tools less" is the wrong
   lever, but RTR-style routing cuts tokens −60%.*
5. **Make PLAN conditional / post-observation, not always-upfront** (new — B/A)
   — Darwin plans before it observes, which **hurts knowledge tasks** ([2308.05960])
   and is a prime suspect for the knowledge over-tooling. *Frameworks: smolagents
   `planning_interval`, CrewAI. Research: ADaPT decomposes only on failure.*
6. **Redundant-tool-call detector** (B/E) — dedupe tool signatures; a cheap
   deterministic loop-breaker. *Frameworks: CrewAI, OpenAI. (Both frameworks
   lacking it flag it as the missing guard.)*
7. **(Self-hosted only) a tool-necessity probe** — tool-need is linearly
   decodable from activations (AUROC 0.89–0.96); Probe&Prefill cuts calls ~48%
   at 1.7% accuracy loss. Calibrate the threshold **per model** ([2605.09252],
   [2605.14038]) — matches our eval's finding that Darwin amplifies per-model
   differences.

**Tier 3 — reflection & structure (quality + robustness):**

8. **Ground REFLECT in an external check, not self-judgment** (new — F/C) —
   Darwin's REFLECT self-scores and emits a heuristic `progress_estimate` with
   no anchor; *intrinsic self-correction is net-negative on reasoning*
   ([2310.01798]). Gate reflection on a tool-result/oracle signal (Reflexion),
   and make REFLECT/UPDATE emit **phase-keyed corrective advice** re-injected
   next iteration (DSPy `Refine`).
9. **Structured/typed phase outputs** (C2) — move PLAN/THINK/REFLECT off
   regex-on-prose toward JSON/native-tool-calls; also unlocks a clean Grimoire
   spell pack (C1) and a typed done-contract. *Frameworks: DSPy `submit`, OpenAI
   `output_type`, LangGraph structured node.*
10. **Conditional phase depth + trajectory pruning** (D2/D-cost) — don't run
    REFLECT/UPDATE every iteration on succeeding steps; prune stale trajectory
    context. *Frameworks: CrewAI isolated context. Research: ADaPT conditional;
    AgentDiet −40–60% tokens.*

**Then measure.** Re-run the exact 70-run matrix with **token capture (D1)** and
**restraint-aware scoring (G1)**, ≥2 trials. Expected deltas: over-tooling ↓,
native convergence ↑ (fewer grace runs), latency ↓ (fast-path + conditional
phases), accuracy held or up. The `darwin+grimoire` prompt-pack arm (C1) tests
whether prompts alone close the gap before touching the architecture.

> **If the team does only three things:** (1) an explicit finish tool +
> in-loop forced-finalize (kills the grace crutch and the 26%), (2) a
> direct-answer heuristic + conditional PLAN (kills knowledge over-tooling),
> (3) externally-grounded REFLECT (stops the net-negative self-judgment). Then
> re-run the benchmark.

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

## Appendix D — References (Part VIII)

**Peer-reviewed / arXiv (adversarially verified, §20):**
- ReAct — *Reasoning + Acting* — arXiv 2210.03629 (ICLR 2023)
- Reflexion — *verbal RL / episodic reflection* — arXiv 2303.11366 (NeurIPS 2023)
- Self-Refine — arXiv 2303.17651 (NeurIPS 2023)
- "LLMs cannot self-correct reasoning yet" — arXiv 2310.01798 (ICLR 2024); TACL 2024 self-correction survey
- Self-Consistency — arXiv 2203.11171
- Plan-and-Solve / BOLAA (plan-first hurts knowledge tasks) — arXiv 2308.05960
- LATS (Language Agent Tree Search) — arXiv 2310.04406 (ICML 2024)
- ADaPT (as-needed decomposition) — arXiv 2311.05772
- ReWOO (reasoning without observation) — arXiv 2305.18323
- When2Call / "To Call or Not to Call" (tool over-calling) — arXiv 2605.18882
- "LLM Agents Already Know When to Call Tools" (Probe&Prefill) — arXiv 2605.09252
- Model-Adaptive Tool Necessity / knowing-doing gap — arXiv 2605.14038
- Agent-loop non-termination empirical study (68 failures) — arXiv 2607.01641
- TUMIX (adaptive early termination) — arXiv 2510.01279
- Route-To-Reason (complexity routing) — arXiv 2505.19435
- AgentDiet (trajectory reduction) — arXiv 2509.23586 (FSE 2026)

**Framework source studied (§18, cloned to `/av/avalon/xrepos`):**
- LangGraph (`langchain-ai/langgraph`) · OpenAI Agents SDK (`openai/openai-agents-python`)
  · smolagents (`huggingface/smolagents`) · CrewAI (`crewAIInc/crewAI`)
  · DSPy (`stanfordnlp/dspy`) · Swarm (`openai/swarm`)

*Note: a few 2605/2607-series arXiv ids are from very recent (2026) preprints
surfaced by the research pass; treat their specific numbers as provisional and
verify the id before citing externally — the findings were cross-checked against
multiple independent sources during verification, but preprint ids occasionally
shift.*
