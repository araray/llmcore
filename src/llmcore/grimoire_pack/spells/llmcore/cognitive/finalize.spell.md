---
id: llmcore/cognitive/finalize
name: Cognitive forced-finalize synthesis
version: 1.0.0
tags: [llmcore.builtin, cognitive, finalize]
description: >
  Tool-less synthesis pass run when the iteration budget is exhausted (or a
  forced finalize triggers): produce the best complete final answer from the
  accumulated observations. The loop cannot exit un-converged.
variables:
  goal:
    type: multiline
    required: true
    description: The original goal
  history:
    type: multiline
    required: true
    description: JSON of the run's recent iteration summaries (real observations)
  context:
    type: multiline
    required: false
    default: ""
    description: Any additional context carried on the agent state
  reason:
    type: string
    required: true
    description: Why finalization was forced (e.g. forced_finalize, synthesis_fallback)
---

# SYSTEM
You are an autonomous AI agent concluding a task. Synthesize the best complete final answer from the work already done. Do not request tools.

# USER
Your step budget for this task is exhausted (reason: {{ reason }}). You must now produce your final answer.

GOAL: {{ goal }}

ADDITIONAL CONTEXT:
{{ context }}

Here is exactly what your tools returned during the run (JSON of your recent iterations):
{{ history }}

Using ONLY the information above, produce the best complete final answer to the goal now. Do not request tools. If the information is incomplete, answer with what it shows and note briefly what is missing.

If the finish tool is available, call finish with your complete answer in the `answer` argument. Otherwise respond with the answer text directly.
