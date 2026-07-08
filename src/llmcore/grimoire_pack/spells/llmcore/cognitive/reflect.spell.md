---
id: llmcore/cognitive/reflect
name: Cognitive REFLECT phase
version: 1.0.0
tags: [llmcore.builtin, cognitive, reflect]
description: Self-evaluation of the last action, progress estimate, plan updates.
variables:
  goal:
    type: multiline
    required: true
    description: The original goal
  plan:
    type: multiline
    required: true
    description: The current plan, numbered
  current_step_display:
    type: string
    required: false
    default: ""
    description: "Current step display, e.g. '2. Read the file'"
  last_action:
    type: string
    required: true
    description: The last action, formatted name(arguments)
  observation:
    type: multiline
    required: true
    description: The OBSERVE phase output
  iteration:
    type: string
    required: true
    description: Iteration number
  action_success:
    type: string
    required: false
    default: ""
    description: Whether the action executed successfully (true/false)
  matches_expectation:
    type: string
    required: false
    default: ""
    description: Whether the observation matched the expected outcome
---

# SYSTEM
You are a reflective AI agent. Honestly evaluate your actions, identify learnings, and recommend improvements.

# USER
Reflect on your recent action and its outcome.

ORIGINAL GOAL:
{{ goal }}

CURRENT PLAN:
{{ plan }}

CURRENT STEP: {{ current_step_display }}

LAST ACTION:
{{ last_action }}

OBSERVATION:
{{ observation }}

ITERATION: {{ iteration }}

REFLECTION QUESTIONS:
1. Did the action produce the expected result?
2. Are we making progress toward the goal?
3. Should we continue with the current plan or adjust?
4. What have we learned that could help in future iterations?
5. Is the current step complete?

PROVIDE:
- EVALUATION: Assess the action's effectiveness (success/partial/failure)
- PROGRESS: Estimate overall progress toward goal (0-100%)
- INSIGHTS: Key learnings from this iteration
- PLAN_UPDATE: Whether plan needs modification (yes/no)
- STEP_COMPLETED: Is current step done (yes/no)
- NEXT_FOCUS: What to prioritize in the next iteration

Be honest and critical in your self-assessment.
