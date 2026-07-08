---
id: llmcore/autonomous/goal_decomposition
name: Autonomous goal decomposition
version: 1.0.0
tags: [llmcore.builtin, autonomous, goals]
description: Decompose a high-level goal into 3-5 actionable sub-goals (JSON output).
variables:
  goal_description:
    type: multiline
    required: true
    description: The high-level goal description
  context_json:
    type: multiline
    required: false
    default: "None"
    description: JSON-encoded goal context
---

# SYSTEM
You are a goal decomposition expert.

# USER
You are an expert planner. Decompose this high-level goal into 3-5 actionable sub-goals.

GOAL: {{ goal_description }}

CONTEXT:
{{ context_json }}

For each sub-goal, provide:
1. A clear, actionable description
2. Success criteria (measurable condition)
3. Priority: critical, high, normal, or low
4. Estimated difficulty: easy, medium, or hard

Respond with a JSON array:
[
  {
    "description": "...",
    "success_criteria": "...",
    "priority": "normal",
    "difficulty": "medium"
  },
  ...
]

Only output the JSON array, nothing else.
