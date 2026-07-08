---
id: llmcore/cognitive/think
name: Cognitive THINK phase
version: 1.0.0
tags: [llmcore.builtin, cognitive, think]
description: ReAct-style decision — a thought plus a tool call or a final answer.
variables:
  goal:
    type: multiline
    required: true
    description: The overall goal
  current_step:
    type: string
    required: true
    description: The current plan step
  history:
    type: multiline
    required: false
    default: "No previous actions."
    description: Bounded JSON of recent iteration summaries
  context:
    type: multiline
    required: false
    default: ""
    description: Retrieved context from PERCEIVE
  tools:
    type: multiline
    required: true
    description: Formatted available-tool definitions
  remaining_steps:
    type: string
    required: false
    default: "unlimited"
    description: Iteration budget remaining for this run
---

# SYSTEM
You are an autonomous AI agent using the ReAct framework. Think step-by-step and use tools effectively.

# USER
You are solving this task:

GOAL: {{ goal }}

CURRENT STEP: {{ current_step }}

RECENT HISTORY:
{{ history }}

RELEVANT CONTEXT:
{{ context }}

AVAILABLE TOOLS:
{{ tools }}

Use the ReAct format:

Thought: [Your reasoning about what to do next]
Action: [Tool name]
Action Input: [Tool arguments]

OR if the task is complete:

Thought: [Final reasoning]
Final Answer: [Complete answer to the goal]

Respond now:
