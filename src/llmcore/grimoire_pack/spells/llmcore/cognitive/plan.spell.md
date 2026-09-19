---
id: llmcore/cognitive/plan
name: Cognitive PLAN phase
version: 1.0.0
tags: [llmcore.builtin, cognitive, plan]
description: Strategic decomposition of the goal into ordered actionable steps.
variables:
  goal:
    type: multiline
    required: true
    description: The goal to plan for
  context:
    type: multiline
    required: false
    default: ""
    description: Retrieved context from PERCEIVE
  constraints:
    type: multiline
    required: false
    default: ""
    description: Planning constraints
  existing_plan_section:
    type: multiline
    required: false
    default: ""
    description: Pre-formatted EXISTING PLAN block when replanning (else empty)
---

# SYSTEM
You are a strategic planning agent. Create clear, actionable plans.

# USER
Create a strategic plan to achieve the following goal:

GOAL:
{{ goal }}

CONTEXT:
{{ context }}

CONSTRAINTS:
{{ constraints }}
{{ existing_plan_section }}

Provide your plan as:
1. A numbered list of concrete, actionable steps
2. Strategic reasoning explaining your approach
3. Any risks or challenges identified

FORMAT:
PLAN:
1. [First step]
2. [Second step]
...

REASONING:
[Your strategic approach]

RISKS:
- [Risk 1]
- [Risk 2]
