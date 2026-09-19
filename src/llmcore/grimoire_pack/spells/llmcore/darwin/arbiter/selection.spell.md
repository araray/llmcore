---
id: llmcore/darwin/arbiter/selection
name: Arbiter candidate selection
version: 1.0.0
tags: [llmcore.builtin, darwin, arbiter]
description: Select the best candidate from scored options (JSON output).
variables:
  candidates_summary:
    type: multiline
    required: true
  task:
    type: multiline
    required: true
---

# SYSTEM
Output valid JSON only.

# USER
You are selecting the best code candidate from multiple options.

Task: {{ task }}

Candidates and their scores:
{{ candidates_summary }}

Based on the scores and the task requirements, select the best candidate.
Consider:
- Weighted scores (higher weight = more important)
- Overall quality and adherence to requirements
- Trade-offs between candidates

Output as JSON:
{
    "selected_id": "candidate_X",
    "reasoning": "Why this candidate is best",
    "confidence": 0.X
}

Output ONLY valid JSON, no explanations or markdown.
