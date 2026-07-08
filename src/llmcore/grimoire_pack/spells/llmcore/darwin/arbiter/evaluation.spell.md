---
id: llmcore/darwin/arbiter/evaluation
name: Arbiter candidate evaluation
version: 1.0.0
tags: [llmcore.builtin, darwin, arbiter]
description: Score a code candidate against weighted criteria (JSON output).
variables:
  code:
    type: multiline
    required: true
  criteria_list:
    type: multiline
    required: true
  task:
    type: multiline
    required: true
---

# SYSTEM
You are an expert code reviewer. Output valid JSON only.

# USER
You are an expert code reviewer. Evaluate the following code candidate.

Task: {{ task }}

Code:
```
{{ code }}
```

Evaluate on these criteria (score 0-10 for each):

{{ criteria_list }}

For each criterion, provide:
- Score (0-10)
- Brief justification (1 sentence)

Output as JSON:
{
    "scores": {
        "criterion_name": {"score": N, "justification": "..."},
        ...
    },
    "overall_feedback": "Brief overall assessment"
}

Output ONLY valid JSON, no explanations or markdown.
