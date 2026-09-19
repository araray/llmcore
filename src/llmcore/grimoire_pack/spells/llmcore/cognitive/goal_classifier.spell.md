---
id: llmcore/cognitive/goal_classifier
name: Goal complexity classifier (LLM fallback)
version: 1.0.0
tags: [llmcore.builtin, cognitive, classifier]
description: LLM fallback classification when the heuristic classifier is uncertain.
variables:
  goal:
    type: multiline
    required: true
    description: The user goal to classify
---

# USER
Classify the following user goal by complexity.

User Goal: "{{ goal }}"

Respond with exactly one line in this format:
COMPLEXITY: [TRIVIAL|SIMPLE|MODERATE|COMPLEX|AMBIGUOUS]
INTENT: [GREETING|FAREWELL|QUESTION|TASK|CREATIVE|ANALYSIS|META|UNKNOWN]
CONFIDENCE: [0.0-1.0]
REQUIRES_TOOLS: [true|false]
MAX_ITERATIONS: [number 1-25]

Classification guidelines:
- TRIVIAL: Greetings, thanks, simple acknowledgments (no tools needed)
- SIMPLE: Single file read, simple search, basic question (1-3 iterations)
- MODERATE: Multi-step task, comparison, debugging (5-15 iterations)
- COMPLEX: Research, analysis with report, building something (15-25 iterations)
- AMBIGUOUS: Unclear what user wants, needs clarification

Respond only with the classification, no explanation.
