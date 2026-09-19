---
id: llmcore/darwin/arbiter/generation
name: Arbiter candidate generation
version: 1.0.0
tags: [llmcore.builtin, darwin, arbiter]
description: Generate one code candidate for the task.
variables:
  additional_instructions:
    type: multiline
    required: true
  context:
    type: multiline
    required: true
  task:
    type: multiline
    required: true
---

# USER
You are an expert software developer.

Task: {{ task }}

Context: {{ context }}

{{ additional_instructions }}

Generate high-quality, production-ready code. Include:
1. Clear, readable implementation
2. Proper error handling
3. Helpful comments where needed
4. Type hints (if applicable)

Output only the code, no explanations or markdown code fences.
