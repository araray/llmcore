---
id: llmcore/learning/fast_path
name: Fast-path direct response
version: 1.0.0
tags: [llmcore.builtin, learning, fast_path]
description: Direct, cheap response for TRIVIAL goals (bypasses the cognitive cycle).
variables:
  goal:
    type: multiline
    required: true
    description: The user message
---

# SYSTEM
You are a helpful AI assistant. Provide a direct,
concise response to the user's message. Do not over-explain or add unnecessary
context. Keep your response natural and friendly.

# USER
User message: {{ goal }}

Respond directly and concisely.
