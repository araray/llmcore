# Model Card Library

> **Version**: 1.0.0  
> **Module**: `llmcore.model_cards`  
> **Status**: Production

The Model Card Library provides comprehensive metadata management for LLM models across all supported providers. Model cards enable intelligent model discovery, capability validation, cost estimation, and context length enforcement throughout the llmcore ecosystem.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Storage Locations](#storage-locations)
5. [Schema Reference](#schema-reference)
   - [Core Fields](#core-fields)
   - [Enumerations](#enumerations)
   - [Nested Structures](#nested-structures)
   - [Provider Extensions](#provider-extensions)
6. [Creating Model Cards](#creating-model-cards)
   - [Method 1: JSON File](#method-1-json-file-recommended)
   - [Method 2: Python API](#method-2-python-api)
7. [Complete Examples](#complete-examples)
   - [Local Ollama Model](#example-1-local-ollama-chat-model)
   - [API Model with Pricing](#example-2-api-model-with-pricing)
   - [Embedding Model](#example-3-embedding-model)
   - [MoE Architecture Model](#example-4-mixture-of-experts-model)
   - [Reasoning Model](#example-5-reasoning-model-with-extended-thinking)
8. [Validation & Troubleshooting](#validation--troubleshooting)
9. [Registry API](#registry-api)
10. [Best Practices](#best-practices)
11. [Migration & Compatibility](#migration--compatibility)

---

## Overview

### What is a Model Card?

A **model card** is a structured metadata document that describes an LLM's:

- **Identity**: Unique ID, display name, provider, model type
- **Technical specs**: Context window, output limits, architecture
- **Capabilities**: Vision, tool use, streaming, reasoning, etc.
- **Commercial info**: Pricing per token, rate limits
- **Lifecycle**: Release date, deprecation status, successor model
- **Provider-specific**: Quantization levels, API features, extensions

### Why Use Model Cards?

| Use Case | Benefit |
|----------|---------|
| **Context enforcement** | Automatically truncate input to `max_input_tokens` |
| **Cost estimation** | Calculate API costs before sending requests |
| **Capability validation** | Check if a model supports vision/tools before use |
| **Model discovery** | List available models, filter by capability |
| **Deprecation warnings** | Alert users when using deprecated models |
| **Alias resolution** | Map `claude-4.5-sonnet` → `claude-sonnet-4-5-20250929` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ModelCardRegistry                           │
│                      (Singleton)                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  Built-in Cards │    │   User Cards    │                    │
│  │  (llmcore pkg)  │    │ (~/.config/...)  │                    │
│  │                 │    │                 │                    │
│  │  Lower Priority │ ←─ │ Higher Priority │  (user overrides)  │
│  └─────────────────┘    └─────────────────┘                    │
├─────────────────────────────────────────────────────────────────┤
│  Indexes:                                                       │
│  • provider → model_id → ModelCard                             │
│  • alias → (provider, model_id)                                │
│  • lowercase_model_id → [(provider, model_id), ...]            │
└─────────────────────────────────────────────────────────────────┘
```

The registry loads cards on first access (lazy loading) and supports:
- **User overrides**: User cards with the same `model_id` replace built-in cards
- **Case-insensitive lookup**: `GPT-4O` resolves to `gpt-4o`
- **Alias resolution**: `claude-4.5-sonnet` → `claude-sonnet-4-5-20250929`
- **Hot reload**: `registry.load(force_reload=True)`

---

## Quick Start

### Lookup a Model Card

```python
from llmcore.model_cards import get_model_card_registry

registry = get_model_card_registry()

# Direct lookup
card = registry.get("openai", "gpt-4o")
if card:
    print(f"Context: {card.get_context_length():,} tokens")
    print(f"Vision: {card.capabilities.vision}")
    print(f"Price: ${card.pricing.per_million_tokens.input}/1M input")

# Lookup by alias
card = registry.get("anthropic", "claude-4.5-sonnet")  # Resolves alias

# Case-insensitive
card = registry.get("ollama", "LLAMA3.2:LATEST")  # Works
```

### List Available Models

```python
# All models for a provider
summaries = registry.list_cards(provider="ollama")
for s in summaries:
    print(f"{s.model_id}: {s.context_length:,} tokens")

# Filter by capabilities
vision_models = registry.list_cards(tags=["vision"])
chat_models = registry.list_cards(model_type="chat")
```

### Estimate Cost

```python
card = registry.get("openai", "gpt-4o")
cost = card.estimate_cost(
    input_tokens=50_000,
    output_tokens=2_000,
    cached_tokens=10_000  # Prompt caching
)
print(f"Estimated cost: ${cost:.4f}")
```

---

## Storage Locations

### Built-in Cards (Package)

```
src/llmcore/model_cards/default_cards/
├── anthropic/
│   ├── claude-sonnet-4-5-20250929.json
│   └── ...
├── deepseek/
├── google/
├── mistral/
├── ollama/
│   ├── llama3.2_latest.json
│   ├── llama3.3_70b.json
│   └── ...
├── openai/
├── qwen/
└── xai/
```

**Shipped with llmcore package. Read-only for end users.**

### User Cards (Override)

```
~/.config/llmcore/model_cards/
├── ollama/
│   └── my-custom-model.json    # Overrides or adds new
├── openai/
│   └── gpt-4o.json             # Overrides built-in gpt-4o
└── ...
```

**User cards take precedence over built-in cards with the same `model_id`.**

### Custom Location (Config)

Set in `llmcore.toml`:

```toml
[model_cards]
user_cards_path = "/path/to/custom/model_cards"
```

Or via environment:

```bash
export LLMCORE_MODEL_CARDS_USER_PATH="/path/to/custom/model_cards"
```

---

## Schema Reference

### Core Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | `string` | ✅ | Unique identifier (e.g., `"gpt-4o"`, `"llama3.2:latest"`) |
| `display_name` | `string` | | Human-readable name for UI |
| `provider` | `Provider` | ✅ | Provider enum value (see below) |
| `model_type` | `ModelType` | ✅ | Model type enum (see below) |
| `architecture` | `ModelArchitecture` | | Architecture details |
| `context` | `ModelContext` | ✅ | Context window configuration |
| `capabilities` | `ModelCapabilities` | | Feature flags (defaults provided) |
| `pricing` | `ModelPricing` | | Token pricing (`null` for local models) |
| `rate_limits` | `Dict[str, RateLimits]` | | Rate limits by tier |
| `lifecycle` | `ModelLifecycle` | | Release/deprecation info |
| `license` | `string` | | License identifier (e.g., `"MIT"`, `"apache-2.0"`) |
| `open_weights` | `boolean` | | Whether weights are publicly available |
| `aliases` | `List[string]` | | Alternative model identifiers |
| `description` | `string` | | Human-readable description |
| `tags` | `List[string]` | | Categorization tags |
| `embedding_config` | `EmbeddingConfig` | | Config for embedding models only |
| `provider_*` | `*Extension` | | Provider-specific extension |
| `source` | `"builtin"\|"user"\|"api"` | | Card origin (auto-set) |
| `last_updated` | `datetime` | | Last modification timestamp |

### Enumerations

#### Provider

```python
class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    KIMI = "kimi"
    XAI = "xai"
    MISTRAL = "mistral"
    COHERE = "cohere"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    REPLICATE = "replicate"
    PERPLEXITY = "perplexity"
    AI21 = "ai21"
    GROQ = "groq"
    LOCAL = "local"
```

#### ModelType

```python
class ModelType(str, Enum):
    CHAT = "chat"              # Conversational LLMs
    COMPLETION = "completion"  # Text completion (legacy)
    EMBEDDING = "embedding"    # Vector embeddings
    RERANK = "rerank"          # Re-ranking models
    IMAGE_GENERATION = "image-generation"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
```

#### ArchitectureType

```python
class ArchitectureType(str, Enum):
    TRANSFORMER = "transformer"  # Standard transformer
    MOE = "moe"                  # Mixture of Experts
    SSM = "ssm"                  # State Space Model (Mamba)
    HYBRID = "hybrid"           # Mixed architecture
```

> ⚠️ **Common Error**: Using invalid values like `"transformer-decoder-multimodal"` will cause silent validation failure. Use only the enum values above.

#### ModelStatus

```python
class ModelStatus(str, Enum):
    ACTIVE = "active"          # Production-ready
    PREVIEW = "preview"        # Early access
    BETA = "beta"              # Beta testing
    DEPRECATED = "deprecated"  # Still works, replacement available
    LEGACY = "legacy"          # Old version, use successor
    RETIRED = "retired"        # No longer available
```

### Nested Structures

#### ModelArchitecture

```json
{
  "architecture": {
    "family": "llama",           // Model family name
    "parameter_count": "70B",    // Total parameters
    "active_parameters": "37B",  // For MoE: active params per forward pass
    "architecture_type": "transformer"  // Must be valid enum
  }
}
```

#### ModelContext

```json
{
  "context": {
    "max_input_tokens": 131072,    // Required: max input context
    "max_output_tokens": 8192,     // Optional: max output length
    "default_output_tokens": 4096  // Optional: default if not specified
  }
}
```

#### ModelCapabilities

All fields default to `false` except `streaming` which defaults to `true`.

```json
{
  "capabilities": {
    "streaming": true,           // Supports streaming responses
    "function_calling": true,    // Supports function/tool calling
    "tool_use": true,            // Supports tool use (modern term)
    "json_mode": true,           // Supports JSON output mode
    "structured_output": true,   // Supports structured output schemas
    "vision": false,             // Supports image input
    "audio_input": false,        // Supports audio input
    "audio_output": false,       // Supports audio output (TTS)
    "video_input": false,        // Supports video input
    "image_generation": false,   // Can generate images
    "code_execution": false,     // Can execute code in sandbox
    "web_search": false,         // Has web search capability
    "reasoning": false,          // Extended thinking / chain-of-thought
    "file_processing": false     // Can process uploaded files
  }
}
```

#### ModelPricing

```json
{
  "pricing": {
    "currency": "USD",
    "per_million_tokens": {
      "input": 2.50,              // Required: input price per 1M
      "output": 10.00,            // Required: output price per 1M
      "cached_input": 1.25,       // Optional: cached/prompt caching price
      "reasoning_output": 15.00   // Optional: for reasoning tokens (o1/o3)
    },
    "batch_discount_percent": 50, // Optional: batch API discount
    "context_tiers": [            // Optional: tiered pricing
      {
        "threshold_tokens": 128000,
        "input_price": 5.00,
        "output_price": 15.00
      }
    ]
  }
}
```

For **local models** (Ollama, etc.), set `"pricing": null`.

#### ModelLifecycle

```json
{
  "lifecycle": {
    "release_date": "2024-12-06",       // ISO date
    "knowledge_cutoff": "2023-12",      // Training data cutoff
    "deprecation_date": null,           // When deprecated/will be
    "shutdown_date": null,              // When retired/will be
    "successor_model": null,            // Replacement model ID
    "status": "active"                  // Must be valid enum
  }
}
```

#### EmbeddingConfig (for embedding models)

```json
{
  "embedding_config": {
    "dimensions_default": 3072,
    "dimensions_configurable": [256, 1024, 3072],
    "supports_matryoshka": true,
    "similarity_metrics": ["cosine", "dot_product", "euclidean"],
    "normalization": "L2",
    "task_types": ["retrieval_query", "retrieval_document"],
    "output_types": ["float", "int8", "binary"],
    "truncation_strategy": "end",
    "prefixes": {
      "query": "query: ",
      "document": "document: "
    },
    "batch_limits": {
      "max_batch_size": 2048,
      "max_tokens_per_batch": 1000000
    },
    "languages_supported": 100,
    "multimodal": false
  }
}
```

### Provider Extensions

Each provider can have a dedicated extension object for provider-specific fields.

#### OllamaExtension (`provider_ollama`)

```json
{
  "provider_ollama": {
    "format": "gguf",                    // "gguf" or "safetensors"
    "quantization_level": "Q4_K_M",      // Quantization level
    "file_size_bytes": 42000000000,      // Model file size
    "digest": "sha256:abc123...",        // Model digest/hash
    "template": "{{ .System }}...",      // Chat template (Go syntax)
    "system_prompt": "You are...",       // Default system prompt
    "modelfile_parameters": {            // Modelfile defaults
      "num_ctx": 8192,
      "temperature": 0.7,
      "top_p": 0.9,
      "stop": ["<|eot_id|>"]
    },
    "gguf_metadata": {                   // GGUF file metadata
      "general.architecture": "llama",
      "llama.context_length": 131072
    },
    "parent_model": "llama3.2",          // Base model
    "modified_at": "2024-12-01T10:00:00Z"
  }
}
```

#### OpenAIExtension (`provider_openai`)

```json
{
  "provider_openai": {
    "owned_by": "openai",
    "supports_reasoning": false,         // o1/o3 series
    "reasoning_effort": "medium",        // "low", "medium", "high"
    "supports_predicted_outputs": true,
    "fine_tuning_available": true,
    "moderation_model": false,
    "tier_requirements": {
      "minimum_tier": 3
    }
  }
}
```

#### AnthropicExtension (`provider_anthropic`)

```json
{
  "provider_anthropic": {
    "extended_thinking": {
      "supported": true,
      "budget_tokens_range": [1024, 128000]
    },
    "computer_use": true,
    "prompt_caching": {
      "cache_write_5m_multiplier": 1.25,
      "cache_write_1h_multiplier": 2.0,
      "cache_read_multiplier": 0.1
    },
    "beta_features": ["context-1m-2025-08-07"]
  }
}
```

#### GoogleExtension (`provider_google`)

```json
{
  "provider_google": {
    "supported_inputs": ["text", "image", "video", "audio", "pdf"],
    "supported_outputs": ["text", "image"],
    "grounding": {
      "google_search": true,
      "maps": false
    },
    "thinking": {
      "supported": true,
      "budget_range": [1024, 32768]
    },
    "live_api": true,
    "url_context": true,
    "safety_settings": [
      {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
    ],
    "versions": {
      "stable": "gemini-2.5-pro-001",
      "preview": "gemini-2.5-pro-preview-0506"
    }
  }
}
```

#### DeepSeekExtension (`provider_deepseek`)

```json
{
  "provider_deepseek": {
    "thinking_mode": {
      "supported": true,
      "requires_think_token": true
    },
    "cache_hit_discount": 0.90,          // 90% off for cache hits
    "fill_in_middle": true,              // Code completion FIM
    "moe_architecture": {
      "total_parameters": "671B",
      "active_parameters": "37B"
    }
  }
}
```

#### MistralExtension (`provider_mistral`)

```json
{
  "provider_mistral": {
    "open_weights": true,
    "license_type": "apache-2.0",
    "fill_in_middle": true,
    "guardrails": {
      "safe_mode": true,
      "custom_guardrails": false
    },
    "fine_tuning": {
      "available": true,
      "methods": ["lora", "full"]
    }
  }
}
```

#### XAIExtension (`provider_xai`)

```json
{
  "provider_xai": {
    "live_search": {
      "enabled": true,
      "cost_per_source": 0.01
    },
    "x_integration": {
      "enabled": true,
      "search_tweets": true
    },
    "server_tools": ["web_search", "X_search", "code_execution"]
  }
}
```

#### QwenExtension (`provider_qwen`)

```json
{
  "provider_qwen": {
    "deployment_regions": ["cn-hangzhou", "us-west-1"],
    "thinking_mode": {
      "supported": true,
      "enable_param": "enable_thinking"
    },
    "context_tiers": [
      {"max_tokens": 32000, "input_price": 0.002},
      {"max_tokens": 128000, "input_price": 0.004}
    ],
    "cache_types": {
      "implicit": 0.5,
      "explicit": 0.25
    },
    "specialized_variant": "coder"  // base, coder, vl, omni, math
  }
}
```

---

## Creating Model Cards

### Method 1: JSON File (Recommended)

Create a JSON file in the appropriate directory.

#### File Naming Convention

| `model_id` | Filename |
|------------|----------|
| `llama3.2:latest` | `llama3.2_latest.json` |
| `gpt-4o` | `gpt-4o.json` |
| `claude-sonnet-4-5-20250929` | `claude-sonnet-4-5-20250929.json` |

**Rule**: Replace `/`, `:`, `\` with `_` in the filename.

#### Minimal Valid Card

```json
{
  "model_id": "my-model:latest",
  "provider": "ollama",
  "model_type": "chat",
  "context": {
    "max_input_tokens": 8192
  }
}
```

#### Complete Card Template

```json
{
  "model_id": "MODEL_ID",
  "display_name": "Human Readable Name",
  "provider": "PROVIDER",
  "model_type": "chat",
  "architecture": {
    "family": "MODEL_FAMILY",
    "parameter_count": "7B",
    "architecture_type": "transformer"
  },
  "context": {
    "max_input_tokens": 32768,
    "max_output_tokens": 4096,
    "default_output_tokens": 2048
  },
  "capabilities": {
    "streaming": true,
    "function_calling": true,
    "tool_use": true,
    "json_mode": true,
    "structured_output": true,
    "vision": false,
    "audio_input": false,
    "audio_output": false,
    "video_input": false,
    "image_generation": false,
    "code_execution": false,
    "web_search": false,
    "reasoning": false,
    "file_processing": false
  },
  "pricing": null,
  "lifecycle": {
    "release_date": "2024-01-01",
    "knowledge_cutoff": "2024-06",
    "status": "active"
  },
  "license": "apache-2.0",
  "open_weights": true,
  "aliases": ["alias1", "alias2"],
  "description": "Brief description of the model and its strengths.",
  "tags": ["open-weights", "local", "coding"],
  "source": "builtin"
}
```

### Method 2: Python API

```python
from datetime import date
from llmcore.model_cards import (
    ModelCard,
    ModelContext,
    ModelCapabilities,
    ModelArchitecture,
    ModelLifecycle,
    OllamaExtension,
    get_model_card_registry,
)

# Create the card
card = ModelCard(
    model_id="my-custom-model:7b",
    display_name="My Custom Model 7B",
    provider="ollama",
    model_type="chat",
    architecture=ModelArchitecture(
        family="llama",
        parameter_count="7B",
        architecture_type="transformer",
    ),
    context=ModelContext(
        max_input_tokens=32768,
        max_output_tokens=4096,
    ),
    capabilities=ModelCapabilities(
        streaming=True,
        function_calling=True,
        tool_use=True,
        json_mode=True,
        vision=False,
    ),
    lifecycle=ModelLifecycle(
        release_date=date(2024, 6, 1),
        knowledge_cutoff="2024-03",
        status="active",
    ),
    license="apache-2.0",
    open_weights=True,
    aliases=["my-model", "my-model:latest"],
    description="A custom fine-tuned model for specialized tasks.",
    tags=["custom", "fine-tuned", "local"],
    provider_ollama=OllamaExtension(
        format="gguf",
        quantization_level="Q4_K_M",
    ),
)

# Save to user directory
registry = get_model_card_registry()
saved_path = registry.save_card(card, user_override=True)
print(f"Saved to: {saved_path}")
# Output: ~/.config/llmcore/model_cards/ollama/my-custom-model_7b.json
```

---

## Complete Examples

### Example 1: Local Ollama Chat Model

```json
{
  "model_id": "gemma3:4b",
  "display_name": "Gemma 3 4B",
  "provider": "ollama",
  "model_type": "chat",
  "architecture": {
    "family": "gemma",
    "parameter_count": "4B",
    "architecture_type": "transformer"
  },
  "context": {
    "max_input_tokens": 131072,
    "max_output_tokens": 8192,
    "default_output_tokens": 4096
  },
  "capabilities": {
    "streaming": true,
    "function_calling": true,
    "tool_use": true,
    "json_mode": true,
    "structured_output": true,
    "vision": true,
    "reasoning": true
  },
  "pricing": null,
  "lifecycle": {
    "release_date": "2025-03-12",
    "knowledge_cutoff": "2024-08",
    "status": "active"
  },
  "license": "gemma",
  "open_weights": true,
  "aliases": ["gemma3", "gemma3:latest"],
  "description": "Gemma 3 4B is Google DeepMind's multimodal model with 128K context, native vision via SigLIP encoder, and 140+ language support.",
  "tags": ["open-source", "local", "multimodal", "vision", "google-deepmind"],
  "provider_ollama": {
    "format": "gguf",
    "quantization_level": "Q4_K_M",
    "file_size_bytes": 4000000000,
    "modelfile_parameters": {
      "num_ctx": 8192,
      "temperature": 0.7
    }
  },
  "source": "builtin"
}
```

### Example 2: API Model with Pricing

```json
{
  "model_id": "gpt-4o",
  "display_name": "GPT-4o",
  "provider": "openai",
  "model_type": "chat",
  "architecture": {
    "family": "gpt",
    "architecture_type": "transformer"
  },
  "context": {
    "max_input_tokens": 128000,
    "max_output_tokens": 16384,
    "default_output_tokens": 4096
  },
  "capabilities": {
    "streaming": true,
    "function_calling": true,
    "tool_use": true,
    "json_mode": true,
    "structured_output": true,
    "vision": true,
    "audio_input": true,
    "audio_output": true,
    "reasoning": false
  },
  "pricing": {
    "currency": "USD",
    "per_million_tokens": {
      "input": 2.50,
      "output": 10.00,
      "cached_input": 1.25
    },
    "batch_discount_percent": 50
  },
  "lifecycle": {
    "release_date": "2024-05-13",
    "knowledge_cutoff": "2023-10",
    "status": "active"
  },
  "license": null,
  "open_weights": false,
  "aliases": ["gpt-4o-2024-05-13", "gpt-4o-2024-08-06"],
  "description": "GPT-4o is OpenAI's flagship multimodal model with vision, audio I/O, and 128K context.",
  "tags": ["flagship", "multimodal", "vision", "audio"],
  "provider_openai": {
    "owned_by": "openai",
    "supports_predicted_outputs": true,
    "fine_tuning_available": true
  },
  "source": "builtin"
}
```

### Example 3: Embedding Model

```json
{
  "model_id": "text-embedding-3-large",
  "display_name": "Text Embedding 3 Large",
  "provider": "openai",
  "model_type": "embedding",
  "architecture": {
    "family": "text-embedding",
    "architecture_type": "transformer"
  },
  "context": {
    "max_input_tokens": 8191,
    "max_output_tokens": null
  },
  "capabilities": {
    "streaming": false,
    "function_calling": false,
    "tool_use": false,
    "json_mode": false,
    "structured_output": false,
    "vision": false
  },
  "pricing": {
    "currency": "USD",
    "per_million_tokens": {
      "input": 0.13,
      "output": 0.0
    }
  },
  "lifecycle": {
    "release_date": "2024-01-25",
    "status": "active"
  },
  "license": null,
  "open_weights": false,
  "aliases": ["text-embedding-3-large-2024-01-25"],
  "description": "OpenAI's most capable embedding model with configurable dimensions up to 3072.",
  "tags": ["embedding", "semantic-search", "rag"],
  "embedding_config": {
    "dimensions_default": 3072,
    "dimensions_configurable": [256, 1024, 3072],
    "supports_matryoshka": true,
    "similarity_metrics": ["cosine", "dot_product", "euclidean"],
    "normalization": "L2"
  },
  "provider_openai": {
    "owned_by": "openai"
  },
  "source": "builtin"
}
```

### Example 4: Mixture of Experts Model

```json
{
  "model_id": "deepseek-r1",
  "display_name": "DeepSeek R1",
  "provider": "deepseek",
  "model_type": "chat",
  "architecture": {
    "family": "deepseek",
    "parameter_count": "671B",
    "active_parameters": "37B",
    "architecture_type": "moe"
  },
  "context": {
    "max_input_tokens": 64000,
    "max_output_tokens": 64000
  },
  "capabilities": {
    "streaming": true,
    "function_calling": false,
    "tool_use": false,
    "json_mode": true,
    "structured_output": false,
    "vision": false,
    "reasoning": true
  },
  "pricing": {
    "currency": "USD",
    "per_million_tokens": {
      "input": 0.70,
      "output": 2.50,
      "cached_input": 0.07
    }
  },
  "lifecycle": {
    "release_date": "2025-01-20",
    "knowledge_cutoff": "2024-11",
    "status": "active"
  },
  "license": "MIT",
  "open_weights": true,
  "aliases": ["deepseek-reasoner"],
  "description": "DeepSeek R1 is a reasoning-optimized 671B MoE model (37B active) rivaling o1 on reasoning benchmarks.",
  "tags": ["reasoning", "chain-of-thought", "open-source", "moe"],
  "provider_deepseek": {
    "thinking_mode": {
      "supported": true,
      "requires_think_token": true
    },
    "fill_in_middle": false
  },
  "source": "builtin"
}
```

### Example 5: Reasoning Model with Extended Thinking

```json
{
  "model_id": "claude-sonnet-4-5-20250929",
  "display_name": "Claude Sonnet 4.5",
  "provider": "anthropic",
  "model_type": "chat",
  "architecture": {
    "family": "claude",
    "architecture_type": "transformer"
  },
  "context": {
    "max_input_tokens": 200000,
    "max_output_tokens": 64000,
    "default_output_tokens": 8192
  },
  "capabilities": {
    "streaming": true,
    "function_calling": true,
    "tool_use": true,
    "json_mode": true,
    "structured_output": true,
    "vision": true,
    "reasoning": true,
    "code_execution": true,
    "web_search": true,
    "file_processing": true
  },
  "pricing": {
    "currency": "USD",
    "per_million_tokens": {
      "input": 3.00,
      "output": 15.00,
      "cached_input": 0.30
    },
    "batch_discount_percent": 50
  },
  "lifecycle": {
    "release_date": "2025-09-29",
    "knowledge_cutoff": "2025-04",
    "status": "active"
  },
  "license": null,
  "open_weights": false,
  "aliases": ["claude-4.5-sonnet", "claude-sonnet-4.5"],
  "description": "Claude Sonnet 4.5 with 200K context, extended thinking, computer use, and advanced coding capabilities.",
  "tags": ["balanced", "extended-context", "coding", "vision", "reasoning"],
  "provider_anthropic": {
    "extended_thinking": {
      "supported": true,
      "budget_tokens_range": [1024, 128000]
    },
    "computer_use": true,
    "prompt_caching": {
      "cache_write_5m_multiplier": 1.25,
      "cache_write_1h_multiplier": 2.0,
      "cache_read_multiplier": 0.1
    },
    "beta_features": ["context-1m-2025-08-07"]
  },
  "source": "builtin"
}
```

---

## Validation & Troubleshooting

### Validate Before Deployment

```python
import json
from llmcore.model_cards import ModelCard

def validate_card(filepath: str) -> bool:
    """Validate a model card JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    
    try:
        card = ModelCard.model_validate(data)
        print(f"✓ Valid: {card.model_id}")
        print(f"  Provider: {card.provider}")
        print(f"  Context: {card.get_context_length():,} tokens")
        print(f"  Capabilities: vision={card.capabilities.vision}, "
              f"tools={card.capabilities.tool_use}")
        return True
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

# Usage
validate_card("my_model.json")
```

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Card not appearing | Invalid enum value | Check `architecture_type`, `status`, `provider` against valid enums |
| Card not appearing | JSON syntax error | Validate JSON with `python -m json.tool file.json` |
| Card not appearing | Wrong directory | Ensure file is in `~/.config/llmcore/model_cards/<provider>/` |
| Card not appearing | Wrong filename | Filename must end in `.json` (not `.json5`, `.jsonc`) |
| Alias not working | Missing from `aliases` | Add alias to the `aliases` array |
| Pricing returns `null` | Missing `pricing` object | Add `pricing` with `per_million_tokens` |

### Debug Loading

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from llmcore.model_cards import get_model_card_registry

registry = get_model_card_registry()
registry.load(force_reload=True)  # Watch debug output for errors
```

Or via environment:

```bash
LLMCORE_LOG_LEVEL=DEBUG python -c "
from llmcore.model_cards import get_model_card_registry
r = get_model_card_registry()
r.load(force_reload=True)
"
```

### Check Registry Stats

```python
from llmcore.model_cards import get_model_card_registry

registry = get_model_card_registry()
stats = registry.stats()

print(f"Total cards: {stats['total_cards']}")
print(f"Providers: {stats['providers']}")
print(f"Aliases: {stats['aliases']}")
print(f"By provider: {stats['cards_by_provider']}")
print(f"Builtin path: {stats['builtin_path']}")
print(f"User path: {stats['user_path']}")
```

---

## Registry API

### Core Methods

```python
from llmcore.model_cards import get_model_card_registry, ModelCard

registry = get_model_card_registry()

# Load/reload cards
registry.load()                          # Lazy load (skips if already loaded)
registry.load(force_reload=True)         # Force reload all cards
registry.load(user_path=Path("/custom")) # Custom user path

# Lookup
card = registry.get("openai", "gpt-4o")              # Direct lookup
card = registry.get("openai", "GPT-4O")              # Case-insensitive
card = registry.get("openai", "gpt-4o", case_sensitive=True)  # Strict
card = registry.get_by_alias("claude-4.5-sonnet")    # Alias (any provider)

# List and filter
cards = registry.list_cards()                        # All cards
cards = registry.list_cards(provider="ollama")       # By provider
cards = registry.list_cards(model_type="embedding")  # By type
cards = registry.list_cards(tags=["vision"])         # By tags
cards = registry.list_cards(include_deprecated=False)# Exclude deprecated

# Convenience methods
length = registry.get_context_length("openai", "gpt-4o")
pricing = registry.get_pricing("openai", "gpt-4o")
providers = registry.get_providers()
models = registry.get_models_for_provider("ollama")

# Save/remove
path = registry.save_card(card, user_override=True)  # Save to user dir
registry.remove_card("ollama", "my-model")           # Remove from memory
```

### Module-Level Functions

```python
from llmcore.model_cards import (
    get_model_card_registry,  # Get singleton
    get_model_card,           # Cached lookup (LRU)
    clear_model_card_cache,   # Clear LRU cache
)

# Cached lookup (faster for repeated calls)
card = get_model_card("openai", "gpt-4o")

# Clear cache after modifying cards
clear_model_card_cache()
```

---

## Best Practices

### 1. Use Accurate Capability Flags

Only set capability flags to `true` if the model genuinely supports them. Inaccurate flags cause runtime failures.

```json
// ✓ Correct: Model actually supports these
"capabilities": {
  "vision": true,     // Verified: accepts image input
  "tool_use": true    // Verified: function calling works
}

// ✗ Wrong: Optimistic/aspirational flags
"capabilities": {
  "vision": true,     // Model can't actually process images
  "reasoning": true   // No extended thinking support
}
```

### 2. Provide Meaningful Aliases

Aliases help users find models with common variations:

```json
"aliases": [
  "claude-4.5-sonnet",       // Common short form
  "claude-sonnet-4.5",       // Alternative ordering
  "claude-sonnet-4-5-latest" // Version-agnostic
]
```

### 3. Document Quantization for Local Models

For Ollama models, specify quantization details:

```json
"provider_ollama": {
  "quantization_level": "Q4_K_M",
  "file_size_bytes": 4200000000,
  "modelfile_parameters": {
    "num_ctx": 8192  // Practical context (vs theoretical max)
  }
}
```

### 4. Keep Cards Updated

Update cards when:
- Model pricing changes
- New capabilities are added
- Model is deprecated
- Context limits change

### 5. Use Tags Consistently

Standard tags for discoverability:

| Tag | Meaning |
|-----|---------|
| `flagship` | Provider's top model |
| `vision` | Image input support |
| `audio` | Audio I/O support |
| `reasoning` | Extended thinking |
| `open-weights` | Weights publicly available |
| `local` | Runs locally (Ollama, etc.) |
| `embedding` | Embedding model |
| `coding` | Optimized for code |
| `multimodal` | Multiple modalities |
| `moe` | Mixture of Experts |

### 6. Test Cards Before Committing

Always validate new cards:

```bash
python -c "
import json
from llmcore.model_cards import ModelCard

with open('my_card.json') as f:
    card = ModelCard.model_validate(json.load(f))
    print(f'Valid: {card.model_id}')
"
```

---

## Migration & Compatibility

### Adding New Provider

1. Add to `Provider` enum in `schema.py` (if not present)
2. Create extension model (optional): `NewProviderExtension`
3. Add `provider_newprovider` field to `ModelCard`
4. Create directory: `default_cards/newprovider/`
5. Add model card JSON files

### Schema Version Changes

The schema uses Pydantic's built-in validation. Future changes should:

1. Add new optional fields with defaults
2. Avoid removing/renaming existing fields
3. Use `model_config = {"extra": "ignore"}` to skip unknown fields
4. Document breaking changes in CHANGELOG

### Backward Compatibility

Cards created for older versions remain compatible:
- Missing optional fields use defaults
- Unknown fields are ignored (not errors)
- Enum values are case-insensitive

---

## Appendix: Valid Enum Values Reference

### Provider Values
`openai`, `anthropic`, `google`, `ollama`, `deepseek`, `qwen`, `kimi`, `xai`, `mistral`, `cohere`, `together`, `fireworks`, `replicate`, `perplexity`, `ai21`, `groq`, `local`

### ModelType Values
`chat`, `completion`, `embedding`, `rerank`, `image-generation`, `audio`, `multimodal`

### ArchitectureType Values
`transformer`, `moe`, `ssm`, `hybrid`

### ModelStatus Values
`active`, `preview`, `beta`, `deprecated`, `legacy`, `retired`

---

*Document generated for llmcore v0.26.0*
