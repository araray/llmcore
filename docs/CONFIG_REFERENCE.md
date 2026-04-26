# LLMCore Configuration Reference

Self-hosted LLM provider abstraction layer with agentic capabilities, RAG, embeddings, observability, and sandbox execution

**Schema version**: 1.0.0  
**App version**: 0.49.6  
**Output format**: toml

---

## ⚙️ Core Settings

Fundamental llmcore configuration — provider selection, embedding model, and diagnostics.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `llmcore.default_provider` | enum |  | `ollama` | Specifies which provider instance (from [providers] below) to use by default whe |
| `llmcore.default_embedding_model` | enum |  | `all-MiniLM-L6-v2` | Specifies which embedding model to use for RAG operations. Required for generati |
| `llmcore.log_level` | enum |  | `INFO` | Configures the logging verbosity for the entire `llmcore` library. Control the a |
| `llmcore.log_raw_payloads` | boolean |  | `False` | If true, `llmcore` will log the exact raw JSON request and response payloads exc |
| `llmcore.admin_api_key` | secret |  | — | Defines the secret key required for accessing administrative endpoints such as l |

### `llmcore.default_provider`

Specifies which provider instance (from [providers] below) to use by default when no provider is explicitly chosen. Allows you to switch between different LLM backends quickly. "openai", "anthropic", "gemini", "openrouter").

**Options:**

- `ollama`: Ollama (local) — Run models locally via Ollama
- `openai`: OpenAI — GPT-4o, GPT-5, o3, o4-mini
- `anthropic`: Anthropic — Claude 4.x family
- `gemini`: Google Gemini — Gemini 3.x / 2.5 family
- `openrouter`: OpenRouter — Unified gateway to 300+ models
- `mistral`: Mistral AI — Mistral Large, Codestral, Magistral
- `vllm`: vLLM (self-hosted) — Self-hosted vLLM inference server
- `poe`: Poe — Gateway to models and community bots

### `llmcore.default_embedding_model`

Specifies which embedding model to use for RAG operations. Required for generating vector embeddings of text chunks in RAG. - For local models (Sentence Transformers): A model name from the Hugging Face model hub (e.g., "all-MiniLM-L6-v2", "all-mpnet-base-v2"). The model will be downloaded automatically. - For service-based models: A string in the format "provider:model_name" (e.g., "openai:text-embedding-3-small", "google:gemini-embedding-001").

**Options:**

- `all-MiniLM-L6-v2`: all-MiniLM-L6-v2 (local, fast) — Sentence Transformers — 384 dims
- `all-mpnet-base-v2`: all-mpnet-base-v2 (local, quality) — Sentence Transformers — 768 dims
- `openai:text-embedding-3-small`: OpenAI text-embedding-3-small — API — 1536 dims
- `openai:text-embedding-3-large`: OpenAI text-embedding-3-large — API — 3072 dims
- `google:gemini-embedding-001`: Gemini Embedding 001 — API — 3072 dims, 100+ languages
- `ollama:mxbai-embed-large`: Ollama mxbai-embed-large (local)
- `ollama:nomic-embed-text`: Ollama nomic-embed-text (local)

### `llmcore.log_level`

Configures the logging verbosity for the entire `llmcore` library. Control the amount of diagnostic information emitted by the library.

**Options:**

- `DEBUG`: Debug
- `INFO`: Info
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical

### `llmcore.log_raw_payloads`

If true, `llmcore` will log the exact raw JSON request and response payloads exchanged with the underlying LLM provider APIs. This logging occurs at the `DEBUG` log level. An indispensable tool for advanced troubleshooting. It allows you to inspect the precise data being sent and received, which is essential for diagnosing provider errors or understanding unexpected behavior.

### `llmcore.admin_api_key`

Defines the secret key required for accessing administrative endpoints such as live configuration reloading. This provides a separate, high-privilege authentication layer independent of tenant API keys. Secures administrative operations that could impact the entire platform, ensuring only authorized administrators can perform sensitive actions like configuration reloads. This key should be treated as a high-privilege secret. In production, ALWAYS set this via the environment variable rather than in config files.

## 📋 Unified Logging

Logging for all components: llmcore, llmchat, semantiscan, and confy. Single source of truth.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `logging.console_enabled` | boolean |  | `False` | Console logging - set to false to suppress most console output. When false, only |
| `logging.console_level` | enum |  | `WARNING` |  *(when `logging.console_enabled == true`)* |
| `logging.console_format` | string |  | `%(levelname)s - %(message)s` |  *(when `logging.console_enabled == true`)* |
| `logging.file_enabled` | boolean |  | `True` | File logging - always recommended to keep enabled |
| `logging.file_level` | enum |  | `DEBUG` |  *(when `logging.file_enabled == true`)* |
| `logging.file_directory` | path |  | `~/.local/share/llmcore/logs` |  *(when `logging.file_enabled == true`)* |
| `logging.file_mode` | enum |  | `per_run` | File mode: "per_run" (new file each invocation) or "single" (persistent + rotati *(when `logging.file_enabled == true`)* |
| `logging.file_name_pattern` | string |  | `{app}_{timestamp:%Y%m%d_%H%M%S}.log` |  *(when `logging.file_enabled == true`)* |
| `logging.file_single_name` | string |  | `{app}.log` | Single-file mode settings (ignored when file_mode = "per_run"): *(when `logging.file_enabled == true && logging.file_mode == 'single'`)* |
| `logging.file_format` | string |  | `%(asctime)s [%(levelname)-8s] %(name)-30s - %(message)s (%(filename)s:%(lineno)d)` |  *(when `logging.file_enabled == true`)* |
| `logging.rotation_max_bytes` | integer |  | `10485760` |  *(when `logging.file_enabled == true && logging.file_mode == 'single'`)* |
| `logging.rotation_backup_count` | integer |  | `5` |  *(when `logging.file_enabled == true && logging.file_mode == 'single'`)* |
| `logging.display_min_level` | enum |  | `INFO` | Minimum level for display=True records on console (even when console_enabled=fal |
| `logging.components.llmchat` | enum |  | `INFO` |  |
| `logging.components.llmcore` | enum |  | `INFO` |  |
| `logging.components.semantiscan` | enum |  | `INFO` |  |
| `logging.components.confy` | enum |  | `WARNING` |  |
| `logging.components.urllib3` | enum |  | `WARNING` | Suppress noisy third-party libraries |
| `logging.components.httpx` | enum |  | `WARNING` |  |
| `logging.components.httpcore` | enum |  | `WARNING` |  |
| `logging.components.asyncio` | enum |  | `WARNING` |  |
| `logging.components.aiosqlite` | enum |  | `WARNING` |  |
| `logging.components.chromadb` | enum |  | `WARNING` |  |
| `logging.components.posthog` | enum |  | `ERROR` |  |

### `logging.console_enabled`

Console logging - set to false to suppress most console output. When false, only messages logged with display=True (via log_display()) will appear on the console.  Set to true for verbose/debug sessions.

### `logging.console_level`

**Options:**

- `DEBUG`: Debug
- `INFO`: Info
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical

### `logging.file_level`

**Options:**

- `DEBUG`: Debug
- `INFO`: Info
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical

### `logging.file_mode`

File mode: "per_run" (new file each invocation) or "single" (persistent + rotation)

**Options:**

- `per_run`: Per-run (new file each invocation)
- `single`: Single (persistent + rotation)

### `logging.display_min_level`

Minimum level for display=True records on console (even when console_enabled=false)

**Options:**

- `DEBUG`: Debug
- `INFO`: Info
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical

### `logging.components.llmchat`

**Options:**

- `DEBUG`: Debug
- `INFO`: Info
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical

### `logging.components.llmcore`

**Options:**

- `DEBUG`: Debug
- `INFO`: Info
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical

### `logging.components.semantiscan`

**Options:**

- `DEBUG`: Debug
- `INFO`: Info
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical

### `logging.components.confy`

**Options:**

- `DEBUG`: Debug
- `INFO`: Info
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical

### `logging.components.urllib3`

Suppress noisy third-party libraries

**Options:**

- `DEBUG`: Debug
- `INFO`: Info
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical

### `logging.components.httpx`

**Options:**

- `DEBUG`: Debug
- `INFO`: Info
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical

### `logging.components.httpcore`

**Options:**

- `DEBUG`: Debug
- `INFO`: Info
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical

### `logging.components.asyncio`

**Options:**

- `DEBUG`: Debug
- `INFO`: Info
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical

### `logging.components.aiosqlite`

**Options:**

- `DEBUG`: Debug
- `INFO`: Info
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical

### `logging.components.chromadb`

**Options:**

- `DEBUG`: Debug
- `INFO`: Info
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical

### `logging.components.posthog`

**Options:**

- `DEBUG`: Debug
- `INFO`: Info
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical

## 🤖 Provider: OpenAI

GPT-4o, GPT-5, o3, o4-mini, and any OpenAI-compatible API.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `providers.openai.base_url` | url |  | — | Override for proxies or OpenAI-compatible APIs (Groq, Together, etc.). Leave emp |
| `providers.openai.api_key` | secret |  | — | OpenAI API key. Strongly recommended to set via OPENAI_API_KEY environment varia |
| `providers.openai.default_model` | enum |  | `gpt-4o` | Default model for this provider instance. |
| `providers.openai.timeout` | integer |  | `120` | Request timeout in seconds. |

### `providers.openai.base_url`

Override for proxies or OpenAI-compatible APIs (Groq, Together, etc.). Leave empty for official OpenAI.

### `providers.openai.api_key`

OpenAI API key. Strongly recommended to set via OPENAI_API_KEY environment variable.

### `providers.openai.default_model`

Default model for this provider instance.

**Options:**

- `gpt-4o`: GPT-4o
- `gpt-4o-mini`: GPT-4o Mini
- `gpt-5`: GPT-5
- `o3`: o3
- `o4-mini`: o4-mini
- `o1`: o1
- `o1-pro`: o1-pro

**Dynamic discovery** (`openai_chat_cards`):

  Load available openai chat models from llmcore model cards
  - Command: `python3 << 'PYEOF'
import json, os, sys
P = 'openai'
dirs = [os.path.join('src', 'llmcore')]
dirs += [os.path.join(p, 'llmcore') for p in sys.path if os.path.isdir(os.path.join(p, 'llmcore', 'model_cards'))]
for b in [os.path.join(d, 'model_cards', 'default_cards', P) for d in dirs]:
    if not os.path.isdir(b): continue
    r = []
    for f in sorted(os.listdir(b)):
        if not f.endswith('.json'): continue
        try:
            d = json.load(open(os.path.join(b, f)))
            if d.get('model_type') != 'chat': continue
            if d.get('lifecycle', {}).get('status') in ('retired', 'deprecated'): continue
            ctx = d.get('context', {}).get('max_input_tokens', '?')
            pi = d.get('pricing', {}).get('per_million_tokens', {}).get('input', '?')
            r.append({'value': d['model_id'], 'label': d.get('display_name', d['model_id']), 'description': f'{ctx} ctx, ${pi}/M in'})
        except Exception: pass
    print(json.dumps(r))
    sys.exit()
print('[]')
PYEOF`
  - Populates: options
  - Merge: replace

## 🧠 Provider: Anthropic

Claude 4.x family.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `providers.anthropic.max_tokens` | integer |  | `4096` | Default max_tokens for Anthropic responses (required parameter). Override per-re |
| `providers.anthropic.api_key` | secret |  | — | Anthropic API key. Strongly recommended to set via ANTHROPIC_API_KEY environment |
| `providers.anthropic.default_model` | enum |  | `claude-sonnet-4-6` | Default model for this provider instance. Use the dash format (canonical API mod |
| `providers.anthropic.timeout` | integer |  | `120` | Request timeout in seconds. Extended thinking responses can take 30-90s; 120s ac |

### `providers.anthropic.max_tokens`

Default max_tokens for Anthropic responses (required parameter). Override per-request.

### `providers.anthropic.api_key`

Anthropic API key. Strongly recommended to set via ANTHROPIC_API_KEY environment variable.

### `providers.anthropic.default_model`

Default model for this provider instance. Use the dash format (canonical API model IDs), not dot format. Current models (April 2026): claude-opus-4-6, claude-sonnet-4-6          (Claude 4.6 family) claude-opus-4-5, claude-sonnet-4-5           (Claude 4.5 family) claude-haiku-4-5, claude-haiku-4-5-20251001  (Claude 4.5 Haiku) claude-opus-4-1, claude-opus-4-0, claude-sonnet-4-0 Dated variants (e.g., claude-sonnet-4-5-20250929) are also accepted.

**Options:**

- `claude-sonnet-4-6`: Claude Sonnet 4.6
- `claude-opus-4-6`: Claude Opus 4.6
- `claude-sonnet-4-5`: Claude Sonnet 4.5
- `claude-opus-4-5`: Claude Opus 4.5
- `claude-haiku-4-5`: Claude Haiku 4.5
- `claude-sonnet-4-0`: Claude Sonnet 4.0
- `claude-opus-4-1`: Claude Opus 4.1
- `claude-opus-4-0`: Claude Opus 4.0

**Dynamic discovery** (`anthropic_chat_cards`):

  Load available anthropic chat models from llmcore model cards
  - Command: `python3 << 'PYEOF'
import json, os, sys
P = 'anthropic'
dirs = [os.path.join('src', 'llmcore')]
dirs += [os.path.join(p, 'llmcore') for p in sys.path if os.path.isdir(os.path.join(p, 'llmcore', 'model_cards'))]
for b in [os.path.join(d, 'model_cards', 'default_cards', P) for d in dirs]:
    if not os.path.isdir(b): continue
    r = []
    for f in sorted(os.listdir(b)):
        if not f.endswith('.json'): continue
        try:
            d = json.load(open(os.path.join(b, f)))
            if d.get('model_type') != 'chat': continue
            if d.get('lifecycle', {}).get('status') in ('retired', 'deprecated'): continue
            ctx = d.get('context', {}).get('max_input_tokens', '?')
            pi = d.get('pricing', {}).get('per_million_tokens', {}).get('input', '?')
            r.append({'value': d['model_id'], 'label': d.get('display_name', d['model_id']), 'description': f'{ctx} ctx, ${pi}/M in'})
        except Exception: pass
    print(json.dumps(r))
    sys.exit()
print('[]')
PYEOF`
  - Populates: options
  - Merge: replace

### `providers.anthropic.timeout`

Request timeout in seconds. Extended thinking responses can take 30-90s; 120s accommodates this.

## 🦙 Provider: Ollama

Locally running models via Ollama.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `providers.ollama.host` | url |  | `http://localhost:11434` | URL of the running Ollama server. |
| `providers.ollama.default_model` | string |  | `llama3` | Default local model to use. Ensure you have run `ollama pull <model_name>`. |
| `providers.ollama.timeout` | integer |  | `240` | Longer timeout is recommended for local models. |

### `providers.ollama.default_model`

Default local model to use. Ensure you have run `ollama pull <model_name>`.

**Dynamic discovery** (`ollama_live_models`):

  Fetch locally pulled Ollama models from running server
  - Command: `curl -sf http://localhost:11434/api/tags 2>/dev/null`
  - Populates: options
  - Merge: replace
  - Condition: `variables.has_ollama`

## 💎 Provider: Google Gemini

Gemini models via Google AI or Vertex AI.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `providers.gemini.location` | string |  | `us-central1` | GCP region for Vertex AI. *(when `providers.gemini.vertex_ai == true`)* |
| `providers.gemini.project` | string |  | — | GCP project ID for Vertex AI. *(when `providers.gemini.vertex_ai == true`)* |
| `providers.gemini.vertex_ai` | boolean |  | `False` | Use Vertex AI backend instead of the Gemini API. Requires GCP authentication. |
| `providers.gemini.api_key` | secret |  | — | Google AI API key. Set via GOOGLE_API_KEY environment variable. |
| `providers.gemini.default_model` | enum |  | `gemini-3.1-flash-lite-preview` | Default model for this provider instance. Current models (April 2026): 3.x (prev |
| `providers.gemini.timeout` | integer |  | `120` |  |
| `providers.gemini.fallback_context_length` | integer |  | `1048576` | Fallback context length when model is not in the card registry or API. Most curr |

### `providers.gemini.default_model`

Default model for this provider instance. Current models (April 2026): 3.x (preview):  gemini-3.1-pro-preview, gemini-3-flash-preview, gemini-3.1-flash-lite-preview 2.5 (stable):   gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite 2.0 (deprecated, shutdown June 2026): gemini-2.0-flash, gemini-2.0-flash-lite

**Options:**

- `gemini-3.1-pro-preview`: Gemini 3.1 Pro (preview)
- `gemini-3-flash-preview`: Gemini 3 Flash (preview)
- `gemini-3.1-flash-lite-preview`: Gemini 3.1 Flash Lite (preview)
- `gemini-2.5-pro`: Gemini 2.5 Pro (stable)
- `gemini-2.5-flash`: Gemini 2.5 Flash (stable)
- `gemini-2.5-flash-lite`: Gemini 2.5 Flash Lite (stable)
- `gemini-2.0-flash`: Gemini 2.0 Flash (deprecated)

**Dynamic discovery** (`google_chat_cards`):

  Load available google chat models from llmcore model cards
  - Command: `python3 << 'PYEOF'
import json, os, sys
P = 'google'
dirs = [os.path.join('src', 'llmcore')]
dirs += [os.path.join(p, 'llmcore') for p in sys.path if os.path.isdir(os.path.join(p, 'llmcore', 'model_cards'))]
for b in [os.path.join(d, 'model_cards', 'default_cards', P) for d in dirs]:
    if not os.path.isdir(b): continue
    r = []
    for f in sorted(os.listdir(b)):
        if not f.endswith('.json'): continue
        try:
            d = json.load(open(os.path.join(b, f)))
            if d.get('model_type') != 'chat': continue
            if d.get('lifecycle', {}).get('status') in ('retired', 'deprecated'): continue
            ctx = d.get('context', {}).get('max_input_tokens', '?')
            pi = d.get('pricing', {}).get('per_million_tokens', {}).get('input', '?')
            r.append({'value': d['model_id'], 'label': d.get('display_name', d['model_id']), 'description': f'{ctx} ctx, ${pi}/M in'})
        except Exception: pass
    print(json.dumps(r))
    sys.exit()
print('[]')
PYEOF`
  - Populates: options
  - Merge: replace

### `providers.gemini.fallback_context_length`

Fallback context length when model is not in the card registry or API. Most current Gemini models support 1M tokens.

## 🔀 Provider: OpenRouter

Unified gateway to 300+ AI models.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `providers.openrouter.api_key_env_var` | string |  | `OPENROUTER_API_KEY` | type = "openrouter"  # auto-detected from section name API Key: Set via environm |
| `providers.openrouter.default_model` | string |  | `openai/gpt-4o-mini` | Default model.  OpenRouter uses "provider/model" format. "google/gemini-3-flash- |
| `providers.openrouter.timeout` | integer |  | `240` |  |
| `providers.openrouter.backend` | enum |  | `openai` | "openai"  (default): Uses the `openai` Python SDK with base_url override. No ext |

### `providers.openrouter.api_key_env_var`

type = "openrouter"  # auto-detected from section name API Key: Set via environment variable for security. api_key = "sk-or-..."

### `providers.openrouter.default_model`

Default model.  OpenRouter uses "provider/model" format. "google/gemini-3-flash-preview", "deepseek/deepseek-chat"

**Dynamic discovery** (`openrouter_chat_cards`):

  Load available openrouter chat models from llmcore model cards
  - Command: `python3 << 'PYEOF'
import json, os, sys
P = 'openrouter'
dirs = [os.path.join('src', 'llmcore')]
dirs += [os.path.join(p, 'llmcore') for p in sys.path if os.path.isdir(os.path.join(p, 'llmcore', 'model_cards'))]
for b in [os.path.join(d, 'model_cards', 'default_cards', P) for d in dirs]:
    if not os.path.isdir(b): continue
    r = []
    for f in sorted(os.listdir(b)):
        if not f.endswith('.json'): continue
        try:
            d = json.load(open(os.path.join(b, f)))
            if d.get('model_type') != 'chat': continue
            if d.get('lifecycle', {}).get('status') in ('retired', 'deprecated'): continue
            ctx = d.get('context', {}).get('max_input_tokens', '?')
            pi = d.get('pricing', {}).get('per_million_tokens', {}).get('input', '?')
            r.append({'value': d['model_id'], 'label': d.get('display_name', d['model_id']), 'description': f'{ctx} ctx, ${pi}/M in'})
        except Exception: pass
    print(json.dumps(r))
    sys.exit()
print('[]')
PYEOF`
  - Populates: options
  - Merge: replace

### `providers.openrouter.backend`

"openai"  (default): Uses the `openai` Python SDK with base_url override. No extra dependencies beyond `openai`. "sdk"   : Uses the native `openrouter` Python SDK (pip install openrouter). Falls back to "openai" mode if the package is not installed.

**Options:**

- `openai`: OpenAI SDK (default)
- `sdk`: Native OpenRouter SDK

## 🅿️ Provider: Poe

Gateway to models and community bots.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `providers.poe.api_key_env_var` | string |  | `POE_API_KEY` | type = "poe"  # auto-detected from section name API Key: Set via environment var |
| `providers.poe.default_model` | string |  | `GPT-4o-Mini` | Default bot to use.  Bot names are case-sensitive as registered on Poe. "Gemini- |
| `providers.poe.timeout` | integer |  | `120` | Request timeout in seconds.  Video/image generation bots may need higher values  |
| `providers.poe.backend` | enum |  | `openai` | "openai" (default): Uses the `openai` Python SDK with base_url override. All sta |

### `providers.poe.api_key_env_var`

type = "poe"  # auto-detected from section name API Key: Set via environment variable for security. api_key = "..."

### `providers.poe.default_model`

Default bot to use.  Bot names are case-sensitive as registered on Poe. "Gemini-3.1-Pro", "Grok-4", "DeepSeek-V3.2-Exp", "Sora-2", "Veo-3.1" (video), "GPT-Image-1.5" (images)

**Dynamic discovery** (`poe_chat_cards`):

  Load available poe chat models from llmcore model cards
  - Command: `python3 << 'PYEOF'
import json, os, sys
P = 'poe'
dirs = [os.path.join('src', 'llmcore')]
dirs += [os.path.join(p, 'llmcore') for p in sys.path if os.path.isdir(os.path.join(p, 'llmcore', 'model_cards'))]
for b in [os.path.join(d, 'model_cards', 'default_cards', P) for d in dirs]:
    if not os.path.isdir(b): continue
    r = []
    for f in sorted(os.listdir(b)):
        if not f.endswith('.json'): continue
        try:
            d = json.load(open(os.path.join(b, f)))
            if d.get('model_type') != 'chat': continue
            if d.get('lifecycle', {}).get('status') in ('retired', 'deprecated'): continue
            ctx = d.get('context', {}).get('max_input_tokens', '?')
            pi = d.get('pricing', {}).get('per_million_tokens', {}).get('input', '?')
            r.append({'value': d['model_id'], 'label': d.get('display_name', d['model_id']), 'description': f'{ctx} ctx, ${pi}/M in'})
        except Exception: pass
    print(json.dumps(r))
    sys.exit()
print('[]')
PYEOF`
  - Populates: options
  - Merge: replace

### `providers.poe.timeout`

Request timeout in seconds.  Video/image generation bots may need higher values (300+) since they perform async long-running work.

### `providers.poe.backend`

"openai" (default): Uses the `openai` Python SDK with base_url override. All standard OpenAI-compatible features supported. No extra dependencies beyond `openai`. "native" : Uses the native `fastapi_poe` SSE client (pip install fastapi-poe). Required for: - Poe-specific custom bot parameters not in OpenAI schema - File attachments via the Attachment protocol - Bot-to-bot invocation (when running as a Poe server bot) Falls back to "openai" mode if the package is not installed or any request error occurs.

**Options:**

- `openai`: OpenAI SDK (default)
- `native`: Native fastapi_poe

## 🚀 Provider: vLLM

Self-hosted vLLM inference server.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `providers.vllm.base_url` | url |  | — | Base URL of your vLLM server (including /v1). Required — no default because vLLM |
| `providers.vllm.default_model` | string |  | `meta-llama/Llama-3.1-8B-Instruct` | Default model name — HuggingFace repo id or whatever name you passed to `vllm se |
| `providers.vllm.timeout` | integer |  | `240` | Request timeout in seconds.  Local cold-starts and large models can be slow; 240 |

### `providers.vllm.base_url`

Base URL of your vLLM server (including /v1). Required — no default because vLLM is self-hosted.

### `providers.vllm.default_model`

Default model name — HuggingFace repo id or whatever name you passed to `vllm serve ...`.  Override per-request via chat_completion(model=...).

**Dynamic discovery** (`vllm_served_models`):

  Query vLLM server for currently served models
  - Command: `curl -sf {{providers.vllm.base_url}}/models 2>/dev/null`
  - Populates: suggestions
  - Merge: prepend
  - Depends on: `providers.vllm.base_url`

### `providers.vllm.timeout`

Request timeout in seconds.  Local cold-starts and large models can be slow; 240s covers most practical cases.

## 🌊 Provider: Mistral AI

Native Mistral AI platform provider.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `providers.mistral.base_url` | url |  | — | Override for custom Mistral endpoints. Leave empty for official API. |
| `providers.mistral.api_key_env_var` | string |  | `MISTRAL_API_KEY` | type = "mistral"  # auto-detected from section name API Key: Set via environment |
| `providers.mistral.default_model` | enum |  | `mistral-large-latest` | Default model. Current models (April 2026): Frontier:   mistral-large-latest (→  |
| `providers.mistral.timeout` | integer |  | `120` | Request timeout in seconds. |

### `providers.mistral.api_key_env_var`

type = "mistral"  # auto-detected from section name API Key: Set via environment variable for security. api_key = "..."

### `providers.mistral.default_model`

Default model. Current models (April 2026): Frontier:   mistral-large-latest (→ mistral-large-2512) Medium:     mistral-medium-latest (→ mistral-medium-2508) Small:      mistral-small-latest (→ mistral-small-2506) Reasoning:  magistral-medium-latest, magistral-small-latest Code:       codestral-latest, devstral-2-latest Ministral:  ministral-3-14b-latest, ministral-3-8b-latest, ministral-3-3b-latest Audio STT:  voxtral-mini-latest (transcription) OCR:        mistral-ocr-latest Embedding:  mistral-embed, codestral-embed-2505

**Options:**

- `mistral-large-latest`: Mistral Large
- `mistral-medium-latest`: Mistral Medium
- `mistral-small-latest`: Mistral Small
- `magistral-medium-latest`: Magistral Medium (reasoning)
- `magistral-small-latest`: Magistral Small (reasoning)
- `codestral-latest`: Codestral (code)
- `devstral-2-latest`: Devstral 2 (code)

**Dynamic discovery** (`mistral_chat_cards`):

  Load available mistral chat models from llmcore model cards
  - Command: `python3 << 'PYEOF'
import json, os, sys
P = 'mistral'
dirs = [os.path.join('src', 'llmcore')]
dirs += [os.path.join(p, 'llmcore') for p in sys.path if os.path.isdir(os.path.join(p, 'llmcore', 'model_cards'))]
for b in [os.path.join(d, 'model_cards', 'default_cards', P) for d in dirs]:
    if not os.path.isdir(b): continue
    r = []
    for f in sorted(os.listdir(b)):
        if not f.endswith('.json'): continue
        try:
            d = json.load(open(os.path.join(b, f)))
            if d.get('model_type') != 'chat': continue
            if d.get('lifecycle', {}).get('status') in ('retired', 'deprecated'): continue
            ctx = d.get('context', {}).get('max_input_tokens', '?')
            pi = d.get('pricing', {}).get('per_million_tokens', {}).get('input', '?')
            r.append({'value': d['model_id'], 'label': d.get('display_name', d['model_id']), 'description': f'{ctx} ctx, ${pi}/M in'})
        except Exception: pass
    print(json.dumps(r))
    sys.exit()
print('[]')
PYEOF`
  - Populates: options
  - Merge: replace

## 💾 Storage: Session & Vector

Persistence backends for conversation history and RAG documents.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `storage.session.type` | enum |  | `sqlite` | Type: Determines the storage backend. - "json": Simple, human-readable. Good for |
| `storage.session.path` | path |  | `~/.llmcore/sessions.db` | Path: Filesystem path for "json" and "sqlite" backends. The `~` character is exp *(when `storage.session.type != 'postgres'`)* |
| `storage.session.db_url` | string |  | — | Database URL: Full connection URL for "postgres". It is STRONGLY recommended to  *(when `storage.session.type == 'postgres'`)* |
| `storage.vector.type` | enum |  | `chromadb` | Type: Determines the vector database backend. - "chromadb": (Default) Easy to se |
| `storage.vector.default_collection` | string |  | `llmcore_default_rag` | Default Collection: A namespace for RAG documents. Allows you to maintain multip |
| `storage.vector.path` | path |  | `~/.llmcore/chroma_db` | Path: Filesystem path for file-based vector stores like "chromadb". *(when `storage.vector.type == 'chromadb'`)* |
| `storage.vector.db_url` | string |  | — | Database URL: Connection URL for "pgvector". It is STRONGLY recommended to set t *(when `storage.vector.type == 'pgvector' || storage.vector.type == 'pgvector_legacy'`)* |

### `storage.session.type`

Type: Determines the storage backend. - "json": Simple, human-readable. Good for low-concurrency apps. - "sqlite": (Default) Excellent for single-process apps and local dev. - "postgres": Recommended for production, especially with multiple workers.

**Options:**

- `sqlite`: SQLite (default)
- `json`: JSON files
- `postgres`: PostgreSQL

### `storage.session.path`

Path: Filesystem path for "json" and "sqlite" backends. The `~` character is expanded to your home directory.

### `storage.session.db_url`

Database URL: Full connection URL for "postgres". It is STRONGLY recommended to set this via an environment variable.

### `storage.vector.type`

Type: Determines the vector database backend. - "chromadb": (Default) Easy to set up for local development. - "pgvector": Excellent for production if you already use PostgreSQL. - "pgvector_legacy": Uses original PgVectorStorage

**Options:**

- `chromadb`: ChromaDB (default)
- `pgvector`: pgvector (PostgreSQL)
- `pgvector_legacy`: pgvector legacy

### `storage.vector.default_collection`

Default Collection: A namespace for RAG documents. Allows you to maintain multiple distinct knowledge bases in the same vector store.

### `storage.vector.db_url`

Database URL: Connection URL for "pgvector". It is STRONGLY recommended to set this via an environment variable.

## 📈 Storage: Observability

Instrumentation, metrics, event logging, and tracing for storage operations.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `storage.observability.enabled` | boolean |  | `True` | Master switch for all observability features. When false, all observability comp |
| `storage.observability.log_queries` | boolean |  | `False` | Log ALL queries at DEBUG level. WARNING: Very verbose. Use only for debugging sp *(when `storage.observability.enabled == true`)* |
| `storage.observability.log_slow_queries` | boolean |  | `True` | Log queries that exceed the slow query threshold at WARNING level. Recommended t *(when `storage.observability.enabled == true`)* |
| `storage.observability.slow_query_threshold_seconds` | float |  | `1.0` | Threshold (in seconds) for slow query detection. Queries taking longer than this *(when `storage.observability.enabled == true`)* |
| `storage.observability.include_query_params` | boolean |  | `False` | Include query parameters in logs. WARNING: Security risk - may expose sensitive  *(when `storage.observability.enabled == true`)* |
| `storage.observability.metrics_enabled` | boolean |  | `True` | Enable metrics collection (counters, histograms, gauges). *(when `storage.observability.enabled == true`)* |
| `storage.observability.metrics_backend` | enum |  | `memory` | Metrics backend type. - "prometheus": Expose metrics via HTTP endpoint for Prome *(when `storage.observability.enabled == true && storage.observability.metrics_enabled == true`)* |
| `storage.observability.metrics_prefix` | string |  | `llmcore_storage` | Prefix for all metric names. *(when `storage.observability.enabled == true && storage.observability.metrics_enabled == true`)* |
| `storage.observability.metrics_port` | port |  | `9090` | Port for Prometheus metrics HTTP endpoint (when metrics_backend = "prometheus"). *(when `storage.observability.enabled == true && storage.observability.metrics_backend == 'prometheus'`)* |
| `storage.observability.event_logging_enabled` | boolean |  | `True` | Enable persistent event logging to database. Events are stored in the storage_ev *(when `storage.observability.enabled == true`)* |
| `storage.observability.event_retention_days` | integer |  | `30` | Days to retain events in the database. Set to 0 to retain events forever (not re *(when `storage.observability.enabled == true && storage.observability.event_logging_enabled == true`)* |
| `storage.observability.event_table_name` | string |  | `storage_events` | Database table name for event storage. *(when `storage.observability.enabled == true && storage.observability.event_logging_enabled == true`)* |
| `storage.observability.event_batch_size` | integer |  | `100` | Number of events to batch before flushing to database. Higher values reduce data *(when `storage.observability.enabled == true && storage.observability.event_logging_enabled == true`)* |
| `storage.observability.event_flush_interval_seconds` | float |  | `5.0` | Interval (in seconds) for flushing events to database. Events are flushed when b *(when `storage.observability.enabled == true && storage.observability.event_logging_enabled == true`)* |
| `storage.observability.tracing_enabled` | boolean |  | `False` | Enable distributed tracing. Requires OpenTelemetry SDK to be installed and confi *(when `storage.observability.enabled == true`)* |
| `storage.observability.tracing_backend` | enum |  | `none` | Tracing backend type. *(when `storage.observability.enabled == true && storage.observability.tracing_enabled == true`)* |
| `storage.observability.tracing_endpoint` | url |  | `http://localhost:4317` | OTLP collector endpoint for trace export. *(when `storage.observability.enabled == true && storage.observability.tracing_enabled == true`)* |
| `storage.observability.tracing_service_name` | string |  | `llmcore-storage` | Service name for traces. *(when `storage.observability.enabled == true && storage.observability.tracing_enabled == true`)* |
| `storage.observability.tracing_sample_rate` | float |  | `1.0` | Sampling rate for traces (0.0 to 1.0). 1.0 = trace all requests, 0.1 = trace 10% *(when `storage.observability.enabled == true && storage.observability.tracing_enabled == true`)* |

### `storage.observability.enabled`

Master switch for all observability features. When false, all observability components are disabled with zero overhead.

### `storage.observability.log_queries`

Log ALL queries at DEBUG level. WARNING: Very verbose. Use only for debugging specific issues.

### `storage.observability.log_slow_queries`

Log queries that exceed the slow query threshold at WARNING level. Recommended to keep enabled for production monitoring.

### `storage.observability.slow_query_threshold_seconds`

Threshold (in seconds) for slow query detection. Queries taking longer than this will be logged and recorded.

### `storage.observability.include_query_params`

Include query parameters in logs. WARNING: Security risk - may expose sensitive data in logs.

### `storage.observability.metrics_backend`

Metrics backend type. - "prometheus": Expose metrics via HTTP endpoint for Prometheus scraping - "memory": In-memory storage for testing/debugging - "none": Disable metrics collection

**Options:**

- `memory`: In-memory (dev/test)
- `prometheus`: Prometheus (production)
- `none`: Disabled

### `storage.observability.event_logging_enabled`

Enable persistent event logging to database. Events are stored in the storage_events table for audit and analysis.

### `storage.observability.event_retention_days`

Days to retain events in the database. Set to 0 to retain events forever (not recommended for production).

### `storage.observability.event_batch_size`

Number of events to batch before flushing to database. Higher values reduce database writes but increase memory usage.

### `storage.observability.event_flush_interval_seconds`

Interval (in seconds) for flushing events to database. Events are flushed when batch_size is reached OR this interval elapses.

### `storage.observability.tracing_enabled`

Enable distributed tracing. Requires OpenTelemetry SDK to be installed and configured.

### `storage.observability.tracing_backend`

Tracing backend type.

**Options:**

- `none`: Disabled
- `opentelemetry`: OpenTelemetry

### `storage.observability.tracing_sample_rate`

Sampling rate for traces (0.0 to 1.0). 1.0 = trace all requests, 0.1 = trace 10% of requests.

## 🃏 Model Card Library

Metadata for LLM models — context limits, capabilities, pricing, lifecycle.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `model_cards.user_cards_path` | path |  | `~/.config/llmcore/model_cards` | Path to user-defined model cards directory. Cards in this directory override bui |
| `model_cards.auto_load` | boolean |  | `True` | Automatically load model cards on first registry access. Set to false if you wan |
| `model_cards.strict_validation` | boolean |  | `False` | Validation strictness for model card files. If true, invalid JSON files will rai |

### `model_cards.user_cards_path`

Path to user-defined model cards directory. Cards in this directory override built-in cards with the same model_id. Organize cards by provider subdirectory: model_cards/<provider>/<model>.json The `~` character is expanded to your home directory.

### `model_cards.auto_load`

Automatically load model cards on first registry access. Set to false if you want to control loading timing manually via registry.load()

### `model_cards.strict_validation`

Validation strictness for model card files. If true, invalid JSON files will raise errors during loading. If false, invalid files are logged as warnings and skipped.

## 📐 Embedding Providers

Provider-specific settings for embedding generation.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `embedding.sentence_transformer.device` | enum |  | `cpu` | Device for local Sentence Transformer models. Auto-detected if not set. |
| `embedding.openai.default_model` | enum |  | `text-embedding-3-small` | API Key for OpenAI embeddings. api_key = "" |
| `embedding.google.default_model` | enum |  | `gemini-embedding-001` | API Key for Google AI (Gemini) embeddings. api_key = "" Default embedding model. |
| `embedding.ollama.default_model` | string |  | `mxbai-embed-large` | Host URL for the Ollama server for embeddings. host = "http://localhost:11434" D |
| `embedding.ollama.timeout` | integer |  | `120` |  |

### `embedding.sentence_transformer.device`

Device for local Sentence Transformer models. Auto-detected if not set.

**Options:**

- `cpu`: CPU
- `cuda`: NVIDIA GPU (CUDA) *(requires `variables.has_gpu`)*
- `mps`: Apple Silicon (MPS)

### `embedding.openai.default_model`

API Key for OpenAI embeddings. api_key = ""

**Options:**

- `text-embedding-3-small`: text-embedding-3-small (1536 dims)
- `text-embedding-3-large`: text-embedding-3-large (3072 dims)
- `text-embedding-ada-002`: text-embedding-ada-002 (legacy)

**Dynamic discovery** (`openai_embed_cards`):

  Load available openai embedding models from llmcore model cards
  - Command: `python3 << 'PYEOF'
import json, os, sys
P = 'openai'
dirs = [os.path.join('src', 'llmcore')]
dirs += [os.path.join(p, 'llmcore') for p in sys.path if os.path.isdir(os.path.join(p, 'llmcore', 'model_cards'))]
for b in [os.path.join(d, 'model_cards', 'default_cards', P) for d in dirs]:
    if not os.path.isdir(b): continue
    r = []
    for f in sorted(os.listdir(b)):
        if not f.endswith('.json'): continue
        try:
            d = json.load(open(os.path.join(b, f)))
            if d.get('model_type') != 'embedding': continue
            if d.get('lifecycle', {}).get('status') in ('retired', 'deprecated'): continue
            ctx = d.get('context', {}).get('max_input_tokens', '?')
            pi = d.get('pricing', {}).get('per_million_tokens', {}).get('input', '?')
            r.append({'value': d['model_id'], 'label': d.get('display_name', d['model_id']), 'description': f'{ctx} ctx, ${pi}/M in'})
        except Exception: pass
    print(json.dumps(r))
    sys.exit()
print('[]')
PYEOF`
  - Populates: options
  - Merge: replace

### `embedding.google.default_model`

API Key for Google AI (Gemini) embeddings. api_key = "" Default embedding model. As of 2026: - "gemini-embedding-001"      — GA, text-only, 3072 dims, 100+ languages - "gemini-embedding-2-preview" — multimodal (text+image+audio+video), 3072 dims DEPRECATED (shutdown Jan 2026): "text-embedding-004", "models/embedding-001"

**Options:**

- `gemini-embedding-001`: gemini-embedding-001 (text, 3072 dims)
- `gemini-embedding-2-preview`: gemini-embedding-2-preview (multimodal)

**Dynamic discovery** (`google_embed_cards`):

  Load available google embedding models from llmcore model cards
  - Command: `python3 << 'PYEOF'
import json, os, sys
P = 'google'
dirs = [os.path.join('src', 'llmcore')]
dirs += [os.path.join(p, 'llmcore') for p in sys.path if os.path.isdir(os.path.join(p, 'llmcore', 'model_cards'))]
for b in [os.path.join(d, 'model_cards', 'default_cards', P) for d in dirs]:
    if not os.path.isdir(b): continue
    r = []
    for f in sorted(os.listdir(b)):
        if not f.endswith('.json'): continue
        try:
            d = json.load(open(os.path.join(b, f)))
            if d.get('model_type') != 'embedding': continue
            if d.get('lifecycle', {}).get('status') in ('retired', 'deprecated'): continue
            ctx = d.get('context', {}).get('max_input_tokens', '?')
            pi = d.get('pricing', {}).get('per_million_tokens', {}).get('input', '?')
            r.append({'value': d['model_id'], 'label': d.get('display_name', d['model_id']), 'description': f'{ctx} ctx, ${pi}/M in'})
        except Exception: pass
    print(json.dumps(r))
    sys.exit()
print('[]')
PYEOF`
  - Populates: options
  - Merge: replace

### `embedding.ollama.default_model`

Host URL for the Ollama server for embeddings. host = "http://localhost:11434" Default Ollama model for embeddings. Ensure it's pulled locally.

**Dynamic discovery** (`ollama_live_models`):

  Fetch locally pulled Ollama models (shared cache)
  - Command: `curl -sf http://localhost:11434/api/tags 2>/dev/null`
  - Populates: suggestions
  - Merge: prepend
  - Condition: `variables.has_ollama`

## 🗄️ Embedding Cache

Two-tier caching (LRU memory + SQLite disk) for embedding vectors.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `embedding.cache.enabled` | boolean |  | `True` | Master switch for embedding caching. Reduce API costs by caching identical embed |
| `embedding.cache.memory_size` | integer |  | `10000` | Maximum number of embeddings to keep in memory (LRU cache). Fast access to frequ *(when `embedding.cache.enabled == true`)* |
| `embedding.cache.disk_enabled` | boolean |  | `True` | Enable SQLite disk persistence for embeddings. Cache survives process restarts. *(when `embedding.cache.enabled == true`)* |
| `embedding.cache.disk_path` | path |  | `~/.cache/llmcore/embeddings.db` | Path to SQLite database for cached embeddings. Persistent storage location. *(when `embedding.cache.enabled == true && embedding.cache.disk_enabled == true`)* |
| `embedding.cache.disk_max_entries` | integer |  | `0` | Maximum number of embeddings to store on disk (0 = unlimited). Limit disk space  *(when `embedding.cache.enabled == true && embedding.cache.disk_enabled == true`)* |
| `embedding.cache.ttl_hours` | integer |  | `0` | Hours before cached embeddings expire (0 = never expire). Ensure cache freshness *(when `embedding.cache.enabled == true`)* |

### `embedding.cache.enabled`

Master switch for embedding caching. Reduce API costs by caching identical embeddings.

### `embedding.cache.memory_size`

Maximum number of embeddings to keep in memory (LRU cache). Fast access to frequently used embeddings.

### `embedding.cache.disk_max_entries`

Maximum number of embeddings to store on disk (0 = unlimited). Limit disk space usage.

### `embedding.cache.ttl_hours`

Hours before cached embeddings expire (0 = never expire). Ensure cache freshness when embedding models change.

## 📑 Context Management & RAG

Prompt assembly, truncation strategy, RAG retrieval, and templates.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `context_management.inclusion_priority` | string |  | `system_history,explicitly_staged,user_items_active,history_chat,final_user_query` | A comma-separated list defining the order in which context components are added  |
| `context_management.truncation_priority` | string |  | `history_chat,user_items_active,rag_in_query,explicitly_staged` | A comma-separated list defining the order in which context types are removed to  |
| `context_management.default_prompt_template` | string |  | `You are an AI assistant specialized in answering questions about codebases
  based on provided context.
  Use ONLY the following pieces of retrieved context to answer the user's question.
  If the answer is not found in the context, state that you cannot answer based on the provided information.
  Do not make up an answer or use external knowledge. Keep the answer concise and relevant to the question.
  Include relevant source file paths and line numbers if possible, based *only* on the provided context metadata.

  Context:
  ---------------------
  {context}
  ---------------------

  Question: {question}

  Answer:` | The template used to format the final query when RAG is enabled. It will be form |
| `context_management.prompt_template_path` | path |  | — | Optional path to a custom prompt template file. Overrides the string above. |
| `context_management.rag_retrieval_k` | integer |  | `3` | Default number of documents to retrieve for RAG. |
| `context_management.reserved_response_tokens` | integer |  | `500` | Number of tokens to leave free in the context window for the model's response. |
| `context_management.minimum_history_messages` | integer |  | `1` | Minimum number of recent chat messages to try to keep during truncation. |
| `context_management.user_retained_messages_count` | integer |  | `5` | Prioritizes keeping the N most recent user messages (and their preceding assista |
| `context_management.max_chars_per_user_item` | integer |  | `40000000` | A safeguard character limit for a single user-provided context item. The default |

### `context_management.inclusion_priority`

A comma-separated list defining the order in which context components are added to the prompt. Gives fine-grained control over the final prompt's structure. Valid Components: "system_history", "explicitly_staged", "user_items_active", "history_chat", "final_user_query". Default: "system_history,explicitly_staged,user_items_active,history_chat,final_user_query"

### `context_management.truncation_priority`

A comma-separated list defining the order in which context types are removed to meet token limits. Controls what information is sacrificed when the context is too long. Valid Components: "history_chat", "user_items_active", "rag_in_query", "explicitly_staged". Default: "history_chat,user_items_active,rag_in_query,explicitly_staged"

### `context_management.default_prompt_template`

The template used to format the final query when RAG is enabled. It will be formatted with `{context}` (retrieved documents) and `{question}` (the user's query). To instruct the LLM on how to use the provided RAG context.

### `context_management.user_retained_messages_count`

Prioritizes keeping the N most recent user messages (and their preceding assistant responses) during truncation.

### `context_management.max_chars_per_user_item`

A safeguard character limit for a single user-provided context item. The default is very large to accommodate large documents.

## 📊 Observability & Cost Tracking

API usage tracking, cost estimation, and latency metrics.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `observability.enabled` | boolean |  | `True` | Top-level observability enable (affects all observability features) |
| `observability.cost_tracking.enabled` | boolean |  | `True` | Enable recording of API usage and cost estimation. Budget monitoring and usage a *(when `observability.enabled == true`)* |
| `observability.cost_tracking.db_path` | path |  | `~/.llmcore/costs.db` | Path to SQLite database for cost records. Persistent storage for usage analytics *(when `observability.enabled == true && observability.cost_tracking.enabled == true`)* |
| `observability.cost_tracking.retention_days` | integer |  | `90` | Days to retain cost records (0 = forever). Manage database size. *(when `observability.enabled == true && observability.cost_tracking.enabled == true`)* |
| `observability.cost_tracking.log_to_console` | boolean |  | `False` | Log usage records to console. Real-time visibility during development. *(when `observability.enabled == true && observability.cost_tracking.enabled == true`)* |
| `observability.cost_tracking.track_latency` | boolean |  | `True` | Record request latency with usage records. Performance monitoring. *(when `observability.enabled == true && observability.cost_tracking.enabled == true`)* |

### `observability.cost_tracking.enabled`

Enable recording of API usage and cost estimation. Budget monitoring and usage analytics.

### `observability.cost_tracking.db_path`

Path to SQLite database for cost records. Persistent storage for usage analytics.

## 🎯 Autonomous: Core & Goals

Autonomous operation — goal management and decomposition.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `autonomous.enabled` | boolean |  | `True` | Master switch — set to false to disable all autonomous capabilities |
| `autonomous.goals.persist_goals` | boolean |  | `True` | Enable persistent goal storage across sessions *(when `autonomous.enabled == true`)* |
| `autonomous.goals.storage_path` | path |  | `~/.local/share/llmcore/goals.json` | Path to goals storage file (JSON format). Supports ~ expansion and ${ENV_VAR} su *(when `autonomous.enabled == true && autonomous.goals.persist_goals == true`)* |
| `autonomous.goals.auto_decompose` | boolean |  | `True` | Automatically decompose high-level goals into sub-goals via LLM *(when `autonomous.enabled == true`)* |
| `autonomous.goals.max_sub_goals` | integer |  | `10` | Maximum number of sub-goals per parent goal (1–50) *(when `autonomous.enabled == true`)* |
| `autonomous.goals.max_goal_depth` | integer |  | `4` | Maximum nesting depth for goal hierarchies (1–10) *(when `autonomous.enabled == true`)* |
| `autonomous.goals.max_attempts_per_goal` | integer |  | `10` | Maximum retry attempts before marking a goal as failed (1–100) *(when `autonomous.enabled == true`)* |
| `autonomous.goals.base_cooldown_seconds` | integer |  | `60` | Base cooldown between retry attempts in seconds (exponential backoff) *(when `autonomous.enabled == true`)* |
| `autonomous.goals.max_cooldown_seconds` | integer |  | `3600` | Maximum cooldown cap in seconds for exponential backoff *(when `autonomous.enabled == true`)* |

### `autonomous.goals.storage_path`

Path to goals storage file (JSON format). Supports ~ expansion and ${ENV_VAR} substitution.

## 💓 Autonomous: Heartbeat

Periodic task scheduler for autonomous operation.

*Visible when*: `autonomous.enabled == true`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `autonomous.heartbeat.enabled` | boolean |  | `True` | Enable the heartbeat task scheduler *(when `autonomous.enabled == true`)* |
| `autonomous.heartbeat.base_interval` | integer |  | `60` | Base heartbeat interval in seconds (1–3600) *(when `autonomous.heartbeat.enabled == true`)* |
| `autonomous.heartbeat.max_concurrent_tasks` | integer |  | `3` | Maximum number of heartbeat tasks running concurrently (1–20) *(when `autonomous.heartbeat.enabled == true`)* |

## 🚨 Autonomous: Escalation

Human escalation framework — webhook and file notification channels.

*Visible when*: `autonomous.enabled == true`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `autonomous.escalation.enabled` | boolean |  | `True` | Enable human escalation *(when `autonomous.enabled == true`)* |
| `autonomous.escalation.auto_resolve_below` | enum |  | `advisory` | Auto-resolve escalations below this level without human intervention. Levels (as *(when `autonomous.escalation.enabled == true`)* |
| `autonomous.escalation.dedup_window` | integer |  | `300` | Deduplication window in seconds — identical escalations within this window are s *(when `autonomous.escalation.enabled == true`)* |
| `autonomous.escalation.webhook.enabled` | boolean |  | `False` |  *(when `autonomous.escalation.enabled == true`)* |
| `autonomous.escalation.webhook.url` | url |  | `${ESCALATION_WEBHOOK_URL}` |  *(when `autonomous.escalation.webhook.enabled == true`)* |
| `autonomous.escalation.webhook.headers.Authorization` | string |  | `Bearer ${WEBHOOK_TOKEN}` |  *(when `autonomous.escalation.webhook.enabled == true`)* |
| `autonomous.escalation.file.enabled` | boolean |  | `True` |  *(when `autonomous.escalation.enabled == true`)* |
| `autonomous.escalation.file.path` | path |  | `~/.local/share/llmcore/escalations.log` |  *(when `autonomous.escalation.file.enabled == true`)* |

### `autonomous.escalation.auto_resolve_below`

Auto-resolve escalations below this level without human intervention. Levels (ascending severity): debug, info, advisory, action, urgent, critical

**Options:**

- `debug`: Debug
- `info`: Info
- `advisory`: Advisory
- `action`: Action
- `urgent`: Urgent
- `critical`: Critical

### `autonomous.escalation.dedup_window`

Deduplication window in seconds — identical escalations within this window are suppressed (0 = no deduplication, max 86400)

## 📟 Autonomous: Resource Monitoring

Hardware limits, API cost limits, and token budgets.

*Visible when*: `autonomous.enabled == true`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `autonomous.resources.enabled` | boolean |  | `True` | Enable resource monitoring *(when `autonomous.enabled == true`)* |
| `autonomous.resources.check_interval` | integer |  | `30` | Resource check interval in seconds (5–600) *(when `autonomous.resources.enabled == true`)* |
| `autonomous.resources.max_cpu_percent` | float |  | `80.0` | Hardware limits *(when `autonomous.resources.enabled == true`)* |
| `autonomous.resources.max_memory_percent` | float |  | `80.0` |  *(when `autonomous.resources.enabled == true`)* |
| `autonomous.resources.max_temperature_c` | float |  | `75.0` |  *(when `autonomous.resources.enabled == true`)* |
| `autonomous.resources.min_disk_free_gb` | float |  | `1.0` |  *(when `autonomous.resources.enabled == true`)* |
| `autonomous.resources.max_hourly_cost_usd` | float |  | `1.0` | API cost limits (USD) *(when `autonomous.resources.enabled == true`)* |
| `autonomous.resources.max_daily_cost_usd` | float |  | `10.0` |  *(when `autonomous.resources.enabled == true`)* |
| `autonomous.resources.max_hourly_tokens` | integer |  | `100000` | Token limits *(when `autonomous.resources.enabled == true`)* |
| `autonomous.resources.max_daily_tokens` | integer |  | `1000000` |  *(when `autonomous.resources.enabled == true`)* |

## 🔄 Autonomous: Context Synthesis

Adaptive context window management for autonomous sessions.

*Visible when*: `autonomous.enabled == true`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `autonomous.context.max_context_tokens` | integer |  | `100000` | Maximum tokens for the synthesized context window *(when `autonomous.enabled == true`)* |
| `autonomous.context.compression_threshold` | float |  | `0.75` | Compression triggers when context usage exceeds this fraction (0.0–1.0) *(when `autonomous.enabled == true`)* |
| `autonomous.context.prioritization_strategy` | enum |  | `recency_relevance` | Context prioritization strategy: recency_relevance — balance recency and semanti *(when `autonomous.enabled == true`)* |

### `autonomous.context.prioritization_strategy`

Context prioritization strategy: recency_relevance — balance recency and semantic relevance (default) relevance_only    — rank purely by semantic similarity recency_only      — rank purely by temporal proximity

**Options:**

- `recency_relevance`: Recency + Relevance (balanced)
- `relevance_only`: Relevance Only
- `recency_only`: Recency Only

## 🤖 Agents: Core

Agent execution limits and sandbox mode selection.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `agents.max_iterations` | integer |  | `10` | Default maximum iterations for agent loops |
| `agents.default_timeout` | integer |  | `600` | Default timeout for agent tasks (in seconds) |
| `agents.sandbox.mode` | enum |  | `docker` | Sandbox mode determines which backend to use for isolation. - docker: Use Docker |
| `agents.sandbox.fallback_enabled` | boolean |  | `True` | Enable fallback to VM if primary mode fails (only applies to hybrid mode) *(when `agents.sandbox.mode == 'hybrid'`)* |

### `agents.sandbox.mode`

Sandbox mode determines which backend to use for isolation. - docker: Use Docker containers (recommended for most use cases) - vm: Use SSH to a dedicated VM (for high-security scenarios) - hybrid: Try Docker first, fall back to VM if Docker unavailable

**Options:**

- `docker`: Docker (recommended)
- `vm`: VM (high-security)
- `hybrid`: Hybrid (Docker → VM fallback)

## 🐳 Agents: Sandbox Docker

Docker container settings for isolated agent code execution.

*Visible when*: `agents.sandbox.mode != 'vm'`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `agents.sandbox.docker.enabled` | boolean |  | `True` | Enable Docker sandbox support *(when `agents.sandbox.mode != 'vm'`)* |
| `agents.sandbox.docker.image` | string |  | `python:3.11-slim` | Default Docker image for sandbox containers. Should be a Python image with neces *(when `agents.sandbox.mode != 'vm'`)* |
| `agents.sandbox.docker.image_whitelist` | list |  | `['python:3.*-slim', 'python:3.*-bookworm', 'llmcore-sandbox:*']` | Whitelist of allowed Docker images (glob patterns supported). Only images matchi *(when `agents.sandbox.mode != 'vm'`)* |
| `agents.sandbox.docker.full_access_label` | string |  | `llmcore.sandbox.full_access=true` | Docker label that grants FULL access level. Containers with this label bypass to *(when `agents.sandbox.mode != 'vm'`)* |
| `agents.sandbox.docker.full_access_name_patterns` | list |  | `['llmcore-trusted-*']` | Container name patterns that grant FULL access (glob patterns). *(when `agents.sandbox.mode != 'vm'`)* |
| `agents.sandbox.docker.memory_limit` | string |  | `1g` | Memory limit for containers (Docker memory limit format). *(when `agents.sandbox.mode != 'vm'`)* |
| `agents.sandbox.docker.cpu_limit` | float |  | `2.0` | CPU limit for containers (number of CPU cores). *(when `agents.sandbox.mode != 'vm'`)* |
| `agents.sandbox.docker.timeout_seconds` | integer |  | `600` | Default timeout for operations in seconds. *(when `agents.sandbox.mode != 'vm'`)* |
| `agents.sandbox.docker.network_enabled` | boolean |  | `False` | Enable network access in containers. WARNING: Enabling network access reduces is *(when `agents.sandbox.mode != 'vm'`)* |

### `agents.sandbox.docker.image`

Default Docker image for sandbox containers. Should be a Python image with necessary tools.

**Dynamic discovery** (`docker_python_images`):

  List locally available Docker images suitable for sandbox
  - Command: `docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -iE 'python|llmcore|sandbox|ubuntu|debian' | sort -u | head -30`
  - Populates: suggestions
  - Merge: prepend
  - Condition: `variables.has_docker`

### `agents.sandbox.docker.image_whitelist`

Whitelist of allowed Docker images (glob patterns supported). Only images matching these patterns can be used.

### `agents.sandbox.docker.full_access_label`

Docker label that grants FULL access level. Containers with this label bypass tool restrictions.

### `agents.sandbox.docker.network_enabled`

Enable network access in containers. WARNING: Enabling network access reduces isolation security.

## 🖥️ Agents: Sandbox VM

SSH-based VM sandbox for high-security agent execution.

*Visible when*: `agents.sandbox.mode != 'docker'`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `agents.sandbox.vm.enabled` | boolean |  | `False` | Enable VM sandbox support (requires SSH access to a VM). |
| `agents.sandbox.vm.host` | string |  | — | VM host address. *(when `agents.sandbox.vm.enabled == true`)* |
| `agents.sandbox.vm.port` | port |  | `22` | SSH port. *(when `agents.sandbox.vm.enabled == true`)* |
| `agents.sandbox.vm.username` | string |  | `agent` | SSH username. *(when `agents.sandbox.vm.enabled == true`)* |
| `agents.sandbox.vm.private_key_path` | path |  | — | Path to SSH private key file. WARNING: Use SSH agent or environment variable in  *(when `agents.sandbox.vm.enabled == true && agents.sandbox.vm.use_ssh_agent == false`)* |
| `agents.sandbox.vm.use_ssh_agent` | boolean |  | `False` | Use SSH agent for authentication instead of key file. *(when `agents.sandbox.vm.enabled == true`)* |
| `agents.sandbox.vm.full_access_hosts` | list |  | — | List of hosts that grant FULL access level. VMs in this list bypass tool restric *(when `agents.sandbox.vm.enabled == true`)* |
| `agents.sandbox.vm.timeout_seconds` | integer |  | `600` | Default timeout for operations in seconds. *(when `agents.sandbox.vm.enabled == true`)* |
| `agents.sandbox.vm.working_directory` | path |  | `/tmp/llmcore_sandbox` | Working directory on the VM. *(when `agents.sandbox.vm.enabled == true`)* |

### `agents.sandbox.vm.private_key_path`

Path to SSH private key file. WARNING: Use SSH agent or environment variable in production.

### `agents.sandbox.vm.full_access_hosts`

List of hosts that grant FULL access level. VMs in this list bypass tool restrictions.

## 📂 Agents: Sandbox Volumes & Tools

Volume mounts, tool access control, and output tracking.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `agents.sandbox.volumes.share_path` | path |  | `~/.llmcore/agent_share` | Host path for shared data that persists across sandbox instances. This directory |
| `agents.sandbox.volumes.outputs_path` | path |  | `~/.llmcore/agent_outputs` | Host path for agent output files. Files saved here persist after sandbox cleanup |
| `agents.sandbox.tools.allowed` | list |  | `['execute_shell', 'execute_python', 'save_file', 'load_file', 'replace_in_file', 'append_to_file', 'list_files', 'file_exists', 'delete_file', 'create_directory', 'get_state', 'set_state', 'list_state', 'get_sandbox_info', 'get_recorded_files', 'semantic_search', 'episodic_search', 'calculator', 'finish', 'human_approval']` | List of tools allowed for RESTRICTED access level sandboxes. FULL access sandbox |
| `agents.sandbox.tools.denied` | list |  | — | List of tools explicitly denied for all access levels. These tools are never ava |
| `agents.sandbox.output_tracking.enabled` | boolean |  | `True` | Enable persistent tracking of agent outputs. |
| `agents.sandbox.output_tracking.max_log_entries` | integer |  | `1000` | Maximum number of execution log entries per run. |
| `agents.sandbox.output_tracking.max_run_age_days` | integer |  | `30` | Maximum age of runs to keep (in days). 0 = keep forever. |
| `agents.sandbox.output_tracking.max_runs` | integer |  | `100` | Maximum number of runs to keep. 0 = no limit. |

### `agents.sandbox.volumes.share_path`

Host path for shared data that persists across sandbox instances. This directory is mounted read-write in sandboxes.

### `agents.sandbox.volumes.outputs_path`

Host path for agent output files. Files saved here persist after sandbox cleanup.

### `agents.sandbox.tools.allowed`

List of tools allowed for RESTRICTED access level sandboxes. FULL access sandboxes bypass this restriction.

### `agents.sandbox.tools.denied`

List of tools explicitly denied for all access levels. These tools are never available, even for FULL access.

## ⚡ Agents: Circuit Breaker

Detect and interrupt failing agent loops.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `agents.circuit_breaker.enabled` | boolean |  | `True` | Enable circuit breaker protection |
| `agents.circuit_breaker.max_iterations` | integer |  | `15` | Maximum iterations (overrides agent max_iterations as hard limit) *(when `agents.circuit_breaker.enabled == true`)* |
| `agents.circuit_breaker.max_same_errors` | integer |  | `3` | Trip after N identical consecutive errors *(when `agents.circuit_breaker.enabled == true`)* |
| `agents.circuit_breaker.max_execution_time_seconds` | integer |  | `300` | Trip after N seconds of total execution time *(when `agents.circuit_breaker.enabled == true`)* |
| `agents.circuit_breaker.max_total_cost` | float |  | `1.0` | Trip if total cost exceeds this amount (USD) *(when `agents.circuit_breaker.enabled == true`)* |
| `agents.circuit_breaker.progress_stall_threshold` | integer |  | `5` | Trip if progress stalls for N iterations (no output change) *(when `agents.circuit_breaker.enabled == true`)* |
| `agents.circuit_breaker.progress_stall_tolerance` | float |  | `0.01` | Progress similarity tolerance (0.0 = exact match, 1.0 = any difference) *(when `agents.circuit_breaker.enabled == true`)* |

## 🎭 Agents: Activity System

XML-based structured output for models without native tool/function calling.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `agents.activities.enabled` | boolean |  | `True` | Enable activity system as fallback for models without native tool support |
| `agents.activities.fallback_to_native_tools` | boolean |  | `True` | Fall back to native tools if activity parsing fails *(when `agents.activities.enabled == true`)* |
| `agents.activities.max_per_iteration` | integer |  | `10` | Maximum activities per LLM output iteration *(when `agents.activities.enabled == true`)* |
| `agents.activities.max_total` | integer |  | `100` | Maximum total activities per session *(when `agents.activities.enabled == true`)* |
| `agents.activities.default_timeout_seconds` | integer |  | `60` | Default timeout for activity execution (seconds) *(when `agents.activities.enabled == true`)* |
| `agents.activities.total_timeout_seconds` | integer |  | `300` | Total timeout for activity loop (seconds) *(when `agents.activities.enabled == true`)* |
| `agents.activities.stop_on_error` | boolean |  | `False` | Stop processing activities on first error *(when `agents.activities.enabled == true`)* |
| `agents.activities.parallel_execution` | boolean |  | `False` | Execute activities in parallel when possible *(when `agents.activities.enabled == true`)* |
| `agents.activities.max_observation_length` | integer |  | `4000` | Maximum observation length in characters *(when `agents.activities.enabled == true`)* |

## ✅ Agents: Capability Check

Pre-flight capability validation before agent execution.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `agents.capability_check.enabled` | boolean |  | `True` | Enable pre-flight capability checking |
| `agents.capability_check.use_model_cards` | boolean |  | `True` | Consult model card registry for capability information *(when `agents.capability_check.enabled == true`)* |
| `agents.capability_check.use_runtime_query` | boolean |  | `True` | Use runtime query to verify capabilities (if model card not found) *(when `agents.capability_check.enabled == true`)* |
| `agents.capability_check.strict_mode` | boolean |  | `False` | Strict mode: fail if model doesn't support required capabilities When false, fal *(when `agents.capability_check.enabled == true`)* |
| `agents.capability_check.suggest_alternatives` | boolean |  | `True` | Suggest alternative models if capability check fails *(when `agents.capability_check.enabled == true`)* |

### `agents.capability_check.strict_mode`

Strict mode: fail if model doesn't support required capabilities When false, falls back to activity system for non-tool models

## 👤 Agents: Human-in-the-Loop

Human approval workflows for sensitive agent operations.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `agents.hitl.enabled` | boolean |  | `True` | Enable HITL approval system |
| `agents.hitl.global_risk_threshold` | enum |  | `medium` | Global risk threshold for requiring approval *(when `agents.hitl.enabled == true`)* |
| `agents.hitl.default_timeout_seconds` | integer |  | `300` | Default timeout for approval requests (seconds) *(when `agents.hitl.enabled == true`)* |
| `agents.hitl.timeout_policy` | enum |  | `reject` | Policy when approval times out *(when `agents.hitl.enabled == true`)* |
| `agents.hitl.batch_similar_requests` | boolean |  | `True` | Batch similar approval requests *(when `agents.hitl.enabled == true`)* |
| `agents.hitl.audit_logging_enabled` | boolean |  | `True` | Enable audit logging of all approval decisions *(when `agents.hitl.enabled == true`)* |
| `agents.hitl.audit_log_path` | path |  | `~/.llmcore/logs/hitl_audit.jsonl` | Path for HITL audit log (JSONL format) *(when `agents.hitl.enabled == true && agents.hitl.audit_logging_enabled == true`)* |

### `agents.hitl.global_risk_threshold`

Global risk threshold for requiring approval

**Options:**

- `low`: Low
- `medium`: Medium
- `high`: High
- `critical`: Critical

### `agents.hitl.timeout_policy`

Policy when approval times out

**Options:**

- `reject`: Reject (safest)
- `approve`: Auto-approve
- `escalate`: Escalate

## 🔀 Agents: Model Routing

Intelligent routing based on task complexity, cost, and capability.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `agents.routing.enabled` | boolean |  | `True` | Enable model routing |
| `agents.routing.strategy` | enum |  | `cost_optimized` | Routing optimization strategy *(when `agents.routing.enabled == true`)* |
| `agents.routing.fallback_enabled` | boolean |  | `True` | Enable fallback to alternative models on failure *(when `agents.routing.enabled == true`)* |
| `agents.routing.tiers.fast` | list |  | `['gpt-4o-mini', 'claude-3-haiku-20240307', 'gemma3:1b']` |  *(when `agents.routing.enabled == true`)* |
| `agents.routing.tiers.balanced` | list |  | `['gpt-4o', 'claude-3-5-sonnet-20241022', 'llama3.3:70b']` |  *(when `agents.routing.enabled == true`)* |
| `agents.routing.tiers.capable` | list |  | `['gpt-4-turbo', 'claude-3-opus-20240229', 'o1-preview']` |  *(when `agents.routing.enabled == true`)* |

### `agents.routing.strategy`

Routing optimization strategy

**Options:**

- `cost_optimized`: Cost Optimized
- `quality_optimized`: Quality Optimized
- `latency_optimized`: Latency Optimized

## 🎯 Agents: Goal Classification

Complexity classification for fast-path routing and iteration limits.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `agents.goals.classifier_enabled` | boolean |  | `True` | Enable goal complexity classification When enabled, goals are analyzed before ex |
| `agents.goals.use_llm_fallback` | boolean |  | `False` | Use LLM for uncertain classifications Falls back to LLM when heuristic confidenc *(when `agents.goals.classifier_enabled == true`)* |
| `agents.goals.heuristic_confidence_threshold` | float |  | `0.9` | Confidence threshold for heuristic classification If heuristic confidence is bel *(when `agents.goals.classifier_enabled == true`)* |
| `agents.goals.trivial_max_iterations` | integer |  | `1` | Maximum iterations by complexity level These override max_iterations based on go *(when `agents.goals.classifier_enabled == true`)* |
| `agents.goals.simple_max_iterations` | integer |  | `5` |  *(when `agents.goals.classifier_enabled == true`)* |
| `agents.goals.moderate_max_iterations` | integer |  | `10` |  *(when `agents.goals.classifier_enabled == true`)* |
| `agents.goals.complex_max_iterations` | integer |  | `15` |  *(when `agents.goals.classifier_enabled == true`)* |

### `agents.goals.classifier_enabled`

Enable goal complexity classification When enabled, goals are analyzed before execution to determine complexity.

### `agents.goals.use_llm_fallback`

Use LLM for uncertain classifications Falls back to LLM when heuristic confidence is below threshold. Slower but more accurate for edge cases.

### `agents.goals.heuristic_confidence_threshold`

Confidence threshold for heuristic classification If heuristic confidence is below this, uses LLM fallback (if enabled).

### `agents.goals.trivial_max_iterations`

Maximum iterations by complexity level These override max_iterations based on goal complexity.

## ⚡ Agents: Fast Path

Fast-path executor for trivial goals — bypasses full cognitive cycle.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `agents.fast_path.enabled` | boolean |  | `True` | Enable fast-path for trivial goals When enabled, trivial goals (like "hello") sk |
| `agents.fast_path.cache_enabled` | boolean |  | `True` | Use cached responses for identical/similar queries *(when `agents.fast_path.enabled == true`)* |
| `agents.fast_path.cache_max_entries` | integer |  | `100` | Maximum cache entries *(when `agents.fast_path.enabled == true && agents.fast_path.cache_enabled == true`)* |
| `agents.fast_path.cache_ttl_seconds` | integer |  | `3600` | Cache TTL in seconds (1 hour default) *(when `agents.fast_path.enabled == true && agents.fast_path.cache_enabled == true`)* |
| `agents.fast_path.templates_enabled` | boolean |  | `True` | Use template responses for known trivial patterns *(when `agents.fast_path.enabled == true`)* |
| `agents.fast_path.max_response_time_ms` | integer |  | `5000` | Maximum response time before timeout (milliseconds) *(when `agents.fast_path.enabled == true`)* |
| `agents.fast_path.temperature` | float |  | `0.7` | Temperature for fast-path LLM calls *(when `agents.fast_path.enabled == true`)* |
| `agents.fast_path.max_tokens` | integer |  | `500` | Maximum tokens for fast-path responses *(when `agents.fast_path.enabled == true`)* |
| `agents.fast_path.fallback_on_timeout` | boolean |  | `True` | Fall back to full cognitive cycle on timeout *(when `agents.fast_path.enabled == true`)* |

### `agents.fast_path.enabled`

Enable fast-path for trivial goals When enabled, trivial goals (like "hello") skip the full cognitive cycle.

## 🧬 Agents: Darwin Enhancements

Failure learning, TDD support, and multi-attempt arbiter.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `agents.darwin.enabled` | boolean |  | `True` | Enable Darwin agent enhancements When false, Darwin uses basic cognitive cycle w |
| `agents.darwin.failure_learning.enabled` | boolean |  | `True` | Enable failure learning system When true, failures are logged and retrieved for  *(when `agents.darwin.enabled == true`)* |
| `agents.darwin.failure_learning.backend` | enum |  | `sqlite` | Storage backend for failure logs - sqlite: File-based storage (recommended for d *(when `agents.darwin.enabled == true && agents.darwin.failure_learning.enabled == true`)* |
| `agents.darwin.failure_learning.db_path` | path |  | `~/.local/share/llmcore/failures.db` | SQLite database path (used when backend = "sqlite") *(when `agents.darwin.failure_learning.enabled == true && agents.darwin.failure_learning.backend == 'sqlite'`)* |
| `agents.darwin.failure_learning.db_url` | string |  | — | PostgreSQL connection URL (used when backend = "postgres") postgresql://user:pas *(when `agents.darwin.failure_learning.enabled == true && agents.darwin.failure_learning.backend == 'postgres'`)* |
| `agents.darwin.failure_learning.max_failures_to_retrieve` | integer |  | `5` | Maximum number of similar failures to retrieve before planning Higher values pro *(when `agents.darwin.failure_learning.enabled == true`)* |
| `agents.darwin.failure_learning.postgres.min_pool_size` | integer |  | `2` | Minimum number of connections in pool *(when `agents.darwin.failure_learning.backend == 'postgres'`)* |
| `agents.darwin.failure_learning.postgres.max_pool_size` | integer |  | `10` | Maximum number of connections in pool *(when `agents.darwin.failure_learning.backend == 'postgres'`)* |
| `agents.darwin.failure_learning.postgres.table_prefix` | string |  | — | Table name prefix (useful for multi-tenant deployments) *(when `agents.darwin.failure_learning.backend == 'postgres'`)* |
| `agents.darwin.tdd.enabled` | boolean |  | `False` | Enable TDD workflow When true, tests are generated before implementation code. *(when `agents.darwin.enabled == true`)* |
| `agents.darwin.tdd.default_framework` | enum |  | `pytest` | Default test framework *(when `agents.darwin.tdd.enabled == true`)* |
| `agents.darwin.tdd.min_tests` | integer |  | `5` | Minimum number of tests to generate *(when `agents.darwin.tdd.enabled == true`)* |
| `agents.darwin.tdd.max_iterations` | integer |  | `3` | Maximum TDD iteration cycles (generate tests → code → verify → retry) *(when `agents.darwin.tdd.enabled == true`)* |
| `agents.darwin.arbiter.enabled` | boolean |  | `False` | Enable multi-attempt generation When true, generates N candidates and uses arbit *(when `agents.darwin.enabled == true`)* |
| `agents.darwin.arbiter.num_candidates` | integer |  | `3` | Number of candidates to generate Higher values increase quality but also cost an *(when `agents.darwin.arbiter.enabled == true`)* |
| `agents.darwin.arbiter.temperatures` | list |  | `[0.3, 0.7, 1.0]` | Temperature values for candidate generation Different temperatures produce diver *(when `agents.darwin.arbiter.enabled == true`)* |
| `agents.darwin.arbiter.use_llm_arbiter` | boolean |  | `True` | Use LLM-based arbiter for selection When true, uses LLM to score and select best *(when `agents.darwin.arbiter.enabled == true`)* |
| `agents.darwin.arbiter.scoring.correctness` | float |  | `0.5` |  *(when `agents.darwin.arbiter.enabled == true`)* |
| `agents.darwin.arbiter.scoring.completeness` | float |  | `0.3` |  *(when `agents.darwin.arbiter.enabled == true`)* |
| `agents.darwin.arbiter.scoring.style` | float |  | `0.2` |  *(when `agents.darwin.arbiter.enabled == true`)* |

### `agents.darwin.enabled`

Enable Darwin agent enhancements When false, Darwin uses basic cognitive cycle without enhancements.

### `agents.darwin.failure_learning.enabled`

Enable failure learning system When true, failures are logged and retrieved for future reference. When false, failure learning is disabled (no-op).

### `agents.darwin.failure_learning.backend`

Storage backend for failure logs - sqlite: File-based storage (recommended for development) - postgres: PostgreSQL database (recommended for production)

**Options:**

- `sqlite`: SQLite (development)
- `postgres`: PostgreSQL (production)

### `agents.darwin.failure_learning.db_url`

PostgreSQL connection URL (used when backend = "postgres") postgresql://user:password@host:port/database

### `agents.darwin.failure_learning.max_failures_to_retrieve`

Maximum number of similar failures to retrieve before planning Higher values provide more context but increase prompt size.

### `agents.darwin.tdd.default_framework`

Default test framework

**Options:**

- `pytest`: pytest
- `unittest`: unittest
- `jest`: Jest (JavaScript)

### `agents.darwin.arbiter.enabled`

Enable multi-attempt generation When true, generates N candidates and uses arbiter to select best.

### `agents.darwin.arbiter.num_candidates`

Number of candidates to generate Higher values increase quality but also cost and latency.

### `agents.darwin.arbiter.temperatures`

Temperature values for candidate generation Different temperatures produce diverse solutions.

### `agents.darwin.arbiter.use_llm_arbiter`

Use LLM-based arbiter for selection When true, uses LLM to score and select best candidate. When false, uses simple heuristics.

## 🔍 Agents: Observability

Structured event logging, metrics, replay, sinks, and performance.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `agents.observability.enabled` | boolean |  | `True` | Master switch for the entire observability system. Quickly disable all observabi |
| `agents.observability.events.enabled` | boolean |  | `True` | Enable/disable structured event logging to file. Capture detailed execution trac *(when `agents.observability.enabled == true`)* |
| `agents.observability.events.log_path` | path |  | `~/.llmcore/events.jsonl` | Path to the JSONL file where events are written. Persistent storage for executio *(when `agents.observability.enabled == true && agents.observability.events.enabled == true`)* |
| `agents.observability.events.min_severity` | enum |  | `info` | Filter events by minimum severity level. Control verbosity of event logging. *(when `agents.observability.enabled == true && agents.observability.events.enabled == true`)* |
| `agents.observability.events.categories` | list |  | — | Whitelist of event categories to capture. Selectively enable/disable logging for *(when `agents.observability.enabled == true && agents.observability.events.enabled == true`)* |
| `agents.observability.events.rotation.strategy` | enum |  | `size` | Rotation strategy: "none", "daily", "size", "both" - none: No rotation, single f *(when `agents.observability.events.enabled == true`)* |
| `agents.observability.events.rotation.max_size_mb` | integer |  | `100` | Maximum log file size in MB before rotation (if strategy includes "size"). *(when `agents.observability.events.enabled == true`)* |
| `agents.observability.events.rotation.max_files` | integer |  | `10` | Number of rotated log files to keep. 0 = keep all. *(when `agents.observability.events.enabled == true`)* |
| `agents.observability.events.rotation.compress` | boolean |  | `True` | Compress rotated files with gzip. *(when `agents.observability.events.enabled == true`)* |
| `agents.observability.buffer.enabled` | boolean |  | `True` | Enable write buffering for better performance. Batch multiple events before writ *(when `agents.observability.enabled == true`)* |
| `agents.observability.buffer.size` | integer |  | `100` | Maximum number of events to buffer before flushing. Balance between I/O efficien *(when `agents.observability.enabled == true && agents.observability.buffer.enabled == true`)* |
| `agents.observability.buffer.flush_interval_seconds` | integer |  | `5` | Maximum time (seconds) to hold events before flushing. Ensure events are persist *(when `agents.observability.buffer.enabled == true`)* |
| `agents.observability.buffer.flush_on_shutdown` | boolean |  | `True` | Flush all buffered events when agent shuts down. Prevent data loss on graceful s *(when `agents.observability.buffer.enabled == true`)* |
| `agents.observability.metrics.enabled` | boolean |  | `True` | Enable/disable execution metrics collection. Track performance, cost, and usage  *(when `agents.observability.enabled == true`)* |
| `agents.observability.metrics.collect` | list |  | — | Whitelist of metric types to collect. Control which metrics are tracked. - "iter *(when `agents.observability.metrics.enabled == true`)* |
| `agents.observability.metrics.track_cost` | boolean |  | `True` | Enable estimated cost calculation for LLM calls. Budget monitoring and optimizat *(when `agents.observability.metrics.enabled == true`)* |
| `agents.observability.metrics.track_tokens` | boolean |  | `True` | Track input/output token counts per call. Usage analysis and optimization. *(when `agents.observability.metrics.enabled == true`)* |
| `agents.observability.metrics.latency_percentiles` | list |  | `[50, 90, 95, 99]` | Percentiles to calculate for latency metrics. Understand latency distribution (p *(when `agents.observability.metrics.enabled == true`)* |
| `agents.observability.replay.enabled` | boolean |  | `True` | Enable execution replay functionality. Debug and analyze past executions step-by *(when `agents.observability.enabled == true`)* |
| `agents.observability.replay.cache_enabled` | boolean |  | `True` | Cache parsed events in memory for faster replay. Speed up repeated replay operat *(when `agents.observability.replay.enabled == true`)* |
| `agents.observability.replay.cache_max_executions` | integer |  | `50` | Maximum number of executions to cache. Limit memory usage. *(when `agents.observability.replay.enabled == true && agents.observability.replay.cache_enabled == true`)* |
| `agents.observability.sinks.file_enabled` | boolean |  | `True` | Primary sink - writes to JSONL file (configured above in events.log_path) *(when `agents.observability.enabled == true`)* |
| `agents.observability.sinks.memory_enabled` | boolean |  | `False` | Store events in memory for debugging/testing. Inspect events without file I/O du *(when `agents.observability.enabled == true`)* |
| `agents.observability.sinks.memory_max_events` | integer |  | `1000` | Maximum events to keep in memory sink. 0 = unlimited. *(when `agents.observability.sinks.memory_enabled == true`)* |
| `agents.observability.sinks.callback_enabled` | boolean |  | `False` | Call a user-defined function for each event. Custom integrations (e.g., send to  *(when `agents.observability.enabled == true`)* |
| `agents.observability.performance.async_logging` | boolean |  | `True` | Write events asynchronously to avoid blocking execution. Minimize observability  *(when `agents.observability.enabled == true`)* |
| `agents.observability.performance.sampling_rate` | float |  | `1.0` | Sample rate for high-frequency events (0.0 to 1.0). Reduce volume for very activ *(when `agents.observability.enabled == true`)* |
| `agents.observability.performance.max_event_data_bytes` | integer |  | `10000` | Maximum size (bytes) for event data fields. Prevent huge events from bloating lo *(when `agents.observability.enabled == true`)* |
| `agents.observability.performance.overhead_warning_threshold_percent` | integer |  | `5` | Warn if logging overhead exceeds this percentage of execution time. Detect perfo *(when `agents.observability.enabled == true`)* |

### `agents.observability.enabled`

Master switch for the entire observability system. Quickly disable all observability features without changing other settings.

### `agents.observability.events.enabled`

Enable/disable structured event logging to file. Capture detailed execution traces for debugging and analysis.

### `agents.observability.events.log_path`

Path to the JSONL file where events are written. Persistent storage for execution events. Events are written as newline-delimited JSON (JSONL/NDJSON).

### `agents.observability.events.min_severity`

Filter events by minimum severity level. Control verbosity of event logging.

**Options:**

- `debug`: Debug
- `info`: Info
- `warning`: Warning
- `error`: Error
- `critical`: Critical

### `agents.observability.events.categories`

Whitelist of event categories to capture. Selectively enable/disable logging for specific event types. "metric", "memory", "sandbox", "rag" Empty list means log ALL categories.

### `agents.observability.events.rotation.strategy`

Rotation strategy: "none", "daily", "size", "both" - none: No rotation, single file grows indefinitely - daily: Rotate at midnight each day - size: Rotate when file exceeds max_size_mb - both: Rotate on either condition

**Options:**

- `none`: No rotation
- `daily`: Daily
- `size`: By size
- `both`: Daily + size

### `agents.observability.buffer.enabled`

Enable write buffering for better performance. Batch multiple events before writing to reduce I/O overhead.

### `agents.observability.buffer.size`

Maximum number of events to buffer before flushing. Balance between I/O efficiency and data freshness.

### `agents.observability.buffer.flush_interval_seconds`

Maximum time (seconds) to hold events before flushing. Ensure events are persisted even during low activity.

### `agents.observability.buffer.flush_on_shutdown`

Flush all buffered events when agent shuts down. Prevent data loss on graceful shutdown.

### `agents.observability.metrics.enabled`

Enable/disable execution metrics collection. Track performance, cost, and usage statistics.

### `agents.observability.metrics.collect`

Whitelist of metric types to collect. Control which metrics are tracked. - "iterations": Cognitive loop iterations - "llm_calls": LLM API calls (latency, tokens, cost) - "activities": Tool/activity executions - "hitl": Human approval metrics - "errors": Error counts and types - "duration": Execution timing - "tokens": Token usage - "cost": Estimated costs Empty list means collect ALL metrics.

### `agents.observability.metrics.track_cost`

Enable estimated cost calculation for LLM calls. Budget monitoring and optimization.

### `agents.observability.metrics.latency_percentiles`

Percentiles to calculate for latency metrics. Understand latency distribution (p50, p95, p99).

### `agents.observability.replay.enabled`

Enable execution replay functionality. Debug and analyze past executions step-by-step. Requires events.enabled = true

### `agents.observability.replay.cache_enabled`

Cache parsed events in memory for faster replay. Speed up repeated replay operations.

### `agents.observability.sinks.memory_enabled`

Store events in memory for debugging/testing. Inspect events without file I/O during development. WARNING: Memory usage grows with event count. Use only for debugging.

### `agents.observability.sinks.callback_enabled`

Call a user-defined function for each event. Custom integrations (e.g., send to external monitoring). Callback must be registered programmatically, this just enables it.

### `agents.observability.performance.async_logging`

Write events asynchronously to avoid blocking execution. Minimize observability overhead on agent performance.

### `agents.observability.performance.sampling_rate`

Sample rate for high-frequency events (0.0 to 1.0). Reduce volume for very active agents. Lifecycle and error events are never sampled.

### `agents.observability.performance.max_event_data_bytes`

Maximum size (bytes) for event data fields. Prevent huge events from bloating log files. Large data is truncated with "[TRUNCATED]" marker.

### `agents.observability.performance.overhead_warning_threshold_percent`

Warn if logging overhead exceeds this percentage of execution time. Detect performance problems from observability.

## 🔬 Semantiscan: Core & Database

RAG/ingestion engine — database and vector backend configuration.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `semantiscan.enabled` | boolean |  | `True` | Top-level semantiscan enablement flag (for future use) |
| `semantiscan.database.type` | enum |  | `chromadb` |  *(when `semantiscan.enabled == true`)* |
| `semantiscan.database.collection_name` | string |  | `default_semantiscan` |  *(when `semantiscan.enabled == true`)* |
| `semantiscan.database.vector_backend` | enum |  | `llmcore` |  *(when `semantiscan.enabled == true`)* |
| `semantiscan.database.user_id` | string |  | `project_123` |  *(when `semantiscan.enabled == true`)* |
| `semantiscan.database.namespace` | string |  | `dev` |  *(when `semantiscan.enabled == true`)* |
| `semantiscan.database.enable_hybrid_search` | boolean |  | `True` |  *(when `semantiscan.enabled == true`)* |
| `semantiscan.database.path` | path |  | `~/.llmcore/chroma_db` | type = "chromadb"  # Currently only ChromaDB is supported by semantiscan Default *(when `semantiscan.enabled == true`)* |

### `semantiscan.database.type`

**Options:**

- `chromadb`: ChromaDB

### `semantiscan.database.vector_backend`

**Options:**

- `llmcore`: llmcore (unified)
- `chromadb`: ChromaDB (standalone)

### `semantiscan.database.path`

type = "chromadb"  # Currently only ChromaDB is supported by semantiscan Default path - should be synchronized with llmcore's vector storage path

## 🗃️ Semantiscan: Metadata Store

External metadata store for ingestion history and Git tracking.

*Visible when*: `semantiscan.enabled == true`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `semantiscan.metadata_store.enable` | boolean |  | `False` |  *(when `semantiscan.enabled == true`)* |
| `semantiscan.metadata_store.type` | enum |  | `sqlite` |  *(when `semantiscan.metadata_store.enable == true`)* |
| `semantiscan.metadata_store.path` | path |  | `~/.local/share/semantiscan/metadata.db` |  *(when `semantiscan.metadata_store.enable == true && semantiscan.metadata_store.type == 'sqlite'`)* |
| `semantiscan.metadata_store.connection_string` | string |  | — |  *(when `semantiscan.metadata_store.enable == true && semantiscan.metadata_store.type == 'postgresql'`)* |
| `semantiscan.metadata_store.table_name` | string |  | `chunk_metadata` | Primary table for storing rich chunk metadata *(when `semantiscan.metadata_store.enable == true`)* |
| `semantiscan.metadata_store.ingestion_log_table_name` | string |  | `ingestion_log` | Table for tracking ingestion state per repo/branch *(when `semantiscan.metadata_store.enable == true`)* |
| `semantiscan.metadata_store.file_history_table_name` | string |  | `file_history` | Table for tracking file changes per commit ('historical_delta' mode) *(when `semantiscan.metadata_store.enable == true`)* |

### `semantiscan.metadata_store.type`

**Options:**

- `sqlite`: SQLite
- `postgresql`: PostgreSQL

## 🧲 Semantiscan: Embeddings

Embedding model configurations for chunking ingestion.

*Visible when*: `semantiscan.enabled == true`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `semantiscan.embeddings.default_model` | string |  | `sentence_transformer_local` |  |
| `semantiscan.embeddings.models.sentence_transformer_local.provider` | string |  | `sentence-transformers` |  |
| `semantiscan.embeddings.models.sentence_transformer_local.model_name` | string |  | `all-MiniLM-L6-v2` |  |
| `semantiscan.embeddings.models.sentence_transformer_local.device` | enum |  | `cpu` |  |
| `semantiscan.embeddings.models.sentence_transformer_local.max_request_tokens` | integer |  | `8000` |  |
| `semantiscan.embeddings.models.sentence_transformer_local.base_url` | string |  | — |  |
| `semantiscan.embeddings.models.sentence_transformer_local.tokenizer_name` | string |  | — |  |
| `semantiscan.embeddings.models.sentence_transformer_local.uses_doc_query_prefixes` | boolean |  | `False` |  |
| `semantiscan.embeddings.models.sentence_transformer_local.query_prefix` | string |  | — |  |
| `semantiscan.embeddings.models.sentence_transformer_local.document_prefix` | string |  | — |  |
| `semantiscan.embeddings.models.openai_ada.provider` | string |  | `openai` |  |
| `semantiscan.embeddings.models.openai_ada.model_name` | string |  | `text-embedding-ada-002` |  |
| `semantiscan.embeddings.models.openai_ada.device` | string |  | `cpu` |  |
| `semantiscan.embeddings.models.openai_ada.api_key_env` | string |  | `OPENAI_API_KEY` |  |
| `semantiscan.embeddings.models.openai_ada.max_request_tokens` | integer |  | `8000` |  |
| `semantiscan.embeddings.models.openai_ada.base_url` | string |  | — |  |
| `semantiscan.embeddings.models.openai_ada.tokenizer_name` | string |  | `cl100k_base` |  |
| `semantiscan.embeddings.models.openai_ada.uses_doc_query_prefixes` | boolean |  | `False` |  |
| `semantiscan.embeddings.models.openai_ada.query_prefix` | string |  | — |  |
| `semantiscan.embeddings.models.openai_ada.document_prefix` | string |  | — |  |
| `semantiscan.embeddings.models.openai_large.provider` | string |  | `openai` |  |
| `semantiscan.embeddings.models.openai_large.model_name` | string |  | `text-embedding-3-large` |  |
| `semantiscan.embeddings.models.openai_large.device` | string |  | `cpu` |  |
| `semantiscan.embeddings.models.openai_large.api_key_env` | string |  | `OPENAI_API_KEY` |  |
| `semantiscan.embeddings.models.openai_large.max_request_tokens` | integer |  | `8000` |  |
| `semantiscan.embeddings.models.openai_large.base_url` | string |  | — |  |
| `semantiscan.embeddings.models.openai_large.tokenizer_name` | string |  | `cl100k_base` |  |
| `semantiscan.embeddings.models.openai_large.uses_doc_query_prefixes` | boolean |  | `False` |  |
| `semantiscan.embeddings.models.openai_large.query_prefix` | string |  | — |  |
| `semantiscan.embeddings.models.openai_large.document_prefix` | string |  | — |  |
| `semantiscan.embeddings.models.ollama_nomic.provider` | string |  | `ollama` |  |
| `semantiscan.embeddings.models.ollama_nomic.model_name` | string |  | `nomic-embed-text:latest` |  |
| `semantiscan.embeddings.models.ollama_nomic.device` | string |  | `cpu` |  |
| `semantiscan.embeddings.models.ollama_nomic.api_key_env` | string |  | — |  |
| `semantiscan.embeddings.models.ollama_nomic.max_request_tokens` | integer |  | `2048` |  |
| `semantiscan.embeddings.models.ollama_nomic.base_url` | url |  | `http://localhost:11434` |  |
| `semantiscan.embeddings.models.ollama_nomic.tokenizer_name` | string |  | — |  |
| `semantiscan.embeddings.models.ollama_nomic.uses_doc_query_prefixes` | boolean |  | `False` |  |
| `semantiscan.embeddings.models.ollama_nomic.query_prefix` | string |  | — |  |
| `semantiscan.embeddings.models.ollama_nomic.document_prefix` | string |  | — |  |

### `semantiscan.embeddings.models.sentence_transformer_local.device`

**Options:**

- `cpu`: CPU
- `cuda`: NVIDIA GPU (CUDA)
- `mps`: Apple Silicon (MPS)

### `semantiscan.embeddings.models.ollama_nomic.model_name`

**Dynamic discovery** (`ollama_live_models`):

  Fetch locally pulled Ollama models
  - Command: `curl -sf http://localhost:11434/api/tags 2>/dev/null`
  - Populates: suggestions
  - Merge: replace
  - Condition: `variables.has_ollama`

## ✂️ Semantiscan: Chunking

File type parsing and chunking strategies for ingestion.

*Visible when*: `semantiscan.enabled == true`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `semantiscan.chunking.default_strategy` | enum |  | `RecursiveSplitter` |  |
| `semantiscan.chunking.strategies.python_code.extensions` | list |  | `['.py', '.pyw']` |  |
| `semantiscan.chunking.strategies.python_code.grammar` | string |  | `Python3` |  |
| `semantiscan.chunking.strategies.python_code.entry_points` | list |  | `['funcdef', 'async_funcdef', 'classdef', 'decorated']` |  |
| `semantiscan.chunking.strategies.python_code.parser` | string |  | — |  |
| `semantiscan.chunking.strategies.python_code.method` | string |  | — |  |
| `semantiscan.chunking.strategies.python_code.strategy_sequence` | list |  | — |  |
| `semantiscan.chunking.strategies.python_code.hybrid_content` | boolean |  | `False` |  |
| `semantiscan.chunking.strategies.java_code.extensions` | list |  | `['.java']` |  |
| `semantiscan.chunking.strategies.java_code.grammar` | string |  | `Java` |  |
| `semantiscan.chunking.strategies.java_code.entry_points` | list |  | `['methodDeclaration', 'classDeclaration', 'interfaceDeclaration']` |  |
| `semantiscan.chunking.strategies.java_code.parser` | string |  | — |  |
| `semantiscan.chunking.strategies.java_code.method` | string |  | — |  |
| `semantiscan.chunking.strategies.java_code.strategy_sequence` | list |  | — |  |
| `semantiscan.chunking.strategies.java_code.hybrid_content` | boolean |  | `False` |  |
| `semantiscan.chunking.strategies.cpp_code.extensions` | list |  | `['.cpp', '.cc', '.cxx', '.hpp', '.h', '.hxx']` |  |
| `semantiscan.chunking.strategies.cpp_code.grammar` | string |  | `CPP14` |  |
| `semantiscan.chunking.strategies.cpp_code.entry_points` | list |  | `['functionDefinition', 'classSpecifier']` |  |
| `semantiscan.chunking.strategies.cpp_code.parser` | string |  | — |  |
| `semantiscan.chunking.strategies.cpp_code.method` | string |  | — |  |
| `semantiscan.chunking.strategies.cpp_code.strategy_sequence` | list |  | — |  |
| `semantiscan.chunking.strategies.cpp_code.hybrid_content` | boolean |  | `False` |  |
| `semantiscan.chunking.strategies.go_code.extensions` | list |  | `['.go']` |  |
| `semantiscan.chunking.strategies.go_code.grammar` | string |  | `Golang` |  |
| `semantiscan.chunking.strategies.go_code.entry_points` | list |  | `['functionDecl', 'methodDecl', 'typeSpec']` |  |
| `semantiscan.chunking.strategies.go_code.parser` | string |  | — |  |
| `semantiscan.chunking.strategies.go_code.method` | string |  | — |  |
| `semantiscan.chunking.strategies.go_code.strategy_sequence` | list |  | — |  |
| `semantiscan.chunking.strategies.go_code.hybrid_content` | boolean |  | `False` |  |
| `semantiscan.chunking.strategies.yaml_format.extensions` | list |  | `['.yaml', '.yml']` |  |
| `semantiscan.chunking.strategies.yaml_format.grammar` | string |  | — |  |
| `semantiscan.chunking.strategies.yaml_format.entry_points` | list |  | — |  |
| `semantiscan.chunking.strategies.yaml_format.parser` | string |  | `PyYAML` |  |
| `semantiscan.chunking.strategies.yaml_format.method` | string |  | — |  |
| `semantiscan.chunking.strategies.yaml_format.strategy_sequence` | list |  | — |  |
| `semantiscan.chunking.strategies.yaml_format.hybrid_content` | boolean |  | `False` |  |
| `semantiscan.chunking.strategies.json_format.extensions` | list |  | `['.json']` |  |
| `semantiscan.chunking.strategies.json_format.grammar` | string |  | — |  |
| `semantiscan.chunking.strategies.json_format.entry_points` | list |  | — |  |
| `semantiscan.chunking.strategies.json_format.parser` | string |  | `json` |  |
| `semantiscan.chunking.strategies.json_format.method` | string |  | — |  |
| `semantiscan.chunking.strategies.json_format.strategy_sequence` | list |  | — |  |
| `semantiscan.chunking.strategies.json_format.hybrid_content` | boolean |  | `False` |  |
| `semantiscan.chunking.strategies.toml_format.extensions` | list |  | `['.toml']` |  |
| `semantiscan.chunking.strategies.toml_format.grammar` | string |  | — |  |
| `semantiscan.chunking.strategies.toml_format.entry_points` | list |  | — |  |
| `semantiscan.chunking.strategies.toml_format.parser` | string |  | `tomli` |  |
| `semantiscan.chunking.strategies.toml_format.method` | string |  | — |  |
| `semantiscan.chunking.strategies.toml_format.strategy_sequence` | list |  | — |  |
| `semantiscan.chunking.strategies.toml_format.hybrid_content` | boolean |  | `False` |  |
| `semantiscan.chunking.strategies.xml_format.extensions` | list |  | `['.xml']` |  |
| `semantiscan.chunking.strategies.xml_format.grammar` | string |  | — |  |
| `semantiscan.chunking.strategies.xml_format.entry_points` | list |  | — |  |
| `semantiscan.chunking.strategies.xml_format.parser` | string |  | `xml.etree.ElementTree` |  |
| `semantiscan.chunking.strategies.xml_format.method` | string |  | — |  |
| `semantiscan.chunking.strategies.xml_format.strategy_sequence` | list |  | — |  |
| `semantiscan.chunking.strategies.xml_format.hybrid_content` | boolean |  | `False` |  |
| `semantiscan.chunking.strategies.markdown_agnostic.extensions` | list |  | `['.md', '.markdown']` |  |
| `semantiscan.chunking.strategies.markdown_agnostic.grammar` | string |  | — |  |
| `semantiscan.chunking.strategies.markdown_agnostic.entry_points` | list |  | — |  |
| `semantiscan.chunking.strategies.markdown_agnostic.parser` | string |  | — |  |
| `semantiscan.chunking.strategies.markdown_agnostic.method` | string |  | `RecursiveSplitter` |  |
| `semantiscan.chunking.strategies.markdown_agnostic.strategy_sequence` | list |  | — |  |
| `semantiscan.chunking.strategies.markdown_agnostic.hybrid_content` | boolean |  | `False` |  |
| `semantiscan.chunking.strategies.text_agnostic.extensions` | list |  | `['.txt']` |  |
| `semantiscan.chunking.strategies.text_agnostic.grammar` | string |  | — |  |
| `semantiscan.chunking.strategies.text_agnostic.entry_points` | list |  | — |  |
| `semantiscan.chunking.strategies.text_agnostic.parser` | string |  | — |  |
| `semantiscan.chunking.strategies.text_agnostic.method` | string |  | `RecursiveSplitter` |  |
| `semantiscan.chunking.strategies.text_agnostic.strategy_sequence` | list |  | — |  |
| `semantiscan.chunking.strategies.text_agnostic.hybrid_content` | boolean |  | `False` |  |
| `semantiscan.chunking.strategies.logs_agnostic.extensions` | list |  | `['.log']` |  |
| `semantiscan.chunking.strategies.logs_agnostic.grammar` | string |  | — |  |
| `semantiscan.chunking.strategies.logs_agnostic.entry_points` | list |  | — |  |
| `semantiscan.chunking.strategies.logs_agnostic.parser` | string |  | — |  |
| `semantiscan.chunking.strategies.logs_agnostic.method` | string |  | `LineSplitter` |  |
| `semantiscan.chunking.strategies.logs_agnostic.strategy_sequence` | list |  | — |  |
| `semantiscan.chunking.strategies.logs_agnostic.hybrid_content` | boolean |  | `False` |  |
| `semantiscan.chunking.strategies.rst_agnostic.extensions` | list |  | `['.rst']` |  |
| `semantiscan.chunking.strategies.rst_agnostic.grammar` | string |  | — |  |
| `semantiscan.chunking.strategies.rst_agnostic.entry_points` | list |  | — |  |
| `semantiscan.chunking.strategies.rst_agnostic.parser` | string |  | — |  |
| `semantiscan.chunking.strategies.rst_agnostic.method` | string |  | `RecursiveSplitter` |  |
| `semantiscan.chunking.strategies.rst_agnostic.strategy_sequence` | list |  | — |  |
| `semantiscan.chunking.strategies.rst_agnostic.hybrid_content` | boolean |  | `False` |  |
| `semantiscan.chunking.parameters.RecursiveSplitter.chunk_size` | integer |  | `1000` |  |
| `semantiscan.chunking.parameters.RecursiveSplitter.chunk_overlap` | integer |  | `150` |  |
| `semantiscan.chunking.parameters.LineSplitter.lines_per_chunk` | integer |  | `50` |  |
| `semantiscan.chunking.parameters.SubChunker.chunk_size` | integer |  | `500` |  |
| `semantiscan.chunking.parameters.SubChunker.chunk_overlap` | integer |  | `50` |  |

### `semantiscan.chunking.default_strategy`

**Options:**

- `RecursiveSplitter`: RecursiveSplitter
- `LineSplitter`: LineSplitter

## 📥 Semantiscan: Ingestion

Ingestion pipeline behaviour and Git-aware modes.

*Visible when*: `semantiscan.enabled == true`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `semantiscan.ingestion.embedding_workers` | integer |  | `4` |  |
| `semantiscan.ingestion.batch_size` | integer |  | `100` |  |
| `semantiscan.ingestion.git.enabled` | boolean |  | `False` |  |
| `semantiscan.ingestion.git.default_ref` | string |  | `main` |  *(when `semantiscan.ingestion.git.enabled == true`)* |
| `semantiscan.ingestion.git.ingestion_mode` | enum |  | `snapshot` |  *(when `semantiscan.ingestion.git.enabled == true`)* |
| `semantiscan.ingestion.git.historical_start_ref` | string |  | — |  *(when `semantiscan.ingestion.git.enabled == true`)* |
| `semantiscan.ingestion.git.enable_commit_analysis` | boolean |  | `False` |  *(when `semantiscan.ingestion.git.enabled == true`)* |
| `semantiscan.ingestion.git.enable_commit_llm_analysis` | boolean |  | `False` |  *(when `semantiscan.ingestion.git.enabled == true && semantiscan.ingestion.git.enable_commit_analysis == true`)* |
| `semantiscan.ingestion.git.commit_llm_provider_key` | string |  | — |  *(when `semantiscan.ingestion.git.enable_commit_llm_analysis == true`)* |
| `semantiscan.ingestion.git.commit_llm_prompt_template` | string |  | — |  *(when `semantiscan.ingestion.git.enable_commit_llm_analysis == true`)* |
| `semantiscan.ingestion.git.commit_message_filter_regex` | list |  | — |  *(when `semantiscan.ingestion.git.enabled == true`)* |

### `semantiscan.ingestion.git.ingestion_mode`

**Options:**

- `snapshot`: Snapshot
- `historical`: Historical
- `historical_delta`: Historical Delta
- `incremental`: Incremental

## 💬 Semantiscan: LLM

LLM configuration for query answering and advanced features.

*Visible when*: `semantiscan.enabled == true`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `semantiscan.llm.default_provider` | string |  | `ollama_llama3` |  |
| `semantiscan.llm.prompt_template_path` | path |  | — |  |
| `semantiscan.llm.enable_query_rewriting` | boolean |  | `False` |  |
| `semantiscan.llm.query_rewrite_provider_key` | string |  | — |  |
| `semantiscan.llm.show_sources_in_text` | boolean |  | `True` |  |
| `semantiscan.llm.tokenizer_name` | string |  | — |  |
| `semantiscan.llm.context_buffer` | integer |  | `200` |  |
| `semantiscan.llm.providers.ollama_llama3.provider` | string |  | `ollama` |  |
| `semantiscan.llm.providers.ollama_llama3.model_name` | string |  | `llama3:8b` |  |
| `semantiscan.llm.providers.ollama_llama3.base_url` | url |  | `http://localhost:11434` |  |
| `semantiscan.llm.providers.ollama_llama3.api_key_env` | string |  | — |  |
| `semantiscan.llm.providers.ollama_llama3.tokenizer_name` | string |  | — |  |
| `semantiscan.llm.providers.ollama_llama3.context_buffer` | integer |  | `250` |  |
| `semantiscan.llm.providers.ollama_llama3.parameters.temperature` | float |  | `0.5` |  |
| `semantiscan.llm.providers.ollama_llama3.parameters.num_ctx` | integer |  | `4096` |  |
| `semantiscan.llm.providers.ollama_llama3.parameters.top_p` | float |  | `0.9` |  |
| `semantiscan.llm.providers.openai_gpt4.provider` | string |  | `openai` |  |
| `semantiscan.llm.providers.openai_gpt4.model_name` | string |  | `gpt-4` |  |
| `semantiscan.llm.providers.openai_gpt4.base_url` | string |  | — |  |
| `semantiscan.llm.providers.openai_gpt4.api_key_env` | string |  | `OPENAI_API_KEY` |  |
| `semantiscan.llm.providers.openai_gpt4.tokenizer_name` | string |  | `cl100k_base` |  |
| `semantiscan.llm.providers.openai_gpt4.context_buffer` | integer |  | `200` |  |
| `semantiscan.llm.providers.openai_gpt4.parameters.temperature` | float |  | `0.2` |  |
| `semantiscan.llm.providers.openai_gpt4.parameters.max_tokens` | integer |  | `4000` |  |
| `semantiscan.llm.providers.openai_gpt4_turbo.provider` | string |  | `openai` |  |
| `semantiscan.llm.providers.openai_gpt4_turbo.model_name` | string |  | `gpt-4-turbo-preview` |  |
| `semantiscan.llm.providers.openai_gpt4_turbo.base_url` | string |  | — |  |
| `semantiscan.llm.providers.openai_gpt4_turbo.api_key_env` | string |  | `OPENAI_API_KEY` |  |
| `semantiscan.llm.providers.openai_gpt4_turbo.tokenizer_name` | string |  | `cl100k_base` |  |
| `semantiscan.llm.providers.openai_gpt4_turbo.context_buffer` | integer |  | `200` |  |
| `semantiscan.llm.providers.openai_gpt4_turbo.parameters.temperature` | float |  | `0.2` |  |
| `semantiscan.llm.providers.openai_gpt4_turbo.parameters.max_tokens` | integer |  | `4000` |  |

### `semantiscan.llm.providers.ollama_llama3.model_name`

**Dynamic discovery** (`ollama_live_models`):

  Fetch locally pulled Ollama models
  - Command: `curl -sf http://localhost:11434/api/tags 2>/dev/null`
  - Populates: suggestions
  - Merge: replace
  - Condition: `variables.has_ollama`

## 🔎 Semantiscan: Retrieval

RAG retrieval parameters — top_k, hybrid search, BM25.

*Visible when*: `semantiscan.enabled == true`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `semantiscan.retrieval.top_k` | integer |  | `10` |  |
| `semantiscan.retrieval.enable_hybrid_search` | boolean |  | `False` |  *(when `semantiscan.enabled == true`)* |
| `semantiscan.retrieval.bm25_k1` | float |  | `1.5` |  *(when `semantiscan.retrieval.enable_hybrid_search == true`)* |
| `semantiscan.retrieval.bm25_b` | float |  | `0.75` |  *(when `semantiscan.retrieval.enable_hybrid_search == true`)* |
| `semantiscan.retrieval.enrich_with_external_metadata` | boolean |  | `False` |  *(when `semantiscan.metadata_store.enable == true`)* |

## 📁 Semantiscan: File Discovery

File inclusion/exclusion rules during ingestion.

*Visible when*: `semantiscan.enabled == true`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `semantiscan.discovery.use_gitignore` | boolean |  | `True` |  |
| `semantiscan.discovery.excluded_dirs` | list |  | `['__pycache__', 'node_modules', '.git', 'venv', '.venv', 'build', 'dist', '.pytest_cache', '.mypy_cache', 'htmlcov', '.tox', '.eggs', '*.egg-info', 'target']` |  |
| `semantiscan.discovery.excluded_files` | list |  | `['.DS_Store', 'Thumbs.db', '*.pyc', '*.pyo', '*.pyd']` |  |

## 📝 Semantiscan: Logging

Semantiscan-specific logging (separate from unified logging).

*Visible when*: `semantiscan.enabled == true`

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `semantiscan.logging.log_level_console` | enum |  | `INFO` |  |
| `semantiscan.logging.log_file_enabled` | boolean |  | `False` |  *(when `semantiscan.enabled == true`)* |
| `semantiscan.logging.log_directory` | path |  | `~/.local/share/semantiscan/logs` |  *(when `semantiscan.logging.log_file_enabled == true`)* |
| `semantiscan.logging.log_filename_template` | string |  | `semantiscan_{timestamp:%Y%m%d_%H%M%S}.log` |  *(when `semantiscan.logging.log_file_enabled == true`)* |
| `semantiscan.logging.log_level_file` | enum |  | `DEBUG` |  *(when `semantiscan.logging.log_file_enabled == true`)* |
| `semantiscan.logging.log_format` | string |  | `%(asctime)s [%(levelname)-8s] %(name)-30s - %(message)s (%(filename)s:%(lineno)d)` |  *(when `semantiscan.logging.log_file_enabled == true`)* |

### `semantiscan.logging.log_level_console`

**Options:**

- `DEBUG`: Debug
- `INFO`: Info
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical

### `semantiscan.logging.log_level_file`

**Options:**

- `DEBUG`: Debug
- `INFO`: Info
- `WARNING`: Warning
- `ERROR`: Error
- `CRITICAL`: Critical
