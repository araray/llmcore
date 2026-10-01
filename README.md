<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/araray/llmcore/main/assets/branding/logo_dark_llmcore.png">
    <img src="https://raw.githubusercontent.com/araray/llmcore/main/assets/branding/logo_light_llmcore.png" alt="llmcore" width="720">
  </picture>
</p>
<p align="center">
  <strong>An async Python framework for LLM applications: chat, agents, RAG,
  generative media, and remote GPU runtimes — behind one interface.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue.svg"/></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"/></a>
  <a href="https://github.com/araray/llmcore"><img alt="Version" src="https://img.shields.io/badge/version-0.53.0-green.svg"/></a>
  <a href="https://github.com/araray/llmcore/actions"><img alt="CI" src="https://img.shields.io/badge/CI-lint%20%2B%20py3.11%20%2B%20py3.12-brightgreen.svg"/></a>
  <a href="#-at-a-glance"><img alt="Providers" src="https://img.shields.io/badge/providers-23-blue.svg"/></a>
  <a href="#-model-cards"><img alt="Model cards" src="https://img.shields.io/badge/model%20cards-2319-blue.svg"/></a>
</p>

<p align="center">
  <a href="#-at-a-glance">At a glance</a> •
  <a href="#-providers">Providers</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-generative-media">Media</a> •
  <a href="#-remote-gpu-runtimes">Runtimes</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-architecture">Architecture</a>
</p>

---

**llmcore** gives you one async interface over 23 model providers, plus the
subsystems that usually get rebuilt per project: conversation persistence,
retrieval, a generative-media layer (image/audio/video), tool-using agents with
sandboxed execution, and provisioning for remotely served open-weights models.

It is a library, not a service. There is no daemon, no broker, and no required
infrastructure — `pip install llmcore[openai]` and a config file are enough. Every
subsystem beyond chat is optional and off until configured.

## 📊 At a glance

Counts are measured from this repository, not estimated.

| | |
|---|---|
| **Providers** | **23** behind one interface (plus 11 alias spellings) |
| **Transport** | **21 of 23** have two transports — 17 direct-first with an SDK fallback, 4 SDK-first with direct available. The other 2 have no vendor SDK |
| **Model cards** | **2,319** across 22 providers, generated from live APIs |
| **Media capabilities** | **19** — image, video, speech, music, SFX, OCR, voice design |
| **Install extras** | **35**, so you install only the providers you use |
| **Tests** | **6,396** collected; 5,622 run in the default (no-infrastructure) profile |
| **Coverage** | **66%** statement+branch, measured on that profile |
| **Source** | ~156k lines across 318 modules |

> Coverage is reported as measured. `pyproject.toml` sets a `fail_under` of 85%,
> which the suite does not currently meet — that gate is a target, not a
> description of today. CI runs the suite without the coverage gate, so this
> number is informational rather than enforced.

### What the model cards cover

Cards carry context windows, capability flags, pricing and lifecycle for every
model llmcore can reach, so cost and capability checks happen locally instead of
by trial and error.

| Model type | Cards | | Model type | Cards |
|---|---:|---|---|---:|
| chat | 1,716 | | audio | 71 |
| image-generation | 156 | | stt | 58 |
| tts | 125 | | multimodal | 13 |
| embedding | 87 | | video-generation | 8 |
| vision | 72 | | ocr / code / other | 13 |

---

## ✨ Features

| Category | What it does |
|----------|--------------|
| **🔌 Providers** | 23 vendors, one `chat()` call. Streaming, tool calling, structured output, reasoning extraction, vision, exact tokenizers where the vendor exposes one |
| **🎨 Generative media** | `llm.media` — image generate/edit/upscale, video generate/interpolate, TTS (+streaming), ASR, music, SFX, voice design. Async jobs with polling, a webhook receiver, and a content-addressed artifact store |
| **🖥️ Remote GPU runtimes** | `llm.runtimes` — size an open-weights model, provision compute, serve it, and attach the endpoint as a provider instance. Spend ceilings and idle reaping are enforced, not optional |
| **🧭 Routing** | `target=`/`pool=`/`lane=` on `chat()` — reach any provider+model without a config section, fail over on 429 / empty wallet / timeout, route by request kind, keep PII on your own hardware, and run as an OpenAI-compatible proxy for agent harnesses |
| **💬 Sessions** | Persistent conversations over SQLite/PostgreSQL/JSON, transient sessions, per-call usage via `chat_with_usage()` |
| **🔍 RAG** | ChromaDB/pgvector, semantic search, context injection, external-RAG bridge |
| **🌐 Web search** | Bright Data, Serper.dev, SerpApi, Semantic Scholar (keyless) |
| **🤖 Agents** | 8-phase cognitive cycle, goal classification, fast-path execution, circuit breaker, personas |
| **🔒 Sandboxing** | Docker and VM/SSH isolation with security policies and output tracking |
| **👤 Human-in-the-loop** | Risk assessment, approval workflows, audit logging |
| **📊 Observability** | Structured events, metrics, execution replay, context diagnostics |
| **📚 Model cards** | 2,319 cards with capability validation and cost estimation, refreshed by `cardctl` |

### Recent additions (v0.51 → v0.53)

- **`llmcore.media`** — a first-class generative-media subsystem. Capability
  protocols rather than per-vendor methods, so `llm.media.images.generate(...)`
  routes to whichever configured provider can serve it. Execution class is
  declared *per capability*, so the same call returns a result from OpenAI and a
  pollable job from fal, and `media.wait()` absorbs the difference.
- **Media adapters** for OpenAI, Google (Veo), Deepgram, ElevenLabs, fal,
  Replicate, Hugging Face and Higgsfield.
- **`VoiceConsent`** — synthetic speech carries whose voice it is and whether
  the vendor considers it cleared, so callers can refuse unverified clones
  without a second API call.
- **Webhook receiver** — signed single-use callbacks for long jobs. Polling
  stays the fallback, so no public ingress is required.
- **`llmcore.runtimes`** — provision remote GPU compute and attach it as a
  provider. Off by default; `up()` refuses without explicit spend confirmation.
- **Dual transport** — 21 of 23 providers now call the vendor API directly with
  the SDK as a fallback (or the reverse, where an SDK owns something llmcore
  should not reimplement). See
  [`PROVIDER_SUPPORT_MATRIX.md`](docs/PROVIDER_SUPPORT_MATRIX.md) §7.1.
- **`cardctl doctor`** — audits that every registered provider has a card
  adapter, so a new provider cannot ship without model cards.
- **`llmcore.routing`** — five composable layers: dynamic targets, failover
  pools with seven selection strategies, classifier-driven lanes, response
  cascades, and prompt transforms. Plus proxy mode, so an unmodified agent
  harness routes through llmcore by setting two environment variables. Off by
  default; see [`Routing_usage.md`](docs/Routing_usage.md).

---

## 🚀 Quickstart

### Simple Chat

```python
import asyncio
from llmcore import LLMCore

async def main():
    async with await LLMCore.create() as llm:
        # Simple question
        response = await llm.chat("What is the capital of France?")
        print(response)
        
        # Streaming response
        stream = await llm.chat("Tell me a story.", stream=True)
        async for chunk in stream:
            print(chunk, end="", flush=True)

asyncio.run(main())
```

### Conversation with Session

```python
async def conversation():
    async with await LLMCore.create() as llm:
        # First message - sets context
        await llm.chat(
            "My name is Alex and I love astronomy.",
            session_id="alex_chat",
            system_message="You are a friendly science tutor."
        )
        
        # Follow-up - LLM remembers context
        response = await llm.chat(
            "What should I observe tonight?",
            session_id="alex_chat"
        )
        print(response)
```

### RAG with Document Context

```python
async def rag_example():
    async with await LLMCore.create() as llm:
        # Add documents to vector store
        await llm.add_documents_to_vector_store(
            documents=[
                {"content": "LLMCore supports multiple providers...", "metadata": {"source": "docs"}},
                {"content": "Configuration uses the confy library...", "metadata": {"source": "docs"}},
            ],
            collection_name="my_docs"
        )
        
        # Query with RAG
        response = await llm.chat(
            "How does LLMCore handle configuration?",
            enable_rag=True,
            rag_collection_name="my_docs",
            rag_retrieval_k=3
        )
        print(response)
```

### Autonomous Agent

```python
from llmcore.agents import AgentManager, AgentMode

async def agent_example():
    async with await LLMCore.create() as llm:
        # Create agent manager
        agent = AgentManager(
            provider_manager=llm._provider_manager,
            memory_manager=llm._memory_manager,
            storage_manager=llm._storage_manager
        )
        
        # Run agent with a goal
        result = await agent.run(
            goal="Research the top 3 Python web frameworks and compare them",
            mode=AgentMode.SINGLE
        )
        print(result.final_answer)
```

---

## 🎨 Generative media

`llm.media` routes image, audio and video work to whichever configured provider
can serve it. Adapters **are** the chat providers, so there is one credential per
vendor and no separate media config tree.

```python
llm = await LLMCore.create(config)

# Routed to the first configured provider that can do it
image = await llm.media.images.generate("a calico cat asleep on books")
speech = await llm.media.audio.speak("Consent is not an afterthought.")
text = await llm.media.audio.transcribe(audio=MediaRef.from_path("call.mp3"))

# Ask what the current configuration can actually do
llm.media.capabilities()                      # every capability available
llm.media.who_can(MediaCapability.VIDEO_GENERATE)   # ['gemini', 'fal', 'replicate']
```

### Execution class is per capability, not per provider

Image generation answers in one request on OpenAI and is a queued job on fal. So
a router call returns either a `MediaResult` or a pollable `MediaJob`, and
`media.wait()` absorbs the difference — the same code works against both.

```python
job = await llm.media.video.generate("a drone shot over a fjord", provider="fal")
result = await llm.media.wait(job, timeout=600)     # polls with capped backoff
print(result.artifacts[0].uri)
```

Long jobs can also arrive by **webhook**: `media.jobs` issues signed, single-use
callback URLs and `create_webhook_app()` returns a plain ASGI app to receive
them. Polling remains the fallback, so no public ingress is required.

### Capabilities

| Group | Capabilities |
|---|---|
| **Image** | `image_generate`, `image_edit`, `image_upscale`, `image_variate` |
| **Video** | `video_generate`, `video_edit`, `video_interpolate`, `video_reframe`, `video_upscale`, `video_extend` |
| **Audio** | `tts`, `tts_stream`, `asr`, `asr_stream`, `music`, `sfx`, `voice_design`, `voice_agent` |
| **Document** | `ocr` |

Artifacts carry bytes or a URI, checksums, dimensions/duration, usage, and
provenance. A content-addressed store can materialize remote artifacts before
vendor URLs expire. For synthetic speech, provenance includes a `VoiceConsent`
record: whose voice it is, whether it is a clone, and whether the vendor
considers it cleared — with `None` meaning *the vendor did not say*, which is
deliberately distinct from *no*.

See [`MEDIA_SUBSYSTEM_SPEC.md`](docs/MEDIA_SUBSYSTEM_SPEC.md).

---

## 🧭 Routing

Five layers that compose. Each is useful on its own, and **nothing is on by
default** — with no `[routing]` section, every call resolves exactly as it did
before.

**Any model, without a config section.** `[providers.*]` sections are presets,
not an allow-list:

```python
await llm.chat("hi", target="xai:grok-4.1-20251117?effort=high")
await llm.chat("hi", target="ollama:llama3.3:70b")        # colons in model names are fine
await llm.chat("hi", target="vllm:Qwen/Qwen3-30B#my-box")  # '#' pins an instance
```

**Failover that distinguishes failures.** A pool is a set of interchangeable
targets; what routing does next depends on *why* the call failed. A 429 cools
down briefly and moves on. An empty wallet cools down for minutes and records a
balance of zero. A 400 does not fail over at all, because it fails everywhere.
A bad key benches the target for the process. A prompt that overflows the
context window is not an error — llmcore knows every model's window from its
card and routes to a bigger one.

```python
await llm.chat("hi", pool="main")     # 7 strategies: priority, round_robin,
                                      # weighted, lowest_latency, lowest_cost,
                                      # least_busy, most_credits
```

**Routing by request kind.** A classifier names a *lane*, never a model, so
swapping models is a config edit:

```python
await llm.chat("rename this variable", lane="trivial")
await llm.chat("[[lane:deep]] walk me through this proof")   # the model can route itself
await llm.chat("summarise this", profile="frugal")
```

The free classifiers (an explicit hint, a marker in the prompt, your own
function, a length/code heuristic) cost nothing. A local 350M encoder scores the
prompt against your lane descriptions zero-shot, and TypeSafe's `choice`
primitive returns a calibrated pick. llmcore orders the chain itself —
cheapest first, and instructions before guesses, so a length heuristic can never
override an explicit `lane=`.

**Privacy by destination, not by redaction.**

```toml
[routing.transforms.pii]
on_detect = "constrain"     # route it somewhere it cannot leak
pool = "local_only"
redact = true               # and redact anyway, as defence in depth
```

A detector that misses one identifier has leaked it, so the guarantee is the
*route*: a prompt with personal data in it goes to a pool that never leaves the
machine. Redaction is stacked on top and documented as a mitigation rather than
a guarantee.

**Proxy mode for agent harnesses.** The harness owns its API call, so llmcore
cannot add an argument to it — but it can set a base URL and a model name:

```bash
llmcore-bridge proxy
export OPENAI_BASE_URL=http://127.0.0.1:8900/v1 OPENAI_MODEL=lane:standard
```

Anything speaking the OpenAI chat-completions API now gets pools, failover,
classifiers, cascades, transforms and cost accounting unchanged. Lanes and pools
appear in `GET /v1/models`, so the harness's own model picker selects routing
policy. The proxy binds to loopback and *refuses* a non-loopback bind without a
bearer token, because the process holds every provider credential you have
configured.

**Explaining itself.** A feature that silently changes which vendor served a
request has to be able to say why:

```python
print(await llm.routing.why("summarise this file"))
#   lane=trivial pool=cheap strategy=priority chosen=gemini:gemini-3.8-flash est=$0.0004
#     classifier: heuristic — ~7 tokens and a simple-task verb
#     -> gemini:gemini-3.8-flash est=0.0004
#        openai:gpt-5.4 — skipped: cooling down for 12s after rate_limit

llm.routing.health()        # per-target cooldowns, latency, failures, balance
```

**Measuring it.** llmcore publishes no accuracy figure for any classifier,
because none has been validated on real traffic and any number would be
invented. `llmcore-routing eval` closes that with your own labelled prompts,
and reports the two error directions separately — routing too cheap produces a
bad answer, routing too expensive only costs money, and a single percentage
hides which one you are buying.

Guide: [`Routing_usage.md`](docs/Routing_usage.md). Design:
[`ROUTING_SUBSYSTEM_SPEC.md`](docs/ROUTING_SUBSYSTEM_SPEC.md).

---

## 🖥️ Remote GPU runtimes

`llm.runtimes` provisions compute elsewhere, serves an open-weights model on it,
and registers the endpoint as a provider instance — so a remotely served model
is reachable through the same `llm.chat()` as a hosted API.

```python
plan = await llm.runtimes.estimate("Qwen/Qwen3-30B-A3B-Instruct-2507", context_length=32768)
print(plan.sku, plan.vram_required_gb, plan.fits)     # free; nothing is provisioned

handle = await llm.runtimes.up(plan.spec.repo_id, name="qwen30", confirm_spend=True)
answer = await llm.chat("Explain GQA briefly.", provider_name="qwen30")
await llm.runtimes.down("qwen30")                      # unregister, then release
```

**This subsystem bills per minute from the moment compute is assigned**, which
makes it unlike every other provider here. The safety rules are enforced in
code, not left to the caller:

- **off by default** — `LLMCore.create()` contacts no backend regardless of config;
- **`up()` refuses** without explicit spend confirmation, while `estimate()` is
  free and works even while the subsystem is disabled;
- **state is written before provisioning returns**, as inspectable JSON under
  `~/.llmcore/runtimes`, so a runtime can always be found and killed;
- **bounded by default** — idle and hard-lifetime deadlines, plus an optional
  compute-unit ceiling, because an idle reaper does not stop a runtime that is
  busy in a loop;
- **`close()` detaches but does not tear down** — a process exiting is not a
  reason to destroy compute you are paying for. `down_all()` is explicit.

The Colab backend is implemented: sizing against Hugging Face metadata, the
bootstrap, the SSH tunnel, keepalive, a liveness probe, the idle reaper, orphan
detection and `bake`. There is a CLI for the commands you need when something
has gone wrong:

```bash
llmcore-runtimes estimate Qwen/Qwen2.5-7B-Instruct --context 16384   # free
llmcore-runtimes up Qwen/Qwen2.5-7B-Instruct --name q7 --yes         # spends
llmcore-runtimes status                                              # incl. orphans
llmcore-runtimes down q7
```

**Verified against a real GPU VM** on 2026-10-01: Qwen2.5-1.5B-Instruct served
by vLLM on a Colab T4, reached through `llm.chat()` and through a pool
containing the runtime, then released — total cost 0.6 compute units. That run
found four bugs that every test had passed, including a session parser that
could not recognise its own session and a `colab exec` that returns 0 even
when the code it ran raised. Details in the CHANGELOG.

Still unproven: a single unattended `up()` with all four fixes applied, and
the "cold start is seconds" claim for a warm environment cache.

Guide: [`Runtimes_usage.md`](docs/Runtimes_usage.md). Design:
[`COLAB_RUNTIME_SPEC.md`](docs/COLAB_RUNTIME_SPEC.md).

---

## 📦 Installation

**Requires Python 3.11 or later.**

### Basic Installation

```bash
pip install llmcore
```

### Provider extras

Each provider is its own extra, so you install only what you use. All 35 extras
are listed in `pyproject.toml`.

```bash
# Chat and reasoning
pip install "llmcore[openai]"        # also covers groq, together, xai, deepseek, kimi
pip install "llmcore[anthropic]"
pip install "llmcore[gemini]"
pip install "llmcore[mistral]"       # httpx + mistralai v3 fallback
pip install "llmcore[zai]"           # GLM family
pip install "llmcore[friendli]"
pip install "llmcore[deepinfra]"
pip install "llmcore[openrouter]"
pip install "llmcore[poe]"
pip install "llmcore[huggingface]"
pip install "llmcore[ollama]"        # local, no API key
pip install "llmcore[vllm]"          # self-hosted

# Media and voice
pip install "llmcore[elevenlabs]"    # TTS, ASR, SFX, music, voice design
pip install "llmcore[deepgram]"      # STT, TTS, Voice Agent
pip install "llmcore[fal]"           # image, video, audio marketplace
pip install "llmcore[replicate]"
pip install "llmcore[higgsfield]"    # image + video

# Typed judgment
pip install "llmcore[typesafe]"

# Web search
pip install "llmcore[brightdata]" "llmcore[serper]" "llmcore[serpapi]" "llmcore[semanticscholar]"
```

### With Storage Backends

```bash
# ChromaDB for vector storage
pip install llmcore[chromadb]

# PostgreSQL with pgvector
pip install llmcore[postgres]
```

### With Sandbox Support

```bash
# Docker sandbox
pip install llmcore[sandbox-docker]

# VM/SSH sandbox
pip install llmcore[sandbox-vm]

# Both sandbox types
pip install llmcore[sandbox]
```

### Full Installation

```bash
# Everything included
pip install llmcore[all]
```

### From Source

```bash
git clone https://github.com/araray/llmcore.git
cd llmcore
pip install -e ".[dev]"
```

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              LLMCore API Facade                           │
│                          (llmcore.api.LLMCore)                            │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Provider   │  │   Session    │  │   Memory     │  │  Embedding   │   │
│  │   Manager    │  │   Manager    │  │   Manager    │  │   Manager    │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                 │                 │                 │           │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐   │
│  │  Providers   │  │   Storage    │  │    RAG       │  │  Embeddings  │   │
│  │  23 vendors  │  │  • SQLite    │  │  • ChromaDB  │  │  • Sentence  │   │
│  │  direct REST │  │  • Postgres  │  │  • pgvector  │  │    Transform │   │
│  │  + SDK       │  │  • JSON      │  │  • external  │  │  • OpenAI    │   │
│  │  fallbacks   │  │              │  │    bridge    │  │  • Google    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                           │
│  ┌──────────────┐  ┌──────────────┐                                       │
│  │    Media     │  │   Runtimes   │   optional subsystems, off until       │
│  │   Manager    │  │   Manager    │   configured                          │
│  └──────┬───────┘  └──────┬───────┘                                       │
│         │                 │                                               │
│  ┌──────┴───────┐  ┌──────┴───────┐                                       │
│  │ image/audio/ │  │ size → up →  │                                       │
│  │ video router │  │ attach as a  │                                       │
│  │ jobs+webhook │  │ provider     │                                       │
│  │ artifacts    │  │ reap/ceiling │                                       │
│  └──────────────┘  └──────────────┘                                       │
│                                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│                            Agent System (Darwin Layer 2)                  │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                         Cognitive Cycle                              │ │
│  │ PERCEIVE → PLAN → THINK → VALIDATE → ACT → OBSERVE → REFLECT → UPDATE│ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │    Goal     │ │  Fast-Path  │ │   Circuit   │ │    HITL     │          │
│  │ Classifier  │ │  Executor   │ │  Breaker    │ │   Manager   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘          │
│                                                                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │   Persona   │ │   Prompt    │ │  Activity   │ │ Capability  │          │
│  │   Manager   │ │   Library   │ │   System    │ │   Checker   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘          │
│                                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│                              Sandbox System                               │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Docker Provider  │  VM Provider  │  Registry  │  Output Tracker     │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│                           Supporting Systems                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Model Card  │ │Observability│ │   Tracing   │ │   Logging   │          │
│  │  Registry   │ │   System    │ │   System    │ │   Config    │          │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘          │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

LLMCore uses [`confy`](https://github.com/araray/confy) for layered configuration with the following precedence (highest priority last):

1. **Package Defaults** → `llmcore/config/default_config.toml`
2. **User Config** → `~/.config/llmcore/config.toml`
3. **Custom File** → `LLMCore.create(config_file_path="...")`
4. **Environment Variables** → `LLMCORE_*` prefix
5. **Direct Overrides** → `LLMCore.create(config_overrides={...})`

### Example Configuration

```toml
# ~/.config/llmcore/config.toml

[llmcore]
default_provider = "openai"
default_embedding_model = "text-embedding-3-small"
log_level = "INFO"

[providers.openai]
# API key via: LLMCORE_PROVIDERS__OPENAI__API_KEY or OPENAI_API_KEY
default_model = "gpt-5.4"
timeout = 60

[providers.anthropic]
default_model = "claude-sonnet-5-5"
timeout = 60

[providers.ollama]
# host = "http://localhost:11434"
default_model = "llama3.2:latest"

[providers.vllm]
# Self-hosted vLLM server. base_url is required (no default).
# base_url = "http://localhost:8000/v1"
default_model = "meta-llama/Llama-3.1-8B-Instruct"
timeout = 240

[providers.zai]
# Z.ai Open Platform (GLM family). API key via ZAI_API_KEY.
# backend = "sdk"       # "sdk" (native zai-sdk, default) | "openai" | "httpx"
# region = "overseas"   # or "china" for the open.bigmodel.cn endpoint
default_model = "glm-5.2"
thinking = "enabled"            # "enabled" | "disabled"
reasoning_effort = "high"       # none|minimal|low|medium|high|xhigh|max
timeout = 300

[providers.friendli]
# FriendliAI. API key via FRIENDLI_TOKEN (FRIENDLIAI_API_KEY also accepted).
# endpoint_type = "serverless"  # "serverless" | "dedicated" | "container"
# backend = "openai"            # "openai" (default) | "httpx" | "sdk"
default_model = "zai-org/GLM-5.3"
parse_reasoning = true          # split reasoning into reasoning_content
# reasoning_effort = "high"     # minimal|low|medium|high|xhigh|max|ultracode
timeout = 300

[providers.typesafe]
# TypeSafe.ai System One (typed judgments, NOT chat). API key via TYPESAFE_API_KEY.
# Use provider.system_one(state, questions) or llm.chat(..., provider_name="typesafe", questions={...}).
default_model = "jev-latest"    # alias -> jev-1.13.0; pin the version if you tune thresholds
timeout = 30
max_retries = 2                 # 408/429/5xx/529 retried, honours Retry-After

[storage.session]
type = "sqlite"
path = "~/.llmcore/sessions.db"

[storage.vector]
type = "chromadb"
path = "~/.llmcore/chroma_db"
default_collection = "llmcore_default"

[agents]
max_iterations = 10
default_timeout = 600

[agents.sandbox]
mode = "docker"

[agents.sandbox.docker]
enabled = true
image = "python:3.11-slim"
memory_limit = "1g"
cpu_limit = 2.0
network_enabled = false

[agents.hitl]
enabled = true
global_risk_threshold = "medium"
default_timeout_seconds = 300
```

### Environment Variables

```bash
# Provider API Keys
export LLMCORE_PROVIDERS__OPENAI__API_KEY="sk-..."
export LLMCORE_PROVIDERS__ANTHROPIC__API_KEY="sk-ant-..."
export LLMCORE_PROVIDERS__GEMINI__API_KEY="..."

# Or use standard provider env vars
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Storage
export LLMCORE_STORAGE__SESSION__TYPE="postgres"
export LLMCORE_STORAGE__SESSION__DB_URL="postgresql://user:pass@localhost/llmcore"

# Logging
export LLMCORE_LOG_LEVEL="DEBUG"
export LLMCORE_LOG_RAW_PAYLOADS="true"
export TYPESAFE_API_KEY="..."          # TypeSafe.ai System One
```

---

## 🔌 Providers

23 providers behind one interface. **Transport** shows which can call the vendor
API directly and which fall back to a vendor SDK — llmcore prefers direct calls
so an SDK lagging the API does not block you, and keeps the SDK where it owns
something non-trivial. Full detail in
[`PROVIDER_SUPPORT_MATRIX.md`](docs/PROVIDER_SUPPORT_MATRIX.md).

### Chat and reasoning

| Provider | Representative models *(from the card registry)* | Transport | Notable |
|---|---|---|---|
| **OpenAI** | `gpt-6-sol`, `gpt-6-astra`, `gpt-5.6-terra`, `gpt-5.5-pro`, `gpt-5.4`, `o4-mini` | direct + SDK | Streaming, tools, vision, images, TTS/ASR, embeddings, native search |
| **Anthropic** | `claude-opus-5-5`, `claude-opus-5`, `claude-sonnet-5-5`, `claude-opus-4-8`, `claude-haiku-4-5-20251001` | SDK + direct | Streaming, tools, vision, extended thinking |
| **Google Gemini** | `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.1-pro`, `gemini-2.5-pro` | SDK + direct | Tools, vision, Imagen, **Veo video**, native TTS, embeddings |
| **xAI** | `grok-4.1-20251117`, `grok-4-heavy` | direct + SDK | Streaming, tools, Live Search |
| **DeepSeek** | `deepseek-v4-pro`, `deepseek-v4-flash`, `deepseek-v3.2`, `deepseek-reasoner` | direct only¹ | Reasoning-content extraction, cache usage |
| **Kimi (Moonshot)** | `kimi-k3`, `kimi-k2.7-code`, `kimi-k2.6`, `kimi-k2-thinking` | direct only¹ | Reasoning, exact tokenizer |
| **Z.ai (GLM)** | `glm-5.3`, `glm-5.2`, `glm-5.1`, `glm-4.7`, `glm-5.3-flash` | SDK → direct | Tools, vision, image/video, TTS/ASR/OCR, embeddings, web search |
| **Mistral** | `mistral-large-3`, `mistral-large-2512`, `magistral-medium-latest`, `magistral-small-latest` | direct + SDK | Tools, vision, FIM, OCR, embeddings, audio |
| **Qwen** | `qwen3-max`, `qwen3-coder-480b` | direct | Streaming, tools |
| **Groq** | Llama, Qwen, Whisper, Kimi on LPU hardware | direct + SDK | Low-latency inference |
| **Together** | Open-weights catalog (Llama, Qwen, DeepSeek, FLUX) | direct + SDK | Chat, images, embeddings |
| **FriendliAI** | `zai-org/GLM-5.3`, `deepseek-ai/DeepSeek-V3.2` + your dedicated endpoints | direct → SDK | Reasoning effort/budget, regex-constrained output, exact tokenizer |
| **DeepInfra** | `Qwen3-235B-A22B`, DeepSeek, Llama, FLUX, Whisper, Kokoro | direct + SDK | Chat, vision, TTS/ASR, images, embeddings |
| **OpenRouter** | 620 cards spanning most vendors | direct + SDK | One key, many vendors |
| **Poe** | 497 cards across Anthropic/OpenAI/Google and others | direct + SDK | Aggregated access |
| **Hugging Face** | 305 cards; any Inference Provider model | SDK + direct | Chat, image, TTS, ASR, **private Inference Endpoints** |
| **Ollama** | `qwen3-vl:4b`, `llama3.3:70b`, `gemma3:12b`, `qwen3-embedding:8b` | SDK + direct | Local, no API key |
| **vLLM** | Anything you serve | direct + SDK | Self-hosted, guided grammars, structured output |

¹ No official vendor Python SDK exists — their own docs point at the `openai`
client, which llmcore already speaks. Verified against PyPI rather than assumed.

### Media, voice and typed judgment

| Provider | What it serves | Transport |
|---|---|---|
| **ElevenLabs** | TTS (+streaming), ASR, SFX, music, voice design — with voice-consent metadata | direct + SDK |
| **Deepgram** | STT (Nova-3, Flux), TTS (Aura-2), Voice Agent, text intelligence | SDK + direct |
| **fal** | 9 capabilities: FLUX image gen/edit/upscale, video, FILM interpolation, SFX, music, TTS, ASR | direct + SDK |
| **Replicate** | One generic prediction adapter driven by each model's published schema | direct + SDK |
| **Higgsfield** | Soul image generation, plus hosted Kling and MiniMax Hailuo video | direct + SDK |
| **TypeSafe.ai** | `jev-1.13.0` (alias `jev-latest`) — typed judgments: `noul` (yes/no probability), `choice`, `score` | direct + SDK |

> **Model names move fast.** The table shows what is in the bundled card
> registry at this release. `cardctl generate <provider>` refreshes it from the
> live API, and `llm.list_models()` reports what your keys can actually reach —
> prefer that over anything written here.

### Switching Providers

```python
# Use default provider
response = await llm.chat("Hello!")

# Override per-request
response = await llm.chat(
    "Explain quantum computing",
    provider_name="anthropic",
    model_name="claude-sonnet-5-5"
)

# With provider-specific parameters
response = await llm.chat(
    "Write a poem",
    provider_name="openai",
    model_name="gpt-5.4",
    temperature=0.9,
    max_tokens=500
)
```

### Voice & Audio (Deepgram)

The **Deepgram** provider adds real-time **voice/audio** — speech-to-text (STT),
text-to-speech (TTS), conversational **Flux** STT, a bidirectional **Voice
Agent** (STT → LLM → TTS over one socket), and **text intelligence**. Deepgram
is not a chat-completion provider, so its media methods are called **directly on
the provider instance** (the `LLMCore` facade has no audio methods):

```python
from llmcore.providers.deepgram_provider import DeepgramProvider

dg = DeepgramProvider({"api_key": "dg_...", "_instance_name": "deepgram"})

# Pre-recorded STT
result = await dg.transcribe_audio(open("call.wav", "rb").read(),
                                   model="nova-3", smart_format=True, diarize=True)
print(result.text)

# TTS
speech = await dg.generate_speech("Hello from Aura.", voice="aura-2-thalia-en")
open("out.mp3", "wb").write(speech.audio_data)

# Live STT (async byte source -> streamed events)
async for ev in dg.transcribe_stream(mic_frames(), model="nova-3",
                                      encoding="linear16", sample_rate=16000,
                                      interim_results=True):
    print(ev.is_final, ev.text)

# Voice Agent (auto-answers client-side tool calls; prompt is never defaulted)
async for event in dg.run_voice_agent(mic_audio(), function_handler=handle,
                                       functions=[weather_fn],
                                       prompt="You are a concise assistant."):
    ...   # CONVERSATION_TEXT / AUDIO / FUNCTION_CALL_REQUEST events

await dg.close()
```

See **[`docs/Deepgram_provider_usage.md`](docs/Deepgram_provider_usage.md)** for
the full guide (config, streaming, Flux, Voice Agent settings shape, token auth)
and the runnable scripts in [`examples/`](examples) (`deepgram_*.py`).

---

## 🧠 Agent System (Darwin Layer 2)

The agent system implements an advanced cognitive architecture for autonomous task execution.

### Cognitive Cycle

The 8-phase cognitive cycle enables sophisticated reasoning:

```
┌───────────────────────────────────────────────────────────────────┐
│                        COGNITIVE CYCLE                            │
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │ PERCEIVE │ →  │   PLAN   │ →  │  THINK   │ →  │ VALIDATE │     │
│  │          │    │          │    │          │    │          │     │
│  │ Analyze  │    │ Generate │    │ Reason & │    │ Check    │     │
│  │ goal &   │    │ strategy │    │ decide   │    │ validity │     │
│  │ context  │    │          │    │ action   │    │          │     │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│       ↑                                               │           │
│       │                                               ↓           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │  UPDATE  │ ←  │ REFLECT  │ ←  │ OBSERVE  │ ←  │   ACT    │     │
│  │          │    │          │    │          │    │          │     │
│  │ Update   │    │ Learn &  │    │ Analyze  │    │ Execute  │     │
│  │ state &  │    │ improve  │    │ results  │    │ action   │     │
│  │ memory   │    │          │    │          │    │          │     │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
└───────────────────────────────────────────────────────────────────┘
```

**Phase Descriptions:**

| Phase | Purpose |
|-------|---------|
| **PERCEIVE** | Analyze goal, extract entities, assess complexity |
| **PLAN** | Generate execution strategy and select approach |
| **THINK** | Reason about next action using CoT/ReAct patterns |
| **VALIDATE** | Check proposed action validity and safety |
| **ACT** | Execute the chosen action (tool call, code, etc.) |
| **OBSERVE** | Analyze results and extract observations |
| **REFLECT** | Learn from outcome, identify improvements |
| **UPDATE** | Update working memory and iteration state |

### Goal Classification

Automatic goal complexity assessment for optimal routing:

```python
from llmcore.agents import GoalClassifier, classify_goal

# Classify a goal
classification = classify_goal("What's 2 + 2?")
print(classification.complexity)  # GoalComplexity.TRIVIAL
print(classification.execution_strategy)  # ExecutionStrategy.FAST_PATH

# Complex goal
classification = classify_goal(
    "Research and compare the top 5 cloud providers, "
    "analyze their pricing, and create a recommendation report"
)
print(classification.complexity)  # GoalComplexity.COMPLEX
print(classification.max_iterations)  # 15
```

**Complexity Levels:**

| Level | Max Iterations | Examples |
|-------|----------------|----------|
| `TRIVIAL` | 1 | Greetings, simple math, factual Q&A |
| `SIMPLE` | 5 | Single-step tasks, translations |
| `MODERATE` | 10 | Multi-step tasks, analysis |
| `COMPLEX` | 15 | Research, multi-source synthesis |

### Fast-Path Execution

Bypass the full cognitive cycle for trivial goals (sub-5s responses):

```python
from llmcore.agents.learning import FastPathExecutor, should_use_fast_path

# Check if fast-path is appropriate
if should_use_fast_path("Hello, how are you?"):
    executor = FastPathExecutor(config=fast_path_config)
    result = await executor.execute(goal="Hello!")
    print(result.response)  # Instant response
```

### Circuit Breaker

Automatic detection and interruption of failing agent loops:

```python
from llmcore.agents import AgentCircuitBreaker, CircuitBreakerConfig

breaker = AgentCircuitBreaker(CircuitBreakerConfig(
    max_iterations=15,
    max_same_errors=3,
    max_execution_time_seconds=300,
    max_total_cost=1.0,
    progress_stall_threshold=5
))

# Circuit breaker trips on:
# - Maximum iterations exceeded
# - Repeated identical errors
# - Timeout exceeded
# - Cost limit exceeded
# - Progress stall detected
```

### Human-in-the-Loop (HITL)

Interactive approval workflows for sensitive operations:

```python
from llmcore.agents.hitl import HITLManager, HITLConfig, ConsoleHITLCallback

# Create HITL manager
hitl = HITLManager(
    config=HITLConfig(
        enabled=True,
        global_risk_threshold="medium",
        timeout_policy="reject"
    ),
    callback=ConsoleHITLCallback()  # Interactive console prompts
)

# Check if activity needs approval
decision = await hitl.check_approval(
    activity_type="execute_shell",
    parameters={"command": "rm -rf /tmp/test"}
)

if decision.is_approved:
    # Execute the activity
    pass
else:
    print(f"Rejected: {decision.reason}")
```

**Risk Levels:**

| Level | Requires Approval | Examples |
|-------|-------------------|----------|
| `LOW` | No | Read files, calculations |
| `MEDIUM` | Configurable | Write files, API calls |
| `HIGH` | Yes | Delete files, network access |
| `CRITICAL` | Always | System commands, credentials |

### Persona System

Customize agent behavior and communication style:

```python
from llmcore.agents import PersonaManager, AgentPersona, PersonalityTrait

# Create a custom persona
persona = AgentPersona(
    name="DataAnalyst",
    description="A meticulous data analyst focused on accuracy",
    personality=[
        PersonalityTrait.ANALYTICAL,
        PersonalityTrait.METHODICAL,
        PersonalityTrait.CAUTIOUS
    ],
    communication_style="formal",
    risk_tolerance="low",
    planning_depth="thorough"
)

# Apply to agent
manager = PersonaManager()
manager.register_persona(persona)
agent_state.persona = manager.get_persona("DataAnalyst")
```

---

## 🐳 Sandbox System

Secure, isolated execution environments for agent-generated code.

### Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         Sandbox Registry                           │
│                    (Manages sandbox lifecycle)                     │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────────────────┐    ┌─────────────────────────┐        │
│  │    Docker Provider      │    │     VM Provider         │        │
│  │                         │    │                         │        │
│  │  • Container isolation  │    │  • SSH-based access     │        │
│  │  • Image management     │    │  • Full VM isolation    │        │
│  │  • Resource limits      │    │  • Network separation   │        │
│  │  • Output capture       │    │  • Persistent storage   │        │
│  └─────────────────────────┘    └─────────────────────────┘        │
│                                                                    │
│  ┌─────────────────────────┐    ┌─────────────────────────┐        │
│  │    Output Tracker       │    │   Ephemeral Manager     │        │
│  │                         │    │                         │        │
│  │  • File lineage         │    │  • Resource cleanup     │        │
│  │  • Execution logs       │    │  • Timeout handling     │        │
│  │  • Artifact collection  │    │  • State management     │        │
│  └─────────────────────────┘    └─────────────────────────┘        │
└────────────────────────────────────────────────────────────────────┘
```

### Usage

```python
from llmcore import (
    SandboxRegistry, DockerSandboxProvider, SandboxConfig, SandboxMode
)

# Create sandbox registry
registry = SandboxRegistry()

# Register Docker provider
docker_provider = DockerSandboxProvider(SandboxConfig(
    mode=SandboxMode.DOCKER,
    image="python:3.11-slim",
    memory_limit="1g",
    cpu_limit=2.0,
    timeout_seconds=300,
    network_enabled=False
))
registry.register_provider("docker", docker_provider)

# Create and use sandbox
async with registry.create_sandbox("docker") as sandbox:
    # Execute Python code
    result = await sandbox.execute_python("""
import math
print(f"Pi = {math.pi}")
    """)
    print(result.stdout)  # "Pi = 3.141592653589793"
    
    # Execute shell command
    result = await sandbox.execute_shell("ls -la")
    print(result.stdout)
    
    # Save file
    await sandbox.save_file("output.txt", "Hello, Sandbox!")
    
    # Read file
    content = await sandbox.load_file("output.txt")
```

### Container Images

Pre-built, security-hardened images organized by tier:

| Tier | Image | Description |
|------|-------|-------------|
| **Base** | `llmcore-sandbox-base:1.0.0` | Minimal Ubuntu 24.04 |
| **Specialized** | `llmcore-sandbox-python:1.0.0` | Python 3.12 development |
| | `llmcore-sandbox-nodejs:1.0.0` | Node.js 22 development |
| | `llmcore-sandbox-shell:1.0.0` | Shell scripting |
| **Task** | `llmcore-sandbox-research:1.0.0` | Research & analysis |
| | `llmcore-sandbox-websearch:1.0.0` | Web scraping |

### Access Levels

| Level | Network | Filesystem | Tools |
|-------|---------|------------|-------|
| `RESTRICTED` | Blocked | Limited | Whitelisted only |
| `FULL` | Enabled | Extended | All tools |

### Security Features

- **Non-root execution**: All containers run as `sandbox` user (UID 1000)
- **No SUID/SGID binaries**: Privilege escalation vectors removed
- **Resource limits**: Memory, CPU, and process limits enforced
- **Network isolation**: Optional network blocking
- **Output tracking**: Full lineage and audit trail
- **AppArmor/seccomp ready**: Compatible with security profiles

---

## 📚 Model cards

**2,319 cards across 22 providers**, generated from live provider APIs and
bundled with the package. Each carries context window, capability flags, pricing
and lifecycle — so capability and cost questions are answered locally instead of
by trial and error against a paid endpoint.

### Keeping them current

```bash
python -m tools.cardctl doctor            # audit coverage — run this first
python -m tools.cardctl generate openai   # refresh one provider from its API
python -m tools.cardctl validate          # schema-check every card
python -m tools.cardctl diff anthropic    # read-only: local cards vs live API
python -m tools.cardctl stats             # coverage dashboard
```

`doctor` cross-checks llmcore's provider registry against cardctl's adapters and
the cards on disk. It exists because three media providers once shipped with no
adapter at all and nothing complained — `generate` only reports on the provider
you name, and `stats` only sees providers that already have cards.

Providers with no catalog endpoint (fal, Higgsfield, Replicate, vLLM) use a
*curated* adapter and every such card is tagged `curated`, so a declared entry
is never mistaken for a discovered one.

### Usage

```python
from llmcore import get_model_card_registry, get_model_card

# Get registry singleton
registry = get_model_card_registry()

# Lookup model card
card = registry.get("openai", "gpt-5.4")
print(f"Context: {card.get_context_length():,} tokens")
print(f"Vision: {card.capabilities.vision}")
print(f"Tools: {card.capabilities.tools}")

# Cost estimation
cost = card.estimate_cost(
    input_tokens=50_000,
    output_tokens=2_000,
    cached_tokens=10_000
)
print(f"Estimated cost: ${cost:.4f}")

# List models by capability
vision_models = registry.list_cards(tags=["vision"])
for model in vision_models:
    print(f"{model.provider}/{model.model_id}")

# Alias resolution
card = registry.get("anthropic", "claude-4.5-sonnet")  # Resolves alias
```

### Cards per provider

| Provider | Cards | | Provider | Cards |
|---|---:|---|---|---:|
| openrouter | 620 | | mistral | 66 |
| poe | 497 | | anthropic | 19 |
| huggingface | 305 | | kimi | 16 |
| deepinfra | 229 | | zai | 13 |
| ollama | 176 | | elevenlabs | 11 |
| deepgram | 147 | | fal | 9 |
| openai | 133 | | deepseek / friendli / replicate | 7 each |
| google | 44 | | higgsfield | 6 |
| | | | qwen · xai · typesafe | 3 · 3 · 1 |
- **xAI**: Grok-4, Grok-4-Heavy
- **DeepInfra**: DeepSeek-V3/R1, Llama 3.x, Qwen, Mistral, FLUX (image), Whisper (STT), Kokoro (TTS), embeddings
- **Deepgram**: Nova-3, Nova-2, Whisper, Flux (STT); Aura-2 (TTS); Voice Agent
- **TypeSafe.ai**: Jev 1.13 (`decision` model type; aliases `jev-latest`, `jev-preview`)

### Custom Model Cards

Add custom cards in `~/.config/llmcore/model_cards/<provider>/<model>.json`:

```json
{
  "model_id": "my-custom-model",
  "display_name": "My Custom Model",
  "provider": "ollama",
  "model_type": "chat",
  "context": {
    "max_input_tokens": 32768,
    "max_output_tokens": 4096
  },
  "capabilities": {
    "streaming": true,
    "tools": false,
    "vision": false
  }
}
```

---

## 📊 Observability

Comprehensive monitoring and debugging for agent executions.

### Event Logging

```python
from llmcore.agents.observability import EventLogger, EventCategory

# Events logged to ~/.llmcore/events.jsonl
logger = EventLogger(log_path="~/.llmcore/events.jsonl")

# Event categories
# - LIFECYCLE: Agent start/stop
# - COGNITIVE: Phase execution
# - ACTIVITY: Tool executions
# - HITL: Human approvals
# - ERROR: Exceptions
# - METRIC: Performance data
# - MEMORY: Memory operations
# - SANDBOX: Container lifecycle
# - RAG: Retrieval operations
```

### Metrics Collection

```python
from llmcore.agents.observability import MetricsCollector

collector = MetricsCollector()

# Available metrics
# - Iteration counts
# - LLM call latency (p50, p90, p95, p99)
# - Token usage (input/output)
# - Estimated costs
# - Activity execution times
# - Error counts by type
```

### Execution Replay

```python
from llmcore.agents.observability import ExecutionReplay

replay = ExecutionReplay(events_path="~/.llmcore/events.jsonl")

# List executions
executions = replay.list_executions()
for exec_id, metadata in executions.items():
    print(f"{exec_id}: {metadata['goal']}")

# Replay specific execution
events = replay.get_execution_events(exec_id)
for event in events:
    print(f"[{event.timestamp}] {event.category}: {event.type}")
```

### Configuration

```toml
[agents.observability]
enabled = true

[agents.observability.events]
enabled = true
log_path = "~/.llmcore/events.jsonl"
min_severity = "info"
categories = []  # Empty = all categories

[agents.observability.events.rotation]
strategy = "size"
max_size_mb = 100
max_files = 10
compress = true

[agents.observability.metrics]
enabled = true
track_cost = true
track_tokens = true
latency_percentiles = [50, 90, 95, 99]

[agents.observability.replay]
enabled = true
cache_enabled = true
cache_max_executions = 50
```

---

## 🔍 RAG System

Retrieval-Augmented Generation for knowledge-enhanced responses.

### Adding Documents

```python
# Add documents with metadata
await llm.add_documents_to_vector_store(
    documents=[
        {
            "content": "LLMCore is a Python library...",
            "metadata": {
                "source": "documentation",
                "version": "0.53.0",
                "category": "overview"
            }
        },
        {
            "content": "Configuration uses the confy library...",
            "metadata": {
                "source": "documentation",
                "category": "configuration"
            }
        }
    ],
    collection_name="project_docs"
)
```

### Semantic Search

```python
# Direct similarity search
results = await llm.search_vector_store(
    query="How does configuration work?",
    k=5,
    collection_name="project_docs",
    metadata_filter={"category": "configuration"}
)

for doc in results:
    print(f"Score: {doc.score:.4f}")
    print(f"Content: {doc.content[:100]}...")
    print(f"Metadata: {doc.metadata}")
```

### RAG-Enhanced Chat

```python
response = await llm.chat(
    "Explain how to configure providers",
    enable_rag=True,
    rag_collection_name="project_docs",
    rag_retrieval_k=3,
    system_message="Answer based ONLY on the provided context."
)
```

### External RAG Integration

LLMCore can serve as an LLM backend for external RAG engines:

```python
# External engine (e.g., semantiscan) handles retrieval
relevant_docs = await external_rag_engine.retrieve(query)

# Construct prompt with retrieved context
context = format_documents(relevant_docs)
full_prompt = f"Context:\n{context}\n\nQuestion: {query}"

# Use LLMCore for generation only
response = await llm.chat(
    message=full_prompt,
    enable_rag=False,  # Disable internal RAG
    explicitly_staged_items=[]  # Optional additional context
)
```

---

## 📖 API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `LLMCore` | Main facade for all LLM operations |
| `AgentManager` | Manages autonomous agent execution |
| `StorageManager` | Handles session and vector storage |
| `ProviderManager` | Manages LLM provider connections |

### Data Models

| Model | Description |
|-------|-------------|
| `ChatSession` | Conversation session with messages |
| `Message` | Individual chat message (user/assistant/system) |
| `ContextDocument` | Document for RAG/context |
| `Tool` | Function/tool definition |
| `ToolCall` | Tool invocation by LLM |
| `ToolResult` | Result of tool execution |
| `ModelCard` | Model metadata and capabilities |

### Exceptions

```python
from llmcore import (
    LLMCoreError,          # Base exception
    ConfigError,           # Configuration issues
    ProviderError,         # LLM provider errors
    StorageError,          # Storage operations
    SessionStorageError,   # Session storage specific
    VectorStorageError,    # Vector storage specific
    SessionNotFoundError,  # Session lookup failure
    ContextError,          # Context management
    ContextLengthError,    # Context exceeds limits
    EmbeddingError,        # Embedding generation
    SandboxError,          # Sandbox execution
    SandboxInitializationError,
    SandboxExecutionError,
    SandboxTimeoutError,
    SandboxAccessDenied,
    SandboxResourceError,
    SandboxConnectionError,
    SandboxCleanupError,
)
```

---

## 📚 Documentation

**Reference**

| Document | What it covers |
|---|---|
| [Configuration reference](docs/CONFIG_REFERENCE.md) | Every config key |
| [Model cards](docs/model_cards.md) | Card schema, the `cardctl` workflow, `doctor` |
| [`chat_with_usage` guide](docs/USAGE_chat_with_usage.md) | Per-call token and cost accounting |
| [Agentic system guide](docs/Agentic_System_Guide.md) | Cognitive cycle, tools, HITL |
| [External RAG integration](docs/External_RAG_integration_guide.md) | Bringing your own retrieval |

**Design and status**

| Document | What it covers |
|---|---|
| [Provider support matrix](docs/PROVIDER_SUPPORT_MATRIX.md) | Per provider: tracked SDK version and commit, transport duality, capability matrix, and a live-validation log recording what was actually called |
| [Provider modernization plan](docs/PROVIDER_MODERNIZATION_PLAN.md) | Phased plan for the remaining gaps in that matrix |
| [Media subsystem spec](docs/MEDIA_SUBSYSTEM_SPEC.md) | Design, the rollout, and what each vendor taught the abstraction |
| [Runtimes usage guide](docs/Runtimes_usage.md) | How to size, start, watch and stop remote GPU runtimes, and what it does not claim |
| [Remote runtime spec](docs/COLAB_RUNTIME_SPEC.md) | Runtime safety model and the Colab backend design |
| [Routing usage guide](docs/Routing_usage.md) | How to use routing: dynamic targets, pools, lanes, cascades, the privacy path, proxy mode, and what it does not claim |
| [Routing subsystem spec](docs/ROUTING_SUBSYSTEM_SPEC.md) | The design: five layers, the failure taxonomy, prior art borrowed, and the measured corrections to it |

**Per-provider guides**

[fal](docs/Fal_provider_usage.md) ·
[ElevenLabs](docs/ElevenLabs_provider_usage.md) ·
[Replicate](docs/Replicate_provider_usage.md) ·
[Hugging Face media](docs/HuggingFace_media_usage.md) ·
[Deepgram](docs/Deepgram_provider_usage.md) ·
[FriendliAI](docs/Friendli_provider_usage.md) ·
[TypeSafe.ai](docs/TypeSafe_provider_usage.md) ·
[Search providers](docs/Search_providers_usage.md)
([rationale](docs/Search_providers_rationale.md))

---

## 📁 Project Structure

```
llmcore/
├── src/llmcore/
│   ├── __init__.py           # Public API exports
│   ├── api.py                # Main LLMCore class
│   ├── models.py             # Core data models
│   ├── exceptions.py         # Exception hierarchy
│   ├── config/               # Configuration system
│   │   ├── default_config.toml
│   │   └── models.py
│   ├── providers/            # 23 provider adapters
│   │   ├── openai_provider.py      # base for deepinfra/vllm/poe/openrouter
│   │   ├── anthropic_provider.py
│   │   ├── gemini_provider.py
│   │   ├── fal_provider.py         # + elevenlabs, replicate, higgsfield,
│   │   └── ...                     #   deepgram, zai, friendli, mistral, …
│   ├── media/                # Generative media subsystem
│   │   ├── manager.py        # routing + capability discovery
│   │   ├── protocols.py      # 19 capability protocols
│   │   ├── jobs.py           # async job polling
│   │   ├── webhooks.py       # signed single-use callbacks
│   │   └── artifacts.py      # content-addressed store
│   ├── runtimes/             # Remote GPU runtimes
│   │   ├── manager.py        # spend ceilings, attach/detach
│   │   ├── protocols.py      # ComputeRuntime
│   │   └── state.py          # inspectable on-disk records
│   ├── storage/              # Storage backends
│   │   ├── sqlite_session.py
│   │   ├── postgres_session_storage.py
│   │   ├── chromadb_vector.py
│   │   └── pgvector_storage.py
│   ├── embedding/            # Embedding models
│   │   ├── sentence_transformer.py
│   │   ├── openai.py
│   │   └── google.py
│   ├── agents/               # Agent system
│   │   ├── manager.py        # AgentManager
│   │   ├── cognitive/        # 8-phase cognitive cycle
│   │   ├── hitl/             # Human-in-the-loop
│   │   ├── sandbox/          # Sandbox execution
│   │   ├── learning/         # Learning mechanisms
│   │   ├── persona/          # Persona system
│   │   ├── prompts/          # Prompt library
│   │   ├── observability/    # Monitoring & logging
│   │   └── routing/          # Model routing
│   ├── model_cards/          # Model metadata
│   │   ├── registry.py
│   │   ├── schema.py
│   │   └── default_cards/
│   └── memory/               # Memory management
├── tools/cardctl/            # Model-card generator (adapters + doctor)
├── container_images/         # Sandbox Docker images
├── examples/                 # Usage examples
├── tests/                    # Test suite
├── docs/                     # Documentation
└── pyproject.toml            # Project configuration
```

---

## 🧪 Testing

```bash
# The profile CI runs: no databases, containers, local servers or API keys
pytest tests --ignore=tests/integration --ignore=tests/adhoc_checks \
  -m "not slow and not integration and not docker and not vm and not sandbox \
      and not requires_postgres and not requires_pgvector and not requires_ollama"

# Everything (needs PostgreSQL + pgvector; Ollama for the local-model tests)
pytest

# Targeted
pytest tests/providers          # provider adapters and transports
pytest tests/media              # media subsystem
pytest tests/runtimes           # runtime safety model
pytest -m sandbox               # sandbox tests only
```

Infrastructure-dependent tests are marked and excluded by default, so a clean
checkout runs green without Postgres, Docker or a GPU. Tests that need those
report *why* they skipped rather than silently passing.

### Invariant guards

A few suites exist to make whole-repository mistakes fail loudly rather than
ship quietly:

| Guard | What it enforces |
|---|---|
| `tests/providers/test_transport_duality.py` | Every provider either offers both transports or declares why not. A declared SDK backend must actually be constructed *and called* — written after one shipped that silently did nothing |
| `tests/tools/test_cardctl_coverage.py` | Every registered provider has a `cardctl` adapter, so a new provider cannot ship without model cards |
| `tests/media/test_media_subsystem.py` | Every declared media capability is backed by its protocol |
| `tests/providers/test_context_length_error_mapping.py` | Static AST check that providers raise `ContextLengthError` with the right keywords |

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Write** tests for your changes
4. **Ensure** all tests pass (`pytest`)
5. **Follow** the existing code style (`ruff check .`)
6. **Commit** with conventional commits (`feat: add amazing feature`)
7. **Push** to your branch
8. **Open** a Pull Request

### Development Setup

```bash
git clone https://github.com/araray/llmcore.git
cd llmcore
pip install -e ".[dev]"
pre-commit install
```

### Code Style

- **Formatter**: Ruff
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style
- **Line Length**: 100 characters

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🔗 Related Projects

- **[llmchat](https://github.com/araray/llmchat)** - CLI interface for llmcore
- **[semantiscan](https://github.com/araray/semantiscan)** - Advanced RAG engine
- **[confy](https://github.com/araray/confy)** - Configuration management library

---

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/araray/llmcore/issues)
- **Discussions**: [GitHub Discussions](https://github.com/araray/llmcore/discussions)

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/araray">Araray Velho</a></sub>
</p>
