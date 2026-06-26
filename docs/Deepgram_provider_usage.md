# Deepgram Provider — Usage Guide

The **Deepgram** provider adds real-time **voice / audio** to llmcore:
speech‑to‑text (STT), text‑to‑speech (TTS), conversational **Flux** STT, a
bidirectional **Voice Agent** (STT → LLM → TTS over one socket), and **text
intelligence** (summary / topics / sentiment / intents). It wraps the official,
async‑first, WebSocket‑native [`deepgram-sdk`](https://github.com/deepgram/deepgram-python-sdk)
(v7.x).

> Deepgram is **not** a text chat‑completion provider. Calling `chat_completion`
> on it raises a clear `ProviderError` (HTTP 400, non‑retryable). Use the media
> methods documented below.

---

## 1. Installation

```bash
pip install "llmcore[deepgram]"
# pulls deepgram-sdk>=7.0.0 and websockets>=12.0
```

## 2. Credentials

The provider authenticates with either an **API key** or a short‑lived
**access token** (Bearer; takes precedence when both are present):

```bash
export DEEPGRAM_API_KEY="dg_..."        # searched automatically
```

or in `~/.config/llmcore/config.toml`:

```toml
[providers.deepgram]
api_key = "dg_..."            # or: access_token = "..."  (Bearer)
# api_key_env_var = "DEEPGRAM_API_KEY"   # override the env var name
```

Self‑hosted / on‑prem deployments can point each Deepgram surface at a custom
base URL:

```toml
[providers.deepgram]
base_url       = "https://dg.internal.example.com"        # listen/speak/read/manage
agent_base_url = "wss://dg-agent.internal.example.com"     # voice agent
```

## 3. Getting a provider instance

Media methods are **provider‑direct** (the high‑level `LLMCore` facade exposes
text/RAG methods, not audio). Two equivalent ways to obtain the provider:

```python
# A) Directly (most explicit; great for scripts/tests)
from llmcore.providers.deepgram_provider import DeepgramProvider

provider = DeepgramProvider({"api_key": "dg_...", "_instance_name": "deepgram"})

# B) Through a fully-initialized LLMCore app (uses packaged [providers.deepgram]
#    defaults + your config.toml). The manager is created during async init.
from llmcore import LLMCore

llm = await LLMCore.create()                 # or your app's factory
provider = llm._provider_manager.get_provider("deepgram")
```

Always `await provider.close()` when finished (it closes the underlying HTTP
client); streaming/agent context managers tear their sockets down on exit.

---

## 4. Batch (pre‑recorded) STT — `transcribe_audio`

```python
transcribe_audio(
    audio_data: bytes | str, *,           # raw bytes OR a path to an audio file
    model: str | None = None,             # default: nova-3
    language: str | None = None,          # "en", "multi", ...
    prompt: str | None = None,            # nova-3 keyterm prompting
    response_format: str = "json",        # accepted for interface parity
    **kwargs,                             # any Deepgram param; plus url=<str>
) -> TranscriptionResult
```

```python
result = await provider.transcribe_audio(
    open("call.wav", "rb").read(),
    model="nova-3", language="en",
    smart_format=True, punctuate=True, diarize=True,
    prompt=["Wairu", "Convergence"],          # boost domain terms
)
print(result.text, result.duration_seconds, len(result.segments))
```

Transcribe a **remote URL** (no local download) via the `url=` keyword:

```python
result = await provider.transcribe_audio(b"", url="https://dpgr.am/spacewalk.wav",
                                          model="nova-3", smart_format=True)
```

`TranscriptionResult` → `.text`, `.language`, `.duration_seconds`,
`.segments[]` (`TranscriptionSegment{text,start,end,speaker}`), `.model`,
`.metadata`. `redact=[...]` is comma‑joined for the SDK automatically.

## 5. Batch TTS — `generate_speech`

```python
generate_speech(
    text: str, *,
    voice: str = "aura-2-thalia-en",      # voice is passed as the model
    model: str | None = None,             # alias for voice if provided
    response_format: str = "mp3",         # mp3 | linear16 | wav | opus | flac | ...
    speed: float = 1.0,
    **kwargs,                             # e.g. sample_rate=24000
) -> SpeechResult
```

```python
speech = await provider.generate_speech("Hello from Aura.",
                                         voice="aura-2-thalia-en",
                                         response_format="mp3")
open("out.mp3", "wb").write(speech.audio_data)
```

`SpeechResult` → `.audio_data` (bytes), `.format`, `.model`, `.voice`,
`.duration_seconds`, `.metadata`.

## 6. Text intelligence — `analyze_text`

```python
analyze_text(
    text: str | None = None, *,           # exactly one of text / url
    url: str | None = None,
    summarize: bool | str | None = None,
    topics: bool | None = None,
    sentiment: bool | None = None,
    intents: bool | None = None,
    language: str | None = None,
    **kwargs,
) -> TextAnalysisResult
```

```python
analysis = await provider.analyze_text(transcript, summarize=True, topics=True,
                                        sentiment=True, intents=True)
print(analysis.summary)
for seg in analysis.topics:  print(seg.get("topics"))
for seg in analysis.intents: print(seg.get("intents"))
print(analysis.sentiments)
```

`TextAnalysisResult` → `.summary` (str|None), `.topics` (list[dict]),
`.intents` (list[dict]), `.sentiments` (dict|None), `.language`, `.model`,
`.request_id`, `.metadata`, `.raw`.

---

## 7. Live (streaming) STT — `transcribe_stream`

One‑call fan‑in/out: pass an **async byte source**, receive
`TranscriptionStreamEvent`s. Audio is pumped concurrently; the socket is always
closed on completion (a `Finalize`/`CloseStream` is sent), and producer errors
are surfaced as `ProviderError`.

```python
async for ev in provider.transcribe_stream(
    mic_frames(),                          # AsyncIterable[bytes]
    model="nova-3", language="en",
    encoding="linear16", sample_rate=16000,
    interim_results=True, smart_format=True,
    # endpointing=300, utterance_end_ms=1000, vad_events=True,
    # keepalive_interval=8.0, finalize_on_close=True,
):
    if ev.type in (StreamEventType.INTERIM, StreamEventType.FINAL):
        print(ev.is_final, ev.text)
    elif ev.type == StreamEventType.UTTERANCE_END:
        print("utterance end @", ev.end)
```

`TranscriptionStreamEvent` → `.type` (`StreamEventType`), `.text`, `.is_final`,
`.speech_final`, `.start`, `.end`, `.confidence`, `.words` (list[dict]),
`.speaker`, `.channel_index`, `.raw`.

**Manual socket** (push/pull yourself):

```python
async with provider.open_transcription_socket(model="nova-3",
                                               encoding="linear16",
                                               sample_rate=16000) as stream:
    await stream.send_audio(chunk)
    await stream.finalize()                # flush a segment
    async for ev in stream: ...
    # stream.close() is implicit on context exit
```

## 8. Conversational STT — Flux (`transcribe_stream_flux`)

Flux (listen.v2) is **turn‑aware**: instead of interim/final hypotheses it emits
`StartOfTurn` / `EagerEndOfTurn` / `EndOfTurn` with end‑of‑turn confidence —
ideal for agent turn‑taking.

```python
async for ev in provider.transcribe_stream_flux(
    mic_frames(), model="flux-general-en",
    encoding="linear16", sample_rate=16000,
    eot_threshold=0.7,                     # eager_eot_threshold=, eot_timeout_ms=
):
    if ev.type == StreamEventType.END_OF_TURN:
        print("turn done:", ev.text, "conf=", ev.confidence)
```

For Flux events, `.start` = audio‑window start, `.end` = audio‑window end,
`.confidence` = end‑of‑turn confidence, `.is_final` = (event is `EndOfTurn`).
`open_flux_socket(...)` gives the manual `DeepgramFluxStream` (with
`send_audio` / `configure` / `close`).

## 9. Live TTS — `stream_speech` (dual‑mode)

```python
# (a) single string  -> REST streaming endpoint yields audio chunks
async for audio in provider.stream_speech("Hello there.",
                                           model="aura-2-thalia-en",
                                           response_format="mp3"):
    play(audio)

# (b) async iterable of text -> a TTS WebSocket; each piece is sent + flushed
async for audio in provider.stream_speech(text_pieces(),         # AsyncIterable[str]
                                           model="aura-2-thalia-en",
                                           response_format="linear16",
                                           sample_rate=24000):
    play(audio)
```

`open_speech_socket(...)` gives the manual `DeepgramSpeechStream`
(`send_text(str)` / `flush()` / `clear()` / `close()`; iterate for audio bytes).

---

## 10. Voice Agent — `run_voice_agent` / `open_voice_agent`

The Voice Agent runs the entire conversational loop (STT → LLM → TTS) on one
socket. **No system prompt is ever defaulted** — pass `prompt=` explicitly (or
inject from grimoire upstream).

### Settings shape (important)

`audio` is a **top‑level** Settings key; `listen` / `think` / `speak` /
`greeting` / `context` nest **under `agent`**. Each of `listen` / `think` /
`speak` wraps a **`provider`** object; `think.prompt` and `think.functions` are
**siblings of** `think.provider`:

```toml
[providers.deepgram.agent.audio.input]
encoding = "linear16"
sample_rate = 24000
[providers.deepgram.agent.audio.output]
encoding = "linear16"
sample_rate = 24000

[providers.deepgram.agent.listen.provider]
type = "deepgram"
model = "nova-3"

[providers.deepgram.agent.think.provider]    # LLM leg (Deepgram-hosted or BYO)
type = "open_ai"
model = "gpt-4o-mini"
# [providers.deepgram.agent.think.endpoint]  # BYO endpoint
# url = "https://api.openai.com/v1/chat/completions"
# headers = { authorization = "Bearer ${OPENAI_API_KEY}" }

[providers.deepgram.agent.speak.provider]
type = "deepgram"
model = "aura-2-thalia-en"
```

`prompt` / `functions` / `greeting` passed to the methods are merged into
`agent.think.prompt`, `agent.think.functions`, and `agent.greeting`.
**Precedence** (low → high): config defaults → `**kwargs` (deep‑merged) →
`prompt`/`functions`/`greeting` → an explicit `settings=` dict (deep‑merged
last, so it always wins).

### High‑level driver

```python
WEATHER = {"name": "get_weather", "description": "...",
           "parameters": {"type": "object",
                          "properties": {"city": {"type": "string"}},
                          "required": ["city"]}}

async def handle(call):                      # call: VoiceAgentFunctionCall
    return json.dumps({"city": call.arguments["city"], "temp_f": 72})

async for event in provider.run_voice_agent(
    mic_audio(),                             # AsyncIterable[bytes] (or None)
    function_handler=handle,                 # auto-answers FunctionCallRequest
    on_event=lambda e: None,                 # side-channel hook (sync or async)
    functions=[WEATHER],
    greeting="Hi! How can I help?",
    prompt="You are a concise voice assistant.",
):
    if event.type == VoiceAgentEventType.CONVERSATION_TEXT:
        print(event.role, event.content)
    elif event.type == VoiceAgentEventType.AUDIO:
        play(event.audio)                    # agent TTS bytes
```

`VoiceAgentEvent` → `.type` (`VoiceAgentEventType`), `.role`, `.content`,
`.audio` (bytes|None), `.function_call` (`VoiceAgentFunctionCall`), `.raw`.
For `FUNCTION_CALL_REQUEST`, **all** functions are in `event.raw["functions"]`;
`event.function_call` is the first, for convenience.

### Low‑level session (manual control)

```python
async with provider.open_voice_agent(prompt="...", functions=[WEATHER]) as s:
    await s.send_audio(chunk)
    await s.inject_user_message("turn on dark mode")
    await s.inject_agent_message("Done!")
    await s.update_prompt("Be even more concise.")
    await s.update_think({"provider": {"type": "open_ai", "model": "gpt-4o"}})
    await s.update_speak({"provider": {"type": "deepgram", "model": "aura-asteria-en"}})
    await s.keepalive()
    async for event in s:
        if event.type == VoiceAgentEventType.FUNCTION_CALL_REQUEST:
            for fn in event.raw["functions"]:
                out = await handle(...)       # build VoiceAgentFunctionCall
                await s.respond_to_function_call(fn["id"], fn["name"], out)
```

The agent protocol has **no close frame**; the session is torn down on context
exit.

---

## 11. Token auth & account

```python
tok = await provider.grant_token(ttl_seconds=30)     # mint a temp key for a browser
# -> {"access_token": "...", "expires_in": 30.0, "raw": {...}}

projects = await provider.get_projects()             # list visible projects
```

The full management API (balances, usage, keys, members, …) is available via the
escape hatch: `provider.client.manage.v1...`.

## 12. Escape hatch

```python
raw_client = provider.client          # the underlying AsyncDeepgramClient
```

---

## 13. Tokens, context length & errors

* **Billing is per audio‑minute (STT) / per‑character (TTS)** — there is no token
  context window. `count_tokens(text)` returns a documented **character count**
  heuristic and `get_max_context_length()` returns a configurable nominal value
  (`fallback_context_length`, default 2000 = the REST‑TTS input character cap).
  These are **not** billing units.
* All Deepgram `ApiError`s are mapped to `ProviderError` with `status_code`,
  `retryable` (5xx/429 → retryable), `retry_after_seconds`, and `headers`
  preserved.

## 14. Models & pricing

Shipped model cards (`provider="deepgram"`):

| Type | Models |
|------|--------|
| STT  | `nova-3`, `nova-3-medical`, `nova-2`, `nova-2-phonecall`, `whisper-large`, `flux-general-en` |
| TTS  | `aura-2-thalia-en`, `aura-2-andromeda-en`, `aura-2-apollo-en`, `aura-asteria-en`, `aura-luna-en` |

```python
from llmcore.model_cards.registry import get_model_card_registry
reg = get_model_card_registry()
reg.list_cards(provider="deepgram")
card = reg.get(provider="deepgram", model_id="nova-3")
card.provider_extension["pricing"]   # -> rate_per_minute, source, as_of, ...
```

### Pricing

Deepgram bills **per audio-minute (STT)** and **per character (TTS)** — not per
token. llmcore's typed `pricing` field is token-centric (and its `get_cost()` is
token-based), so the cards keep `pricing = null` and record the real
pay-as-you-go rates in **`card.provider_extension["pricing"]`** with correct
units, `source`, and `as_of` date. Only rates actually published on Deepgram's
pricing page are given as numbers; models the page doesn't price are marked
`"status": "not_listed"` (no fabricated figure), and any inferred mapping is
flagged `"status": "inferred"` + `"assumption"`.

Rates below were captured from <https://deepgram.com/pricing> on **2026-06-25**
(pay-as-you-go; Growth = prepaid). Streaming STT currently has limited-time
promotional pricing, so STT cells show the standard rate with the promotional
streaming rate noted. **Always treat the live page as authoritative.**

**Speech-to-Text (USD / audio-minute)**

| Card | Standard | Promo (streaming) | Growth |
|------|---------:|------------------:|-------:|
| `flux-general-en` (Flux English) | $0.0077 | $0.0065 | $0.0057–0.0065 |
| `nova-3` (Nova-3 monolingual) | $0.0077 | $0.0048 | $0.0042–0.0065 |
| `nova-3-medical` *(inferred = Nova-3)* | $0.0077 | $0.0048 | $0.0042–0.0065 |
| `nova-2`, `nova-2-phonecall` | not listed | — | — |
| `whisper-large` | not listed | — | — |

Multilingual Nova-3 (not a separate card) is $0.0058–$0.0092/min PAYG; Flux
Multilingual is $0.0078/min PAYG. STT add-ons (PAYG): Redaction $0.0020/min,
Keyterm Prompting $0.0013/min, Diarization $0.0020/min, Smart Formatting included.

**Text-to-Speech (USD / 1k characters)**

| Card | PAYG | Growth | per 1M chars |
|------|-----:|-------:|-------------:|
| `aura-2-*` (Thalia/Andromeda/Apollo) | $0.030 | $0.027 | $30.00 |
| `aura-asteria-en`, `aura-luna-en` (Aura-1) | $0.0150 | $0.0135 | $15.00 |

**Voice Agent (USD / minute, PAYG)** — not a model card; configured via the
`agent` settings:

| Tier | PAYG | Growth |
|------|-----:|-------:|
| Standard | $0.075 | $0.068 |
| Standard – BYO TTS | $0.065 | $0.051 |
| Custom – BYO LLM | — | $0.059 |
| Custom – BYO LLM + TTS | $0.050 | $0.041 |
| Advanced | $0.163 | $0.146 |
| Advanced – BYO TTS | $0.122 | $0.110 |

**Audio Intelligence** (`analyze_text`) — Summarization $0.0003/1k input tokens +
$0.0006/1k output tokens (PAYG); Topic/Sentiment/Intent included with it.

> All rates opt into Deepgram's Model Improvement Program (reflected as
> `model_improvement_program: true` in the STT records).

## 15. Examples

Runnable scripts in [`examples/`](../examples):

| File | Demonstrates |
|------|--------------|
| `deepgram_batch_stt.py`         | bytes + URL pre‑recorded STT |
| `deepgram_batch_tts.py`         | Aura TTS → file |
| `deepgram_streaming_stt.py`     | live STT (self‑contained TTS→PCM→STT) |
| `deepgram_flux.py`              | Flux turn events |
| `deepgram_streaming_tts.py`     | REST string + WebSocket iterable TTS |
| `deepgram_text_intelligence.py` | summary / topics / sentiment / intents |
| `deepgram_voice_agent.py`       | end‑to‑end Voice Agent + client‑side tool |

## 16. References

1. Deepgram docs — https://developers.deepgram.com/ (append `.md` for Markdown)
2. Pre‑recorded STT — https://developers.deepgram.com/docs/pre-recorded-audio
3. TTS (REST) — https://developers.deepgram.com/docs/text-to-speech
4. Streaming STT — https://developers.deepgram.com/docs/live-streaming-audio
5. Streaming TTS — https://developers.deepgram.com/docs/streaming-text-to-speech
6. Flux — https://developers.deepgram.com/docs/flux/quickstart
7. Voice Agent — https://developers.deepgram.com/docs/voice-agent
8. Text Intelligence — https://developers.deepgram.com/docs/text-intelligence
9. Token‑based auth — https://developers.deepgram.com/docs/token-based-authentication
10. Python SDK — https://github.com/deepgram/deepgram-python-sdk
