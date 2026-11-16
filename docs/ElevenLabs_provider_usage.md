# ElevenLabs provider — usage guide

[ElevenLabs](https://elevenlabs.io) is the deepest *voice* vendor llmcore
reaches. In llmcore it is a **media-only provider** — it serves no chat API, and
`chat_completion()` raises with a pointer to `llm.media`.

| Capability | Default model | Router call |
|---|---|---|
| `tts` | `eleven_v4` | `llm.media.audio.speak(...)` |
| `tts_stream` | `eleven_v4_turbo` | `llm.media.audio.stream_tts(...)` |
| `asr` | `scribe_v3` | `llm.media.audio.transcribe(...)` |
| `asr_stream` | `scribe_v2_realtime` | `provider.open_transcription_session(...)` |
| `sfx` | `eleven_text_to_sound_v3` | `llm.media.audio.sfx(...)` |
| `music` | `music_v2_5` | `llm.media.audio.music(...)` |
| `voice_design` | `eleven_ttv_v3` | `provider.design_voice_media(...)` |

---

## 1. Install and configure

```bash
pip install "llmcore[elevenlabs]"   # httpx, elevenlabs>=2.78, websockets (realtime STT)
```

```toml
[providers.elevenlabs]
# api_key is read from ELEVENLABS_API_KEY (then ELEVEN_API_KEY) when omitted
timeout = 120
backend = "auto"              # "auto" | "httpx" | "sdk"
# voice_id = "EXAVITQu4vr4xnSDxMaL"   # premade "Sarah"; needs no verification
# output_format = "mp3_44100_128"
# resolve_consent = true

[providers.elevenlabs.models]
tts = "eleven_v4"
asr = "scribe_v3"
asr_stream = "scribe_v2_realtime"
```

`eleven_v4` is the default because it is GA (no alpha access, standard rate) and
covers 85 languages — verified against the live `/v1/models` lineup, not assumed.

---

## 2. Consent is attached to the audio

This is what makes the ElevenLabs adapter different from the other voice
providers. ElevenLabs tracks consent state per *voice* — whether it is a clone,
who owns it, whether it passed verification — but only on the voice resource.
A caller who wanted to refuse audio from an unverified clone would have to know
to make a second API call.

llmcore does that lookup for you and hangs the result off the artifact:

```python
speech = await llm.media.audio.speak("Hello there.", provider="elevenlabs")
consent = speech.artifacts[0].provenance.consent

consent.voice_name              # 'Sarah - Mature, Reassuring, Confident'
consent.category                # 'premade' | 'cloned' | 'professional' | 'generated'
consent.is_cloned               # True when the voice imitates a real person
consent.is_owner                # whether this account owns the voice
consent.requires_verification   # provider's requirement
consent.is_verified             # provider's verdict
consent.verification_failures   # ('captcha_failed', ...)
consent.verification_satisfied  # the one you probably want
```

### `None` is not `False`

`verification_satisfied` is deliberately **tri-state**:

| Value | Meaning |
|---|---|
| `True` | Verification is not required, or is required and passed |
| `False` | Required and **not** passed |
| `None` | **The provider said nothing** |

Treating "unknown" as "fine" is a policy decision, and it belongs to you, not to
this library. So write all three branches:

```python
if consent and consent.verification_satisfied is False:
    raise RuntimeError("voice is not cleared for use")
if consent and consent.verification_satisfied is None:
    log.warning("consent state unknown for %s", consent.voice_id)
```

### Cost and failure behaviour

Lookups are **cached per voice** for the life of the provider, so a thousand
synthesis calls on one voice cost one extra request. Call
`get_voice_consent(voice_id, refresh=True)` after verifying a voice.

If the lookup **fails**, you still get your audio. The consent record comes back
with every field `None` and `provider_declared=False` — which correctly reads as
*we do not know*, not as *this is fine*. Losing the metadata should not lose the
audio you asked for.

Set `resolve_consent = false` to skip lookups entirely; `provenance.consent` is
then `None`.

### Generated audio carries no consent record

SFX and music return `provenance.consent is None`, because nothing there is
anyone's voice. An empty consent record would imply a question that does not
apply.

---

## 3. Speech

```python
speech = await llm.media.audio.speak(
    "Consent is not an afterthought.",
    provider="elevenlabs",
    voice="EXAVITQu4vr4xnSDxMaL",
    speed=1.1,
)
artifact = speech.artifacts[0]      # bytes, mime, sample rate
speech.usage.characters             # ElevenLabs bills per character, not per token
```

Streaming, when first-byte latency matters more than a handle:

```python
async for chunk in llm.media.audio.stream_tts("...", provider="elevenlabs"):
    player.write(chunk)
```

`output_format` drives both the encoding and the reported MIME/sample rate —
`mp3_44100_128`, `pcm_24000`, `opus_48000_128`, `ulaw_8000`, and so on.

---

## 4. Transcription

```python
# Local bytes are uploaded as multipart.
result = await llm.media.audio.transcribe(
    audio=MediaRef.from_path("/audio/interview.mp3"), provider="elevenlabs",
    diarize=True, timestamps=True,
)
result.artifacts[0].text
result.artifacts[0].provider_metadata["words"]       # word-level timings
result.artifacts[0].provider_metadata["language_code"]
```

A **remote** `MediaRef` is handed to ElevenLabs as a `source_url`
(`cloud_storage_url` is deprecated), so the bytes never round-trip through your
process:

```python
await llm.media.audio.transcribe(
    audio=MediaRef.from_url("https://example.com/call.mp3"), provider="elevenlabs"
)
```

### Realtime transcription (`asr_stream`)

Realtime Scribe runs over a WebSocket. The session has the same shape as
Deepgram's: push raw audio, iterate `TranscriptionStreamEvent`s.

```python
provider = llm._provider_manager.get_provider("elevenlabs")
async with provider.open_transcription_session(
    sample_rate_hz=16000,          # pcm_8000 ... pcm_48000, or audio_format="ulaw_8000"
    language="en",
    commit_strategy="vad",         # server commits on silence; "manual" -> finalize()
    include_timestamps=True,       # FINAL events carry word timings
    keyterms=["llmcore"],
) as session:
    session.set_context("Earlier conversation text")   # optional, sent with the first chunk
    for chunk in mic_chunks():                         # raw PCM bytes
        await session.send_audio(chunk)
    await session.finalize()
    async for event in session:
        if event.type == "interim":
            show(event.text)
        elif event.type == "final":
            commit(event.text, event.words)
        elif event.type == "error":
            raise RuntimeError(event.text)
```

How server messages map to events:

| Server message | Event |
|---|---|
| `partial_transcript` | `interim` |
| `committed_transcript` | `final`; with `include_timestamps`, the `_with_timestamps` twin replaces it, carrying `words`, `start` and `end` |
| `session_started` | `open` |
| `auth_error`, `quota_exceeded`, `rate_limited`, `input_error`, ... | `error` |
| entity, redaction, edit and language-detection results | `other` (see `event.raw`) |

Every other documented option passes through as a query parameter:
`secondary_languages`, `no_verbatim`, the `vad_*` and `min_*_ms` thresholds,
`entity_detection`, `entity_redaction`, `entity_redaction_mode`,
`transcript_edit`, `filter_background_audio`, `enable_logging`, and `token` (a
single-use token in place of the API key).

Verified live on 2026-11-16. TTS audio of "The quick brown fox jumps over the
lazy dog." at `pcm_16000`, streamed in 100 ms chunks, produced three interim
events and then that exact sentence as the final event; the timestamped run
returned 17 word entries.

---

## 5. Sound effects and music

```python
sfx = await llm.media.audio.sfx(
    prompt="a heavy wooden door creaking open", provider="elevenlabs",
    duration_seconds=3,
)
music = await llm.media.audio.music(
    prompt="a slow ambient piano motif", provider="elevenlabs", duration_seconds=30,
)
```

`duration_seconds` is the protocol's unit; llmcore converts to the
milliseconds the music API expects.

**ElevenLabs sound generation is text-conditioned only.** Passing `video=` raises
rather than being silently ignored — dropping it would hand you audio unrelated
to the footage you supplied. For video-conditioned foley, route to a provider
that does it (fal's MMAudio).

---

## 6. Voice design

Voice design generates *candidate voices* from a description, not speech from
text. Each preview is a separate artifact carrying the `generated_voice_id` you
need to keep it:

```python
previews = await provider.design_voice_media(
    "a calm elderly storyteller with a slight rasp"   # min. 20 characters
)
for p in previews.artifacts:
    p.provider_metadata["generated_voice_id"]
    p.provenance.consent.category       # 'generated' — imitates no one
```

A designed voice states `category="generated"` and
`requires_verification=False` explicitly, rather than leaving consent `None`,
because "this imitates nobody" is a known fact, not an unknown one.

---

## 7. Backends

| Backend | What it does | When it is chosen |
|---|---|---|
| `httpx` | Direct REST against `api.elevenlabs.io` | **Default** |
| `sdk` | The official `elevenlabs` package | Only when pinned explicitly |

These are plain JSON/multipart endpoints returning audio bytes, so the direct
path keeps the vendor SDK off the critical path and the dependency optional.

---

## 8. Troubleshooting

| Symptom | Cause |
|---|---|
| *"unavailable on this account's plan (the API key is valid)"* | **Music, voice design and some other features require a paid plan.** Your key is fine — llmcore reports `402` and `403 feature_not_available` as plan gating rather than as an auth failure, so you are not sent hunting a credential problem you do not have |
| `401` / a genuine `403` | Real auth failure — check `ELEVENLABS_API_KEY` |
| `422 string_too_short` on voice design | `voice_description` must be at least 20 characters |
| `422` on TTS | Usually `voice_settings`; ranges differ per model |
| `consent.provider_declared is False` | The consent lookup failed. The audio is still valid; the metadata is unknown |
| `is_owner` is `None` | The single-voice endpoint omits it where the list endpoint reports it. `None` means *not stated*, per the tri-state rule |

---

## 9. Related

- the media subsystem design spec — subsystem design; §5.3
  covers what ElevenLabs changed about it
- the provider support matrix — SDK versions and
  the live validation log
- [`Fal_provider_usage.md`](Fal_provider_usage.md) — the other broad media adapter
