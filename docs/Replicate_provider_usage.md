# Replicate provider — usage guide

[Replicate](https://replicate.com) hosts tens of thousands of community models
behind **one** prediction API. In llmcore it is a **media-only provider** —
`chat_completion()` raises with a pointer to `llm.media`.

The important thing to understand: this is **one generic adapter**, not a class
per model. It works for models llmcore has never heard of.

| Capability | Default model | Router call |
|---|---|---|
| `image_generate` | `black-forest-labs/flux-schnell` | `llm.media.images.generate(...)` |
| `image_edit` | `black-forest-labs/flux-kontext-pro` | `llm.media.images.edit(...)` |
| `image_upscale` | `nightmareai/real-esrgan` | `llm.media.images.upscale(...)` |
| `video_generate` | `minimax/video-01` | `llm.media.video.generate(...)` |
| `asr` | `openai/whisper` | `llm.media.audio.transcribe(...)` |
| `tts` | `jaaari/kokoro-82m` | `llm.media.audio.speak(...)` |
| `music` | `meta/musicgen` | `llm.media.audio.music(...)` |

---

## 1. Install and configure

```bash
pip install "llmcore[replicate]"
```

```toml
[providers.replicate]
# api_key is read from REPLICATE_API_TOKEN when omitted
timeout = 300
backend = "auto"       # "auto" | "httpx" | "sdk"
use_schema = true      # see §2 — this is the whole design

[providers.replicate.models]
image_generate = "black-forest-labs/flux-dev"
video_generate = "minimax/video-01"
```

Model references are `owner/name` or `owner/name:version`.

---

## 2. How one adapter serves the whole catalog

Every Replicate model publishes an OpenAPI schema describing its own inputs.
`flux-schnell` takes `prompt` and `num_outputs`; `whisper` takes `audio` and
`language`. Rather than hardcoding "the prompt field is called prompt", llmcore
**reads the schema** and maps its canonical protocol arguments onto whatever
that model actually calls them:

```python
await llm.media.images.generate("a cat", provider="replicate", n=2)
# flux-schnell receives: {"prompt": "a cat", "num_outputs": 2}

await llm.media.audio.transcribe(audio=ref, provider="replicate", language="en")
# whisper receives:     {"audio": "...", "language": "en"}
```

Neither mapping is hardcoded. A model that calls its image input `input_image`
gets your image in `input_image`. A model llmcore has never seen works, and a
model that renames a field next week keeps working.

**Explicit keyword arguments always win**, because you know your model better
than the mapping does:

```python
await llm.media.images.generate(
    "a cat", provider="replicate",
    model="some/unusual-model",
    my_custom_field=0.7,       # passed through verbatim
)
```

If the schema lookup fails, llmcore falls back to canonical field names and lets
the API answer — a `422` naming the field is far more useful than a silently
dropped input. Set `use_schema = false` to always send canonical names.

---

## 3. Everything is a prediction

Every capability returns a `MediaJob`:

```python
job = await llm.media.images.generate("a calico cat on books", provider="replicate")
print(job.provider_job_id, job.status)     # pred-id, queued

result = await llm.media.wait(job, timeout=300)
print(result.artifacts[0].uri)
print(result.usage.compute_seconds)        # from Replicate's own metrics
```

Cancellation is real here — Replicate stops the prediction and stops billing it:

```python
job = await llm.media.jobs.cancel(job)
```

---

## 4. Output shapes vary per model

Because outputs are schema-defined, they differ per model rather than per
vendor. llmcore walks the shapes the schemas actually produce:

| Output | Result |
|---|---|
| `["https://…a.png", …]` | one image artifact each |
| `"https://…a.mp4"` | one video artifact |
| `{"transcription": "…", "srt_file": "https://…"}` | a **text** artifact plus a file artifact |
| anything unrecognised | no artifacts; the full payload stays on `job.provider_metadata["prediction"]` |

**MIME types may be `None`.** Replicate output URLs often carry no file
extension. llmcore falls back to the `output_format` you *requested*, and
otherwise reports `None` rather than guessing — an honest unknown beats a
plausible lie when you are choosing a decode path.

---

## 5. Inputs

```python
# Remote: passed straight through.
await llm.media.images.upscale(
    image=MediaRef.from_url("https://example.com/photo.jpg"), provider="replicate"
)

# Local: sent as a data URI, so there is no upload step to configure.
await llm.media.audio.transcribe(
    audio=MediaRef.from_path("/audio/call.mp3"), provider="replicate"
)
```

---

## 6. Webhooks

Replicate supports callbacks natively, and this adapter opts into llmcore's
webhook receiver. Configure `media.jobs.webhook_base_url` and every prediction
registers a single-use callback URL automatically. Polling remains the fallback.

---

## 7. Troubleshooting

| Symptom | Cause |
|---|---|
| `429` mentioning "burst of 1" | **Accounts under \$5 of credit** are throttled to ~6 prediction creations per minute with a burst of 1. This is the single most confusing Replicate behaviour — it looks like a bug. Add credit or space out calls |
| *"rejected for billing reasons (the token is valid)"* | A `402`. Check the account's spend limit or payment method — not the token |
| `404` on a model that clearly exists | Should not happen: community models need a version pin, and llmcore resolves one from the model's schema lookup automatically |
| `422` with unfamiliar field names | The model's schema uses names the mapping did not cover. Pass them explicitly as keyword arguments |
| `mime_type` is `None` | The output URL had no extension and no `output_format` was requested. The bytes are fine |

---

## 8. Related

- the media subsystem design spec — §5.5 on what this
  proved about the generic-adapter bet
- the provider support matrix — SDK versions and
  the live validation log
- [`Fal_provider_usage.md`](Fal_provider_usage.md) — the other marketplace adapter
