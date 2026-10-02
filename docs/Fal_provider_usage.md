# fal provider — usage guide

[fal](https://fal.ai) is a **media model marketplace**: thousands of
image, video, audio and speech models behind one queue API. In llmcore it is a
**media-only provider** — it does not serve chat, and `chat_completion()` raises
with a pointer to `llm.media`.

It is the broadest single media adapter llmcore has, covering nine capabilities:

| Capability | Default endpoint | Router call |
|---|---|---|
| `image_generate` | `fal-ai/flux/schnell` | `llm.media.images.generate(...)` |
| `image_edit` | `fal-ai/flux-pro/kontext` | `llm.media.images.edit(...)` |
| `image_upscale` | `fal-ai/clarity-upscaler` | `llm.media.images.upscale(...)` |
| `video_generate` | `fal-ai/minimax-video` | `llm.media.video.generate(...)` |
| `video_interpolate` | `fal-ai/film` | `llm.media.video.interpolate(...)` |
| `sfx` | `fal-ai/mmaudio-v2` | `llm.media.audio.sfx(...)` |
| `music` | `fal-ai/stable-audio` | `llm.media.audio.music(...)` |
| `tts` | `fal-ai/kokoro` | `llm.media.audio.speak(...)` |
| `asr` | `fal-ai/whisper` | `llm.media.audio.transcribe(...)` |

---

## 1. Install and configure

```bash
pip install "llmcore[fal]"
```

That pulls `httpx` (the default transport) and `fal-client` (the optional SDK
backend). The direct REST path is the default; see §5.

```toml
[providers.fal]
# api_key is read from FAL_KEY or FAL_API_KEY when omitted
timeout = 300          # generation is slow; this is a per-request ceiling
backend = "auto"       # "auto" | "httpx" | "sdk"

# Endpoint paths are configurable because the gallery moves faster than
# llmcore releases. Anything you leave out keeps the default above.
[providers.fal.models]
image_generate = "fal-ai/flux-pro/v1.1-ultra"
video_generate = "fal-ai/kling-video/v2/master/text-to-video"
```

**Credentials.** fal's own convention is `FAL_KEY`; llmcore also accepts
`FAL_API_KEY`, in the order `config.api_key → FAL_KEY → FAL_API_KEY`.

---

## 2. Everything is a job

Unlike OpenAI (where image generation answers synchronously), **every fal call
is a queue submission** — so every capability reports
`MediaExecution.ASYNC_JOB` and returns a `MediaJob`:

```python
llm = await LLMCore.create(config)

job = await llm.media.images.generate(
    "a calico cat asleep on a stack of books, soft window light",
    provider="fal",
)
print(job.status, job.queue_position)      # MediaJobStatus.QUEUED 3

result = await llm.media.wait(job, timeout=300)
print(result.artifacts[0].uri)             # https://v3b.fal.media/files/...
print(result.usage.compute_seconds)        # 1.86
```

`llm.media.wait()` polls with capped exponential backoff and jitter, so the same
call works whether the model takes two seconds or twenty minutes. If you would
rather drive the loop yourself:

```python
while not job.is_terminal:
    await asyncio.sleep(2)
    job = await llm.media.jobs.poll(job)
```

Because the execution class is declared **per capability**, code that goes
through `media.wait()` is portable across providers — the identical call returns
a `MediaResult` on OpenAI and a `MediaJob` on fal, and `wait()` absorbs both.

---

## 3. Inputs are URLs

fal models take URLs, not inline bytes. llmcore handles both sides of that:

```python
# Already remote: passed straight through, no round trip through this process.
await llm.media.images.upscale(
    image=MediaRef.from_url("https://example.com/photo.jpg"), provider="fal"
)

# Local bytes or a path: uploaded to fal storage first, automatically.
await llm.media.images.edit(
    "make it winter",
    image=MediaRef.from_path("/photos/house.png"),
    provider="fal",
)
```

This makes chaining cheap — a fal artifact URL feeds straight back into the next
fal call without ever being downloaded:

```python
speech = await llm.media.wait(
    await llm.media.audio.speak("The abstraction did not bend.", provider="fal")
)
text = await llm.media.wait(
    await llm.media.audio.transcribe(
        audio=MediaRef.from_url(speech.artifacts[0].uri), provider="fal"
    )
)
print(text.artifacts[0].text)      # " The abstraction did not bend."
```

To keep a copy locally, materialize it through the artifact store:

```python
local = await llm.media.artifacts.materialize(result.artifacts[0], force=True)
print(local.uri)                                    # file:///…/<sha256>.png
print(local.provider_metadata["source_uri"])        # the original fal URL
```

**`force=True` matters here.** fal URLs are CDN-hosted and not permanent, but
fal does not publish an expiry — so the default `on_expiry` policy has nothing
to trigger on and leaves the artifact remote. Either pass `force=True` per call,
or set the policy once:

```toml
[media]
artifact_materialize = "always"
```

---

## 4. Cancellation is a request, not a guarantee

```python
job = await llm.media.video.generate("a drone shot over a fjord", provider="fal")
job = await llm.media.jobs.cancel(job)

print(job.provider_metadata["cancellation"])       # "requested"
print(job.provider_metadata["cancellation_note"])  # ...may still complete...
```

fal answers `202 CANCELLATION_REQUESTED`, and a request a runner already picked
up **may still finish and still bill**. llmcore marks the job `CANCELED` and
records that caveat rather than implying the work stopped.

If fal answers `400 ALREADY_COMPLETED`, the job finished before the cancel
landed. That is not an error: llmcore fetches the result and returns a
**succeeded** job with `cancellation = "already_completed"`.

---

## 5. Backends

| Backend | What it does | When it is chosen |
|---|---|---|
| `httpx` | Direct calls to `queue.fal.run` | **Default.** No vendor SDK needed |
| `sdk` | `fal-client`'s `AsyncClient` | Only when pinned explicitly |

The direct path is the default because it is the whole API surface fal
documents, it keeps the dependency optional, and it lets llmcore address queue
routes and storage backends that the SDK version in the venv may not know about.
Set `backend = "sdk"` if you would rather track the vendor client.

Both backends are functionally equivalent for submit, poll, result, cancel and
upload.

---

## 6. Picking endpoints

`fal-ai/flux/schnell` is a fast, cheap default, not a recommendation for
production. Browse [fal.ai/models](https://fal.ai/models) and pin what you want
either in config (§1) or per call:

```python
await llm.media.images.generate(
    "a product shot on seamless white",
    provider="fal",
    model="fal-ai/flux-pro/v1.1-ultra",
)
```

Model **input** schemas differ per model, not per capability — FILM takes a
start/end image pair, Kling takes a duration and aspect ratio. Unknown keyword
arguments are forwarded to fal verbatim, so a model-specific field just works:

```python
await llm.media.images.generate(
    "a tabby cat", provider="fal", num_inference_steps=8, seed=1234,
)
```

Output shapes vary too, and llmcore's extractor walks the common keys
(`images`, `image`, `video`, `audio`, `audio_url`, `audio_file`, `text`). A
model returning something unrecognised yields no artifacts rather than an
error — the full payload is always kept in `job.provider_metadata["result"]`.

---

## 7. Troubleshooting

| Symptom | Cause |
|---|---|
| `405` while polling | Should not happen: fal namespaces its queue by **application** (`fal-ai/flux`), not by model path (`fal-ai/flux/schnell`). llmcore uses the URLs fal returns at submission and falls back to an app-scoped path |
| `404 Application "<x>" not found` | The endpoint path is wrong; check [fal.ai/models](https://fal.ai/models) |
| `422 Field required` | The model wants a field this capability does not send. Pass it as a keyword argument — it is forwarded verbatim |
| `400 Invalid storage type` on upload | Should not happen: llmcore tries CDN v3 first, then the older signed-URL flow, because accounts differ in which they offer |
| Job never leaves `QUEUED` | Concurrency limit for your plan; check the fal dashboard |

---

## 8. Webhooks

fal supports webhook delivery (`?fal_webhook=`), and the adapter will send a
`webhook_url` from config if you set one. llmcore has **no webhook receiver
yet** — that is phase M9 of
the media subsystem design spec. Until then, a configured
webhook is delivered to an endpoint you run and reconcile yourself; polling
remains the supported path.

---

## 9. Related

- the media subsystem design spec — the subsystem design,
  and §5.2 on what fal proved about it
- the provider support matrix — SDK versions and
  the live validation log
- [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md) — every config key
