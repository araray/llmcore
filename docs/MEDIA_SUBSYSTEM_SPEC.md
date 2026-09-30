# Media Subsystem — Design & Specification

Generative image, audio and video as a first-class `llmcore` subsystem, plus the
provider adapters that sit behind it.

- **Status:** **M1 + M2 implemented** (core subsystem; Deepgram migrated behind
  the audio protocols). M3 onward not started — see §5.
- **Written:** 2026-09-29
- **Primary input:** `/av/data/repos/docs/llmcore/researches/image-audio-video_providers_2026september.md`
  (the provider survey and priority matrix; this document is the llmcore-side design)
- **Related:** [`PROVIDER_SUPPORT_MATRIX.md`](PROVIDER_SUPPORT_MATRIX.md),
  [`PROVIDER_MODERNIZATION_PLAN.md`](PROVIDER_MODERNIZATION_PLAN.md)

---

## 1. Problem statement

llmcore already reaches media APIs, but through a surface that does not
generalize:

- `BaseProvider` carries five optional media methods (`generate_speech`,
  `transcribe_audio`, `generate_image`, `ocr`, `create_embeddings`), each
  defaulting to `NotImplementedError`.
- Seven providers implement some subset: `zai` (the most complete — image, TTS,
  STT, OCR, video, web search), `deepinfra`, `mistral`, `huggingface`,
  `openai`, `friendli`, `deepgram`.
- `deepgram` is the extreme case: 12 provider-specific public methods
  (`open_voice_agent`, `transcribe_stream_flux`, `stream_speech`, …) because
  the facade has nowhere to put realtime audio.
- `zai.generate_video()` is the only video generation in the codebase, under a
  provider-specific name with no shared job model.

Three structural problems follow:

1. **No shared execution model.** Image generation is request/response, TTS is a
   byte stream, video is a long-running job. The current surface assumes
   request/response and forces the other two into provider-private methods.
2. **No capability discovery.** A caller cannot ask "which configured provider
   can do video interpolation?" without hardcoding provider names.
3. **No normalized results.** `models_multimodal.py` has `SpeechResult`,
   `TranscriptionResult`, `ImageGenerationResult`, `GeneratedImage`, `OCRResult`
   — but nothing for video, nothing for jobs, and no common artifact type.

**The fix is not more provider classes.** It is a media subsystem the providers
plug into.

---

## 2. Design

### 2.1 Shape

```
LLMCore
  ├── ProviderManager        (existing — chat)
  └── MediaManager           (new)
        ├── AudioRouter      tts · asr · music · sfx · realtime
        ├── ImageRouter      generate · edit · upscale · variate
        ├── VideoRouter      generate · edit · interpolate · reframe · upscale
        ├── MediaJobManager  poll · webhook · cancel · resume
        └── ArtifactStore    bytes/URI persistence + checksums
```

Routers resolve `(capability, model?, provider?)` → an adapter, using model
cards for capability metadata (§2.5) and a selection policy (§2.7). They do not
contain vendor logic.

### 2.2 Three execution classes — the core distinction

| Class | Examples | Surface |
|---|---|---|
| **Request/response** | image generation & edit, batch ASR, one-shot TTS | `await media.images.generate(...) -> MediaResult` |
| **Byte / event stream** | streaming TTS, realtime ASR, voice agents | `async for chunk in media.audio.stream_tts(...)` / async session object |
| **Long-running job** | Veo, fal queue, Replicate predictions, Luma generations | `job = await media.video.generate(...)` → `MediaJob`, then poll or webhook |

Modelling streaming as a job (or a job as request/response) is the main
abstraction mistake to avoid. Deepgram already proves the streaming semantics
are irreducible.

### 2.3 Core types

New module `src/llmcore/media/models.py`. These are the provider-independent
contract; adapters translate to and from vendor shapes.

```python
class MediaKind(StrEnum):
    AUDIO = "audio"; IMAGE = "image"; VIDEO = "video"; TEXT = "text"

class MediaCapability(StrEnum):
    # audio
    TTS = "tts"; TTS_STREAM = "tts_stream"
    ASR = "asr"; ASR_STREAM = "asr_stream"
    VOICE_AGENT = "voice_agent"
    MUSIC = "music"; SFX = "sfx"; VOICE_DESIGN = "voice_design"
    # image
    IMAGE_GENERATE = "image_generate"; IMAGE_EDIT = "image_edit"
    IMAGE_UPSCALE = "image_upscale"; IMAGE_VARIATE = "image_variate"
    OCR = "ocr"
    # video
    VIDEO_GENERATE = "video_generate"; VIDEO_EDIT = "video_edit"
    VIDEO_INTERPOLATE = "video_interpolate"; VIDEO_REFRAME = "video_reframe"
    VIDEO_UPSCALE = "video_upscale"; VIDEO_EXTEND = "video_extend"

class MediaExecution(StrEnum):
    REQUEST_RESPONSE = "request_response"; STREAM = "stream"; ASYNC_JOB = "async_job"

class MediaJobStatus(StrEnum):
    QUEUED = "queued"; RUNNING = "running"; SUCCEEDED = "succeeded"
    FAILED = "failed"; CANCELED = "canceled"; EXPIRED = "expired"
```

`MediaArtifact` — one produced asset:

```python
@dataclass(frozen=True, slots=True)
class MediaArtifact:
    kind: MediaKind
    uri: str | None = None            # provider URL, or artifact-store URI
    data: bytes | None = None         # inline bytes (small results)
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    sample_rate_hz: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    checksum_sha256: str | None = None
    expires_at: datetime | None = None    # provider URLs are usually temporary
    provenance: MediaProvenance | None = None
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)
```

`MediaUsage` — normalized billing units (§2.6), `MediaJob` — the async handle,
`MediaResult` — the request/response wrapper (`artifacts`, `usage`, `model`,
`provider`, `raw`).

> **Addition beyond the research doc:** `expires_at` and `checksum_sha256` are
> load-bearing, not nice-to-have. Every aggregator returns short-lived URLs; a
> caller that stores the URI instead of the bytes gets a dead link hours later.
> The `ArtifactStore` uses both to decide what to materialize and to dedupe.

### 2.4 Capability protocols

`typing.Protocol` classes in `src/llmcore/media/protocols.py`, one per
capability group. Adapters implement only what they support; routers check with
`isinstance(adapter, ImageGenerationProvider)` (runtime-checkable).

```python
@runtime_checkable
class ImageGenerationProvider(Protocol):
    async def generate_image_media(
        self, prompt: str, *, model: str | None = None, n: int = 1,
        size: str | None = None, seed: int | None = None,
        reference_images: Sequence[MediaRef] | None = None, **kwargs: Any,
    ) -> MediaResult | MediaJob: ...

@runtime_checkable
class StreamingTTSProvider(Protocol):
    async def stream_tts(
        self, text: str, *, model: str | None = None, voice: str | None = None,
        sample_rate_hz: int | None = None, **kwargs: Any,
    ) -> AsyncIterator[bytes]: ...

@runtime_checkable
class VideoGenerationProvider(Protocol):
    async def generate_video(
        self, prompt: str, *, model: str | None = None,
        first_frame: MediaRef | None = None, last_frame: MediaRef | None = None,
        duration_seconds: float | None = None, resolution: str | None = None,
        with_audio: bool | None = None, **kwargs: Any,
    ) -> MediaJob: ...
```

`MediaRef` is the input counterpart of `MediaArtifact`: a URL, local path, raw
bytes, or a previously produced `MediaArtifact`. Adapters upload/inline as their
API requires — callers never hand-roll base64.

> Note the method name `generate_image_media`, not `generate_image`: the latter
> is already taken on `BaseProvider` with a different return type. §4 covers the
> migration; the new names are temporary and collapse at the 1.0 boundary.

### 2.5 Model cards carry the capabilities

Extend the existing card schema rather than adding provider-specific tables in
code. New optional `media` block:

```json
{
  "model_id": "fal-ai/film/video",
  "provider": "fal",
  "model_type": "video",
  "media": {
    "kind": "video",
    "capabilities": ["video_interpolate"],
    "execution": "async_job",
    "supports_webhooks": true,
    "inputs": {"video": true, "image": false, "text": false},
    "outputs": {"video": true},
    "max_duration_seconds": 10,
    "resolutions": ["720p", "1080p"]
  },
  "sourcing": {
    "model_owner": "google-research",
    "model_family": "film",
    "model_license": "apache-2.0",
    "hosting_provider": "fal",
    "hosting_policy": "fal-aup"
  },
  "policy": {
    "supports_custom_weights": false,
    "provider_policy_applies": true,
    "commercial_use": "model_specific"
  }
}
```

Two deliberate choices from the research, both adopted:

- **Aggregators need four separate concepts**, not one `provider` field:
  `provider` (who we call) vs `model_owner` / `model_family` /
  `model_license` (whose weights) vs `hosting_policy` (whose AUP). Without this,
  "can I use this commercially?" is unanswerable for fal/Replicate/HF.
- **There is no "uncensored" boolean.** The survey found no mainstream managed
  API that credibly promises policy-free generation. The honest representation
  is two orthogonal facts: `supports_custom_weights` (can modified/ablated
  weights run?) and `provider_policy_applies` (does the host still enforce an
  AUP?). Replicate, HF Endpoints and BFL's open-weight route are `true`/`true`.

`cardctl` gains a `--kind media` mode and per-provider media adapters.

### 2.6 Cost normalization without pretending

Media vendors bill in incompatible units: per-image, per-megapixel, per-second
of video, per-audio-minute, per-character, per-token, per-compute-second.
`MediaUsage` keeps whichever units the vendor reported *and* an
`estimated_cost_usd` with a `pricing_as_of` stamp and a `basis` label. Callers
that need exactness read the native fields; dashboards read the estimate and can
see how stale the pricing is. Never synthesize a token count for a video.

### 2.7 Provider selection policy

`media.<router>.<op>(...)` resolves in this order:

1. Explicit `provider=` → use it, or raise if it lacks the capability.
2. Explicit `model=` → resolve the owning provider from cards.
3. Configured `[media.routing]` preference list for that capability.
4. The research doc's workload defaults (§"Final provider choices") as built-in
   fallbacks — ElevenLabs for TTS, Deepgram for ASR, OpenAI/Google for frontier
   image, fal for breadth, etc.
5. Otherwise raise `MediaCapabilityError` listing which configured providers
   *could* satisfy it if enabled.

Constraint filters compose with all of the above: `require_commercial_use=True`,
`require_custom_weights=True`, `max_cost_usd=...`, `exclude_providers=[...]`.

### 2.8 Webhooks as a core facility

Async jobs need a callback path. A generic receiver — not per-provider:

- `MediaJobManager` issues a signed, single-use callback token per job.
- An optional ASGI app (`llmcore.media.webhooks:app`) mounts at a configured
  path; the existing `llmcore[bridge]` server can host it.
- Providers that support webhooks register the URL; providers that don't are
  polled with capped exponential backoff.
- **Polling is always the fallback**, so llmcore stays usable with no public
  ingress — the common case for local development.

### 2.9 Artifact store

Reuse llmcore's storage layer rather than inventing one. `ArtifactStore`
persists bytes to a configured backend (filesystem by default, with the existing
SQLite/Postgres metadata store for the index), keyed by
`checksum_sha256`. Policy: `materialize = "always" | "on_expiry" | "never"`.
Default `on_expiry` — fetch and store before the provider URL dies.

---

## 3. Public API

```python
async with await LLMCore.create() as llm:
    # request/response
    img = await llm.media.images.generate("an orange tabby", size="1024x1024")
    img.artifacts[0].uri

    # edit with a reference
    edited = await llm.media.images.edit(
        "make it night-time", image=MediaRef.from_path("cat.png")
    )

    # streaming TTS
    async for chunk in llm.media.audio.stream_tts("Hello there", voice="rachel"):
        speaker.write(chunk)

    # long-running job
    job = await llm.media.video.generate("a drone shot over dunes",
                                         duration_seconds=8, with_audio=True)
    job = await llm.media.jobs.wait(job, timeout=600)     # poll or webhook
    await llm.media.artifacts.download(job.artifacts[0], "dunes.mp4")

    # capability discovery
    llm.media.capabilities()                      # {capability: [providers]}
    llm.media.who_can(MediaCapability.VIDEO_INTERPOLATE)
```

---

## 4. Migration and backward compatibility

Non-negotiable: **no existing call site breaks.**

1. **Phase M1 adds only new code.** `llmcore.media` lands with the types,
   protocols, routers and job manager, plus an in-repo fake adapter for tests.
2. **`BaseProvider`'s five media methods stay**, and keep their current return
   types (`SpeechResult`, `TranscriptionResult`, `ImageGenerationResult`,
   `OCRResult`). They become thin shims that call the new subsystem when the
   provider has an adapter, and keep their current implementation otherwise.
3. **`models_multimodal.py` types become views over `MediaArtifact`.** They are
   public API today (returned by seven providers), so they get
   `from_artifact()` / `to_artifact()` and a deprecation note, not deletion.
4. **Deepgram is the reference migration** (research doc's second merge, adopted).
   Its 12 provider-specific methods stay as the compatibility surface while the
   streaming protocols are implemented behind them. It is the only integration
   that already exercises batch + realtime WebSocket + voice agent, so it
   validates the hard parts of the abstraction before any new vendor lands.
5. **`zai.generate_video()`** is renamed into the protocol with an alias kept.

---

## 5. Implementation plan

Order follows the research doc's rollout, with llmcore-specific gates.

| Phase | Scope | Gate |
|---|---|---|
| **M1** ✅ | `llmcore.media` core: types, protocols, routers, `MediaJobManager` (poll only), `ArtifactStore`, config section, fake adapter + tests. *Card schema blocks deferred to M2, where the first real adapter needs them.* | Landed 2026-09-30, 134 tests |
| **M2** ✅ | Refactor **Deepgram** behind the audio protocols; keep its public methods. Added the `models_multimodal` ↔ `MediaArtifact` bridge (§4.3). | Landed 2026-09-30, 74 tests. Live: TTS → artifact → ASR round trip, plus streaming TTS |
| **M3** | **OpenAI** media: images generate/edit, TTS, ASR, realtime audio | Lowest marginal cost — adapter already exists |
| **M4** | **Google** media: Imagen/Nano-Banana images, **Veo** video (async job), native TTS | First true async-job provider; validates §2.8 |
| **M5** | **fal** — queue/webhook lifecycle, URL inputs, video, SFX, FILM interpolation | The provider-neutrality test: if the abstraction bends here, fix the abstraction |
| **M6** | **ElevenLabs** — batch + realtime STT, TTS, SFX, music, voice design | Consent/provenance as first-class metadata |
| **M7** | **Replicate** — one generic prediction adapter + model-schema descriptors | Explicitly *not* a class per model |
| **M8** | **Hugging Face Inference Endpoints** — configurable endpoint/schema adapter | Custom weights / private repos path |
| **M9** | Webhook receiver, then direct specialists (BFL, Luma, Stability) when justified: lower unit cost, first-party-only feature, data contract, or pre-aggregator access | Otherwise fal/Replicate already cover it |

Runway stays on the watchlist — the research run could not verify its current
API contract, and the doc is explicit about not freezing a guessed model id.

---

## 6. Corrections and additions to the research

Flagging these because the research snapshot and the code disagree, or the
research could not see llmcore internals:

1. **OpenAI Sora is deprecated.** The survey lists Sora video as a P0 reason to
   extend the OpenAI adapter. But `openai` 3.1.0 (2026-08-14) shipped
   *"**api:** deprecate Sora video APIs"* — confirmed in
   `/av/avalon/xrepos/openai-python/CHANGELOG.md`. **Do not build the Sora
   adapter.** Frontier video comes from **Veo (M4)** and fal-hosted models (M5).
   This is the single most important correction: M3 shrinks to images + speech.
2. **Reuse `models_multimodal.py`, don't parallel it.** The research proposes
   fresh types without knowing llmcore already returns `SpeechResult` /
   `ImageGenerationResult` from seven providers. §4.3 keeps them as views.
3. **Deepgram's surface is bigger than the survey assumes** — 12 public methods
   including a bidirectional voice agent. The M2 refactor is a larger job than
   "refactor behind protocols" implies; budget for it.
4. **`zai` is already the most media-complete provider** (image, TTS, STT, OCR,
   video, web search). It is a better second migration target than the survey
   suggests, and a free second data point on whether the protocols generalize.
5. **Env var naming.** fal's own convention is `FAL_KEY`; the key in
   `/av/data/dbs/.env` is `FAL_API_KEY`. Adapters should accept both, in the
   order `config → api_key_env_var → FAL_KEY → FAL_API_KEY` — the same
   multi-spelling tolerance the Friendli provider uses.
6. **Provenance is emerging as a real field.** Several vendors now emit C2PA /
   content-credential metadata. `MediaProvenance` is in the artifact from day
   one so it is not retrofitted later.
7. **Idempotency.** Long-running media jobs are expensive; a retried submit
   must not double-bill. `MediaJobManager` sends a client-generated
   idempotency key where the vendor supports one, and always records the
   submitted key so a resumed process re-attaches instead of resubmitting.

---

## 7. Testing

- **Offline**: `respx` for httpx adapters, `AsyncMock` on vendor SDK clients;
  pin transport `backend` explicitly (the rule in the modernization plan §9).
- **Fake adapter**: an in-repo provider implementing every protocol, driving
  router/selection/job tests with no network.
- **Contract tests**: parametrized over all registered media adapters —
  protocol conformance, artifact normalization, usage population, error mapping.
- **Job lifecycle**: simulated queue→running→succeeded, failure, cancel,
  expiry, webhook-vs-poll equivalence.
- **Live smokes**: key-gated under `tests/integration/`, one per vendor, using
  the cheapest model and smallest output.

---

## 8. Configuration sketch

```toml
[media]
default_image_provider = "openai"
default_audio_provider = "elevenlabs"
default_video_provider = "google"
artifact_materialize = "on_expiry"   # always | on_expiry | never
artifact_path = "~/.llmcore/media"

[media.routing]
tts = ["elevenlabs", "openai", "deepgram"]
asr = ["deepgram", "openai", "elevenlabs"]
image_generate = ["openai", "google", "fal"]
video_generate = ["google", "fal", "replicate"]
video_interpolate = ["fal", "replicate"]

[media.jobs]
poll_initial_seconds = 2
poll_max_seconds = 30
job_timeout_seconds = 1800
webhook_base_url = ""                # empty = poll only

[media.providers.fal]
# api_key_env_var = "FAL_KEY"        # FAL_API_KEY also accepted
[media.providers.elevenlabs]
# api_key_env_var = "ELEVENLABS_API_KEY"
```

---

## 9. Open questions

1. **`[media.providers.*]` vs `[providers.*]`** — a separate section (clean
   separation, some duplication for OpenAI/Google which appear in both), or
   extend the existing provider sections (single source of credentials, but
   `[providers.fal]` would be a chat provider that cannot chat)? *Recommendation:
   extend `[providers.*]`, with `[media.routing]` separate — one credential per
   vendor, and the capability matrix already tolerates non-chat providers
   (`deepgram`, `typesafe`).*
2. **Artifact retention default** — `on_expiry` costs disk silently. Acceptable?
3. **Does the bridge expose media?** The gRPC/HTTP bridge would need artifact
   streaming; defer to a later phase or design in now?
4. **Keys still needed**: `REPLICATE_API_TOKEN` (M7), `BFL_API_KEY` /
   `LUMA_API_KEY` (M9). `ELEVENLABS_API_KEY` and `FAL_API_KEY` are available.
