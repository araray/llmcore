# Media Subsystem — Design & Specification

Generative image, audio and video as a first-class `llmcore` subsystem, plus the
provider adapters that sit behind it.

- **Status:** **M1–M6 implemented** (core subsystem; Deepgram, OpenAI and
  Gemini migrated behind the protocols, fal as the first marketplace adapter,
  ElevenLabs as the first with consent metadata). M5's neutrality gate passed
  unchanged; **M6 deliberately extended the core**, which is what its gate
  asked for. M7 onward not started — see §5.
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
`checksum_sha256`. Policy: `artifact_materialize = "always" | "on_expiry" | "never"`.
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
| **M3** ✅ | **OpenAI** media: images generate/edit, TTS (+streaming), ASR, and provider-level embeddings. Realtime audio deferred to a later phase with Gemini Live. | Landed 2026-09-30, 42 tests. Live: TTS → artifact → ASR round trip, streaming TTS, embeddings |
| **M4** ✅ | **Google** media: images (dual transport), **Veo** video (async job), native TTS, embeddings | Landed 2026-09-30, 52 tests. Live: Veo job submitted + polled through `MediaJobManager`; 2 MB image; 112 KB PCM TTS. **`MediaJob` validated against a real vendor — no changes to the abstraction were needed.** |
| **M5** ✅ | **fal** — queue lifecycle, URL inputs, video, SFX, music, FILM interpolation; 9 capabilities, all async-job | Landed 2026-09-30, 79 tests. Live: image, TTS → ASR round trip, CDN upload, FILM interpolation, upscale, music, cancel. **The provider-neutrality test passed — no core type changed.** See §5.2 |
| **M6** ✅ | **ElevenLabs** — TTS, streaming TTS, batch STT, SFX, music, voice design | Landed 2026-09-30, 69 tests. Live: TTS + consent, streaming TTS, STT round trip, SFX. Music/voice design are **plan-gated on the current account** — implemented and unit-tested, not live-validated. **Consent is now first-class: new `VoiceConsent` type and `VoiceDesignProvider` protocol.** Realtime STT deferred — see §5.3 |
| **M7** ✅ | **Replicate** — one generic prediction adapter + model-schema descriptors | Landed 2026-09-30, 70 tests. Live: schema-driven field mapping, image generation, community-model ASR, cancel. **One adapter, seven capabilities, zero per-model classes.** See §5.5 |
| **M8** | **Hugging Face Inference Endpoints** — configurable endpoint/schema adapter | Custom weights / private repos path |
| **M9a** ✅ | **Webhook receiver** — signed single-use tokens, generic ASGI app, fal callback parsing | Landed 2026-09-30, 38 tests. Polling stays the fallback; webhook/poll equivalence tested |
| **M9b** | Direct specialists (BFL, Luma, Stability) when justified: lower unit cost, first-party-only feature, data contract, or pre-aggregator access | Otherwise fal/Replicate already cover it |

Runway stays on the watchlist — the research run could not verify its current
API contract, and the doc is explicit about not freezing a guessed model id.

---

### 5.1 What M4 changed about the design

Nothing in the core abstraction. `MediaJob`, `MediaJobManager` and the polling
protocol absorbed a real vendor's long-running-operation shape unmodified,
which is the main thing this phase was meant to find out.

Three provider-level facts did emerge, all from live calls rather than docs:

1. **Imagen's dedicated endpoints are Vertex-only.** `generate_images`,
   `edit_image` and `upscale_image` fail on the Gemini Developer API with
   *"only supported in Gemini Enterprise Agent Platform mode"*. Capability
   declaration is therefore **mode-aware**: `image_edit` / `image_upscale` are
   advertised only when `vertex_ai = true`. This is the first provider whose
   capability set depends on configuration rather than on the class.
2. **Image generation needs two transports.** On Vertex it is Imagen; on the
   Developer API it is `generate_content` with an `IMAGE` response modality.
   Same capability, different call — which is exactly what the protocol
   indirection is for, and the caller never sees the difference.
3. **Veo cannot be cancelled.** `cancel_media_job()` raises rather than
   reporting a cancellation that did not happen, because a false success would
   let a caller believe billing had stopped.

### 5.2 What M5 changed about the design

**Nothing in the core abstraction** — which is the answer this phase existed to
get. fal is structurally unlike the first four adapters: a marketplace rather
than a first-party vendor, every capability queued rather than only the slow
ones, inputs addressed by URL rather than by bytes, and output schemas that vary
per model instead of following one house style. It needed no new field on
`MediaJob`, no new `MediaExecution` member, and no change to `MediaJobManager`.

Three existing design decisions did the work:

1. **Execution class is per capability, not per provider.** `image_generate` is
   `REQUEST_RESPONSE` on OpenAI and `ASYNC_JOB` on fal. Callers that use
   `media.wait()` never notice, because the router already returns whichever of
   `MediaResult` / `MediaJob` the provider reports.
2. **`provider_metadata` absorbed everything fal-specific** — the endpoint path,
   the raw submission, the result and cancel URLs, the cancellation caveat.
   None of it leaked into the shared types.
3. **`MediaRef` already distinguished remote from local.** A fal input that is
   already a URL passes straight through; only local bytes are uploaded. No
   artifact round-trips through the process just to be re-uploaded.

Four provider-level facts emerged, all of them from live calls rather than docs:

1. **The queue is namespaced by application, not by model path.** A request
   submitted to `fal-ai/flux/schnell` is tracked at `fal-ai/flux/requests/{id}`;
   polling the full model path returns `405`. The adapter therefore prefers the
   absolute `status_url` / `response_url` / `cancel_url` that fal returns at
   submission, and only falls back to a reconstructed, app-scoped path.
   *Generalizable rule: when a provider hands back URLs, use them — do not
   rebuild routes it owns.*
2. **Storage is a separate host with two backends.** Uploads go to
   `rest.fal.ai`, not `fal.run` (which reads `storage/upload` as an owner/app
   pair and 404s). On that host, `storage_type=gcs` answers *"Invalid storage
   type"* for newer accounts, which use `fal-cdn-v3`. The adapter mirrors the
   official client and tries CDN v3 first, then the signed-URL flow, so it works
   across account vintages rather than for whoever wrote it.
3. **Cancellation is a request, not a guarantee.** fal answers
   `202 CANCELLATION_REQUESTED` and may still complete work already running, and
   a `400 ALREADY_COMPLETED` means the job finished — not that the call failed.
   The adapter treats that 400 as success, fetches the result, and records the
   caveat in `provider_metadata` rather than implying billing stopped. This is
   the *third* distinct cancellation semantic across adapters (Gemini: refuses;
   Deepgram: n/a; fal: best-effort) and the handle models all three.
4. **Model input schemas are per model, not per capability.** FILM takes
   `start_image_url` / `end_image_url`, not a frame list. Endpoint paths are
   configurable per capability under `[providers.fal.models]` precisely because
   the gallery moves faster than a release cycle.
5. **The default `ON_EXPIRY` materialization policy does not protect fal
   artifacts.** fal CDN URLs are not permanent, but fal does not publish a TTL,
   so `MediaArtifact.expires_at` is `None` and the policy has nothing to fire
   on. The adapter deliberately does **not** invent an expiry — a fabricated
   timestamp is worse than a missing one, because callers would trust it.
   Documented instead: pass `force=True` or set `artifact_materialize = "always"` when
   fal output must survive. *Open design question for a later phase: whether
   `ON_EXPIRY` should treat "provider known to expire artifacts, TTL unknown"
   as a third state rather than collapsing it into "no expiry".*

One open consequence, **since closed by M9a**: fal supports webhooks
(`?fal_webhook=`), which the adapter could send but nothing in llmcore could
receive. The generic receiver now issues fal a single-use callback URL per job
and parses its delivery — see §5.4.

### 5.3 What M6 changed about the design

Unlike M5, this phase **did** extend the core — which is what its gate asked
for. Two additions:

**1. `VoiceConsent`, hung off `MediaProvenance`.** Synthetic speech raises a
question no other media kind does. A generated image resembles no one in
particular; a cloned voice belongs to a person who either did or did not agree
to it. ElevenLabs tracks that state — `category`, `is_owner`, `safety_control`,
`voice_verification` — but only on the *voice* resource, so a caller wanting to
refuse audio from an unverified clone would have to know to make a second API
call. The adapter resolves it (cached per voice) and attaches it to the
artifact.

The design decision worth recording is the **tri-state**.
`verification_satisfied` returns `None` when the provider said nothing, which is
*not* `False`. Collapsing the two would force a default: either silently
treating unknown voices as cleared, or refusing audio from every provider that
reports nothing. Both are policy, and policy belongs to the caller. The same
reasoning drives two related choices:

* A **failed consent lookup** yields `provider_declared=False` with every field
  `None`, rather than raising. The caller asked for speech; losing the metadata
  should not lose the audio, and the `None`s read correctly as *we do not know*.
* **Designed voices** state `category="generated"`, `requires_verification=False`
  explicitly instead of leaving consent `None` — "this imitates nobody" is a
  known fact, not an unknown one.
* **SFX and music** carry no consent record at all, because nothing there is
  anyone's voice and an empty record would imply a question that does not apply.

**2. `VoiceDesignProvider`.** `MediaCapability.VOICE_DESIGN` had been mapped to
`TTSProvider` as a placeholder since M1. ElevenLabs is the first provider to
actually implement it, and voice design is not TTS: it returns *candidate
voices* from a description rather than speech from text, so the result is a set
of previews each carrying the id needed to keep it. The M1 protocol-coverage
invariant caught the placeholder immediately when the real protocol landed —
the guard working as intended.

Two provider-level findings, both from live calls:

1. **Plan gating must not be reported as an auth failure.** A perfectly valid
   key on a free plan returns `402 paid_plan_required` or `403
   feature_not_available`. The first implementation reported the 403 as
   *"authentication failed — check ELEVENLABS_API_KEY"*, which would send a
   caller hunting a credential problem they do not have. Now mapped as plan
   gating, explicitly stating the key is valid. *Generalizable: 403 is not
   always about credentials.*
2. **ElevenLabs sound generation is text-conditioned only.** The `SFXProvider`
   protocol accepts a `video` reference for foley; ElevenLabs cannot use it. The
   adapter **raises** rather than ignoring it, because silently dropping it
   would return audio unrelated to the footage the caller passed. A capability
   two providers both "have" can still differ in what it accepts, and the
   honest move is to refuse the part that cannot be honoured.

**Deferred: realtime STT.** ElevenLabs offers a realtime speech-to-text
websocket, and `ASR_STREAM` routing already names it. It is not implemented
here. Realtime ASR is duplex — the caller pushes audio *and* consumes events —
so it needs the session shape Deepgram already established, and doing it
properly is its own piece of work rather than a sixth capability bolted onto
this one. The adapter therefore does **not** declare `asr_stream`, so routing
falls through to Deepgram instead of advertising something that would fail.

### 5.4 What the webhook receiver settled

Three decisions are worth recording, because each had a tempting wrong answer.

**1. Polling is the fallback, not the backup plan.** `wait()` races the callback
against its existing backoff sleep and then polls anyway. It would have been
simpler to branch — await the callback when configured, poll when not — but that
produces two code paths with two sets of bugs, and the webhook path would be the
one nobody tests. Racing means a missing, late, duplicated or malformed delivery
can only cost latency, never correctness, and it is why `webhook_base_url = ""`
(the default) is a fully supported mode rather than a degraded one.

**2. The callback URL is issued before the job exists.** Vendors want it *at
submission*, but the job id only exists once submission returns. So a token is
**reserved**, handed to the vendor, then **bound** to the job that comes back —
and a delivery arriving in that window is refused, because there is nothing it
could correctly report on. Tokens are HMAC-signed, verified in constant time and
single-use; the token→job binding lives server-side, which is what stops a token
being transplanted onto another job.

**3. A callback URL is only offered to adapters that opt in.** Most media
adapters forward unknown keyword arguments straight into the vendor payload —
fal does exactly this, deliberately, so model-specific fields work without an
llmcore change. Passing `webhook_url` blindly would therefore post our callback
URL to a diffusion model as a generation parameter. Adapters set
`accepts_webhook_url = True`; everything else is never offered one, and fal
lifts the value out of its payload before the request is built.

A fourth, smaller one: a provider that cannot *parse* its callback still
benefits from receiving it, because the delivery says *when* to look even if not
*what* happened. Those adapters fall back to a single poll, which is most of the
latency win at no correctness cost.

**Security posture.** A delivery may only report on a job llmcore already
submitted. It cannot create a job, redirect one to another provider, or attach
an artifact to work that was never submitted. An unknown or spent token gets
`404` rather than `401`, because confirming that a token exists but is used
tells an unauthenticated caller something they should not learn.

---

### 5.5 What M7 proved about the generic-adapter bet

The spec's requirement — *one generic prediction adapter, explicitly not a class
per model* — only works if a model can describe itself. Replicate's do: every
model publishes an OpenAPI schema naming its own inputs and outputs.
`flux-schnell` requires `prompt` and returns an array of URIs; `whisper`
requires `audio` and returns an object with a `transcription` field.

So the adapter maps llmcore's canonical protocol arguments onto whatever each
model actually calls them, read from that model's schema. Verified live: `n`
became `num_outputs` for flux while `audio` stayed `audio` for whisper, with
neither hardcoded. A model llmcore has never heard of works, and a model that
renames `image` to `input_image` next week keeps working.

Three supporting decisions:

1. **Explicit keyword arguments always win** over the mapping, because the
   caller knows their model better than a candidate list does.
2. **A schema lookup failure degrades, it does not raise.** The schema is an
   optimization for field naming; losing it falls back to canonical spellings
   and lets the API answer. A 422 naming the field is far more useful than a
   silently dropped input.
3. **An unrecognised output shape yields no artifacts rather than an error**,
   with the untouched payload kept on the job — the same tolerance fal needed.

Two provider-level findings, both from live calls:

1. **Replicate has two creation routes and the reference does not say which.**
   *Official* models run unversioned at `/v1/models/{owner}/{name}/predictions`;
   *community* models `404` there and must be run by version at
   `/v1/predictions`. `openai/whisper` is the second kind. The obvious fix —
   try the first, fall back on the 404 — **costs two creation requests**, and
   Replicate throttles accounts under $5 of credit to a burst of **1**, so the
   fallback reliably turned a working call into a `429`. The version is
   therefore resolved from the model lookup already made for the schema, which
   is a `GET` and does not count against prediction-creation limits.
   *Generalizable: a retry-based fallback is not free when the thing you are
   retrying is the rate-limited operation.*
2. **Output URLs frequently carry no file extension**, so MIME detection came
   back `None`. Rather than guessing a format, the adapter falls back to the
   `output_format` that was *requested* — grounded in the call rather than
   invented — and stays `None` when neither is available, because callers key
   decode paths off this field and a plausible lie is worse than an honest
   unknown.

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
