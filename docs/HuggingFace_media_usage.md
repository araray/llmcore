# Hugging Face — media usage guide

The `huggingface` provider serves **chat and media**, unlike the media-only
adapters (fal, Replicate, ElevenLabs). This guide covers the media half; see
`CONFIG_REFERENCE.md` for the chat side.

| Capability | Default model | Router call |
|---|---|---|
| `image_generate` | `black-forest-labs/FLUX.1-schnell` | `llm.media.images.generate(...)` |
| `tts` | `hexgrad/Kokoro-82M` | `llm.media.audio.speak(...)` |
| `asr` | `openai/whisper-large-v3-turbo` | `llm.media.audio.transcribe(...)` |

---

## 1. What makes Hugging Face different

It is not a vendor in the way the other adapters are — it is a **routing layer**
over third-party inference providers (fal-ai, nscale, deepinfra, together, plus
HF's own `hf-inference`). Two consequences you will actually notice:

**A model is not served for every task by every provider.** `Kokoro-82M` does
TTS on fal-ai and deepinfra but *not* on hf-inference. And each provider knows
the model by **its own id** — the Hub says `black-forest-labs/FLUX.1-schnell`,
fal-ai says `fal-ai/flux/schnell`.

llmcore reads the Hub's routing table and resolves both automatically. You pass
the Hub id; the right provider and the right id are looked up (once, cached).

```python
routing = await provider.get_inference_routing("hexgrad/Kokoro-82M")
# {'text-to-speech': ('fal-ai', 'fal-ai/kokoro/american-english')}
```

Pin a provider if you want one:

```toml
[providers.huggingface]
provider = "nscale"
```

---

## 2. Custom weights and private repos

This is the reason to use HF media rather than a marketplace. A private model is
**not a model id on a shared router** — it is a dedicated Inference Endpoint you
deployed, at your own URL, possibly serving weights nobody else can see.

```toml
[providers.huggingface.endpoints]
asr = "https://xxxxx.us-east-1.aws.endpoints.huggingface.cloud"
```

Once set, that capability:

- posts to your URL **verbatim** — no provider routing, no model id in the path;
- switches to **direct HTTP**, because your deployment speaks the standard HF
  task schema and you control it;
- leaves every other capability routing as normal.

---

## 3. Which transport is used, and why

Hugging Face is llmcore's **one documented exception** to the direct-REST-first
rule. The reason is concrete: the router hands third-party providers *their own*
request shape. `hf-inference` takes `{"inputs": ...}`; the same model on fal-ai
wants `{"prompt": ...}` and returns `422 Field required` otherwise. That shape
belongs to the provider and changes on their schedule, so reimplementing the
mapping in llmcore would mean tracking every provider's schema forever —
absorbing it is exactly what `huggingface_hub` is for.

Both transports are available; only the default differs.

| Traffic | Backend | Why |
|---|---|---|
| Router, JSON body (image, TTS) | **SDK** | The body shape is the provider's, not HF's |
| Binary input (ASR) | **direct** | The SDK sends raw audio with no `Content-Type`, which hf-inference rejects; and there is no provider-specific body to translate |
| Dedicated endpoint | **direct** | Your deployment, standard schema, your URL |

Override globally if you want:

```toml
[providers.huggingface]
media_backend = "httpx"   # "auto" (default) | "sdk" | "httpx"
```

---

## 4. Using it

```python
image = await llm.media.images.generate(
    "a calico cat asleep on books", provider="huggingface", size="1024x768"
)
speech = await llm.media.audio.speak("Inference providers work.", provider="huggingface")
text = await llm.media.audio.transcribe(
    audio=MediaRef.from_bytes(speech.artifacts[0].data,
                              mime_type=speech.artifacts[0].mime_type),
    provider="huggingface",
)
print(text.artifacts[0].text)     # ' Inference providers work.'
```

Note that ASR takes **bytes**, not a URL — unlike fal and Replicate. A remote
`MediaRef` is fetched here first, so the audio does pass through your process.

TTS artifacts carry **no** `VoiceConsent`: HF serves open-weight voices and
tracks no per-voice consent, so claiming anything there would be inventing it.
Contrast ElevenLabs, where the vendor does track it.

---

## 5. Troubleshooting

| Symptom | Cause |
|---|---|
| `402` / *"no inference credits left"* | Your token is fine; the account is out of inference credit. Inference Providers bill per call; PRO includes a monthly allowance |
| `400 Model not supported by provider` | That provider does not serve that *task* for that model. Clear `provider` to let llmcore route, or pick a provider the Hub lists as `live` |
| `422 Field required: prompt` | A provider-native body reached a route expecting the HF schema. Should not happen on `auto`; check `media_backend` |
| `Content type "None" not supported` | Binary input sent without a MIME type. Set `mime_type` on the `MediaRef` |
| `503` | The model is cold-loading. Retryable — llmcore marks it so |

---

## 6. Related

- the media subsystem design spec — §5.6 on the routing
  problem and the transport exception
- the provider support matrix — live validation log
