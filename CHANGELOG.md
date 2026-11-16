# Changelog

All notable changes to **llmcore** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added — Gemini Live: realtime transcription and a voice agent

- `open_transcription_session()` (`asr_stream`, `gemini-3.5-transcribe-live`)
  and `open_voice_agent()` (`voice_agent`, `gemini-3.8-live`) over the Live
  API, with the same session surfaces as Deepgram and OpenAI. Live fixes the
  system instruction at connect time, so `update_prompt` raises.
- Both are declared on the Developer API, where they were verified live:
  exact transcription, and a full tool loop with a spoken answer.

### Added — OpenAI: Realtime voice agent (`voice_agent`)

- `open_voice_agent()` opens a Realtime speech-to-speech session with
  Deepgram's agent surface: `send_audio`, `inject_user_message`,
  `inject_agent_message`, `update_prompt`, `respond_to_function_call`, plus
  `interrupt` for barge-in. It emits `VoiceAgentEvent`s (audio, both sides'
  transcripts, function calls, speaking markers, errors), and takes tools,
  a greeting, a voice and `settings` overrides.
- Verified live on `gpt-realtime-2.1-mini`. A full tool loop ran
  (preamble → `get_weather({"city": "Lisbon"})` → result → spoken answer),
  with about 8 s of 24 kHz audio.

### Added — OpenAI: realtime transcription (`asr_stream`)

- `open_transcription_session()` uses the Realtime API in transcription mode
  (`wss …/v1/realtime?intent=transcription`), with the same session shape
  as Deepgram and ElevenLabs. Events:
  - deltas → `interim` (the text so far);
  - completed → `final`;
  - VAD → `speech_started` / `utterance_end`.

  Options: `language`, `prompt`, `keywords`, `delay`, `noise_reduction`,
  `turn_detection`, logprobs.
- The default `gpt-4o-transcribe` runs with server VAD.
  `gpt-live-transcribe` rejects turn detection, so it runs on manual
  commits.
- Verified live: both models returned the exact sentence.

### Added — Mistral, Z.ai and Gemini media reachable through `llm.media`

- **Mistral** (TTS, ASR, OCR) and **Z.ai** (TTS, ASR, OCR, images, video)
  had working legacy methods that the media manager never routed to, because
  neither provider was a media adapter. A shared bridge
  (`providers/_media_bridge.py`) now exposes them through the media
  protocols. Z.ai video is a polled `MediaJob` on `/async-result/{id}`, and
  its cancel refuses honestly because Z.ai has no cancel endpoint.
- **Mistral speech** sent no model unless one was given, and the API now
  answers 422 "No model provided for speech". It defaults to
  `voxtral-mini-tts-3` (configurable as `tts_model`).
- **Gemini `asr`** via `gemini-3.5-transcribe` on the Developer API
  (`generateContent`). It answers with an `audio_transcription` part, which
  `response.text` ignores.
- Verified live: Mistral and Gemini transcription returned the exact text.
  Mistral OCR and TTS, and Z.ai video, reach the API and stop only on those
  accounts' quota or balance.
- README: a table of which provider serves which media capability.

### Fixed — agents `ModelRouter` defaults named only retired models

- `DEFAULT_MODELS` listed gpt-3.5-turbo, o1-preview, Claude 3, Gemini 1.5 and
  deepseek-chat/-reasoner, with 2024 prices typed by hand. It is now built
  lazily from the bundled model cards. Prices, windows and capabilities are
  read from each card.
- The tiers are:
  - **FAST:** Haiku 5.5, GPT-5.4 nano/mini, Gemini 2.5 Flash(-Lite), qwen3.5:4b;
  - **BALANCED:** Sonnet 5.5, GPT-5.4, Gemini 2.5 Pro, DeepSeek V4 Pro,
    qwen3.5:35b;
  - **CAPABLE:** Opus 5.5.
- A model whose card has no price is left out instead of being treated as
  free; Gemini 3.x and GPT-5.5+ cards are unpriced today.

### Fixed — model cards: one file per model, Ollama windows from the registry

- **Duplicates removed.** 80 Ollama ids and 7 Hugging Face ids each had two
  card files (cardctl's `a--b-c.json` next to an older `a__b_c.json`), and
  the registry picked a winner by origin. For Ollama that meant, for example,
  qwen3:32b reporting 262,144 tokens instead of 40,960. Each pair is now one
  card at cardctl's filename:
  - curated fields come from the hand-written copy;
  - server facts come from the generated copy;
  - the HF placeholders (128,000) are dropped.

  Two cards filed under the wrong provider directory were removed, including
  an `openai/` copy of Google's retired `text-embedding-004` that still said
  active. A test now fails on any same-id duplicate.
- **Every Ollama context window checked** against the GGUF header of its
  current registry tag (a range read of the model blob, no download).
  `qwen3:4b` is 262,144 (the 2507 retag). The `ministral-3-*` cards had ids
  that cannot be pulled; they are now `ministral-3:{3b,8b,14b}`, with the old
  ids as aliases and 262,144-token windows.

### Added — Deepgram: Flux TTS over the WebSocket

- Incremental text into `stream_speech()` and `open_speech_socket()` with a
  Flux voice used to raise "not supported yet". These calls now use
  `speak.v2` (Speak / Flush / Interrupt / Close; `expressivity`,
  `normalize_volume`, `tag`). Aura stays on `speak.v1`.
- `clear()` sends `Interrupt` on Flux.
- In-band `Error` frames raise `ProviderError`, with the Deepgram code.
- Verified live: `flux-haley-en` over v2 and Aura over v1 each returned
  their own audio.

### Added — ElevenLabs: realtime transcription (`asr_stream`)

- **`open_transcription_session()`** speaks Realtime Scribe
  (`scribe_v2_realtime`) over the direct WebSocket
  `/v1/speech-to-text/realtime`, with the same session shape as Deepgram's
  (`send_audio`, `finalize`, `close`, async iteration). Mapped events:
  - partial → `interim`;
  - committed → `final` (words, start and end with `include_timestamps`);
  - typed errors → `error`.

  Every documented query option passes through, including single-use
  `token` auth. Verified live: a TTS sentence streamed as `pcm_16000` came
  back verbatim, with 17 word timings.
- `websockets>=13` added to the `elevenlabs` extra. The usage guide's stale
  defaults (`scribe_v1`, SFX v2, `cloud_storage_url`) are corrected.

### Fixed — Command Code: gateway Jev is text only

Re-checked after the lane providers changed (2026-11-16). Messages
(`claude-haiku-5-5`), Chat (`deepseek/deepseek-v4-flash`) and Responses
(`gpt-5.4-mini`) all answer live, plain and streaming.

- **`system_one` refuses image `assets` and Box questions.** The gateway
  rejects Box with a 400. Worse, it accepts an image request and answers
  without the image: a red-square Noul came back 0.25, against 0.99 on the
  direct `typesafe` provider. The error points to the `typesafe` provider.

### Fixed — Higgsfield: live parameter enums, first_frame, SDK errors, request deletion

Audited against openapi.json 2.0.0, the live API and higgsfield-client 0.2.0
(the latest) on 2026-11-16. A Soul image with `size="1024x768"` completed
live on both backends.

- **Soul image `size`** was sent as `resolution`, a 422. The live API takes
  `720p`/`1080p` and seven aspect ratios, while the published openapi.json
  still lists 2K/4K and ten ratios. A `"WxH"` size now picks the resolution
  and the nearest accepted ratio.
- **Video**:
  - The media protocol's `first_frame` was ignored, which put a `MediaRef`
    into the JSON body. It now selects the image-to-video endpoint.
  - `duration_seconds` snaps to the integer enums the API enforces (Hailuo
    6/10, Kling 5/10).
  - Portable arguments a model does not declare are dropped with a warning.
- **SDK backend**: `HiggsfieldClientError` (status only on `__cause__`) used
  to escape raw. It now maps to `ProviderError` with the HTTP status (live:
  422, non-retryable).
- **`delete_request_data(request_id)`** wraps `DELETE /requests/{id}`
  (stored inputs and outputs). An unknown id returns `False`, verified live.

### Fixed — Replicate: SDK backend, Cancel-After, enum-aware inputs, Hailuo 2.3

Audited against the HTTP API reference and replicate 1.0.7 (the latest) on
2026-11-16. Every default model was checked live. Kokoro on the SDK backend,
FLUX on the direct backend with `Cancel-After`, and the 422 and 429 mappings
were verified live.

- **SDK backend**:
  - It always ran by `model=`, so community models (whisper, kokoro,
    musicgen) got a 404. It now runs by the resolved version, like the
    direct path.
  - It forwards `webhook_events_filter`.
  - `ReplicateError` maps to `ProviderError` with its status: a 402 is
    explained and a 422 is non-retryable.
- **`cancel_after`** (config or per call) sends the documented
  `Cancel-After` header, so a stuck prediction stops billing. It only works
  on the direct backend; the SDK takes no headers.
- **Enum-aware inputs**: Cog enums behind `allOf`/`$ref` are resolved, and
  portable numeric arguments snap to the nearest allowed value. Hailuo's
  `duration` is `[6, 10]`, so 8.5 becomes 10.
- **Default video model**: `minimax/hailuo-2.3` (official, text- or
  image-to-video) replaces `minimax/video-01`. The card, cardctl, config and
  docs are updated to match.

### Fixed — fal: retired default endpoints, image size, SDK backend parity

Audited against fal's Platform API, the per-endpoint OpenAPI schemas and
fal-client 1.0.3 (the latest) on 2026-11-16. Image, TTS, text-to-SFX and
error paths were verified live.

- **Default video endpoint** `fal-ai/minimax-video` was gone (404). It is
  now `fal-ai/minimax/hailuo-2.3/standard/text-to-video`. A `first_frame`
  goes to the image-to-video sibling, and `duration_seconds` maps onto
  Hailuo's `"6"`/`"10"` enum.
- **SFX**: `fal-ai/mmaudio-v2` requires a video, so a text-only request now
  goes to `fal-ai/mmaudio-v2/text-to-audio`.
- **TTS** defaults to `fal-ai/kokoro/american-english`; the bare id has no
  schema.
- **Image size**: `size="1024x768"` is sent as `image_size={"width",
  "height"}`. The raw string used to be sent, which FLUX does not accept.
  Presets still pass through.
- **SDK backend**:
  - The per-job webhook URL is forwarded.
  - A `Completed` status that carries an error fails the job.
  - `FalClientHTTPError` maps to `ProviderError` with fal's status, so a
    422 is non-retryable, as on the direct backend.
- **Cards**: the minimax-video and bare kokoro cards were removed. Hailuo
  2.3 (t2v/i2v), MMAudio text-to-audio and Kokoro American English were
  added.

### Added — TypeSafe: Jev 1.14, Box questions, image assets, long context

Audited against typesafe-sdk 0.7.5 and the live API on 2026-11-16. Text,
image and Box questions were verified live on both transports.

- **Jev 1.14**: `jev-latest` and `jev-preview` point to `jev-1.14.0` (the
  API answers with that id). There is a new card, cardctl maps the aliases
  there, and 1.13 keeps its own card. Both take images.
- **Box questions** locate things in an image (`Box`, `BoxAnswer`,
  `result.boxes`). Box answers used to be discarded as an unknown type.
- **Image assets**: `system_one(..., assets={"ref": png_or_jpeg_bytes |
  path | dict})` with the `TypeSafe-Danger-Zone-Images` header added
  automatically. `long_context = true` sends the Long-Context header and
  budgets 262k tokens.
- **SDK backend**: every failure was `retryable=True`, a 400/401/422
  included. Failures are now classified by type and status. Headers,
  timeouts and assets are forwarded.
  `typesafe-sdk>=0.7.5,<0.8`.

### Changed — Deepgram: Flux TTS, SDK transcription parameters, newer fields

Audited against deepgram-sdk 7.19 and the live API on 2026-11-16. Flux TTS
and transcription were verified live on both transports.

- **Flux TTS** (`flux-<voice>-<lang>`, GA 2026-08-12) goes to `/v2/speak`
  (`speak.v2`) for synthesis and REST streaming, with `expressivity`,
  `normalize_volume` and `priority`. Aura stays on `/v1/speak`. WebSocket
  input with a Flux voice raises a clear error; it uses a different protocol
  (follow-up).
- **SDK transcription** forwarded `channels` and `sample_rate`, which the
  SDK's `transcribe_file`/`transcribe_url` don't take, so it raised
  `TypeError`. They are now sent on the direct transport only.
- New fields: `language_hint` (nova-3 `multi` now serves 55 languages),
  `redact_usage`, `diarize_model` on streaming, and `numerals`,
  `profanity_filter`, `redact`, `redact_usage` on Flux STT.
  `deepgram-sdk>=7.19,<8`. Note that 7.18 stopped retrying
  `RemoteProtocolError`.

### Changed — ElevenLabs: scribe_v3, SFX v3, SDK backend fidelity, STT list fields

Audited against the live API and elevenlabs 2.78 on 2026-11-16. Verified
live on both backends: TTS, scribe_v3 transcription, streaming and SFX v3.

- **Defaults**: ASR `scribe_v3` (GA 2026-11-09; `scribe_v1` is
  deprecated), SFX `eleven_text_to_sound_v3` (v2 is removed on 2027-03-01),
  streaming TTS `eleven_v4_turbo`.
- **STT list fields** such as `secondary_language_codes` and `keyterms` were
  sent as Python's `str(list)`. They now go as JSON strings; repeated form
  fields return 500 (measured). Remote audio goes as `source_url`
  (`cloud_storage_url` is deprecated). Results carry the duration and the
  `edited_transcript`.
- **SDK backend**: TTS, streaming TTS, SFX and music sent only text, model
  and format. `voice_settings`, `seed`, `audio_effects`, `loop`,
  `prompt_influence`, `force_instrumental` and the rest were dropped. Every
  field the SDK has a keyword for is now forwarded, and the rest are named in
  a warning.
- WAV output is labelled `audio/wav`; it was `audio/mpeg`.
  `elevenlabs>=2.78,<3`.

### Fixed — FriendliAI: SDK 0.16 streaming, per-model effort, reasoning replay, retired models

Audited against friendli 0.16.6 (released 2026-11-04) and the live catalogue
on 2026-11-16. The team account has no credits (inference returns 403), so
streaming was verified with the real SDK over a mocked transport.

- **friendli 0.16 removed `chat.stream`.** Streaming is now
  `chat.complete(stream=True)`, so `backend = "sdk"` streaming raised
  `AttributeError`; the extra's open pin allowed 0.16. Both are handled now.
  The pin is `friendli>=0.16.6,<0.17` (pre-1.0 SDKs break across minor
  versions).
- **effort= follows each model's catalogue `reasoning_options`**: GLM-5.3
  low/high/max, GLM-5.2 toggle plus high/max, GLM-5.1 and gemma-4 toggle
  only. "none" turns the toggle off.
- **Reasoning replay**: `assistant_replay_metadata()` returns
  `reasoning_content`. GLM-5.3 on Friendli preserves thinking by default, so
  it has to come back. Streams accumulate it on the direct and openai
  backends; the SDK's models drop it.
- DeepSeek-V3.2 and MiniMax-M2.5 (deprecated 2026-10) are gone from the
  tables and docs, and their cards are marked deprecated.

### Fixed — Z.ai: streaming on the default backend, GLM-5.3 rules, preserved thinking

Audited on 2026-11-16 against docs.z.ai and the real zai-sdk 0.2.3
signature. The account was out of balance. The GLM-5.3 rules below were
confirmed by the API's own validation, and the error mapping was checked live
on every transport.

- **Streaming was broken on the default `sdk` backend.** zai-sdk's
  `create()` has a fixed signature without `stream_options` or `user`, so
  every stream raised `TypeError`. The old tests mocked `create()` with a
  bare MagicMock. Unknown keys now go in `extra_body`; verified to reach the
  server.
- **GLM-5.3** always thinks: `thinking: "disabled"` is a 400 (code 1210)
  and is kept enabled with a warning. Its effort levels are low/high/max
  (`medium` is a 400); other values move to the nearest.
  Default `glm-5.3`. Flash/FlashX are marked vision, the
  release dates are corrected, and a `glm-5.3-flashx-legal` card was added.
- **Preserved thinking**: `clear_thinking` (config or kwarg) and
  `assistant_replay_metadata()`. Streams accumulate `reasoning_content` onto
  the finish chunk.
- **Errors**: the out-of-balance 429 (code 1113) is non-retryable and says
  so. The direct stream read no error body. `_raw_post` shares the mapping.
  `model_context_window_exceeded` is reported as `length`.
- `effort=` (none → thinking off where allowed). `temperature=0` is sent
  as `do_sample=false`; it used to be clamped to 0.01, so sampling stayed
  on. `native_search=True` adds the built-in web_search tool, and the search
  default is `search-prime`. `responses_api = "always"` uses
  `/api/v1/responses`.

### Fixed — Ollama: silent context truncation, tool-loop replay, direct-backend listing

Audited against a live server (0.32.14) and ollama-python 0.6.3 on
2026-11-16. Verified live on both backends.

- **Silent truncation.** The server loads models with its own default
  context: 4096 on the test GPU (`/api/ps`), while llmcore assumed the
  trained 262,144. A new `num_ctx` setting is sent with every request.
  Without it, the effective window is read from `/api/ps` and reported by
  `get_max_context_length()`, with a one-time warning. The server's own
  figures now come before the static table, which had qwen3 at 262,144
  instead of 40,960.
- **Tool loops.** Assistant turns were sent without their `tool_calls` and
  `thinking`. Both are replayed now, with arguments as objects. Tool results
  get `tool_name`, resolved from the call id or by position, since the SDK
  drops ids. Tool calls keep the server's `id`. `assistant_replay_metadata()`
  returns the thinking, and the final stream chunk carries all of it.
- **Direct backend**: model listing crashed (a dict read as an object); it
  now uses the tags' own capabilities and windows. It honours `OLLAMA_HOST`
  and sends `api_key` / `OLLAMA_API_KEY` (ollama.com cloud).
- `effort=` maps to `think` ("none" turns it off). Usage reports cached
  tokens and the finish reason when the server sends them.
- Default `qwen3.5:4b` (tools, thinking, vision); config and code used to
  disagree, on `llama3` and `gemma3:4b`, neither of which has tools. cardctl
  takes types and capabilities from the server instead of guessing from the
  family, so embedding models stop being carded as chat models.

### Changed — Hugging Face: router catalogue, reasoning replay, broken config section, hub 2.4

Audited against the live router and huggingface_hub 2.4 on 2026-11-16.
Verified live through llmcore: discovery and context, effort, tool loops
with replay (plain and streamed), and batched embeddings.

- **The packaged config section was broken.** Its scalar keys followed the
  `[providers.huggingface.endpoints]` subtable, so TOML put
  `api_key_env_var`, `default_model` and `timeout` inside `endpoints`. They
  never applied, and the provider saw three bogus "dedicated endpoints". The
  keys now come first.
- **Discovery** reads the router's own `/v1/models`: the 132 chat models
  actually served, with each live provider's context window, tools and
  pricing. Model cards cover the rest. Context follows the route: the
  smallest live provider's window, or the pinned one's. Llama-3.3-70B is
  12,288 on one of its providers; the table said 131,072.
- **Tool loops**: first-class `Message.tool_calls` were ignored (only the
  legacy metadata channel was read). Reasoning (`message.reasoning`) is
  extracted, streamed, and replayed as `reasoning`. The router rejects
  `reasoning_content` on its default route (400), so that spelling is
  avoided.
- **Parameters**: `effort` / `reasoning_effort`, `top_k` (advertised but
  silently dropped), `min_p`, `repetition_penalty` and
  `chat_template_kwargs` travel in `extra_body`. `logprobs`, `n`,
  `tool_prompt` and `stream_options` are forwarded. Streams request usage.
- **Media and embeddings**: the legacy `generate_speech` default
  `facebook/mms-tts-eng` has no inference provider any more; the served media
  defaults are used instead. Every SDK failure is wrapped as `ProviderError`
  (hub 2.x raises httpx2 errors, and even `StopIteration` for an unserved
  model). Embeddings go out as one batched request with `dimensions`; they
  used to be one request per text.
- Default `openai/gpt-oss-120b` (7 live providers, tools, reasoning).
  `huggingface-hub>=2.4,<3`. The Orpheus TTS cards no longer claim their
  backbone's 131,072-token window, and cardctl no longer reads a window from
  a TTS/STT/image model's config.

### Fixed — vLLM: guided decoding silently ignored, discovery behind tunnels, newer parameters

Audited against the vLLM source (v0.34.0rc0, `c02ebf136e`). No live server
was available, so this follows the source.

- **`guided_json` / `guided_regex` / `guided_choice` / `guided_grammar` were
  removed in vLLM 0.12**, and the server now treats them as unknown extras:
  HTTP 200 with unconstrained output. llmcore forwarded them as-is. They are
  now translated into `structured_outputs`, with a deprecation warning.
- **Discovery** used its own HTTP client without the configured
  `default_headers`, so behind an authenticated tunnel (the gpu.ai runtime's
  Basic auth) it failed and the context fell back to 4096. It uses the
  provider's client now. Without a configured `default_model`, the first
  served model becomes the default.
- **Embeddings** default to `default_embedding_model` or the served model.
  The inherited default was OpenAI's `text-embedding-3-small`.
- The newer chat fields are accepted: `thinking_token_budget`,
  `include_reasoning`, `stop_token_ids`, `truncate_prompt_tokens`,
  `bad_words`, `allowed_token_ids`, `cache_salt`, `priority`,
  `media_io_kwargs`, `documents`, `vllm_xargs`, `repetition_detection`,
  `return_token_ids`.
- Reasoning replay works through the shared OpenAI-compatible layer (vLLM
  renames `reasoning_content` to `reasoning` on input).

### Fixed — gpu.ai: streaming flags, refused parameters, embeddings contract

Audited against the live API on 2026-11-16 (verified on both transports).

- 12 of 15 chat models list no `supported_parameters` (null or empty) yet
  stream; they were all reported as non-streaming. Only a non-empty list
  without `stream` counts as a negative now, in the provider and in cardctl.
- The endpoint answers 400 to `tools`, `response_format` and array content.
  Tools now raise a clear error before the request, text content parts are
  joined into a string, images raise, models report no tools or vision, and
  `responses_api` is forced to "never" (`/v1/responses` returns 404).
- `create_embeddings` follows the base contract (`input_texts`, an
  OpenAI-shaped dict); it took `texts` and returned bare lists. `dimensions`
  works (64 → 64).
- Discovery skips models whose `status` is not active. cardctl prefers the
  exact microcents prices over rounded cents for cached input.

### Fixed — DeepInfra: dead STT default, model types, schema-2.4 cards, effort levels, dimensions

Audited against the live API on 2026-11-16. Verified live through llmcore:
discovery, embeddings with `dimensions`, the TTS→STT round trip, tool loops
with reasoning replay (plain and streamed), and the Responses route.

- **STT default** `openai/whisper-large` returned 404 and has left the
  catalogue. It is now `openai/whisper-large-v3-turbo`.
- **Model types**: the live tags are short (`embed`, `tts`, `stt`,
  `image-gen`, `video-gen`, `vlm`). None matched, so all 178 models were "chat
  with tools". The classifier now recognises them and reports `reasoning`.
- **cardctl**: the `/openrouter/models` feed moved to schema 2.4, which nests
  modalities, context, parameters and prices. Every model parsed with no
  context, tools, reasoning or price, and 210 cards carried stringified dicts
  as tags. The adapter now reads 2.4 (and still reads the flat shape) and
  dedupes the feed (519 rows, 352 ids); the 210 card tags are repaired.
- **Parameters**: `reasoning_effort` takes none through max (all verified),
  `service_tier` takes `flex`, and the newer body fields are accepted
  (`chat_template_kwargs`, `prompt_cache_options`, `cache_salt`, `fail_fast`,
  `models`, `stop_token_ids`, `media_io_kwargs`, `continue_final_message`,
  `ignore_eos`).
- **Embeddings** keep `dimensions`, live-verified; it was dropped after older
  docs. `input_type` and `task` pass through.
- Default chat model `deepseek-ai/DeepSeek-V4-Flash`.

### Changed — Poe: effort per bot, live context windows, working native backend, retired bots

Audited against the live API and fastapi_poe 0.0.83 on 2026-11-16. Verified
live through llmcore: effort on GPT, Gemini and Claude bots, search, a tool
loop, native history and stream, the 413, and balance.

- **`effort=` follows each bot.** `/v1/models` declares each bot's custom
  parameters with schemas, and reasoning is spelled differently per bot.
  Effort now maps onto `reasoning_effort` / `output_effort` /
  `thinking_level` (nearest enum value), `thinking_budget` (scaled within its
  range) or `enable_thinking`. Measured: gpt-5.4-nano 0 vs 51 reasoning
  tokens, gemini-3-flash 0 vs 472. `native_search=True` sets `web_search` on
  bots that declare it.
- **Discovery** reads the API's own context windows, max output, tools and
  reasoning flags, and display names; `created` is converted from
  milliseconds. Context lookups use them. A failed listing raises; it used to
  return "no bots". cardctl gets the same fields and types video, image and
  audio bots (it used to call them all chat with tools).
- **Native backend**: assistant turns are sent as `bot`. The protocol has no
  `assistant` role, so any history failed validation outside the fallback.
  Custom parameters and `stop` are forwarded, and streams are primed so the
  fallback can engage.
- A 413 ("Your message is too long") is a `ContextLengthError`. `store`,
  `metadata` and `prediction` join the stripped parameters.
  `get_balance_details()` returns the full balance.
- Default `gpt-5.4-mini` (ids are lowercase now). 216 cards for bots the API
  no longer serves (they answer 500) are marked retired. `fastapi-poe` joins
  the `poe` extra.

### Changed — OpenRouter and OpenAI-compatible vendors: reasoning replay, in-band errors, full catalogue, working SDK backend

Audited against the live API and openrouter SDK 1.3.179 on 2026-11-16. All
of it was verified live on the openai, httpx and openrouter-SDK backends.

- **Reasoning replay for every OpenAI-compatible vendor** (OpenAIProvider):
  `assistant_replay_metadata()` returns `reasoning_details` and the reasoning
  text, and the next assistant turn sends them back. The text goes under the
  vendor's field: `reasoning` for OpenRouter and vLLM, `reasoning_content`
  elsewhere. Streams accumulate both onto the finish chunk. Gemini 3 through
  OpenRouter returns its reasoning only as an encrypted `reasoning_details`
  item, which OpenRouter requires to be sent back unmodified and in order.
  OpenAI itself is unaffected.
- **In-band errors**: an OpenRouter failure inside an HTTP 200 (an `error`
  body, or a stream chunk with `finish_reason: "error"`) now raises
  `ProviderError`. It used to come back as empty content.
- **OpenRouter vendor profile**: provider routing, `models` fallbacks,
  plugins, `session_id`, caching and sampling extensions are accepted
  parameters. `native_search=True` adds the `web` plugin, and
  `provider_preferences` in config are merged into every request (e.g.
  `zdr`, `data_collection`). Usage includes `cost`, `cost_details` and
  `is_byok`.
- **Catalogue**: `/models` is paginated and lists only text-output models
  unless asked; discovery now loads all 724 models (not 468), raises on
  failure, and clamps `effort=` to each model's `supported_efforts`. In
  cardctl, `supports_tools` checked `"tool_use"`, which no live model reports;
  it now checks `"tools"` (405 models). It also maps reasoning, vision and
  non-chat model types.
- **SDK backend** (`backend = "sdk"`): it used to drop tool calls, images and
  names from history, yield text-only stream chunks, ignore `base_url` and
  `timeout`, and fall back by raising `ValueError` on routing kwargs. It now
  sends the same body as the default path and returns the same shapes.
  `openrouter>=1.3,<2` is part of the `openrouter` extra.
- Default `openai/gpt-5.4-mini`. `X-OpenRouter-Title` and
  `X-OpenRouter-Categories` headers. "not a valid model ID" is recognised as
  model-not-found. tiktoken is looked up without the `vendor/` prefix.

### Changed — Kimi: K3 family, reasoning replay, learned fixed values, dynamic tools, Responses

Audited against the live Moonshot API on 2026-11-16 (funded account; every
item below verified live through llmcore).

- **K3** (`kimi-k3`, `kimi-k3.1-pro`, `-highspeed`; 1M context) is
  supported. It always reasons, and its depth is `reasoning_effort`
  (low/high/max). `effort=` maps onto it, and "none" turns reasoning off.
  `kimi-k2.7-code` refuses `thinking: disabled` (a 400); llmcore warns and
  keeps it enabled. Fixed sampling values (temperature 1, top_p 0.95) are
  stripped for K3 and k2.7. A new "invalid X: only Y is allowed" rejection is
  dropped, retried and remembered.
- **Reasoning replay**: `assistant_replay_metadata()` returns
  `reasoning_content`, and for streams the finish chunk carries all of it.
  Measured: without it, a K3 tool loop stops reasoning after the first call.
- **Default `kimi-k3.1-pro`**. The moonshot-v1 series, kimi-k2.5 and the k2
  previews now return 404; their cards are marked retired, and K3.1 Pro
  cards were added.
- **Responses API**: `responses_api = "always"` routes chat through
  `/v1/responses` (plain, tool loop and stream verified).
- **Dynamic tools** (K3): a system message with `metadata["tools"]` becomes
  `{"role": "system", "tools": [...]}`.
- `max_tokens` is sent as `max_completion_tokens` (the former is deprecated).
  New parameters: `prompt_cache_options`, `prediction`, `logprobs`.
- Usage reads `prompt_tokens_details.cached_tokens` and `cache_write_tokens`.
- Kimi Code: with `base_url = "https://api.kimi.com/coding/v1"`, the key
  defaults to `KIMI_API_KEY`.
- DeepSeek, Kimi and OpenAI share one SDK `/responses` helper
  (`openai_responses.sdk_create`).

### Changed — DeepSeek: V4.1-Flash, low effort, reasoning replay in streams, built-in tools, Responses

Audited against api-docs.deepseek.com and the live `GET /models` on 2026-11-16.
The test account had no balance (every chat call returned 402), so chat
behaviour follows the documentation; discovery, balance and the 402 mapping
were checked live.

- **Default `deepseek-flash`** (DeepSeek-V4.1-Flash: image input, 1M
  context). V4 Flash was retired on 2026-09-10; `deepseek-v4-flash` is still
  accepted and served by V4.1-Flash. `deepseek-chat` and `deepseek-reasoner`
  were retired on 2026-07-24, and their cards say so.
- **Effort**: the levels are now low/high/max. `low` used to be folded into
  `high`. llmcore's `effort=` maps as DeepSeek documents (minimal/low → low,
  medium/high/xhigh → high, max → max), and `effort="none"` turns thinking off.
- **Tool loops while streaming**: DeepSeek returns a 400 unless every earlier
  assistant turn carries its `reasoning_content`, which a stream only sends in
  pieces. The finish chunk now carries the whole of it, and
  `assistant_replay_metadata()` returns it. A 400 about it explains the fix.
- **Built-in tools** on `deepseek-flash`: `native_search=True` adds
  `deepseek_web_search`, `extra_tools` passes `deepseek_read_url`, and
  `safety_check` is supported.
- **`responses_api = "always"`** uses DeepSeek's native `/responses`.
- Discovery reads the listing's context window, image input and effort
  levels. A 402 is not retryable. `get_balance()` was added.

### Changed — Mistral: reasoning chunks and replay, per-model effort, learned parameters, Responses, fixed streaming

Audited against the live API and mistralai 3.7 on 2026-11-16. On the test
plan, Small, Medium and TTS-3 have a quota of 0 requests per minute, so
reasoning follows docs.mistral.ai. Everything else was verified live on both
transports.

- **Streaming was broken.** `chat_completion(stream=True)` and
  `fim_completion(stream=True)` returned an un-awaited coroutine instead of an
  iterator. Streams are now primed, so HTTP errors raise from the call, with
  the server's message instead of only the status code.
- **Reasoning** (Small 4, Medium 3.5, Large 4): `content` arrives as thinking
  and text chunks. It is flattened to `content` plus `reasoning_content`, and
  the raw chunks are kept as `mistral_content`, which
  `assistant_replay_metadata()` returns. Mistral requires the thinking chunk
  to be replayed. Reasoning support comes from `/v1/models` capabilities, not
  from the `magistral` name (Small and Medium reason now; `magistral-*` are
  their aliases). `effort=` maps per family: none/high on Small and Medium,
  low/medium/high on Large 4.
- **New parameters**: `top_k`, `min_tokens`, `repetition_penalty`,
  `logprobs`, `prompt_cache_key`, `service_tier`. Mistral rejects some per
  model ("… is not enabled for this model"). Such a parameter is dropped, the
  request retried, and the rejection remembered.
- **Errors**: a plan without access to a model gets a 429 with a 0 req/min
  limit. That is now reported as "no quota" and is not retryable. SDK errors
  map exactly like REST errors.
- **`responses_api = "always"`** routes chat through Mistral's stateless
  `/v1/responses`, with tool loops and streaming (verified live).
- **Media**: TTS no longer sends OpenAI's "alloy" voice; it requires `voice`
  or `ref_audio`, and new options pass through. STT forwards extra fields,
  sends the right MIME type, and no longer leaks the file handle. SDK
  embeddings keep `output_dimension` and friends.
- Default `mistral-small-latest`; `mistral-large-latest` is not offered to
  every plan. The context table comes from the live listing. The cardctl
  adapter reads `reasoning` from the API, and 13 cards were corrected.
  `mistralai>=3.7,<4`.

### Fixed — shared Responses builder posted `extra_body` verbatim

`build_responses_request` sent a caller's `extra_body` as a literal field.
Servers that ignore unknown fields, such as vLLM, answered 200 without the
vendor parameters. It is now merged into the body. `OpenAIProvider` also
implements `assistant_replay_metadata()` (Responses reasoning items).

### Fixed — Gemini: tool calling, thought signatures, SDK 3.x, retired models

Audited against the live Developer API and `google-genai` 3.3 (2026-11-16).
Tool use was unusable:

- **Every call with tools raised `AttributeError`.** It used
  `types.FunctionDeclaration.from_dict`, which google-genai has never had.
  Declarations now use `parameters_json_schema`.
- **Tool loops could not continue.** Assistant tool calls were never replayed
  (so the model re-issued them), and Gemini 3+ rejects a replayed function call
  without its thought signature ("Function call is missing a
  thought_signature"). Signatures are now captured: per call in
  `tool_calls[i].extra_content.google.thought_signature` (Google's own
  convention), and the whole turn as `message["gemini_parts"]`, returned by
  `assistant_replay_metadata()`. Turns are replayed with them. Hand-built or
  foreign history gets Google's documented placeholder. Tool results carry the
  call `id` and the function name (the id was used as the name).
- **Streaming** now emits tool calls, the finish reason and usage, and the
  final chunk carries the parts to replay. All three were dropped.
- **google-genai 3.x** (`>=3.3,<4`): the Developer API is pinned to `v1beta`
  (`api_version`), because the SDK's new `v1` default has no TTS ("Unknown name
  speechConfig"), no explicit caching and no `-latest` aliases. The extra
  declares `httpx` (3.x installs httpx2 only) and no longer requires
  `google-api-core`, which made the provider unavailable without it.
- **Video**: the Developer API shut its Veo previews down on 2026-10-22, and
  SDK 3.x refuses `generate_videos` outside Vertex. `video_generate` is now
  declared on Vertex only (`veo-3.1-generate-001`).
- **Embeddings**: `gemini-embedding-2` returned one aggregated vector for a
  list of texts. Each text is now its own input. The default is
  `gemini-embedding-2`.
- `effort=` maps to `thinking_level` (Gemini 4 and 3.8 have no `minimal`;
  `xhigh` only on Hydrogen), and to a budget on 2.5. `thinking_display` returns
  thought summaries. Thinking tokens count as completion tokens; they are
  billed as output and were left out.
- `ContextLengthError` only for the input-too-long error. Any message
  mentioning "token" used to qualify. SDK errors carry status codes.
- Defaults: `gemini-3.5-flash-lite` (`gemini-3.1-flash-lite-preview` is shut
  down), TTS `gemini-3.8-flash-tts`, images `gemini-nano-banana-2.1`. The token
  table is refreshed from the API's own limits. cardctl sends the key in a
  header instead of the URL.

### Changed — Anthropic: SDK 2.x / API 2026-10-21, per-model rules, thinking round trip

Audited against the live API and `anthropic` 2.8 (2026-11-16):

- **SDK 2.x and API version 2026-10-21** (`anthropic>=2.8,<3`). That version
  removes `temperature`, `top_p` and `top_k`, which Claude 5.x already
  rejected. They are dropped with a one-time warning; on SDK 1.x they used to
  crash the SDK path with a `TypeError`.
- **"No thinking" is per model, not per generation.** llmcore sent
  `between_tools` to every 4.7+ model, which is a 400 everywhere except Sonnet
  5.5. Per-model capabilities are now read from the Models API once per
  process, with a measured snapshot as fallback. `effort="none"` becomes
  `disabled` (Haiku 5.5, Opus 4.7/4.8, Sonnet 5, 4.6), `between_tools` (Sonnet
  5.5) or the lowest effort (Opus 5.5 and Fable, which have no off switch).
  `xhigh` is kept on models that have it instead of being folded to `high`.
- **Forced `tool_choice`** (`any` or a tool name) on Opus 5.5, Sonnet 5.5 and
  Fable is sent as `auto` with a warning instead of failing.
- **Thinking round trip.** Thinking blocks (with signatures), redacted
  thinking and server-tool blocks come back as `message["anthropic_content"]`
  (streamed calls: on the final chunk) and are replayed verbatim.
  `assistant_replay_metadata(response)` is a new `BaseProvider` method that
  returns what to attach to the next assistant message, for any provider.
- `native_search=True` adds the web-search server tool, and `extra_tools=[...]`
  passes raw server tools. `thinking_display="summarized"` returns readable
  thinking.
- API 2026-10-21 rules are applied: documents are dropped from assistant
  turns, and citations are stripped from user text.
- `ContextLengthError` only for "prompt is too long". Any 400 mentioning
  tokens used to qualify. `model_context_window_exceeded` stops are reported as
  `length`. SDK-path errors carry status codes and the original exception.
- Default model `claude-sonnet-5-5` (`claude-sonnet-4-5-20250929` retires
  2026-11-30). Context windows come from the Models API: 1M for every current
  model, 200K for the 4.5 line.
- Fixed: `_model_generation` read `claude-sonnet-4-20250514` as generation
  (4, 20250514), and `supports_thinking` was true for every model.
- cardctl: the adapter read capability keys that do not exist (`tool_use`,
  `vision`, `extended_thinking`), so every Anthropic card claimed no tools,
  vision or reasoning. It now reads `image_input`, `structured_outputs`,
  `thinking.types` and `effort`, plus windows and lifecycle. 13 cards were
  repaired in place, and cards were added for `claude-haiku-5-5` (with its
  >100K context tier) and `claude-fable-5-2`. The enrichment prices were 2024
  family prefixes (`claude-opus-4` = $15/$75 would have priced Opus 4.8 at
  three times its $5/$25 on regeneration); they are rebuilt from the
  vendor-scraped cards.

### Fixed — xAI, Groq, Together: retired search, retired defaults, vendor profiles

No keys were available for these three, so this follows their API docs and
unauthenticated endpoint checks (2026-11-16), not live calls.

- **xAI search was dead.** `search_parameters` (Live Search) answers HTTP 410,
  and xAI replaced it with the server-side `web_search` and `x_search` tools,
  which exist only on its Responses API. `native_search=True` now routes there
  with those tools. Plain chat stays on Chat Completions; `responses_api =
  "always"` opts in to xAI's recommended route for everything.
- **Groq:** `native_search=True` adds the built-in `browser_search` tool on the
  gpt-oss models, which are the only ones that have it.
- **Together:** base URL `https://api.together.ai/v1`. The old
  `api.together.xyz` serves chat but answers 404 on `/responses`.
- **Retired defaults replaced:** xAI `grok-4.1-20251117` → `grok-4.7`; Groq
  `llama-3.3-70b-versatile` (shut down for free/dev tiers 2026-08-16) and
  Together `Llama-3.3-70B-Instruct-Turbo` (left serverless 2026-10-22) →
  `openai/gpt-oss-120b`.
- **Vendor profiles.** These vendors are instances of `OpenAIProvider`, so
  they inherited OpenAI's media endpoints and default media models. Routing
  could then send them TTS, image-edit or Whisper calls they do not serve.
  Each now declares only what it serves in OpenAI's shape: xAI image
  generation (`grok-imagine-image`), Groq ASR (`whisper-large-v3-turbo`),
  Together none yet. Their documented chat extensions are accepted and sent in
  `extra_body`: Groq `reasoning_format`, `include_reasoning` and
  `search_settings`; Together `reasoning`, `chat_template_kwargs`, `top_k` and
  `min_p`. They used to raise `ValueError`.
- `extra_body` is accepted on every OpenAI-compatible instance.

### Changed — OpenAI: Responses API, per-model parameter rules, current models

Audited against the live API and `openai` 3.51 (2026-11-16). Several things
were broken for every current model:

- **Responses API.** `responses_api = "auto"` (the default against OpenAI)
  sends the o-series and GPT-5+ to `/responses` and keeps gpt-4.x, search and
  audio models on Chat Completions. `gpt-5-pro`, `gpt-5.x-pro` and
  `gpt-5.3-codex` exist only on `/responses` and answered 404 before. A
  Chat-only parameter (`n`, `logprobs`, `seed`, `stop`, …) keeps a call on
  Chat Completions, a Responses-only one (`reasoning`, `text`, `include`, …)
  moves it, and `responses=True|False` decides per call. Both routes return the
  same shape. Encrypted reasoning comes back as `reasoning_items`
  (`extract_reasoning_items()`) and goes out again from
  `metadata["reasoning_items"]`, so reasoning survives tool loops.
- **Parameters are learned per model.** OpenAI models no longer share one
  parameter set: gpt-6-luna rejects `temperature` unless effort is `none`,
  gpt-5 takes effort `minimal` but not `none`, o3 rejects `top_p`. With
  `param_policy = "adapt"` (the default against OpenAI) a rejected parameter is
  renamed, dropped or moved to the nearest supported effort level, the call is
  retried once, and the rule is remembered for that model. `strict` fails
  instead.
- `max_tokens` is sent as `max_completion_tokens` to OpenAI. GPT-5+ rejected
  it. Compatible endpoints keep `max_tokens`.
- `effort=` (llmcore's canonical knob) is accepted, and `reasoning_effort`
  takes `none` through `max`.
- `native_search=True` uses the Responses `web_search` tool, which every model
  has. Chat Completions search works only on `*-search-api` models.
- Context windows are the **input** limits OpenAI enforces, measured from its
  own overflow errors: 272K for gpt-5/5.1/5.2/5.4-mini/5.4-nano, 922K for
  gpt-5.4/5.5/5.6/6.x, and 1,047,576 for gpt-4.1. gpt-4.1 used to resolve to
  8,192.
- `ContextLengthError` is raised by error code on both routes and every
  wording. It had stopped matching from SDK 3.38, whose error text no longer
  carries the body.
- Chat streams ask for usage, so streamed calls now have token counts and
  cost. Errors raised mid-stream are `ProviderError`, not raw SDK exceptions.
- Direct transport: `extra_body` is merged into the body instead of being sent
  as a literal field, and the `OpenAI-Project` header is sent.
- Images default to `gpt-image-2` (`gpt-image-1` no longer exists), and
  `response_format` is sent only to DALL-E models (GPT image models reject it).
- Defaults: `gpt-6-luna` (`gpt-4o` shuts down 2027-01-14). `openai>=3.38`.
- Reasoning text is extracted from every OpenAI-compatible server
  (`reasoning` or `reasoning_content`). Cache-write tokens are reported.
- cardctl: GPT-5+ cards were all-false placeholders (no tools, vision or
  reasoning) because the adapter knew only `o1`/`o3`/`o4` as reasoning models.
  The inference is fixed, and 59 existing cards were repaired in place
  (capabilities only where the block was the placeholder; windows to the
  measured limits; prices untouched).
- Command Code's `/responses` lane now uses OpenAIProvider's Responses code
  instead of its own copy.

### Added — Command Code provider

`commandcode` (aliases `command_code`, `command-code`) reaches Command Code's
Provider API: one key for 98+ models. The gateway speaks three protocols and
each model answers on one, so the provider routes **per model** from the live
`/models` catalog. Claude goes over the Anthropic wire (thinking, effort and
caching come from `AnthropicProvider`), the other chat models over the OpenAI
wire, and `typesafe/jev` through `system_one()`. Spend and spans are attributed
to `commandcode` and priced from its own cards.

- Short names resolve like the Command Code CLI (`kimi-k3` →
  `moonshotai/Kimi-K3`). An ambiguous one raises rather than guessing.
- An upstream failure sent as **HTTP 200** with `finish_reason: "error"` and
  partial content now raises a retryable `ProviderError`, in streams too. The
  fragment is never returned as an answer.
- Double-encoded error messages are unwrapped, with the gateway's upstream
  attempts. Every error carries `trace_id`, `error_code` and `failure_kind`.
- A stale catalog heals itself: told a model lives on another route, the
  provider re-reads the catalog and retries once.
- Zero data retention: config `zdr`, `CMD_ZDR=1`, or per call `zdr=True`.
- **OpenAI Responses API** on `/responses`. GPT models use it by default
  (`responses_api = "openai"`; also `"all"` / `"never"`, or per call
  `responses=True|False`). Requests are stateless with encrypted reasoning, and
  `extract_reasoning_items()` plus `metadata["reasoning_items"]` carry a
  reasoning model's chain across tool loops. Answers come back in the Chat
  Completions shape. The conversion is reusable:
  `llmcore.providers.openai_responses`.
- Direct HTTP by default; `transport = "sdk"` uses the official OpenAI and
  Anthropic SDKs.
- 100 model cards from a new `cardctl` adapter (public catalog, no key).
  Prices come from Command Code's pricing page, with peak rates for DeepSeek
  V4.x and the off-peak rate kept alongside. Capabilities are inherited from
  the origin vendor's card unless that card is a placeholder.

### Changed — routing honours a provider's own failure classification

`classify_failure` now reads a `failure_kind` attribute when an exception
carries one, before inferring from status and text. A provider that knows a
403 means "not served in your region" (fail over) rather than "bad key"
(bench the target) can now say so.

### Fixed — cardctl kept curated cache-write prices

The card builder dropped `cache_write` from enrichment pricing, so a curated
cache-write rate disappeared on every regeneration. It is now carried through.

### Fixed — transcription keeps word timings, speakers and language

`transcribe_audio` threw away most of what OpenAI, DeepInfra, Mistral and
Deepgram return. `TranscriptionSegment` gains `words: list[TranscriptionWord]`
(`word`, `start`, `end`, `probability`, `speaker`; empty by default), and the
four providers fill it:

- **OpenAI / DeepInfra (Whisper):** a plain request to a Whisper model now asks
  for `verbose_json` with `["word", "segment"]` granularities (free for
  Whisper); words are placed in the segment that contains them
  (`attach_words_to_segments`). `gpt-4o-*-transcribe` stays on `json`; a
  `diarized_json` response keeps its speaker labels. A caller's language hint
  wins; otherwise the provider's value is kept as given (Whisper's
  `verbose_json` says `"portuguese"`, not `"pt"`).
- **Deepgram:** words (punctuated form, confidence, speaker) from utterances,
  else paragraph sentences, else one segment over all words; the detected
  language (`detected_language`, or the dominant `languages` entry for
  `multi`) instead of only the hint.
- **Mistral:** the request was a multipart body labelled
  `application/json` (the client's default header won), and array fields
  (`timestamp_granularities`, `context_bias`) went as one JSON string; both
  now match Mistral's own SDK. Segments keep `speaker_id`.

### Fixed — DeepInfra TTS sends its extra fields

`generate_speech` put `instructions` and extra kwargs under a literal
`"extra_body"` key in a raw httpx JSON body, so the API never saw them. They
are now top-level body fields (named arguments win on a clash).

### Added — an ownership rule for strays on a shared account

Every instance a rental backend's state store did not know was reported as an
**orphan**, with the command to adopt it. On an account shared between tools —
the same gpu.ai key used by hand from a CLI and by an application built on
llmcore — that offers to take over, and so to terminate, someone else's
running job.

- `llmcore.runtimes.Ownership(name_prefix=..., known_ids=...)` says which
  untracked compute is the caller's. Pass it to `GpuAiRuntime(...,
  ownership=)` / `DeepInfraRuntime(..., ownership=)`, per call to
  `status(ownership=)`, or set `runtimes.<backend>.owned_name_prefix` /
  `owned_ids`.
- With a rule, a stray the rule claims is still an orphan (ours, record
  lost). One it does not claim is reported as **"not ours"**, with its real
  phase, and no adopt hint.
- `RuntimeStatus.owned` says which: `True` for anything in the state store or
  claimed, `False` for compute started elsewhere, `None` when no rule was given.
  Without a rule nothing changes.

### Added — account balance, spending-limit and usage reads

The runtimes subsystem could rent, price and terminate, but not answer "how
much money is left, and how fast is it going". All in
`llmcore.runtimes.accounts`, exported from `llmcore.runtimes`:

- `GpuAiRuntime.balance()` → `AccountBalance` from `GET /billing/balance`:
  balance, live **burn rate** (every live instance, provisioning included),
  auto-pay, and `runway_hours` (balance ÷ burn, `None` where undefined).
- `GpuAiRuntime.spending_limit()` → `SpendingLimit` from
  `GET /billing/spending-limit`: monthly/daily caps and spend, percent used,
  enforcement. A 404 is reported as `configured=False`.
- `GpuAiRuntime.usage(bucket="day", group_by="instance_id", start=, end=)`
  → `UsageReport` from `GET /usage`, all pages: **the real meter**, with
  `total_cost_usd` and `by_instance()`. `status()`'s accrued spend is rate ×
  elapsed and over-counts (provisioning is unbilled, unreachable windows are
  credited back); this is what was charged.
- `DeepInfraRuntime.balance()` from `GET /payment/checklist`: DeepInfra's
  inverted `stripe_balance` (negative = funds) is turned the right way up;
  usage since the last invoice, the account cap, suspension, and model-scoped
  credit reported apart. Billing address and payment-method details are not
  carried. `DeepInfraRuntime.spending_limit()` from `GET /payment/config`
  (negative = none).
- `openrouter_balance(api_key=None, api_key_env_var="OPENROUTER_API_KEY")`
  from OpenRouter's `GET /credits`: credit bought minus credit used.

**These reads never raise.** A failed read — no key, an HTTP error, a dropped
connection — comes back with `error` set and every figure `None`, so a panel
that shows three vendors does not go blank because one timed out. Every figure
is in dollars, and `None` always means unknown: gpu.ai's `null` burn rate is
kept as `None`, never shown as zero.

### Added — SSH keys for rented instances

A bare instance is reached over SSH, and a gpu.ai launch installs keys by id.
llmcore could pass `runtimes.gpuai.ssh_key_ids` but not find or register one.

- `GpuAiRuntime.ssh_keys()` lists the account's keys (all pages) and
  `GpuAiRuntime.create_ssh_key(name, public_key, idempotency_key=None)`
  registers one. `DeepInfraRuntime.ssh_keys()` reads DeepInfra's
  (`GET /v1/ssh_keys`), for its container rentals.
- Both return `llmcore.runtimes.SshKey` (`id`, `name`, `fingerprint`,
  `public_key`, `created_at`).
- `create_ssh_key` refuses anything that is not a single OpenSSH **public**
  key, and refuses a private key outright, before contacting gpu.ai
  (`llmcore.runtimes.accounts.check_public_key`).

### Added — rent a bare gpu.ai instance

`GpuAiRuntime.up()` always deploys the vllm template with `env.MODEL` and
waits for its HTTPS app URL. A caller that runs its own code on the GPU — its
own environments, models and server, reached over SSH — had no way to rent
the machine.

- `GpuAiRuntime.rent(InstanceRequest, *, name, idempotency_key=None)` rents a
  bare instance and waits for `running` (SSH-ready), not for an app URL. The
  handle (`api_style="ssh"`, `recipe="instance"`, no `base_url`) carries
  `ssh_command`, `hostname`, `port`, `terminal_url`, `auto_terminate_at` and
  the price in `metadata`.
- `InstanceRequest` maps `POST /instances` field for field: `environment` or
  `image` (with `entrypoint`/`cmd`/`ports`), `disk_gb`, `ssh_key_ids`, `env`,
  `auto_terminate_hours`, a pinned `offering_id`/`region`, and the
  `max_price_per_hour`/`viewed_price_per_hour` ceilings.
  `InstanceRequest.from_offer(offer, disk_gb=...)` pins a `GpuOffer` with its
  price, rounded up to the cent, as both ceilings. `problems()` finds what
  gpu.ai would refuse (image vs environment, port 22, reserved env names,
  lifetime bounds…) and `rent()` refuses before sending anything.
- Same guarantees as `up()`: recorded before the create, idempotency-keyed
  (pass your own key to make a post-crash retry return the same instance),
  fail-closed, and `auto_terminate_hours` defaults to the configured lifetime
  rounded up. `env` values are never recorded, only their names. A name that
  already records a live instance is refused rather than overwritten.
- Public `GpuAiRuntime.get(id, credentials=False)` and
  `GpuAiRuntime.terminate(id)`. `terminate` is idempotent (404/409 are
  "already gone") but, unlike the internal fail-closed path, **raises** when
  the DELETE fails, and marks any record of that instance STOPPED.

### Added — gpu.ai's market as raw offers

`GpuAiRuntime.catalogue()` keeps one rung per shape, the cheapest, and drops
what kind of machine it is. For RTX 6000 Ada that silently picks a $0.65 VM
that boots in 3–5 minutes with a fixed 350 GB disk over a $0.74 container that
boots in about one.

`GpuAiRuntime.offers(gpu_type=, region=, tier="on_demand",
include_unavailable=False)` returns every matching row as a `GpuOffer`,
cheapest first: `offering_id`, region, price, units available,
`instant_boot`, `deployment_type` (`vm`/`container`), `capacity_class`,
disk allowance and price, host specs, per-GPU VRAM and the certified
image's CUDA and minimum driver (`OfferEnvironment`). Unreported specs (the
API's `0`) are `None`, not zero. `GpuOffer.cost_per_hour(disk_gb)` prices
only disk beyond the included allowance, as the contract bills it. Free and
keyless, like `catalogue()`.

**Fixed:** `/pricing` is paged and `catalogue()` read only the first page,
which could drop the cheapest offering. Both now follow the cursor.

### Fixed — rental backends record a runtime before they rent it

The manager's docstring promised "state is written before provisioning
begins". The rental backends wrote it only after `up()` returned, and the wait
inside `up()` is up to 35 minutes. A crash there left an instance billing with
no record; only orphan detection or gpu.ai's `auto_terminate_hours` would ever
catch it.

- **gpu.ai** saves a STARTING handle carrying the create's `Idempotency-Key`
  *before* `POST /instances`, then adds the operation id, then the instance id
  the moment the operation names it. The request is recorded as a summary
  (`metadata["requested"]`) with `env` reduced to its key names, so an
  `HF_TOKEN` sent to the template is not written down.
- If the poll is lost before the operation names an instance, the record stays
  STARTING with the operation id. `status()` and `down()` resolve the operation
  to its instance (or to "made nothing"), so `down(name)` terminates what the
  create made. `down()` on a create still in flight says so and keeps the
  record rather than marking it stopped. A record saved before the operation id
  came back is matched to a live instance by the name it was sent with.
- A create that fails outright is recorded FAILED (with the idempotency key to
  retry safely); one that failed after its instance existed is terminated and
  recorded STOPPED with the cause.
- **DeepInfra** saves a STARTING handle before `POST /deploy/llm` and records
  the `deploy_id` before waiting.

### Fixed — runtime state files no longer hold credentials

A runtime state file is meant to be read with `cat`, which is exactly why it
gets copied, backed up and pasted into bug reports. Two backends wrote a
credential into one: `DeepInfraRuntime` stored the **account API key** in the
handle's metadata, and `GpuAiRuntime` stored the vllm tunnel's **HTTP Basic
header** (the instance's app password). Both files are mode 0600, which does
not survive a copy.

- `RuntimeHandle.secrets` holds credentials **in memory only**. `to_dict()`
  never serializes it, and also drops any key in `SECRET_METADATA_KEYS`
  (`api_key`, `default_headers`, `app_password`, `authorization`, `token`)
  from `metadata`, so a third-party backend that puts one in the wrong place
  still cannot leak it.
- State files written before this change still load: a credential found in
  their metadata is moved to `secrets` for this process, and the next save
  rewrites the file without it.
- Each rental backend re-derives credentials at use time with
  `await backend.credentials(handle)`: DeepInfra resolves its key from the
  environment again; gpu.ai re-reads the tunnel login with
  `GET /instances/{id}?include=credentials`, the only read that returns it.
- `RuntimeManager.attach()` and the liveness probe read
  `handle.credential(...)`. The probe also now **sends** the tunnel's Basic
  header: without it every probe of a gpu.ai template was a 401, and a healthy
  rental would have been marked DEGRADED after three of them.

### Added — the artifact cache can actually be collected

`ArtifactStore.gc()` existed and had **no callers**, for two reasons. It
took a *keep-set* and nothing computed one — a collector with no way to say
what was still in use. And eviction was not recoverable: materialization
re-points an artifact's `uri` at the local copy, so deleting that copy left
the artifact pointing at a dead path, even though the provider URL had been
preserved under `provider_metadata["source_uri"]`.

**Recovery comes first.** `materialize()` now falls back to that preserved
source URL when the local copy is missing. That is what makes this store a
*cache* rather than a store of record: a file can always be re-fetched, so
throwing one away is a tuning decision rather than data loss. Verified end
to end — evict, then re-materialize, and the bytes come back from the
original URL.

**Then eviction.** `gc()` takes three composable policies:

- `keep_checksums` — reference-based; anything unnamed is evicted.
  `None` still means *no-op*, so a mistaken bare call cannot wipe the store.
  An empty set is distinct: a positive statement that nothing is live.
- `max_age_days` — evict files older than this.
- `max_total_bytes` — evict oldest-first until the store fits.

A file named in `keep_checksums` is never removed by the other two, so a
caller who knows what is live cannot be overruled by an age rule.
`dry_run=True` reports without removing.

`live_checksums(artifacts)` builds the keep-set, and `entries()` /
`total_bytes()` expose what is stored. Reachable as
`MediaManager.artifacts.gc(...)`.

**Breaking:** `gc()` returns a `GcReport` (removed, freed bytes, kept,
dry-run flag, and a per-file reason for auditing a surprise) rather than a
bare count. `ArtifactStore` is publicly exported, but nothing in this repo
or the sibling repos called `.gc(`, so the change is breaking in principle
and affects nothing today. Declared rather than slipped in.
### Added — pricing for models that are not billed per token

Image, video, speech and transcription models are often priced per image,
per megapixel, per second of output, per character or per audio minute.
`per_million_tokens` cannot express any of that — and because it was a
**required** field, such a model could not have a `ModelPricing` at all.
That is why those cards carried `pricing: null` and were invisible to every
cost path, and why providers like `elevenlabs`, `deepgram`, `fal` and
`higgsfield` sit on the unpriceable list.

`ModelPricing.per_million_tokens` is now optional, and a new
`ModelPricing.per_unit` carries the non-token rates. `get_cost()`,
`ModelCard.estimate_cost()` and `LLMCore.estimate_cost()` accept unit
quantities — `images`, `megapixels`, `videos`, `video_seconds`,
`audio_minutes`, `characters`, `seconds`, `requests` — and token and unit
rates add, since a model may charge for its prompt *and* per image.

Three things kept honest:

- A unit the model states no rate for contributes **nothing**, and
  `UnitPricing.unpriced_units()` reports it, so "we have no rate" stays
  distinguishable from "free".
- `rates_for()` returns `None` for a model with no token rates rather than
  zeros, and `registry.get_pricing()` **omits** the token keys rather than
  reporting them as 0.0.
- No conversion is attempted between dollars-per-image and
  dollars-per-token. Comparison is only meaningful within one unit, which
  is what a routing pool of like-for-like models provides.

Existing token-only pricing is unchanged, and omitting the new arguments
changes no existing answer.

Card pricing documentation now also states plainly that these figures are a
reference rather than an invoice: rate cards change, promotional rates
lapse, and some providers bill reasoning tokens as output without reporting
the split, so real cost should be checked against billing.

### Fixed — `[agents.fast_path]` settings now take effect

`FastPathExecutor` accepts a `config`, but `single_agent` constructed it
without one, so it fell back to its own module-level defaults. Everything in
the user's `[agents.fast_path]` section except `enabled` was inert: a
`cache_max_entries` or `cache_ttl_seconds` a user set did nothing, because
the executor also built `ResponseCache()` with no arguments.

Two classes are named `FastPathConfig` — the user-facing pydantic section in
`config.agents_config`, and the executor's runtime class in
`agents.learning.fast_path`. They disagree about field names
(`cache_enabled` vs `use_cache`, `templates_enabled` vs `use_templates`) and
about which fields exist: the runtime class had no cache-sizing fields at
all, so the executor could not have honoured them even if handed a config.

The runtime class gains `cache_max_entries` and `cache_ttl_seconds`, passes
them to `ResponseCache`, and gains
`FastPathConfig.from_agents_config()`. That translation is written out field
by field deliberately — anything automatic (`model_dump()` into `**kwargs`)
would drop precisely the renamed fields and silently leave them defaulted,
reproducing the bug while appearing correct. A test pins the fact that the
two classes still differ, so unifying them later is a deliberate act rather
than an accident.

Building `FastPathExecutor()` with no config still yields the previous
defaults; it is public API.
### Changed — capability flags report unknown instead of False

`ModelDetails.supports_tools` / `supports_vision` / `supports_reasoning` are
now `bool | None`, defaulting to `None`. A card whose capability block was
never filled in says nothing, and reporting `False` asserted a negative that
had not been established — 26% of packaged chat cards carry a generator's
`streaming=True` and every other flag false. `None` is falsy, so truthiness
checks are unaffected.

`ModelCapabilities.is_populated()` now owns that test, and the agent
capability checker uses it rather than its own copy.

Also hardens `FastPathConfig.from_agents_config` against values that exist
but are not usable. Coercion alone was not enough: a `MagicMock`
implements `__float__` and becomes `1.0`, which would have given the
fast-path cache a one-second TTL. Values must now be the right kind of
thing, not merely convertible.

### Changed — the agent circuit breaker's budgets now sit above normal work

`max_total_cost` defaulted to **$1.00**. Measured across 4,140 real agent
turns, priced from model cards with cache-aware accounting, the **median**
turn costs **$1.04** and the 90th percentile **$10.02** — so the guard cut
off slightly over half of ordinary turns mid-run. It only looked harmless
while cost accounting was understating Anthropic-shaped runs roughly
tenfold; correcting that made the cap bite as written.

The default is now **$25.00**, which trips on roughly 3% of real turns —
the tail where runaway behaviour actually lives, since the most expensive 1%
of turns account for about a quarter of all spend. This is a dollar figure
and therefore model-dependent, so it is worth lowering for unattended runs.

`max_execution_time_seconds` is raised from **300** to **1800** for the same
reason: five minutes sat below the median measured duration of an agent
turn, so it could not distinguish "stuck" from "doing tool work". The
evidence is weaker here — usable timing came from a single harness, about 6%
of the corpus, median 295s and p90 2,340s — and that is stated in the
config rather than implied.

These defaults are declared in four places: the breaker's pydantic config,
its dataclass fallback, its `__init__` signature, and `default_config.toml`.
The TOML is the one that actually loads, and it had already drifted from the
code. A test now pins all of them together.

### Added — the breaker reports a trip coming, not just a trip

`CircuitBreakerResult` gains `projected_total_cost` and
`cost_budget_fraction`. The breaker previously reported a budget only once
it was already gone, which on a long run means discarding most of the work;
a projection at iteration three is actionable in a way a trip at iteration
twenty is not.

The projection extrapolates linearly from spend-per-iteration, which the
shape of real agent traffic supports: a turn's cost is dominated by
re-sending a roughly constant context on every step, so it grows close to
linearly in step count rather than accelerating. It returns 0.0 before any
iteration has been priced rather than guessing from nothing.
### Fixed — a failed evaluation is no longer scored as a mid-range result

`MultiAttemptArbiter` filled a failed candidate evaluation with 5.0 across
every criterion, at three sites: an exception, a timeout and an unparseable
response. On a 0–10 scale that is exactly `min_acceptable_score`, so a
candidate nobody managed to judge entered selection looking like a mid-range
result — it could beat one that genuinely scored 4, and in the
highest-score fallback could win outright (a missing score mapped to 0.0,
while a *failed* one got 5.0).

`CandidateScore.weighted_total` is now `float | None` with
`evaluation_failed`, unjudged candidates are excluded from ranking, the
selection summary shows them as "not evaluated" rather than a number, and an
all-unjudged set is logged as an arbitrary choice instead of being presented
as a selection. `ArbiterDecision.confidence` is nullable, since a confidence
derived from a non-score was itself invented.

### Added — `get_cost` reads the two remaining pricing fields

`reasoning_output` and `batch_discount_percent` were declared on the pricing
model and consulted nowhere, so a card could state them and a caller still
be charged the plain rate. `get_cost` and `ModelCard.estimate_cost` now take
`reasoning_tokens` (a subset of output, clamped to it) and `batch`. Omitting
them changes no existing answer.
### Fixed — the fast-path response cache could serve another conversation's answer

Three defects, each demonstrable, in a cache that PR #43 had just made
configurable and therefore more likely to be exercised:

- **The key was the prompt text alone.** One session's answer to `"continue"`
  was returned to another session asking the same thing. Entries are now
  namespaced by a caller-supplied scope, and `single_agent` passes the
  session id.
- **Lookup used Jaccard over word *sets*,** which ignores order entirely:
  `"delete the old file"` and `"the file delete old"` scored 1.0 and
  returned each other's response. Matching is now exact by default;
  a threshold below 1.0 enables fuzzy matching that respects sequence.
- **Nothing stopped a state-dependent prompt being stored.** `"continue"` is
  not a question, it is a reference to whatever was happening, and the
  correct answer changes every time. Such prompts are now refused at write
  time, so the mistake cannot be made later at lookup either.

Measured on 6,694 real harness prompts: 35.1% are exact duplicates, but of
the 483 repeated prompts **92% recur inside a single session** and **63%
produced outputs differing by more than 2×** — the same text with a
different correct answer. The single most repeated prompt was
`"continue... ensure you commit often"` (78 occurrences).

The default threshold moved from 0.8 to 1.0 for the same measurement:
normalising those prompts raised the duplicate rate only from 35.1% to
35.4%, so fuzzy matching buys almost nothing while risking a wrong answer.

### Fixed — the capability pre-check refused every model newer than its table

`CapabilityChecker` resolved models against `DEFAULT_MODEL_INFO`, a static
table of **13 entries** from early 2024 (`gpt-4-turbo`, `claude-3-opus`,
`gemini-pro`…), and returned `compatible=False` for anything absent from it.
`capability_check.enabled` and `strict_mode` both default to `True`, and
`single_agent` returns early with `success=False` and zero iterations when
the check fails — so `EnhancedAgentManager.run()` (whose default mode is
`SINGLE`) **refused to run any current model out of the box**, while a model
card existed for every one of them.

`agents.capability_check.use_model_cards` has defaulted to `True` since it
was added and promised exactly this lookup. Nothing read it.

Three changes:

- `CapabilityChecker` takes `use_model_cards` and falls back to the
  model-card registry for models absent from the static table, so the
  thousands of packaged cards are reachable. `single_agent` passes the
  configured value.
- **An unknown model is unknown, not incompatible.** It now yields
  `compatible=True` with a warning saying capabilities could not be
  established, instead of a hard error that `strict_mode` turned into a
  refusal.
- A card whose capability block was never filled in is read as *unknown*
  rather than as a set of negatives. Card generators default `streaming` to
  true and leave the rest false, so a card whose only true flag is
  `streaming` would otherwise make the checker assert that a frontier model
  cannot call tools. Context windows are still trusted on such cards, being
  numbers rather than defaulted flags.

A model whose card genuinely states it lacks a required capability is still
a hard error, and a context-window shortfall is still a hard error. Both are
covered by tests, since the easy way to "fix" this would be to stop checking.

### Fixed — cache token counts reach session statistics

`SessionTokenStats.total_cached_tokens` has always existed, and
`get_session_token_stats` has always read `cached_tokens` off each
interaction record — but nothing ever wrote that key, so the field was
permanently zero. The cognitive cycle measures the number and it was being
dropped at every handoff.

The whole chain now carries it: `PhaseUsage` →
`CycleIteration.total_cached_tokens` / `total_cache_write_tokens` →
`EnhancedAgentState` → `record_agent_usage` → `SessionTokenStats`.
`SessionTokenStats` gains `total_cache_write_tokens`, and
`CycleIteration.to_history_summary()` reports both.

`record_agent_usage` clamps `cached_tokens` to `prompt_tokens`: cached
tokens are part of the prompt, so a larger count is a reporting error rather
than extra tokens, and letting it through would imply a cache-hit ratio
above 100%. Records omitting the new keys behave exactly as before.

Also corrects `record_agent_usage`'s docstring, which claimed agent runs
"bypass the chat path that normally records
`session.metadata["interactions"]`". Nothing writes that list for the chat
path either — `record_agent_usage` is its only writer, and plain chat
sessions fall back to estimating from message token counts.

### Fixed — the agent's cost accounting now sees prompt caching

`extract_usage` priced every iteration as
`get_cost(input_tokens=prompt_tokens, output_tokens=completion_tokens)`,
reading no cache counters at all. The circuit breaker's `COST_LIMIT` acts on
that number, and on cache-heavy agent traffic cache reads are the
overwhelming majority of input tokens — so the limit was wrong by an order
of magnitude, in **opposite directions depending on the provider**:

- **Anthropic-style** reports the prompt count as *fresh tokens only*, with
  `cache_read_input_tokens` alongside. The cached bulk never reached
  pricing: a real turn with 6 fresh and 922,403 cached tokens priced at
  $0.0193 instead of $0.2037 — **10.6× under**. Against the default
  `max_total_cost` of $1.00 the breaker tripped at iteration 52 instead of 5.
- **OpenAI-style** reports the whole prompt, with the cached part under
  `prompt_tokens_details`. Every cached token was charged at the fresh rate:
  $3.7089 for the same turn — **18.2× over**, tripping at iteration 1 and
  killing every run.

`extract_usage` now reads both conventions, decides which contract applies
from the provider's own key names, and normalises to this codebase's rule
that `cached_tokens` is a subset of `prompt_tokens` while cache writes are
additional. Both shapes now yield the same cost for the same turn.
`PhaseUsage` carries `cached_tokens` and `cache_write_tokens`, so iteration
totals and session stats can report them too.

A usage block with no cache information behaves exactly as before.

### Fixed — context-tiered pricing is actually applied

`ModelPricing.context_tiers` was declared, documented, and populated on four
packaged cards — and read by nothing. `get_cost` used the flat rates whatever
the prompt size, so a long prompt on a model that charges more for long
context was priced at roughly **half** its true cost. Gemini 2.5 Pro and
Sonnet 4.5 both charge about double beyond a 200k prompt; a 500k-token Gemini
prompt came out at $0.725 instead of $1.40.

`ModelPricing` gains `tier_for()` and `rates_for()`, and both cost paths
(`get_cost` and `LLMCore.estimate_cost`) resolve the bracket before pricing
anything. `CostEstimate.context_tier_threshold` reports which bracket was
applied, so an unexpected number can be explained without re-deriving it.

**`threshold_tokens` is the tier's inclusive upper bound**, which the field's
original description left ambiguous. That is the reading that matches
published pricing: Gemini 2.5 Pro declares thresholds 200,000 and 1,048,576
with input $1.25 and $2.50, and its real rates are $1.25 up to a 200k prompt
and $2.50 beyond — so a 500k prompt belongs to the 1,048,576 bracket.
Treating the threshold as a floor would price that prompt at $1.25 and halve
the bill. A prompt larger than every threshold falls to the largest bracket,
never back to the cheapest.

Cache-write tokens count toward the bracket, since they are part of the
prompt even though providers report them separately from `input_tokens`.

A tier need only state input and output. Where it does not state cached or
cache-write rates, the base rates are **rescaled by the tier's input ratio**
rather than carried across unchanged: `cached_input` is definitionally a
discount off input, so carrying an absolute figure over a bracket boundary
would silently change the discount — 25% of input in one bracket and 12% in
the next, which no vendor publishes. `ContextTier` now also accepts explicit
`cached_input` and `cache_write`, which override that rule, so a vendor who
does something different can be recorded exactly.

### Added — cache-write pricing, and a daily price refresh

**`TokenPricing.cache_write`.** Cost estimates priced fresh input and cached
reads but had no concept of writing the cache, which providers bill at a
premium (Anthropic charges 1.25x input for a 5-minute cache). For
cache-heavy workloads that term can dominate the bill, so omitting it does
not approximate the total — it answers a different question.

`ModelPricing.get_cost`, `ModelCard.estimate_cost` and
`LLMCore.estimate_cost` all take `cache_write_tokens` now, and `CostEstimate`
reports `cache_write_cost`. Cache writes are *additional* to `input_tokens`
rather than a subset, matching how providers report usage — Anthropic returns
`cache_creation_input_tokens` alongside `input_tokens`. Omitting the argument
changes no existing answer.

Where a card states no cache-write rate, the plain input rate is used. That
understates the premium, but inferring it from the multipliers in
`AnthropicExtension.prompt_caching`, which no code reads, would silently
diverge from the published number, and the 5-minute vs 1-hour variant is not
recoverable from a token count.

### Fixed — model card loading is deterministic

`ModelCardRegistry._load_directory` walked the tree with an unsorted
`rglob`, so load order was filesystem dependent. The packaged tree contains
model ids that appear in two files under different filename spellings
(`vendor--model.json`, `vendor__model.json`), and some of those pairs
disagree about whether the model has pricing — so whether a model appeared
priced could depend on directory iteration order, and routing reads pricing
to rank targets.

Loading is now sorted, and a duplicate `model_id` *within* one tier is
resolved by the card's own declared `source` (hand-written beats generated,
matching the card-refresh convention) rather than by load order. A
cross-tier collision is still the documented override: user cards replace
packaged ones. Collisions are summarised in one warning, with the full list
at debug level.

The loading tier no longer destroys the card's declared `source`; it is kept
alongside so precedence can be decided.

### Added — `tools/kairos/pricewatch.yaml`

Definition of a scheduled job that refreshes model pricing from vendor APIs
and vendor pricing pages into `~/.config/llmcore/model_cards/`, which the
registry already loads over the packaged cards. The tool it invokes lives
outside this repo, since it reaches the network on a timer; only the
machine-neutral definition is versioned here. See `tools/kairos/README.md`.

### Added — `llmcore.routing`: pools, lanes, failover and proxy mode

Five composable layers, all off until configured. With no `[routing]` section
every call resolves exactly as it did before. Design and the reasoning behind
each decision: the routing subsystem design spec;
usage: [`Routing_usage.md`](docs/Routing_usage.md).

**Config stops being an allow-list.** `[providers.*]` sections were acting as
one: a provider registered in `PROVIDER_MAP`, with its key in the environment
and its base URL already known, was still unreachable. Any provider+model is
now addressable by spec string, and llmcore builds the instance on demand:

```python
await llm.chat("hi", target="xai:grok-4.1-20251117?effort=high")
await llm.chat("hi", target="ollama:llama3.3:70b")
await llm.chat("hi", target="vllm:Qwen/Qwen3-30B#my-box")
```

`routing.autoprovision = false` restores the closed set for deployments that
want one. (Found along the way: `xai`, `groq` and `together` had no
`[providers.*]` section at all, so all three were unreachable despite being
registered — sections added.)

**Pools, with a failure taxonomy rather than one retry rule.** Conflating
failures is how naive failover burns money or loops, so each gets its own
handling: a 429 cools down briefly and honours `Retry-After`; an empty wallet
cools down for minutes and records a balance of zero; a 5xx is retried once
*in place* before moving, since it is usually one bad node and moving would
discard a warm prompt cache; a bad key benches the target for the process; a
400 does not fail over at all; a content refusal does not either, unless
`on_refusal = "failover"` — retrying a refusal elsewhere is "shop until
someone says yes", which should be opted into. A prompt that overflows the
context window is treated as a routing signal, not an error, and is sent to a
model with a larger window; llmcore also skips a target *before* calling it
when the card already proves the prompt will not fit.

Seven selection strategies: `priority`, `round_robin`, `weighted`,
`lowest_latency`, `lowest_cost`, `least_busy`, `most_credits`. Order tiers
(`?order=1`) express "my own GPU, and only pay a vendor if it is down".
Session affinity is on by default, because failing over mid-conversation drops
the cached prefix, shifts output style under few-shot expectations, and
invalidates preserved reasoning blocks.

Three places where the strategies refuse to guess, because treating "unknown"
as a number inverts them: an **unpriced** target ranks after every priced one;
an **unknown balance** ranks after a known one but ahead of a known-*empty*
one; `lowest_latency` explores an unmeasured target first, since it cannot
prefer low latency without a measurement. Self-hosted providers price at zero
rather than unknown, so a local model wins outright.

**Lanes and classifiers.** A classifier names a lane, never a model, so
swapping models is a config edit. That one indirection covers complexity
tiers, speed tiers, domain routing and a privacy class with one mechanism.
Nine classifiers ship, from a free `hint` to a local 350M zero-shot encoder to
TypeSafe's `choice` primitive. llmcore orders the chain itself: cheapest
first, and within a cost band instructions before guesses — a `lane=` argument
and a length heuristic are both free, and running the guess first would
override what the caller asked for.

Authority has four levels (`caller` > `policy` > `prompt` > `inferred`)
because a routing marker found in *content* is not as trustworthy as an
argument on the call. Markers are the point of the feature — in an agent
harness the model's text is the only channel that passes through, so that is
how an agent routes itself — but in a RAG path that text may have come from a
retrieved document, where `[[lane:deep]]` would be a one-line prompt
injection, or a way out of the private lane. Markers are always stripped
before egress, acted on or not.

**Cascades** (opt-in): answer cheaply, verify, escalate only if the answer
fell short. Verdicts are three-valued, and *could not judge* is kept distinct
from both — read as a fail it escalates every unjudgeable answer and inverts
the saving; read as a pass it silently disables the quality floor the moment
the judge breaks.

**The privacy path, where routing is the guarantee and redaction is not.** A
detector that misses one identifier has leaked it, and none catches
everything, so `on_detect = "constrain"` changes the *destination*: a prompt
with personal data goes to a pool that never leaves the machine, and a miss
stays on your own hardware. Redaction can be stacked and is documented as
defence in depth. Three ways it refuses to degrade quietly: `constrain` with
no pool configured blocks rather than sending; a constrain pool that does not
exist blocks; and a detector that raises blocks (`fail_closed`), because a
detector that crashed has not cleared the prompt. Findings carry a 16-char
hash, never the value — including in the exception message.

**Proxy mode** (`llmcore-bridge proxy`): an OpenAI-compatible endpoint, so an
unmodified agent harness gets all of the above by setting a base URL and a
model name. `model` accepts `lane:deep`, `pool:main`, `profile:frugal`, a
target spec, a bare model name or `auto`; lanes and pools appear in
`GET /v1/models` so the harness's own picker selects routing policy. Usage
reports the target that *actually* answered, because under a pool that is not
the model requested and a harness logging spend should not be lied to. It
binds to loopback and **refuses** a non-loopback bind without a bearer token,
since the process holds every provider credential in the config.

**Effort and parameters.** The existing effort vocabulary is extended rather
than replaced, with precedence card → provider config → target → lane →
profile → per-call keyword. Under a pool an unsupported parameter is dropped
with a warning, because members genuinely have different parameter surfaces;
outside a pool it still raises.

**Everything is overridable.** Config is a warm-up, not a cage: every
`[routing]` setting resolves config → environment → request, and the request
wins. A misspelled override raises rather than being silently dropped.

**Explaining itself.** `llm.routing.explain()`/`why()` report the lane, the
classifier that chose it, the chosen target and *why each other candidate was
not used*, without making a call. `health()` shows per-target cooldowns,
latency and balance. Every decision also emits a structured event.

New skills in the bundled grimoire pack: `skills/llmcore/proxy`,
`skills/llmcore/routing`, `skills/llmcore/cost`.

### Fixed — Anthropic thinking/effort, mapped per generation and verified live

Reasoning depth is expressed differently by different Claude generations, and
llmcore forwarded whatever the caller passed -- so a reasonable request became
an API error purely because of which model served it. Pools make that routine,
since members span generations.

`effort` is now a first-class Anthropic parameter and means the same thing
everywhere; the provider maps it to whatever the target accepts.

**The boundaries were measured against the live API**, one request per model,
and there are **three of them, which do not coincide.** The first draft of this
fix assumed a single cutoff at 4.6 and was wrong:

| generation | `type=enabled` + `budget_tokens` | `type=adaptive` | `type=disabled` |
|---|---|---|---|
| ≤ 4.5 | 200 — the only option | **400** not supported | 200 |
| **4.6** | 200 | 200 | 200 |
| ≥ 4.7 | **400** not supported | 200 | **400** not supported |

So:

* **≥ 4.7** gets `{"type": "adaptive"}` plus `output_config.effort`; a caller's
  `budget_tokens` is converted with a warning quoting the 400 it would have
  caused.
* **4.6 is an overlap where both forms work**, so a caller's explicit token
  budget is *honoured* there rather than discarded. The single-cutoff draft
  would have silently overridden a valid, deliberate choice.
* **≤ 4.5** gets `{"type": "enabled", "budget_tokens": N}` mapped from the
  effort level, and a request for `adaptive` is converted rather than failing.
* **An unparseable model id gets the newest behaviour**, because the older
  families are the shrinking set.

Two further bugs the live run found, both of which would have 400'd:

* **`{"type": "disabled"}` is rejected from 4.7**, and the API names the
  replacement itself: *"To turn thinking off on this model, send
  `thinking: {"type": "between_tools"}`"*. So `effort="none"` now sends
  whichever spelling the target accepts. (It disables rather than requesting
  adaptive-at-low, which would quietly spend reasoning tokens the caller
  explicitly asked not to spend.)
* **`max_tokens` must be *strictly* greater than `budget_tokens`** -- equal
  values are a 400 -- and a budget below 1024 is rejected outright. So
  `effort="high"` with `max_tokens=512` was unsendable. The budget is now
  clamped to fit, and where no valid budget exists thinking is turned off with
  a warning saying how to get it back, because sending a request that cannot
  succeed is worse than answering without reasoning and saying so.

`xhigh` folds to `high` on 4.6+, following the documented policy of folding to
the nearest supported rung rather than dropping an effort level and losing the
caller's intent.

**Validated live**: 12 of 12 cases across `claude-haiku-4-5`,
`claude-sonnet-4-6` and `claude-sonnet-5-5`, asserting both the response and
the exact payload sent. Total cost under $0.02.

### Fixed — every Anthropic call without a system message failed on the SDK transport

Found while validating the above, and unrelated to it. The SDK path passed
`system=system_prompt` unconditionally, so with no system message the SDK
serialised `"system": null` into the body and the API answered 400 *"system:
Input should be a valid array"*. The httpx path already omitted it when
absent; the two transports have to agree, and now do.

This is exactly the class of bug the dual-transport work exists to surface:
the direct path is the default, so the SDK fallback was never exercised
against a real account until credits were available.

### Added — a classifier evaluation harness, and the CLI to run it

llmcore publishes no accuracy figure for any classifier, because none has been
validated on real traffic and any number would be invented. That gap is still
open but is now **closable**: `llmcore.routing.evaluation` and
`llmcore-routing eval` measure a chain against labelled prompts.

The harness is built around one property a single accuracy figure destroys:
**the two error directions are not interchangeable.** Routing too cheap
produces a bad answer; routing too expensive only costs money. The same 67%
can be usable or unusable depending on which way it errs, so every report
separates them and names the consequence. A test asserts that two classifiers
with identical 0% accuracy produce opposite reports, which is the whole reason
the harness exists.

Two further scoring choices, both deliberate: an abstention is **not** counted
as an error (a classifier that declines is passing the turn down the chain,
which is the designed behaviour -- scoring it as wrong would make the most
honest classifier look like the worst), and with no lane order supplied the
report declines to guess a direction rather than inventing one.

`llmcore-routing` also gets `why`, `health` and `show`. A 29-case starter set
ships at `docs/examples/lane_eval_starter.jsonl` so replacing it with real
traffic is an edit rather than a blank page; its README is explicit that it is
not a benchmark. Unlabelled traffic (a `.txt` file) is accepted too, which
measures coverage and latency before anyone has decided labelling is worth it.

### Fixed — a privacy-relevant misroute, found by the new harness on its first run

The free `heuristic` classifier routes

> "Here is my patient record: John Doe, DOB 1971-03-02, diagnosed with
> hypertension. Summarise it."

to the **trivial** lane, because "summarise" is a simple-task verb and the
heuristic is documented as not looking for personal data at all.

That is survivable only because the privacy guarantee does not depend on the
classifier: transforms run *after* target selection and change the
destination, so the prompt is still constrained to a local-only pool before
anything is sent. This is now asserted rather than assumed, in
`TestLayeringSurvivesAMisclassification` -- one test proves the classifier
really does get it wrong, so the premise cannot rot silently, and another
proves the prompt still cannot reach a remote target.

### Changed — the Colab backend is verified against a real GPU VM

The R3 gate -- "one real model served end to end and reachable through
`llm.chat()`" -- is **met**. Qwen2.5-1.5B-Instruct served by vLLM on a Colab
T4, reached through `llm.chat(provider_name=...)`, and routed to through a
pool containing the runtime:

```
[2.6s] 'Reply with exactly: COLAB'  ->  COLAB
plan: pool=local strategy=priority chosen=llmcore-e2e:Qwen/Qwen2.5-1.5B-Instruct
                                   ->  ROUTED
```

Teardown released the VM and `colab usage` confirmed 0.00/hr afterwards. Total
cost of validation: 0.6 compute units.

One honest caveat: in the successful run the server was started by the
corrected bootstrap script invoked by hand on the VM, because the
orchestration's own first attempt had died on the torchaudio mismatch below.
Everything *around* it -- the assignment guard, the tunnel, provider
attachment, `chat()`, pool routing, status and teardown -- ran through the
orchestration. A single unattended `up()` with every fix applied has not been
repeated.

### Fixed — four bugs the live Colab run found, each silent

Every one of these passed the test suite and would have failed on a VM that
was already billing:

* **The session parser could not find its own session.** `colab sessions`
  prints `[name] external-id | Hardware: T4 | ...`, not the box-drawn table
  the generic scraper assumed. The scraper *appeared* to handle it -- it
  returned a row rather than raising -- with the whole `[name] id` chunk as
  the name, so the assignment guard never recognised its own session and
  released a healthy T4 after 180 seconds. Being forgiving is not the same as
  being right. The real format is now matched explicitly and captured verbatim
  as a fixture.

* **`colab exec` returns 0 even when the code it ran raised.** The bootstrap
  script correctly detected its own failure and exited -- and the
  orchestration read rc=0, opened a tunnel to a server that had never started,
  and waited 45 minutes. Success is now the explicit `[llmcore] READY` marker,
  printed only after the server answers on the VM, and a failure message leads
  with the line that explains it rather than the tail of the progress chatter.

* **A CUDA-mismatched torch companion killed the server on import.**
  Installing vLLM upgraded torch to `2.13.0+cu130` while Colab's preinstalled
  `torchaudio 2.11.0+cu128` stayed put; transformers imports torchaudio
  unconditionally, and torchaudio refuses to load against a different CUDA. The
  bootstrap now removes torch companions whose CUDA tag disagrees with torch,
  which is cheaper and more reliable than chasing a matching multi-gigabyte
  build on a billing VM.

* **The environment cache wrote and read the wrong directory, twice.** First
  `/usr/lib/python3/dist-packages` (Colab installs into `/usr/local`), then
  `site.getsitepackages()[-1]`, which resolved to the same wrong path on a real
  VM. Both now use `sysconfig.get_paths()["purelib"]` -- the path pip itself
  resolves, and so the only one that cannot disagree with the installer.

Also: `PIP_CACHE_DIR` pointed at the Google Drive mount, so every wheel
downloaded through FUSE to Drive. The finished environment tarball belongs on
Drive; the pip cache is scratch and now stays on local disk.

### Fixed — a time-of-day flake in the routing tests

Three `most_credits` tests recorded a balance without an injected clock while
asserting against a fixed one, so a probed-empty cooldown was measured from
real wall-clock UTC. They passed in the morning and failed in the afternoon.
Verified clock-independent by re-running the file with its fixed moment
shifted 25 years into the past.

### Added — the Colab runtime backend (R2-R6)

`llmcore.runtimes` can now actually provision: sizing, the Colab backend, a
supervisor that enforces the deadlines, and `llmcore-runtimes` for the
commands you need when something has gone wrong.

**Sizing** (R2) is read-only and free, and prints its own arithmetic — a sizer
that answers "use an A100" and shows nothing is impossible to argue with.
Every unknown rounds toward *needing more*: overestimating buys a bigger GPU,
underestimating OOMs on the VM after billing has started. Context shrinks
before a bigger GPU is chosen, and when nothing fits the sizer refuses with a
concrete alternative rather than sending the caller to guess — the next guess
is also a launch.

**The Colab backend** (R3) enforces four rules in code: state is written
before compute can be assigned; any bootstrap failure releases the VM; nothing
is connected to until the session actually exists; deadlines are set at
creation. If a release *also* fails, the log says MAY STILL BE BILLING in
those words. The VM-side server is started with `setsid`, because as a child
of the kernel a kernel restart would kill it silently while the VM kept
billing. Secrets travel on stdin or through `colab exec --env`, never in argv.

**Supervision** (R4). `reap()` existed but nothing called it, which made the
idle and hard deadlines documentation rather than limits; `up()` now starts a
supervisor. Liveness marks a runtime DEGRADED after three consecutive
failures, not one, and never tears it down — letting a transient network
problem destroy an expensive VM would be worse than the problem. The idle
reaper needs a last-used time and vLLM exposes no such metric, so `attach()`
wraps the provider's `chat_completion`.

`runtimes.colab.auth` selects the CLI's authentication strategy, so a machine
authenticated with `gcloud auth application-default login` can set
`auth = "adc"` instead of running the CLI's interactive code-paste flow.

**`llmcore-runtimes`** (R6) — estimate, up, status, down, logs, adopt, bake,
cache. `up` requires `--yes`; `estimate` works while the subsystem is
disabled, because deciding whether to spend should not require enabling spend.
`status` lists orphans with the command that adopts them.

Three corrections the real tools forced on the spec:

* `colab new --gpu` accepts T4, L4, G4, H100, A100 and nothing else, so the
  spec's `A100-40`/`A100-80` rungs could never have been provisioned. The
  ladder now uses the CLI's names (A100 sized conservatively at 40 GB) and a
  test asserts every SKU maps to a value the CLI accepts. The old spellings
  still resolve.
* `Quantization` had one `GGUF` member, but Q4 and Q8 differ by 2x in weight
  bytes — routinely the difference between fitting a 24 GB card and not.
* The spec's KV fallback estimated 0.5 GB for a 70B model at 8k context, about
  5x under. It now scales from the parameter count at a figure calibrated
  against models where the real numbers are available.

**Three gates are left explicitly unmet**, and the spec's phase table says so
rather than claiming completion: R3's "one real model served end to end" and
R5's "cold start is seconds" both require provisioning a real GPU VM and
spending real compute units, and R6's agent-lens migration guide is not
written because it would be telling another project to depend on an unproven
path. `cache gc` is also not implemented — `cache` lists, nothing deletes.

### Fixed — the test suite wrote into the real `~/.llmcore/runtimes`

That directory is the record of what is currently costing money — the spec
calls it a safety mechanism rather than a cache, and `llmcore-runtimes status`
reads it. Running the test suite left a phantom entry claiming a READY L4 VM
was running, which is exactly the false signal the subsystem exists to
prevent: someone checking whether they were being billed would have been told
yes, by their own test run.

### Fixed — two config keys promised behaviour that does not exist

A full audit of `default_config.toml` (660 keys) against the code found two
with no reader anywhere. Both are now marked **NOT ENFORCED** in the config
rather than quietly removed, because a control that silently does nothing is
worse than one that is absent — someone has to be able to find out:

* `llmcore.admin_api_key` documented itself as protecting administrative
  endpoints "such as live configuration reloading". Nothing reads it, and the
  bridge's `ControlService/ReloadConfig` has no reference to it, so a
  deployment that set it believing its reload endpoint was protected was
  wrong. What does protect the bridge is transport-level: mTLS and
  `--auth authflow`.
* `context_management.minimum_history_messages` — truncation does not honour
  it.

Everything else audited clean. All 48 `[routing]` keys and all 14
`[runtimes]` keys are wired; the 116 unread `semantiscan.*` keys belong to
that package, which llmcore only carries defaults for.

### Fixed — Gemini targets had no pricing or context window, silently

The model-card alias map ran the wrong way: provider type `gemini` was mapped
*to* a `gemini` card namespace, but the cards are filed under `google/`.
Nothing raised — the lookup returned `None`, `None` means "unknown", and every
caller handles unknown quietly — so `lowest_cost` could not price any Gemini
target and the pre-call context-window check never fired, for one of the most
used providers in the library. `tests/routing/test_cards.py` now audits every
provider type against the packaged card tree, read from disk rather than from
the registry singleton that other suites reset.

### Fixed — tests could load the repo's `.env` and reach real vendors

The test fixtures built confy `Config` objects with its default
`load_dotenv_file=True`, which exports `.env` into `os.environ`. That made a
credential-discovery assertion depend on the developer's machine, and it meant
a test run could reach a real vendor and spend real money. Fixtures now pass
`load_dotenv_file=False`.

### Changed — measured corrections to the routing spec

Two claims in the design document were wrong and are corrected in place rather
than quietly dropped:

- §4.4 asserted that a local classifier adds "<50 ms p50 on CPU". Measured on
  8 CPU threads with `LFM2.5-Encoder-350M-Prompt-Router`: **191 ms for 2
  lanes, 246 ms for 5, 314 ms for 9**, plus ~40 s once to load. Wrong by about
  5x, which is why the gate said *measured, not assumed*. The encoder is off
  by default, runs its forward pass in a worker thread so it cannot stall the
  event loop, and is documented as a batch/agent feature.
- The same model's raw top score is meaningless without reading it against
  chance: 0.20 across five lanes is exactly uniform, i.e. *no opinion*, and
  taking it as 20% confidence would route on noise. Confidence is now reported
  chance-corrected, so one floor means the same thing at any lane count.

The heuristic classifier also lost its short-prompt rule, because prompt length
does not predict request complexity — "write a 2000-word essay on X" is ten
tokens. Length now only ever *vetoes* the trivial lane.

### Added — `LLMCore.discard_transient_state()`

Drops the cached per-turn introspection and raw response for a session. Needed
by any long-running embedder — the routing proxy uses one synthetic session per
request, and without this those caches would grow for the life of the process.


### Added — dual transport for every provider that has a vendor SDK

llmcore's rule is to call each API directly and fall back to the vendor SDK
where one exists. Six providers did not follow it. All six now do, taking
coverage to **21 of 23**; the two that remain single-transport have no official
vendor SDK.

**Fixed a false claim first.** The transport audit asserted that "TypeSafe
publishes no Python SDK". `typesafe-sdk` 0.7.2 is official and was already
cloned in the vendor repos — the evidence was in llmcore's own support matrix.
TypeSafe now has an SDK fallback, and the remaining exemptions cite what was
actually checked: the PyPI names `deepseek` (Deskpai.com), `deepseek-sdk` (Sifat
Hasan) and `kimi-sdk` (no stated author or repository) are **not** vendor
packages, which is why DeepSeek and Kimi stay direct-only.

**SDK fallbacks added** (direct stays the default):

- **`typesafe`** — `typesafe-sdk`, via a response shim so neither existing
  parser changes. One honest limitation: request ids are unavailable on that
  path, because the SDK does not surface response headers.
- **`mistral`** — `mistralai` v3 for chat, streaming, models and embeddings.
  OCR, audio, classification, moderation and FIM stay on direct REST, because
  the SDK models those with its own typed resources rather than OpenAI-shaped
  dicts. That split is logged per call, so `backend = "sdk"` never silently
  does nothing for half the surface.

**Direct REST paths added** (the SDK stays the default, for stated reasons):

- **`anthropic`** — `/v1/messages`, streaming included. The SDK keeps the
  default because it owns prompt-caching and beta headers plus retry behaviour.
- **`gemini`** — Developer-API `generateContent` and `streamGenerateContent`.
  Vertex mode is *forced* onto the SDK, since Vertex authenticates with Google
  ADC rather than an API key.
- **`deepgram`** — `/v1/listen` and `/v1/speak`. The realtime surfaces
  (streaming STT/TTS, voice agent, Flux v2) are duplex WebSocket protocols and
  stay on the SDK, logged per call.
- **`ollama`** — `/api/chat` with NDJSON streaming, and `/api/tags`.

Throughout, there is **one normalization path per provider rather than one per
transport**: Anthropic's stream normalizer consumes event dicts from either
source, Gemini's reads a wire shim that presents REST JSON the way the SDK's
typed objects look, and Deepgram's batch parser reads a JSON attribute wrapper.
Each direct path maps failures to the same exceptions the SDK path raises,
because a dual transport that reports failures differently is not really dual.

### Fixed — five bugs found by live calls rather than by reading

- The **Mistral SDK appends its own version prefix**, so passing llmcore's
  `base_url` verbatim produced `/v1/v1/...` and a "no Route match" 404.
- The two **Mistral streaming paths returned different shapes** — the SDK path
  an async generator, the existing httpx path a coroutine resolving to one.
  Matched to the existing contract rather than "improved", since callers depend
  on it.
- **Gemini's `finish_reason`** arrives as a plain string but readers call
  `.name`, so enum-valued wire fields are now wrapped.
- **Gemini's `text` means different things at different levels** of the
  response graph — the joined non-thought parts on a response, the part's own
  string on a part. A shim property that only did the join returned `""` for
  every part and made streaming yield empty deltas.
- **Deepgram's credential attribute** is `api_key`, not `_api_key`, and the
  scheme differs between an API key (`Token`) and an access token (`Bearer`).
- **Ollama double-wrapped its own errors**: a `ProviderError` raised by the
  direct path was re-caught by the generic handler and reported as "An
  unexpected error occurred", burying the actionable message and making the two
  transports describe the same condition differently.

### Changed — the transport audit now checks more

`test_transport_duality.py` gained checks that a provider whose SDK stays the
default still has a direct client, a transport selector and its own error
mapper. It also walks the MRO, since `deepinfra` and `vllm` inherit their
selector from `OpenAIProvider` — reading only a provider's own module
misclassified both as single-transport.

### Added — Higgsfield provider

- **New `higgsfield` provider**: generative image (Soul) and video (hosted Kling
  and MiniMax Hailuo) behind one async request API. Media-only;
  `chat_completion()` raises. Direct REST by default, following the fal pattern
  deliberately rather than inventing a second one for the same shape.
- **Credentials are a pair.** The scheme is `Authorization: Key {id}:{secret}`,
  not one opaque token, so `api_key_id` / `api_key_secret` can be configured
  separately and a single-token credential warns at construction rather than
  failing with a 401 much later.
- **`nsfw` is handled as its own terminal state.** Higgsfield reports it
  alongside `failed`, but one is a content refusal and the other a malfunction.
  Both end the job; the error says explicitly that a refusal will not succeed on
  retry, and the raw state is preserved in `provider_metadata`.
- **Text-to-video and image-to-video are separate endpoints**, so supplying a
  conditioning image switches the path instead of posting an image to a
  text-only endpoint and getting a 422. An unmapped model warns rather than
  silently failing.
- Opts into the webhook receiver and parses its own deliveries; a mismatched
  `request_id` falls back to polling.
- **Dual transport, per the house rule**: direct REST by default, with
  `higgsfield-client` 0.2.0 as the `backend = "sdk"` fallback covering the whole
  `submit` / `status` / `result` / `cancel` lifecycle. Live-validated end to end
  on **both** transports.

### Added — `test_transport_duality.py`: the direct-vs-SDK audit, enforced

llmcore prefers calling each API directly with the vendor SDK as a fallback, but
nothing checked it. The Higgsfield provider initially shipped with a
`backend = "sdk"` option that its resolver accepted, an SDK import, and an
`self._sdk` attribute that was **never assigned and never called** — so
selecting it silently did nothing.

The audit now encodes the rule: a provider offering one transport must declare
why, dual-transport providers must actually construct *and call* their SDK
client, direct must be the default, and every SDK fallback must be installable
from its extra. Known gaps are listed with a `GAP:` prefix so they read as
outstanding work; a provider that gains a second transport and stays on the list
fails the audit.

The audit walks the MRO rather than a provider's own module, because
`deepinfra` and `vllm` inherit their selector from `OpenAIProvider` and have
dual transport without a line of their own about it — the first version read
only the own-module source and misclassified both.

Result: **15 dual-transport**, 3 single by design (`deepseek`, `kimi`,
`typesafe` — no vendor SDK published), and **5 recorded gaps**: `anthropic`,
`gemini`, `deepgram` and `ollama` are **SDK-only with no direct path**, and
`mistral` is **direct-only with an unused mistralai v3 SDK**.

### Fixed — cardctl was missing adapters for eight providers

`fal`, `elevenlabs`, `replicate`, `deepgram`, `groq`, `together`, `vllm` and
`higgsfield` were registered in `PROVIDER_MAP` with **no cardctl adapter**, so
their model cards could never be generated. `openai_compat`'s own docstring had
claimed Groq and Together since it was written, but neither was ever registered,
so `cardctl generate groq` simply failed.

Nothing caught this because `generate` only reports on the provider you name and
`stats` only sees providers that already have cards — a provider with no adapter
was invisible to both.

### Added — `cardctl doctor`, and a test that enforces coverage

- **`cardctl doctor`** cross-checks llmcore's provider registry against the
  adapter registry and the cards on disk, separating *errors* (a registered
  provider with no adapter — cards can never be generated) from *warnings* (no
  cards yet, or a key not set) and *info* (an adapter with no provider, usually
  an alias). `--strict` fails on warnings too.
- `tests/tools/test_cardctl_coverage.py` asserts the same invariant, so adding a
  provider without an adapter now fails the suite instead of shipping quietly.

### Added — cardctl support for providers with no catalog endpoint

- **`CuratedAdapter`**: emits cards from a declared set for providers that
  publish no listing route — fal and Higgsfield address models by endpoint path,
  Replicate's catalog is tens of thousands of community models, and vLLM serves
  whatever a deployment loaded. Every such card is tagged **`curated`**, so a
  declared entry is never mistaken for a discovered one.
- **`ReplicateAdapter` enriches** each curated entry from the model's live
  schema, recording the owner's description and the model's real input field
  names — the same schema the provider reads at runtime, so the card documents
  what will actually be sent.
- **`VLLMAdapter`** requires `--base-url` and says why: there is no vendor
  catalog, and defaulting to localhost would silently card whatever happens to
  be running.

### Added — media capabilities on model cards

- `ModelCapabilities` gains `image_generation`, `image_edit`, `image_upscale`,
  `video_generation`, `video_interpolation`, `speech_synthesis`,
  `transcription`, `music_generation`, `sfx_generation` and `voice_design`,
  mirroring `llmcore.media.MediaCapability`. `ModelType` gains
  `video-generation` and `media`. All default off, so existing chat cards are
  unaffected.
- The card builder maps them through. Without this the media cards claimed **no
  capabilities at all**, which reads as "this model does nothing" rather than
  "nobody filled this in".

### Fixed — two cardctl bugs found while generating

- **Adapter aliases built duplicate card trees.** `cardctl generate gemini`
  created a second `gemini/` directory alongside `google/`, because the
  directory is named after whatever string the caller passed.
  `cards_dir_for_provider()` now canonicalizes aliases, so there is exactly one
  directory per provider.
- **Deepgram cards were written once per model *and language*.** Its
  `/v1/models` lists a record per language, so 553 records collapsed into 144
  files with whichever variant came last deciding each card's language metadata.
  Records are now grouped by canonical name with their language coverage merged.

### Changed — all model cards regenerated

2319 cards across 22 providers, 0 validation failures.

### Added — `llmcore.runtimes` core: remote GPU runtimes (R1)

- **New `llmcore.runtimes` subsystem**, reachable as `llm.runtimes`: provision
  compute elsewhere, serve an open-weights model on it, and attach the endpoint
  as a provider instance — so a remotely served model is reachable through the
  same `llm.chat()` as any hosted API. Phase R1 of
  the Colab runtime design spec: core types, the `ComputeRuntime` protocol, the state
  store, `RuntimeManager`, and a `FakeRuntime`. **No real backend yet, so
  nothing can spend money.**
- **The safety model is enforced, not just documented.** Unlike every other
  provider, a runtime bills per minute from the moment it is assigned, whether
  or not anyone calls it. So: the subsystem is **off by default**;
  `LLMCore.create()` builds the manager without contacting any backend; `up()`
  raises `SpendNotConfirmedError` unless confirmation is explicit; and
  `estimate()` is free and works **while disabled**, because deciding whether to
  spend should not require enabling spend.
- **State is written before provisioning returns**, because the dangerous window
  is a crash between assignment and bookkeeping — money burning with nothing
  tracking it. One indented-JSON file per runtime under `~/.llmcore/runtimes`,
  so anyone who suspects they are being billed can find out with `ls` and `cat`.
- **Bounded by default**: idle (45 min) and hard-lifetime (240 min) deadlines
  come from config, not from the caller remembering. A **compute-unit ceiling**
  is checked *before* either, because an idle reaper does not protect against a
  runtime that is busy in a loop.
- **Fail closed**: if attach fails after `up()` succeeded, the runtime is
  released rather than left burning, and the error says so.
- **`close()` detaches; it does not tear down.** A process exiting is not a
  reason to destroy compute someone is paying for and may still want, so
  `LLMCore.close()` unregisters the provider instances and leaves the state
  files for the next session. `down_all()` is the explicit way to stop spending.
- Provider attachment needs **no new provider class**: the endpoint is
  OpenAI-compatible, so `attach()` registers a `vllm`-type instance
  (`ephemeral=True`, `replace=True`) and `api_style` decides the type, so a
  future TGI or llama.cpp recipe attaches a different one without touching the
  runtime layer.
- Robustness where the state directory is already wrong: an unknown `phase`
  parses as `DEGRADED` rather than raising, and one corrupt record is skipped
  with a warning rather than failing the listing — a bad file must not hide the
  runtimes still running.
- New config section `[runtimes]`, disabled by default.

### Added — direct `httpx` transport for OpenAI and its four subclasses

- **`OpenAIProvider` now has a direct REST transport** alongside the `openai`
  SDK, selected with `transport = "httpx"`. Because this class is the base for
  **`deepinfra`, `vllm`, `poe` and `openrouter`**, one transport gives five
  providers a dual approach at once — the highest-leverage item in Phase 3 of
  the provider modernization plan (§5.8).
- **The key is `transport`, not `backend`.** OpenRouter and Poe already use
  `backend` to choose between their native vendor SDK and OpenAI-compatible
  mode; overloading it would have made one of the two settings unreachable.
- **The default stays `"sdk"`, deliberately.** Four subclasses' test suites
  mock `AsyncOpenAI`, so flipping the default would route five providers past
  their own tests — the same failure the Z.ai SDK backend caused when it
  changed auto-resolution and broke 21 tests. Direct transport is opt-in per
  instance.
- **The transports are interchangeable**, which is the actual requirement:
  every `extract_*` method parses response *dicts*, so the direct path returns
  exactly what `model_dump(exclude_none=True)` produces and maps failures to
  the same typed exceptions — `ContextLengthError`, and the actionable
  model-not-found `ProviderError` that names the provider's default model.
  Verified live against OpenAI: identical response keys, identical extracted
  tool calls, identical 137-model listings, streaming on both. Also verified
  live on DeepInfra, confirming the base-class leverage is real.
- Request shaping — parameter validation, native search, reasoning-model
  parameter naming, tool payloads — happens **before** the transport branch, so
  the direct path inherits it rather than reimplementing it. That is what keeps
  the two from drifting.
- **OpenRouter's `HTTP-Referer` / `X-Title` headers reach the direct transport
  too**, so both identify the application to OpenRouter the same way.
- Streaming parses server-sent events, treats `[DONE]` as a sentinel rather
  than JSON, and skips an unparseable chunk instead of failing the stream.
- New config: `transport` and `[providers.<name>.default_headers]`.

### Added — Hugging Face media adapter and Inference Endpoints (M8)

- **The `huggingface` provider is now a media adapter** for `image_generate`,
  `tts` and `asr`, behind the media protocols. It remains a chat provider too —
  the first adapter to be both.
- **Dedicated Inference Endpoints are the custom-weights / private-repo path**,
  which is this phase's gate. A private model is not a model id on a shared
  router; it is a deployment you own. Configure one under
  `[providers.huggingface.endpoints]` and it is used verbatim — no routing
  lookup, no model id in the path — and that capability switches to direct HTTP.
- **Provider routing is discovered, not guessed.** A model is not served for
  every task by every provider, and each provider knows the model by *its own
  id*: the Hub says `black-forest-labs/FLUX.1-schnell`, fal-ai says
  `fal-ai/flux/schnell`. llmcore reads the Hub's `inferenceProviderMapping`
  (cached) and resolves provider, id and URL shape. A failed lookup degrades to
  `hf-inference` rather than raising.
- **TTS artifacts carry no `VoiceConsent`**: HF serves open-weight voices and
  tracks no per-voice consent, so claiming anything there would be inventing it.

### Changed — Hugging Face is llmcore's one exception to direct-REST-first

Every other media adapter defaults to direct REST. Hugging Face defaults to the
SDK for router traffic, and the reason was measured rather than assumed: **the
router hands third-party providers their own request shape.** `hf-inference`
takes `{"inputs": ...}`, while the same model routed to fal-ai wants
`{"prompt": ...}` and answers `422 Field required` otherwise. That shape belongs
to the provider and changes on their schedule, so reimplementing the mapping
would mean tracking N third-party schemas forever — absorbing it is what
`huggingface_hub` exists for.

The dual approach is intact: both transports ship and `media_backend` selects.
The split is per capability — SDK for router JSON bodies, direct HTTP for
dedicated endpoints and for binary-input tasks such as ASR, where the SDK sends
raw audio with **no `Content-Type`** and hf-inference rejects it outright.

### Added — Replicate provider: one adapter for the whole catalog (M7)

- **New `replicate` provider**: image generate/edit/upscale, video generation,
  ASR, TTS and music — **seven capabilities, zero per-model classes**, which is
  what the spec required. Media-only; `chat_completion()` raises.
- **Model-schema descriptors are the design.** Replicate hosts tens of thousands
  of community models, so nothing can be hardcoded. Every model publishes an
  OpenAPI schema naming its own inputs, and the adapter maps llmcore's canonical
  protocol arguments onto whatever that model actually calls them. Verified
  live: `n` became `num_outputs` for `flux-schnell` while `audio` stayed `audio`
  for `whisper`. A model llmcore has never heard of works.
- Explicit keyword arguments always beat the mapping — the caller knows their
  model better than a candidate list does. A schema lookup failure **degrades**
  to canonical spellings rather than raising, because a 422 naming the field is
  more useful than a silently dropped input.
- **Ready for the webhook receiver**: declares `accepts_webhook_url` and parses
  its own deliveries; a payload whose `id` does not match falls back to polling.
- **Dual transport**: direct REST by default, `replicate` SDK opt-in. New
  `replicate` extra.

### Fixed — Replicate community models were unreachable

Found by live validation: Replicate has **two** prediction-creation routes and
the model reference does not say which one applies. *Official* models run
unversioned at `/v1/models/{owner}/{name}/predictions`; *community* models —
including `openai/whisper` — `404` there and must be run by version at
`/v1/predictions`.

The obvious fix, trying the first and falling back on the 404, costs **two**
creation requests. Replicate throttles accounts under $5 of credit to a burst of
**1**, so that fallback reliably turned a working call into a `429`. The version
is now resolved from the model lookup already made for the input schema — a
`GET`, which does not count against prediction-creation limits — so a community
model runs in a single request. *A retry-based fallback is not free when the
operation being retried is the rate-limited one.*

Also: `402` is reported as a billing stop that states the token is valid, and
`429` explains the under-$5 burst limit, which otherwise reads as a bug.

### Changed — Replicate MIME types fall back to the requested format

Replicate output URLs often carry no file extension, so MIME detection returned
`None`. It now falls back to the `output_format` that was *requested* — grounded
in the call rather than guessed — and stays `None` when neither is available,
because callers key decode paths off this field and a plausible lie is worse
than an honest unknown.

### Added — generic webhook receiver for media jobs (M9a)

- **`llmcore.media.webhooks`**: a vendor-neutral callback receiver. `WebhookRegistry`
  issues signed, single-use callback URLs; `create_webhook_app()` returns a plain
  ASGI app (no web framework is imposed) that can be mounted anywhere.
- **Polling remains the fallback, by design.** `wait()` races the callback
  against its existing backoff sleep and polls anyway, so webhook and poll
  delivery converge on one code path instead of two. A missing, late, duplicated
  or malformed callback costs latency, never correctness — and
  `webhook_base_url = ""` (the default, and the normal case with no public
  ingress) stays a fully supported mode.
- **Reserve-then-bind.** Vendors want the callback URL at submission, before the
  job exists, so a token is reserved, handed over, then bound to the job that
  comes back. A delivery arriving in that window is refused. Tokens are
  HMAC-signed, verified in constant time, single-use, and bound server-side so
  they cannot be transplanted onto another job.
- **Callback URLs are only offered to adapters that opt in** via
  `accepts_webhook_url`. Most media adapters forward unknown keyword arguments
  into the vendor payload — fal does so deliberately — so passing one blindly
  would post the callback URL to a model as a generation parameter.
- **fal now parses its own callbacks** and receives a per-job URL. A delivery
  whose `request_id` does not match falls back to polling: anyone who learns a
  callback URL can POST to it, and a wrong artifact is worse than a slow one.
- Providers that cannot parse a callback still fall back to a single poll — the
  delivery says *when* to look even when it cannot say *what* happened.
- New config: `media.jobs.webhook_base_url`, `media.jobs.webhook_secret`.

### Added — ElevenLabs provider, and consent as first-class metadata (M6)

- **New `elevenlabs` provider** (alias `eleven_labs`): TTS, streaming TTS, batch
  STT, sound effects, music and voice design. Media-only — `chat_completion()`
  raises with a pointer to `llm.media`.
- **New `VoiceConsent` type on `MediaProvenance`.** Synthetic speech raises a
  question no other media kind does: a generated image resembles no one in
  particular, but a cloned voice belongs to a person who either did or did not
  agree to it. ElevenLabs tracks that state only on the *voice* resource, so
  llmcore resolves it (cached per voice) and attaches it to the artifact —
  a caller can refuse audio from an unverified clone without a second API call.
- **`verification_satisfied` is deliberately tri-state.** `None` means *the
  provider said nothing*, which is not `False`. Collapsing the two would force a
  default — silently clearing unknown voices, or refusing audio from every
  provider that reports nothing — and that is policy belonging to the caller.
  The same reasoning drives three related choices: a failed consent lookup
  yields `provider_declared=False` with `None` fields rather than raising (you
  keep your audio, and the `None`s read as *we do not know*); designed voices
  state `category="generated"` explicitly rather than leaving consent open; SFX
  and music carry no consent record at all, because nothing there is a voice.
- **New `VoiceDesignProvider` protocol.** `VOICE_DESIGN` had been mapped to
  `TTSProvider` as a placeholder since M1. Voice design is not TTS — it returns
  candidate *voices* from a description, each preview carrying the id needed to
  keep it. The M1 protocol-coverage invariant caught the placeholder the moment
  the real protocol landed.
- **Dual transport**: direct REST against `api.elevenlabs.io` by default, with
  the `elevenlabs` SDK as an opt-in backend. New `elevenlabs` extra.
- Default models verified against the live `/v1/models` lineup rather than
  assumed: `eleven_v4` for TTS (GA, standard rate, 85 languages), `scribe_v1`
  for STT, `music_v2_5`, `eleven_text_to_sound_v2`, `eleven_ttv_v3`.

### Fixed — ElevenLabs plan gating was reported as an authentication failure

Found by live validation: a perfectly valid API key on a free plan returns
`402 paid_plan_required` or `403 feature_not_available` for music and voice
design. The first implementation mapped that 403 to *"authentication failed —
check ELEVENLABS_API_KEY"*, which would send a caller hunting a credential
problem they do not have. Both are now reported as plan gating, stating
explicitly that the key is valid. A genuine `403` still reads as auth.

### Changed — ElevenLabs SFX refuses a video reference rather than ignoring it

The `SFXProvider` protocol accepts a `video` reference for video-conditioned
foley; ElevenLabs sound generation is text-conditioned only. Passing one raises,
because silently dropping it would return audio unrelated to the footage the
caller supplied, with nothing to indicate why.

### Not included — ElevenLabs realtime STT

`ASR_STREAM` routing names ElevenLabs, but realtime transcription is duplex —
the caller pushes audio *and* consumes events — and needs the session shape
Deepgram established. The adapter deliberately does **not** declare
`asr_stream`, so routing falls through to Deepgram rather than advertising a
capability that would fail.

### Added — fal provider: the media marketplace adapter (M5)

- **New `fal` provider** (alias `fal_ai`), llmcore's broadest single media
  adapter: `image_generate`, `image_edit`, `image_upscale`, `video_generate`,
  `video_interpolate`, `sfx`, `music`, `tts` and `asr`. It is **media-only** —
  `chat_completion()` raises with a pointer to `llm.media`.
- **The provider-neutrality gate passed: no core type changed.** fal is
  structurally unlike the first four adapters — a marketplace rather than a
  vendor, everything queued rather than only the slow things, inputs addressed
  by URL rather than by bytes, output schemas that vary per model. It needed no
  new `MediaJob` field, no new `MediaExecution` member and no change to
  `MediaJobManager`. See the media subsystem design spec §5.2.
- **Every capability is an async job**, including image generation. The same
  `llm.media.images.generate(...)` call returns a `MediaResult` on OpenAI and a
  `MediaJob` on fal; `llm.media.wait()` absorbs both, because execution class is
  declared per capability rather than per provider.
- **Per-capability endpoints are configurable** under `[providers.fal.models]`.
  fal model ids are endpoint paths and the gallery moves faster than a release
  cycle, so llmcore ships a small starting set rather than a catalog. Unknown
  keyword arguments are forwarded verbatim, so model-specific fields work
  without an llmcore change.
- **Dual transport**: direct REST against `queue.fal.run` by default, with the
  `fal-client` SDK as an opt-in backend. New `fal` extra.
- **Cancellation is reported honestly.** fal answers `202
  CANCELLATION_REQUESTED` and work already running may still complete and still
  bill, so the job is marked cancelled *and* the caveat is recorded in
  `provider_metadata`. A `400 ALREADY_COMPLETED` is treated as success — the
  result is fetched and the job succeeds. This is the third distinct
  cancellation semantic across adapters (Gemini refuses, Deepgram n/a, fal
  best-effort) and the shared handle models all three.

### Fixed — three fal API contract bugs, all found by live validation

None of these were visible from the docs or from mocked tests:

- **Queue routes are namespaced by application, not by model path.** A request
  submitted to `fal-ai/flux/schnell` is tracked at `fal-ai/flux/requests/{id}`;
  polling the full model path returns `405`. The adapter now prefers the
  absolute `status_url` / `response_url` / `cancel_url` fal returns at
  submission and only falls back to a reconstructed, app-scoped path.
- **Uploads go to a different host, with two backends.** `fal.run` reads
  `storage/upload` as an owner/app pair and 404s; storage lives at
  `rest.fal.ai`. There, `storage_type=gcs` answers *"Invalid storage type"* for
  newer accounts. The adapter now mirrors the official client: CDN v3 first,
  signed-URL flow as fallback, both causes reported if both fail.
- **FILM takes an explicit image pair**, not a frame list — a frames list
  returned `422 Field required` for `start_image_url` / `end_image_url`.
  `interpolate_video_media()` now maps the frames it is given onto that pair.

### Added — Gemini media adapter, the first async-job provider (M4)

- **Gemini is now a media adapter** for `image_generate`, `image_edit`,
  `image_upscale`, `tts` and `video_generate`, plus provider-level embeddings.
- **Veo makes it the first true async-job provider.** `generate_video_media()`
  returns a `MediaJob`; `poll_media_job()` refreshes it through the
  long-running-operations API. **The `MediaJob` abstraction absorbed a real
  vendor's operation shape without modification** — which is what this phase
  was meant to find out.
- **Capability declaration is now mode-aware.** Imagen's `generate_images`,
  `edit_image` and `upscale_image` are Vertex-only: the Developer API rejects
  them with *"only supported in Gemini Enterprise Agent Platform mode"*
  (verified live). `image_edit` / `image_upscale` are therefore advertised only
  when `vertex_ai = true`. This is the first provider whose capability set
  depends on configuration rather than on its class.
- **Image generation has two transports.** Imagen on Vertex;
  `generate_content` with an `IMAGE` response modality on the Developer API.
  Same capability, different call, invisible to the caller.
- **`cancel_media_job()` raises rather than lying.** Veo offers no
  cancellation, and reporting success would let a caller believe billing had
  stopped when it has not.
- Uses google-genai's non-deprecated `source=` argument for `generate_videos`
  (`prompt=`/`image=` are deprecated, removal no earlier than 2026-07-31).
- Generated imagery and video carry `MediaProvenance(watermarked=True)`, since
  Google watermarks its generative output.
- Unlike OpenAI, Imagen supports `negative_prompt` natively, so it is forwarded
  rather than dropped — the protocol parameter is honoured wherever the vendor
  honours it.

Live-validated: a 2 MB PNG through the Developer-API path, 112 KB of 24 kHz PCM
from native TTS, 256-dimension embeddings, and a Veo job submitted and polled
through `MediaJobManager`. 52 new tests; full unit suite 5470 passed.

### Added — OpenAI media adapter (M3)

- **OpenAI is now a media adapter** for `image_generate`, `image_edit`, `tts`,
  `tts_stream` and `asr`. Image generation, TTS and ASR delegate to the existing
  provider methods; **image editing (`POST /v1/images/edits`) and streaming TTS
  are new**.
- **`create_embeddings()` on the provider** — previously OpenAI embeddings were
  reachable only through the separate `[embedding.openai]` subsystem, so a
  caller holding a provider could not embed with it. Closes the gap recorded in
  the support matrix.
- **Sora is deliberately not offered.** `openai` 3.1 deprecated the video APIs,
  so `video_generate` is absent and a test asserts it stays that way.
- Parameters with no OpenAI equivalent (`seed`, `negative_prompt`,
  `sample_rate_hz`) are dropped with a debug log rather than forwarded, where
  forwarding would 400. Supplying `reference_images` routes to the edit
  endpoint, which is how OpenAI expresses reference-conditioned generation.

### Fixed — audio format was hard-coded on upload

- **`OpenAIProvider.transcribe_audio()` labelled every raw-bytes upload
  `audio.wav`.** OpenAI infers the container format from the upload filename, so
  passing mp3 bytes was rejected with *"This model does not support the format
  you provided"*. Found by feeding a TTS artifact straight back in as an ASR
  input — the exact chaining the media subsystem makes natural.
- `transcribe_audio()` gained an optional `filename` parameter (defaulting to
  the previous `"audio.wav"`, so existing callers are unaffected), and the media
  adapter derives the right name from the `MediaRef`'s mime type, filename or
  URL.

### Changed — capability-less providers are no longer registered as adapters

- Four providers subclass `OpenAIProvider` and therefore inherit its media
  protocol *methods* — but not the endpoints behind them. Each now declares its
  own `_MEDIA_CAPABILITIES` (DeepInfra: image/TTS/ASR; vLLM, Poe and OpenRouter:
  none), and a test asserts **every** subclass declares explicitly, so a future
  one cannot silently inherit and advertise endpoints that 404.
- `MediaManager.from_provider_manager()` now skips providers that implement the
  protocols but declare no capabilities, so `adapter_names` keeps meaning "can
  actually do something".

Live-validated: TTS (55 KB mp3) → artifact → ASR round trip transcribed
correctly, streaming TTS, and 256-dimension embeddings. 42 new tests; full unit
suite 5410 passed.

### Added — Deepgram behind the media protocols (M2)

- **Deepgram is now a media adapter**, implementing `MediaCapableProvider`,
  `ASRProvider`, `TTSProvider`, `StreamingTTSProvider` and
  `StreamingASRProvider`. It declares exactly the five capabilities it can
  serve (`asr`, `asr_stream`, `tts`, `tts_stream`, `voice_agent`) and no
  execution class it does not have — Deepgram is request/response or live
  stream only, never an async job.
- Deepgram was chosen as the **reference migration** because it is the only
  integration that already exercises batch STT, realtime WebSocket STT *and* a
  bidirectional voice agent, so it stress-tests the hard parts of the
  abstraction before any new vendor lands.
- **The new methods delegate; they do not duplicate.** `transcribe_media`,
  `synthesize_speech_media`, `stream_speech_media` and
  `open_transcription_session` translate `MediaRef` in and `MediaArtifact` out,
  then call the existing implementations — one code path per operation rather
  than two that can drift. A remote `MediaRef` is handed to Deepgram's own
  `transcribe_url` path rather than downloaded locally.
- **All twelve provider-specific methods are untouched** and still return the
  legacy types; the three existing Deepgram test suites pass unchanged.

### Added — `models_multimodal` ↔ `MediaArtifact` bridge

- `SpeechResult`, `TranscriptionResult`, `OCRResult`, `GeneratedImage` and
  `ImageGenerationResult` gained `to_artifact()` / `to_artifacts()`, and the two
  round-trippable ones gained `from_artifact()`. These types are public API
  returned by seven providers, so they are **bridged, not replaced** (spec §4.3).
- Data that must survive the conversion does: audio format → MIME type,
  diarization segments and timings, `revised_prompt`, OCR page structure. Base64
  image payloads are decoded to real bytes, because the media layer deals in
  bytes; malformed base64 degrades to the URI path instead of raising.

Live-validated end to end: TTS through the router produced 146 KB of WAV, the
resulting artifact was fed straight back in as an ASR input via
`MediaRef.from_artifact()` and transcribed correctly, and streaming TTS yielded
39 chunks. 74 new tests; full unit suite 5372 passed.

### Added — media subsystem core (M1)

- **`llmcore.media`**, reached through `llm.media`: a sibling subsystem to chat
  providers and search providers for generative image, audio and video.
  Implements phase M1 of the media subsystem design spec; no vendor adapters yet,
  which is the gate the spec requires before any provider work lands.
- **Three execution classes, not one** — `MediaResult` for request/response,
  `AsyncIterator[bytes]` for streams, `MediaJob` for long-running work. Image
  generation, TTS and video generation genuinely differ, and collapsing them
  into one shape is the modelling mistake the design avoids.
- **Types**: `MediaKind`, `MediaCapability` (19 capabilities), `MediaExecution`,
  `MediaJobStatus`, `MediaRef` (url/path/bytes/artifact inputs, so callers never
  hand-roll base64), `MediaArtifact` (with `expires_at` + `checksum_sha256`,
  because every aggregator returns short-lived URLs), `MediaProvenance`,
  `MediaUsage` (keeps the vendor's native billing units rather than inventing a
  token count), `MediaResult` and `MediaJob`.
- **Capability protocols** (`typing.Protocol`, runtime-checkable) — routers
  discover what an adapter can do with `isinstance`, so routing logic never
  names a provider. A capability declared but not backed by its protocol is
  dropped with a warning rather than failing at call time.
- **`MediaManager`** with per-modality routers (`images`, `audio`, `video`),
  capability discovery (`capabilities()`, `who_can()`), and a documented
  resolution order: explicit provider → explicit model → `[media.routing]` →
  built-in defaults → any capable adapter. **Adapters are the chat providers**:
  any `[providers.*]` instance implementing the protocols becomes a media
  adapter, so there is one credential per vendor and nothing to duplicate.
- **`MediaJobManager`** owns polling, capped exponential backoff with jitter,
  timeouts and cancellation, so no adapter writes its own poll loop. A timeout
  raises **without cancelling the job** — the handle stays valid and can be
  waited on again, because an expensive video generation must not be discarded
  over a client-side deadline.
- **`ArtifactStore`** — content-addressed, sharded by SHA-256, atomic publish,
  with `always` / `on_expiry` (default) / `never` materialization policies. The
  byte fetcher is injected, so the store has no hard dependency on `httpx`.
- **`FakeMediaProvider`** ships inside the package (not under `tests/`) so
  downstream projects building adapters can use it too. It implements every
  protocol, and is what the 134 new tests run against — no network, no account.
- **`[media]` config section** (artifact policy/path, `[media.routing]`
  preferences per capability, `[media.jobs]` poll/timeout policy). Entirely
  optional: omit it and `llm.media` still works on built-in defaults.
- **Backward compatible**: `BaseProvider`'s five media methods and the
  `models_multimodal` result types are untouched. Providers gain routing when
  they are migrated in M2 onward; until then `llm.media.adapter_names` is empty
  and reports so honestly.

### Added — dynamic provider registration

- **`ProviderManager.register_instance()` / `unregister_instance()`** plus
  `is_ephemeral()` / `ephemeral_instances`. Providers were previously only
  constructible during `__init__`; subsystems that *create* endpoints need to
  add one afterwards. This is the single capability shared by the media program
  and the remote-runtime program (the Colab runtime design spec), where a Colab
  VM's OpenAI-compatible endpoint is registered as a `vllm` instance.
- Guards that matter: a name collision raises unless `replace=True` (so a live
  provider is never silently swapped out from under its callers), the configured
  default provider cannot be unregistered, construction failures surface as
  `ConfigError`, and a failing `close()` during unregister is logged rather than
  blocking teardown.

### Fixed

- **Job-polling backoff could overflow.** `2 ** attempt` stops converting to
  float past ~1024 polls, so the wait loop would die with `OverflowError` — a
  multi-hour video job polled every few seconds would actually reach that. The
  exponent is now capped; regression tested at 10,000 polls.

### Added — media and remote-runtime specifications

- the media subsystem design spec — design and specification for a first-class
  `llmcore.media` subsystem: `MediaArtifact` / `MediaUsage` / `MediaJob`,
  capability `Protocol`s per modality, the three execution classes
  (request/response, byte stream, async job), capability-oriented model cards
  with the aggregator sourcing split, generic webhooks with polling fallback,
  an artifact store, a selection policy, and a nine-phase vendor rollout
  (Deepgram refactor → OpenAI → Google/Veo → fal → ElevenLabs → Replicate → HF
  Endpoints → direct specialists). Includes a backward-compatibility path that
  keeps `BaseProvider`'s five media methods and the `models_multimodal` types
  working. Notably corrects the source research: **OpenAI's Sora video APIs
  were deprecated in `openai` 3.1**, so frontier video comes from Veo and
  fal-hosted models instead.
- the Colab runtime design spec — design and specification for a
  `llmcore.runtimes` subsystem that provisions and controls remote GPU runtimes
  (Google Colab first) and attaches the resulting OpenAI-compatible endpoint as
  a provider instance, so a remotely served model is reachable through the
  normal `llm.chat(provider_name=...)` path. Built on the study of agent-lens's
  implemented design and BellaVox's process; proposes that llmcore own the
  abstraction and agent-lens delegate to it. Adds a spend-ceiling
  (`max_lifetime_minutes`) on top of the reference idle reaper, since an idle
  reaper does not protect against a busy runaway runtime.

Both documents are specification only — no implementation.

### Changed — provider SDK majors (Phase 0 of the modernization plan)

- **`openai` `>=3.0.0,<4`** (was `>=2.31.0`), **`anthropic` `>=1,<2`** (was
  `>=0.94.0`), **`google-genai` `>=2,<3`** (was `>=1.72.0`), plus
  `ollama>=0.6.3`, `deepgram-sdk>=7.11.0`, `zai-sdk>=0.2.3`. All three majors
  were adopted at once because `openai` 3.x and `anthropic` 1.x share the same
  breaking change: their HTTP layer moved from `httpx` to **httpx2**.
- **New extras for six providers that had none**: `mistral`, `kimi`, `poe`,
  `openrouter`, `vllm`, `huggingface`. These import `httpx` but previously
  worked only because `openai` installed it transitively — under `openai>=3`
  they would have failed at import. All six are in `[all]`.
- **TLS behaviour change documented**: httpx2 verifies against the OS trust
  store, not `certifi`, which can break minimal containers and TLS-inspecting
  proxies. `CONFIG_REFERENCE.md` gained an "HTTP transport and TLS" section
  with the `SSL_CERT_FILE` / `SSL_CERT_DIR` escape hatches.
- **No provider code changes were needed for httpx2**: llmcore only ever passes
  numeric timeouts to the vendor clients, never `httpx` objects, and no `respx`
  test routes traffic through a vendor SDK. Verified before bumping.
- **Z.ai's native SDK backend is now exercised.** `zai-sdk` was never installed,
  so the provider's preferred transport was dead code in CI. Its tests now pin
  `backend` explicitly (the pattern the plan's §9 mandates) instead of depending
  on what happens to be installed, so the SDK can be installed safely — and the
  SDK backend is validated live for the first time.
- **CI installs `.[dev,all]`** instead of a hand-maintained extras subset, so a
  new extra is exercised the moment it is added. The previous `zai-sdk`
  carve-out is gone, since tests can no longer be bypassed by an installed SDK.

Live-validated after the upgrade: OpenAI (3.22.1), Google Gemini (2.25.0, 47
models discovered), Z.ai (SDK backend), DeepSeek. Full unit suite green (5164
passed). **Anthropic 1.9.0 is import- and test-verified but not live-validated —
no `ANTHROPIC_API_KEY` is available in this environment.**

### Added — provider audit documents

- the provider support matrix — the ongoing tracker: per provider, the
  vendor SDK clone with tag/commit/date, our pin, the installed version, the
  transport shape, and a capability matrix extracted from the provider classes.
  Section 6 is a runnable refresh procedure.
- the provider modernization plan — the phased program that closes the
  gaps, built on the dual-transport and one-contract principles.

### Added — FriendliAI provider

- **FriendliAI provider**: first-class `FriendliProvider` covering all three
  Friendli inference surfaces through one `[providers.friendli]` section,
  selected with `endpoint_type`:
  `"serverless"` (Friendli Model APIs — the hosted pay-per-token catalog),
  `"dedicated"` (Dedicated Endpoints; the `model` field is the **endpoint ID**,
  or `ENDPOINT_ID:ADAPTER_ROUTE` for Multi-LoRA), and `"container"`
  (self-hosted Friendli Engine; `base_url` required, API key optional).
- **Dual transport**: `backend = "openai" | "httpx" | "sdk"`, auto-resolving
  **openai → httpx → sdk**. The vendor `friendli` SDK is supported but ranked
  last on purpose: its generated response models ignore unknown fields, so
  `reasoning_content` / `reasoning` are silently dropped and there is no
  `extra_body` escape hatch. The provider warns at startup when `backend =
  "sdk"` is combined with `parse_reasoning`.
- **Reasoning controls**: `reasoning_effort`
  (`minimal|low|medium|high|xhigh|max|ultracode`), `reasoning_budget`,
  `parse_reasoning`, `include_reasoning`, plus the chat-template switches
  `enable_thinking` / `clear_thinking` folded into `chat_template_kwargs`.
  Parsed chains of thought are surfaced by `extract_reasoning_content()` and
  `extract_delta_reasoning_content()` in both streaming and non-streaming mode.
- **Friendli Engine sampling**: `top_k`, `min_p`, `min_tokens`,
  `repetition_penalty`, `eos_token`, and XTC (`xtc_threshold` /
  `xtc_probability`) routed through `extra_body`; mutually exclusive body
  fields (`tools` vs `min_tokens`/`response_format`) are dropped with a warning
  instead of 422-ing.
- **Structured output** including Friendli's `regex` `response_format`, tool
  calling with first-class `Message.tool_calls` (R-2), and multimodal input via
  `metadata["inline_images"|"inline_audio"|"inline_videos"|"content_parts"]`.
- **Rich model discovery**: `GET /models` reports context length, max completion
  tokens, per-token pricing, a `functionality` block, modalities, reasoning
  options, `base_model` and `mode`; the catalog is cached, primed by
  `warm_up()`, and drives `get_max_context_length()`.
- **Auxiliary surfaces**: `tokenize()` / `detokenize()` / `render_chat()`,
  `text_completion()`, `transcribe_audio()`, and — on dedicated/container only
  — `create_embeddings()` / `generate_image()`, each gated with an actionable
  error on the wrong endpoint type. `get_team_cost()` / `get_team_usage()` read
  the Friendli Suite billing APIs for the configured team.
- **Team scoping**: `team_id` (or `FRIENDLI_TEAM_ID` / `FRIENDLIAI_TEAM_ID`) is
  sent as `X-Friendli-Team` on every request.
- **Token counting**: local by default (tiktoken `cl100k_base`, then a
  character-ratio estimate); `native_token_count = true` routes counts through
  Friendli's exact `/tokenize` endpoint at the cost of one API request per
  count, with a local fallback when that call fails.
- **Config**: new `[providers.friendli]` section in `default_config.toml`
  (`api_key`/`api_key_env_var` → `FRIENDLI_TOKEN` → `FRIENDLIAI_API_KEY` →
  `FRIENDLI_API_KEY`, `team_id`/`team_id_env_var`, `endpoint_type`, `backend`,
  `base_url`, `suite_base_url`, `default_model`, `timeout`, the reasoning
  defaults, `native_token_count`, `fallback_context_length`) and a matching
  `provider_friendli` section in the confy schema.
- **Model cards**: `FriendliAdapter` for cardctl derives context, pricing,
  capabilities, modalities and reasoning options straight from the live
  catalog, with a `friendli.toml` enrichment overlay for architecture and
  display names; seven generated cards under
  `model_cards/default_cards/friendli/`.
- **Packaging**: `llmcore[friendli]` extra (`openai`, `httpx`, and the optional
  `friendli` SDK), included in `llmcore[all]`; `friendli` registered in
  `ProviderManager` with the `friendliai` / `friendli_ai` aliases.
- **Errors**: 401/403/404/429 map to actionable `ProviderError`s (429 marked
  retryable so `chat_completion_with_retry` applies), and context-overflow
  wording on 400/422 maps to `ContextLengthError`.
- **Docs, examples & tests**: `docs/Friendli_provider_usage.md`,
  `examples/friendli_example.py`, README updates, and a 120-test offline suite
  (`tests/providers/test_friendli_provider.py`) covering credential/backend
  resolution, parameter splitting, payload building, both direct backends,
  discovery, tokenization, endpoint gating, Suite APIs and error mapping.

### Fixed — `ContextLengthError` construction in three providers

- **OpenAI, DeepSeek and Z.ai raised `TypeError` instead of
  `ContextLengthError` on every context overflow.** All three constructed the
  exception with a keyword set it has never accepted
  (`provider_name` / `model` / `max_tokens` / `requested_tokens`) rather than
  its real signature `(model_name, limit, actual, message)`, so the `raise`
  statement itself blew up inside `__init__`. Callers catching
  `ContextLengthError` — including llmcore's own context-management and agent
  retry paths — never saw it, and the user got an opaque `TypeError` with no
  model or limit attached. Fixing `OpenAIProvider` also fixes its subclasses
  (DeepInfra, vLLM, Poe, OpenRouter). Anthropic, Mistral, Gemini, Kimi and the
  new Friendli provider already used the correct signature.
- **Regression coverage** (`tests/providers/test_context_length_error_mapping.py`):
  a static AST check asserts that *every* `ContextLengthError(...)` call site
  in `src/llmcore` uses keywords the constructor accepts — covering providers
  with no error-path tests and any added later — plus behavioural tests that
  drive the real `chat_completion()` failure path of each fixed provider and
  assert the mapped exception carries the model name and context limit. Both
  guards were verified to fail against the pre-fix code.

### Notes

- Friendli's documented `/detokenize` and `/chat/render` routes currently
  return 404 on Model APIs (verified 2026-09-20); they are implemented and work
  on Dedicated Endpoints / Container, and every caller degrades gracefully.
- Model APIs rate limits are tier-based; tier 0 is "adaptive" and in practice
  allows only a couple of requests per minute, which is why native token
  counting is opt-in and `examples/friendli_example.py` paces its calls.

### Added — gpu.ai provider (text surface)

[gpu.ai](https://gpu.ai) sells serverless inference behind an
OpenAI-compatible API, media generation, and GPU rental. This adds the text
surface: 14 chat models and one embedding model, with all 49 catalogue
models carded.

Its `/v1/models` is the most complete catalogue of any provider integrated
here — `context_length`, `pricing`, `supported_parameters` and `aliases`
inline — so cards are generated from discovery rather than hand-written.
Image and video models are filtered out of *discovery* while keeping their
cards: listing them as chat models would make them selectable by a router
that cannot call them.

Three provider-specific behaviours, each found by calling the API:

- **`max_tokens` includes reasoning.** On `gpuai/gpt-oss-120b`,
  `max_tokens=8` returned empty content having spent all 8 tokens, while
  256 returned `"OK"` using 45. A layer trimming `max_tokens` to save money
  produces *nothing*, not less; `MIN_REASONING_MAX_TOKENS` documents a floor.
- **Reasoning tokens are billed as output with no breakdown**, so cost
  estimates assume the output rate. Recorded per card under
  `provider_metadata.reasoning_billing` rather than left implicit.
- **`encoding_format` is rejected outright** — the parameter, not a value.
  The OpenAI SDK injects it, so embeddings build their payload directly.

### Changed — `ModelCard.context` is optional

An image or video generation model has no token context window, and because
`context` was **required** such a model could not be carded at all — 34 of
gpu.ai's 49 models failed validation. This is the same structural blocker as
the required `per_million_tokens`, one field over.

`get_context_length()` now returns `None` rather than 0 where the question
does not apply, since 0 reads as "no room". Two unguarded readers in
`schema.py` are fixed; `api.py` already defended against absence.

### Rentable compute beyond Colab: gpu.ai and DeepInfra runtime backends

The runtimes subsystem could provision from one place, and its data model said
so in several ways that had nothing to do with Colab being first and everything
to do with Colab being the only one. Two rental backends now ship alongside it,
and the places where the model assumed Colab are named and fixed rather than
worked around.

**The billing unit is now explicit.** A new `CostUnit` travels with every rate
and every ceiling. Colab bills in compute units and publishes no dollar value
for them; gpu.ai and DeepInfra bill in dollars. There is deliberately no
conversion between the two: a fabricated exchange rate is the same class of bug
as every other one this cost model has had — a plausible number standing in for
an absent implementation.

- `RuntimeHandle.max_compute_units` / `compute_units_used` are now `max_spend`
  / `spend_used` with a `spend_unit`. The spend ceiling is the only guard
  against a runtime *busy* in a loop, which the idle reaper never touches, so
  it has to compare like with like. State files written before this keep
  loading, with their old keys read and their unit inferred: a file on disk
  describes something that may still be billing, and dropping its ceiling
  because a field was renamed would remove the guard.
- `RuntimeManager.up(max_compute_units=...)` still works and warns. Passing
  both names raises rather than picking one.
- `runtimes status` gained a `SPENT` column. `-` there means the backend does
  not report consumption — not that the runtime is free.
- `Plan` gained `cost_unit`, `gpu_count`, `region` and `offering_id`, and a
  `shape` property. The CLI previously printed any backend's rate with the
  literal suffix "compute units/hour", so a $2.50 GPU displayed as "2.5
  compute units/hour".
- `Sizer` takes a `skus` catalogue. Colab's five-rung `GPU_SKUS` table stays as
  Colab's menu; a rental backend passes its own, read from its own API, because
  live per-region dollar prices are not something a table in this source could
  keep honest.

**gpu.ai** (`backend="gpuai"`) reads `GET /v1/pricing` and `/v1/gpu-types`,
neither of which needs a key — so `estimate` stays genuinely free. It keeps the
cheapest available offering per shape, records which region that was, and pins
the launch to that `offering_id`: a catalogue row is a quote, not a booking, and
without the pin the platform may place the instance on a pricier row than the
one approved. The quoted rate includes the instance disk, which is billed
separately from the GPU. Serving uses gpu.ai's own `vllm` template, and because
its HTTPS tunnel authenticates with HTTP Basic rather than a bearer token, the
handle carries a header.

**DeepInfra** (`backend="deepinfra"`) drives dedicated LLM deployments, which
reserve hardware by the hour. Two traps it handles: a deployment's name must be
prefixed with the account's display name or DeepInfra refuses it, and
`/deploy/list` returns serverless per-token model references alongside real
rentals — reporting those as untracked runtimes would raise a cost alarm about
something with no hourly cost. Accrued spend comes from DeepInfra's own meter
rather than elapsed wall-clock, because a deployment stopped and restarted has
billed for less time than it has existed. `down()` deletes, since reserving is
what bills; `down(release=False)` stops it and records `detached`.

Both fail closed — a failure after the instance exists terminates it before
raising — and both report compute llmcore did not start as an **orphan** with
the command to adopt it. That matters more here than on Colab: a forgotten
Colab session expires on its own, and a forgotten rental bills until something
kills it. gpu.ai's `auto_terminate_hours` is also set from the configured
lifetime, rounded up, so the platform enforces a kill even if llmcore dies.

`ComputeRuntime` itself needed no new verbs. The audit behind this work is in
the design docs; the protocol was written with other backends in mind and held
up.

### Configured request headers reached only one of two transports

`default_headers` in a provider config was applied to the direct httpx
transport and silently dropped by the SDK client — which is the default — though
the code claimed both transports send the same request. It now reaches both.
This matters for any endpoint whose authentication *is* a header, such as a
runtime served behind an HTTP Basic tunnel.

### Fixed — an unstated context window is now absent, not 128,000

The card generator had no way to say "nobody told me". It defaulted
`max_input_tokens` to 4,096, which was far too low and produced broken context
budgets at runtime; that was fixed by changing the default to 128,000, which is
far too low for some models and **meaningless** for others. Both were the same
error one layer down — answering with a plausible number instead of admitting
the question had no answer. #48 made `context` optional precisely so unknown
could be represented; this makes the generator use that.

Measured before the change: **986 of 2,279 effective cards (43%) claimed
exactly 128,000 tokens.** Among them, 51 speech-to-text models whose input is
audio, 114 text-to-speech voices whose published limit is in characters, and
124 image generators with no token window of any kind.

- `CardBuilder._build_context()` returns `None` when no source states a window.
  There is no fallback, deliberately.
- The Poe adapter's catch-all `128_000` and its `4_096` for media endpoints are
  gone. Its per-family values are researched and stay: `gpt-4o` really is
  128,000 and Claude really is 200,000.
- The OpenRouter adapter no longer substitutes 128,000 when `context_length`
  is absent.
- **396 cards** had the fabricated window removed: every speech, image, video
  and audio card that carried one. No media card claims 128,000 any more, and
  430 cards now state no window at all, which llmcore reads as unknown.

Windows that a vendor actually publishes were **kept**, because they are real
numbers in the wrong unit rather than numbers with no referent: ElevenLabs
states a per-model character limit (5,000–40,000), and Whisper's decoder really
does stop at 448.

`tests/model_cards/test_absent_context.py` audits this so it cannot come back,
including that an unrecognised Poe model gets no guess while `gpt-4o` keeps its
researched value.

**Not fixed:** 757 chat, vision and embedding cards still claim 128,000. For
those the field is meaningful and the value may even be right, so they need a
vendor source rather than a blanket strip. The HuggingFace embedding cards are
the most suspect — a BERT-family encoder stops at 512, not 128,000.

### Fixed — `PoeProvider.get_max_context_length()` could return `None`

It read `card.get_context_length()` straight out, which now returns `None` for
a card stating no window — breaking the method's `int` contract and, through
it, `ModelDetails` validation for the whole Poe model listing.

The fallback stays in this method on purpose. A card states what is *known*
about a model; this method is a **budgeting assumption** callers use to decide
how much history to send, so it has to produce a number. The card does not
claim 128,000, and nothing reading cards will believe it does.

### Fixed — context windows now come from the sources we already had

Follow-on from the previous entry. Having stopped the generator inventing
windows, this uses the real sources that existed all along, and admits unknown
where none does. **Effective cards claiming 128,000: 986 → 320.**

**HuggingFace (210 → 19), via each repo's own `config.json`.** Added to
pricewatch rather than here, since it is a vendor-native fetch. 197 repos state
a window and **only 5 of those were actually 128,000**: `BAAI/bge-base-en-v1.5`
is **512**, overstated 250-fold, and `Qwen/QwQ-32B` is 40,960 — while
`ByteDance-Seed/UI-TARS-1.5-7B` genuinely is 128,000, which is why this reads
rather than infers. 63 of the windows live under `text_config`, where a
multimodal repo keeps its language model's; `model_max_length` is never
consulted, because it is tokenizer config and routinely holds the sentinel
`1000000000000000019884624838656`.

**Poe (170 → 45).** 75 cards claimed 128,000 purely from the catch-all removed
above — including `canvas-creator`, `code-editor` and `elevenlabs-music`, which
are Poe bots, not frontier models. Those are now unknown. A further **50
disagreed with a family rule the adapter already held**, 8 of them really
1,000,000, so llmcore was truncating work it could have done. The 45 that
remain are the ones a researched rule actually backs.

**Anthropic (8 → 0).** No Claude model has ever had a 128,000-token input
window — they are 200,000, or 1,000,000 for 5.x — so unlike chat cards in
general, that value on a Claude card is always wrong. `claude-opus-5-5`,
`claude-sonnet-5-5` and `claude-fable-5-1` now carry the 1,000,000 that
Anthropic's own comparison table publishes. The other five are unknown:
Anthropic publishes only the current generation, and llmcore's provider table
stops at 4.6, so nothing states them.

**OpenAI: 6 cards contradicted the provider's own table.** `gpt-4` claimed
128,000 where `DEFAULT_OPENAI_TOKEN_LIMITS` says **8,000** — a 16-fold
overstatement — and the `gpt-3.5-turbo` family claimed it where the table says
16,000.

Three audits keep each of these from returning, and they deliberately assert
different things. An **exact-id** table (OpenAI's) is ground truth a card may
not contradict. A **family-pattern** table (Poe's, which answers 200,000 for
anything matching "claude") is only a fallback, and 73 Poe cards legitimately
disagree with it because the *table* is the stale one — it would call
`claude-sonnet-5.5` 200,000 and `deepseek-v4-pro` 128,000 when the cards
correctly say 1,000,000 and 1,048,576. So the Poe audit only forbids an
unbacked 128,000.

**Still 320**, nearly all needing a source rather than a correction: `openai`
100 (newer models absent from its table), `deepinfra` 99, `openrouter` 46,
`poe` 45 (backed by a family rule), `huggingface` 19 (gated or config-less
repos). 505 cards now state no window at all, which llmcore reads as unknown.

### Fixed — a model listing can now say it does not know the context window

Caught by checking the consequences of the previous two entries rather than
assuming them. Once 227 packaged cards stopped claiming a window they were
never told, two providers that fed card values straight into `ModelDetails`
broke — and one of them had been broken all along.

**`ModelDetails.context_length` is now `int | None`, defaulting to `None`.** It
was `int` with a default of **4,096** — the same fabrication the card generator
used to make, one layer up — so an explicit `None` was refused outright. A
caller could not tell a real 4k model from one nobody had described.

- **Deepgram's `get_models_details()` raised `ValidationError`.** Its 144
  speech cards state no token window, because their input is audio, and the
  provider passed that through. It failed on the first model.
- **HuggingFace's `get_models_details()` raised `AttributeError`, and always
  had.** It called `registry.get_provider_cards()`, which does not exist on the
  registry — and behind that first failure sat three more:
  `card.context.get(...)` and `card.architecture.get(...)` treat pydantic
  models as dicts, and `card.capabilities.tool_calling` is not a field, it is
  `tool_use`. Four independent bugs in one function, with no test touching it.
  It now lists all 298 cards, 83 of them reporting no window.

Both consumers of the field were adjusted rather than left to meet a `None`:
`MemoryManager._get_precise_context_length()` falls through to the provider's
own fallback instead of returning it (the method promises an `int`, and callers
size a context budget with it), and the gRPC bridge sends `0`, which is how
that wire format already spells unset.

A sweep of every other provider's listing found no further dead code: the seven
that fail without a network genuinely call their vendor's API.

### Added — a per-turn step and spend budget, enforced in the proxy

The cost-control design's layer 4, which was spec'd and left unimplemented.
The agent circuit breaker bounds an agent *run*; the chat path had no per-turn
budget at all — and `chat()` is what the proxy exposes to agent harnesses,
which is where the expensive turns in the measurements actually came from.

**Why steps rather than prompt complexity.** Measured on this project's own
traffic: routing by predicted prompt complexity is worth about **2.2% of
spend**, while turns over 200 steps are **62%** of it. Cost is roughly
`steps × context × cached_rate` with context per step close to flat, so the
quantity worth bounding is the one observable directly, cheaply and exactly.

`llmcore.routing.budget` adds `TurnBudget`, `BudgetPolicy`, `BudgetVerdict`
and `BudgetAction` (`report` / `warn` / `stop`), with a cost projection from
the mean cost per step. Configured under `[routing.budget]`, and the proxy
refuses a step past the ceiling with **429** and `type: "budget_exceeded"`,
reporting state in the `llmcore` usage block of every response. Streaming is
counted too, or `stream=true` would be a way to spend unseen.

**There are no default limits.** An unset dial is unbounded. The agent circuit
breaker shipped with a `$1.00` default that would have cut off 50.7% of this
project's normal turns, and a number nobody chose is worse than no number.

Three questions the design left open, and how they resolved:

- **Where the accumulator lives:** caller-owned and explicitly fed, like
  `AgentCircuitBreaker`. Building it on routing's event stream was the
  tempting alternative and is wrong — `emit_attempt` is guarded by
  `has_sinks()`, so the budget would silently stop counting whenever nobody
  was listening. A spend guard must not depend on observability being on.
- **Mid-turn `constrain`:** not implemented, because it is not safe. Changing
  model inside a conversation changes behaviour, and with preserved-thinking
  models it invalidates reasoning blocks already in the history.
- **Who sets it:** `caller > policy > config`, mirroring the classifier chain.
  `with_policy()` replaces the dials without resetting the meter, so raising a
  limit cannot clear what a turn has already spent.

**Unknown cost is not zero.** A step against an unpriceable target records its
cost as unknown rather than `0.00`; the step still counts and
`cost_is_partial` goes true, with the verdict saying the spend ceiling cannot
be enforced. A budget that treated unpriced calls as free would never trip —
the failure this cost model has had at several layers. A step ceiling still
works regardless, which is part of why it is the better dial.

**It can only count a turn the harness identifies.** A step budget needs a
stable key across the calls of one turn, and the proxy has one only when the
caller sends `user` or `llmcore.session_id`. Without it every request gets a
fresh synthetic session, so "steps so far" would always be zero; the proxy
declines to track the turn rather than enforce something meaningless, and
omits the `budget` block so the difference is visible.
### Fixed — a failed `LLMCore.create()` left the process unable to exit

`StorageManager` is constructed *before* the providers, and its aiosqlite
backend starts a **non-daemon** worker thread. When provider setup then failed,
that thread outlived the exception — so a caller that caught the `ConfigError`
and carried on could never exit, because interpreter shutdown blocks forever
joining it.

Not hypothetical: it is why `llmcore-runtimes estimate` computed the right
answer and then hung until killed, with stdout never flushed. Nothing in the
error message pointed at storage. Initialization now releases what it opened
before raising, best-effort, and never lets a cleanup error replace the failure
worth reporting. The `ConfigError` also now chains its cause.

### Fixed — the commands that stop you spending no longer need the whole library

`runtimes/cli.py` exists because, in its own words, *"the commands that matter
most here are the ones you need when something has gone wrong... the thing you
are trying to do is stop paying for a VM."* It then built a full `LLMCore` for
every command — every provider SDK, the prompt pack, storage, embeddings — so
one missing vendor library, or a `default_provider` that no longer loads, took
`status` and `down` down with it.

`estimate`, `status`, `down`, `logs`, `bake`, `cache`, `doctor` and `spend` now
fall back to a `RuntimeManager` built straight from config, printing what
degraded. `up` and `adopt` still require the full stack, because they attach
the runtime as a provider instance.

### Added — `llmcore-runtimes doctor`

Preflight checks, which llmcore had no answer to. Subsystem state, whether the
state directory is writable, today's spend, anything currently billing, and
each backend's own preconditions: credential present, catalogue reachable, CLI
executable, SSH keys registered.

Exit code is the worst finding — `2` blocks, `1` billing or degraded, `0` clean
— so `doctor && up` will not provision into a broken state. Severity is graded
by *consequence* rather than state, and `$$$` outranks everything: a broken
backend costs nothing, a forgotten instance costs money every hour.

The first live run found a real problem. llmcore's default Colab `cli_path` is
a bare `colab`, so where that CLI lives in a venv llmcore could not drive Colab
at all — and would only have said so at provision time, after the decision to
spend. Credential checks report whether a variable is set, never its value.

Backends opt in with `async def diagnose()`, discovered structurally like the
backend protocol itself; all three shipped backends publish checks.

### Added — a daily spend ledger and cap for provisioned compute

`max_spend` bounds one runtime and `[routing.budget]` bounds one turn. Neither
bounded a **day**, so ten runtimes each under its own ceiling could spend ten
times the intent with nothing objecting. Found by writing the agent-lens
migration guide, which has had a ledger and a daily cap all along.

```toml
[runtimes.budget]
max_usd_per_day = 20.00
max_compute_units_per_day = 150
```

**Two dials, because one cap cannot span two units** — there is no defensible
rate between Colab's compute units and dollars, and inventing one to produce a
single tidy number would be this cost model's recurring bug in a new place.

The cap is checked **before** `up()` contacts the backend, because a cap
applied after compute is assigned is not a cap; there is a test asserting the
backend is never asked. It raises `DailyBudgetExceededError`, its own type so
"cannot afford this" can be caught apart from "this is broken". A plan with no
known unit is *not* blocked: refusing it would turn "we do not know the price"
into "you may not have it".

`llmcore-runtimes spend` reports per day and per unit. The ledger is
append-only JSON lines at `<state_dir>/spend.jsonl`, written when a runtime is
released — the moment its cost stops moving — and a write failure never fails
the release, since a ledger that could block a teardown would be a way to keep
paying for something.

**Unknown spend is not zero.** A run whose cost could not be determined is
recorded as *unpriced*, and every total containing one says so; when the cap
refuses it states that the real figure is higher than the one quoted.

### Fixed — `RuntimeStatus` accepted a string where a `CostUnit` belongs

It is a plain dataclass, so nothing validated it: a backend author passing
`spend_unit="usd"` — the exact string value of `CostUnit.USD` — got an
`AttributeError` from `spend_so_far` much later, one layer from the cause. The
unit is now coerced, with an unrecognised one falling back rather than raising,
since a status describes something that may be billing and refusing to build it
would hide the runtime entirely. Found by making the mistake while writing the
diagnostics tests.

### Fixed — CI was red on `main`, in two ways this work was responsible for

Found by reading this branch's own CI logs rather than trusting a local run.
Both failures were already on `main`, and neither was visible locally — the
first because a local pricewatch override masked it, the second because the
local run was a subset that skipped `tests/tools`.

**`cardctl` had no gpu.ai adapter.** PR #49 registered `gpuai` in
`PROVIDER_MAP`, and `tests/tools/test_cardctl_coverage.py` exists to catch
exactly that: a provider whose cards can never be generated. It had been
failing ever since. `tools/cardctl/adapters/gpuai_adapter.py` now reads the
catalogue, verified live against all 49 models.

gpu.ai prices four modalities four different ways, so the adapter converts
cents-per-1M-tokens for chat and embedding, and routes images
(`per_mpxl_microcents`) and video (`per_video_microcents`,
`per_video_second_cents`) through unit pricing instead of coercing them into a
per-token rate they do not have. Media models get **no** context window rather
than a defaulted one, and `supported_parameters` is read as a real negative —
gpu.ai enumerates what each endpoint accepts.

**Six Claude cards had `pricing: null`.** `claude-opus-5-5`,
`claude-sonnet-5-5`, `claude-fable-5-1` and their predecessors carried no
pricing at all, so `get_cost()` on llmcore's most-used model returned nothing
and four cache-accounting tests failed on `NoneType`. The rates now come from
Anthropic's own published pricing page, via the same `scrape:anthropic` feed
that supplied the 1M context windows — not inferred.

A lesson worth recording: a locally installed pricewatch override set makes the
packaged cards look more complete than they are. Tests that read the registry
pass on a developer's machine and fail in CI, which is the wrong way round.

### Changed — the architecture diagrams are SVG

The eight box-and-arrow diagrams in the README, the agentic system guide and
the model-cards doc were ASCII art. They are now SVG, drawn with a shared
design system and rendered through Krita to verify each layout before it
shipped — which is how several problems were caught: a title and a path label
overlapping by 5px, a `RESULT` box sitting 26px below the bottom of its own
canvas, a loop label running off the left edge, and connector arrows whose
heads were longer than their 4px shafts.

The diagrams carry a light card background rather than a transparent one.
GitHub renders a README on white *or* near-black, and dark strokes on
transparency vanish on the dark one.

Each file has a `<title>` and a `<desc>`, so a screen reader gets the
explanation rather than a filename, and the markdown alt text says what the
diagram shows. Text stays live `<text>` with a portable font stack, so it
scales and selects; every file was re-rendered through librsvg — a different
renderer with different fonts — to confirm nothing overflows when the viewer's
font is not the one it was drawn with.

**File trees and the exception hierarchy stay as text.** They were caught by
the same search, but an image of a directory listing is worse than the
listing: it cannot be copied, searched or diffed.

The generator lives in `tools/diagrams/`, so the pictures can be regenerated
rather than hand-edited: `python -m tools.diagrams` writes them and
`--check` fails if the committed files have drifted from the source. That
check is a test, so a forgotten regeneration fails the suite instead of
shipping a diagram that no longer matches the code that drew it.

`docs/Diagrams_usage.md` covers adding one, the layout DSL, the design
system, and how to look at a change before committing it — including the
`--krita` mode, which emits font-substituted fragments for the av-krita-mcp
server. Geometry bugs are invisible in markup and obvious on screen, which is
how all of the above were found.

## v0.53.0

### Added — TypeSafe.ai (System One) provider

- **TypeSafe.ai provider**: first-class `TypeSafeProvider` for TypeSafe's
  System One typed-judgment API (Jev models). It is **not a chat model**:
  `POST /v1/systemone` evaluates a `state` (string / JSON object / array)
  against typed questions and returns calibrated, structured answers —
  `noul` (yes/no probability), `choice` (pick one option; per-option
  probabilities + `confidence`), `score` (rubric level; per-level
  probabilities + `confidence`).
- **Typed surface**: `await provider.system_one(state, questions, model=…)`
  with the `Noul` / `Choice` / `Score` builders (raw dicts accepted;
  `normalize_questions()` validates locally), a `SystemOneResult` with
  `.nouls` / `.choices` / `.scores` accessors, `usage`, `request_id`
  (`x-typesafe-request-id`) and `raw`; `await provider.list_models()`
  (`GET /v1/models`).
- **Chat bridge**: `llm.chat(msg, provider_name="typesafe", questions={…})`
  sends the conversation as the state (or an explicit `state=`) and returns
  the answers as a JSON string; usage flows into cost tracking. Streaming
  and tool calling raise a non-retryable `ProviderError` (400).
- **Transport**: plain `httpx` against the two REST endpoints (no vendor
  SDK). In-provider retries for `408/429/5xx/529`, timeouts and connection
  errors with exponential backoff honouring `Retry-After` / `retry-after-ms`;
  errors map to `ProviderError` with `status_code`, `retryable`,
  `retry_after_seconds`, the request id and the server's validation detail.
- **Config**: new `[providers.typesafe]` section (`api_key`/`api_key_env_var`
  → `TYPESAFE_API_KEY`, `base_url` ← `TYPESAFE_BASE_URL`, `default_model`
  ← `TYPESAFE_DEFAULT_MODEL`, `timeout`, `max_retries`,
  `retry_backoff_initial`/`_max`, `fallback_context_length`) in
  `default_config.toml`, with a matching `provider_typesafe` section in the
  confy schema (whose `app.version` now tracks the package again).
- **Model cards**: new `ModelType.DECISION` (`"decision"`) and a builtin
  `typesafe/jev-1.13.0` card carrying the `jev-latest` / `jev-preview`
  aliases, the 64k-per-request / 32k state+longest-question budget, the
  $0.042 per 1M input tokens price (output free), rate limits and question
  types. `get_models_details()` lists the versioned id and its aliases.
- **cardctl**: `TypeSafeAdapter` (collapses the listed aliases onto the
  versioned id; registered as `typesafe` / `jev`) and a `typesafe.toml`
  enrichment overlay so regenerated cards match the builtin one.
- **Packaging**: `llmcore[typesafe]` extra (`httpx`), included in
  `llmcore[all]`; `typesafe` registered in `ProviderManager` with the `jev`
  alias.
- **Docs & examples**: `docs/TypeSafe_provider_usage.md` (surface, builders,
  answers, confidence-gated routing, chat bridge, errors/retries, limits),
  README / `CONFIG_REFERENCE.md` / `model_cards.md` updates, and
  `examples/typesafe_example.py`.
- **Tests**: 92-test offline provider suite (config/env precedence, builders,
  `system_one`, error mapping + retry loop, `list_models`, model cards, chat
  bridge, tokens, lifecycle, registration), builtin-card + registry alias
  tests, cardctl adapter tests, and a key-gated live smoke
  (`tests/integration/test_typesafe_live.py`).

### Fixed

- DeepSeek DSML text-format tool calls are normalized at the provider
  boundary; OpenRouter model cards refreshed (both landed after the 0.52.0
  entry was written).

## v0.52.0

### Changed — Grimoire is THE control plane (breaking)

- **Hard dependency on `grimoire>=0.4.0`.** llmcore ships a packaged
  `llmcore-builtin` grimoire pack (`src/llmcore/grimoire_pack/`) covering
  EVERY agent prompt as spells — the cognitive phases (`plan`, `think`,
  `validate`, `reflect`, plus the new `finalize`), the activity (XML) protocol,
  the goal classifier, the fast path (+ canned responses as
  `llmcore/fast_path/*` promptlets), the five builtin personas (definitions in
  spell `attributes.persona`), the Darwin arbiter/TDD prompts, and autonomous
  goal decomposition. Zero config → the bundled pack loads; user layers
  override by id.
- **Fail-loud, no silent fallbacks (breaking).** Every inline/f-string prompt
  fallback is DELETED. `prompt_registry` is required by
  `SingleAgentMode`/`CognitiveCycle` (raise on `None`);
  `EnhancedAgentManager` self-builds a bundled-only adapter when constructed
  directly. New `[grimoire]` config section (layers, prompt_map, strict,
  metrics) with startup validation that renders every required template and
  raises `ConfigError` naming template, spell, winning layer, and cause.
- **Prompt-consumer tail routed through the registry**: goal classifier LLM
  fallback, fast-path executor (system+user via `render_messages`; canned
  responses via promptlets), `PersonaManager.load_from_grimoire()` (spells
  tagged `llmcore.persona`; hardcoded builtins remain only for legacy
  no-grimoire construction), `MultiAttemptArbiter` + `TestGenerator`/
  `TDDManager` (`darwin_arbiter_*`/`darwin_tdd_*` templates; the
  `ArbiterPrompts` class and TDD prompt constants are gone),
  `GoalManager._decompose_goal` (`goal_decomposition` — also fixes a latent
  `MessageRole` ImportError that silently disabled LLM decomposition),
  `agents/activities/prompts.py` DELETED (activity prompts render via
  `activity_system`/`activity_execute`), and the four cognitive default
  templates removed from `template_loader` (the variable-mismatch source;
  `PromptRegistry.with_defaults()` now seeds snippets only).
- **Builtin tools are catalog-driven**: `GrimoireToolCatalog` builds the five
  builtins (`finish`, `human_approval`, `semantic_search`, `episodic_search`,
  `calculator`) from bundled rune contracts with real parameter schemas and
  risk/approval/OWASP metadata; unbound runes are visible in `contracts()`
  but never registered.
- **Deprecations**: the legacy `agents/cognitive_cycle.py` + `prompt_utils.py`
  stack (outside the control plane) — `AgentManager.run_agent_loop()` now
  emits a `DeprecationWarning`; removal next minor. Unwired
  `reasoning/react.py`/`reflexion.py` carry deprecation notes.

### Added/Fixed — Darwin convergence & accounting (plan Phase 2)

- **Finish-tool convergence** (2.1): native `finish`/`final_answer` tool calls
  terminate THINK (with act-phase defense in depth); `finish` gets a real
  `{answer}` schema; new `TerminationReason` enum stamped at every stop site
  and mirrored on streaming results (`termination_reason`).
- **`remaining_steps` + in-cycle forced finalize** (2.2): the cycle can no
  longer exit un-converged on budget paths — a tool-less finalize pass (new
  `finalize_prompt` spell, `tool_choice="required"` where supported) or the
  synthesis fallback guarantees an answer; hosts' "grace" crutches are
  redundant.
- **Deterministic guards under `skip_validation`** (2.3): tool-registry +
  dangerous-pattern prechecks always run; only the LLM judge is skipped
  (`ValidationConfig.deterministic_guards` escape hatch).
- **Redundant-call detector** (2.4): repeated tool signatures skip
  VALIDATE/ACT with a corrective observation; ≥3 repeats → forced finalize.
- **Conditional PLAN** (2.5): `PlanningConfig.mode`
  (`always|first|complex_only|on_failure`, default `complex_only`) + replan
  budget — planning no longer precedes observation on knowledge tasks.
- **Grounded structured REFLECT** (2.6): JSON reflection (where the provider
  supports it) with text fallback; `action_success=False` clamps
  self-judgment; reflection gated to action iterations; THINK's
  `expected_outcome` finally reaches OBSERVE.
- **End-to-end token/cost accounting** (2.7): `PhaseUsage` per LLM phase,
  iteration/state totals, circuit-breaker cost double-count fixed, and new
  `api.record_agent_usage(session_id, records)` so `get_session_token_stats`
  covers agent runs (hosts flush per run/segment).

## v0.51.0

### Added — Z.ai (GLM) provider

- **Z.ai provider**: first-class `ZaiProvider` for the Z.ai Open Platform,
  serving the GLM model family (`glm-5.2`, `glm-5.1`, `glm-4.7`, the `glm-*v`
  vision models, and `embedding-3`).
- **Selectable transport backend** (`backend` config): the provider prefers
  the official synchronous `zai-sdk` (`"sdk"`, the default when installed,
  bridged to async via threads), and falls back to the `openai` SDK
  (`"openai"`, OpenAI-compatibility mode) and/or direct `httpx` REST calls
  (`"httpx"`). Unset/`"auto"` auto-detects in that order. All three backends
  share one request-building path and normalize responses to a common shape.
  Chat, embeddings, and every media API honor the selected backend. Built on
  the chat endpoint (`https://api.z.ai/api/paas/v4`) with:
  - GLM **thinking mode** (`thinking = {"type": "enabled" | "disabled"}`) and
    `reasoning_effort` (`none|minimal|low|medium|high|xhigh|max`).
  - `reasoning_content` extraction in both streaming and non-streaming modes.
  - Open-interval `(0, 1)` clamping of `temperature`/`top_p` (matching the
    Z.ai API constraint).
  - Platform extras (`do_sample`, `request_id`, `user_id`, `seed`,
    `watermark_enabled`, `sensitive_word_check`, `tool_stream`) routed via
    `extra_body`.
  - Tool calling, cache/reasoning token usage accounting, and `embedding-3`
    embeddings.
  - Region selection (`overseas` default, or `china` for the
    `open.bigmodel.cn` endpoint).
- **Z.ai multimodal media APIs**: the provider implements llmcore's full
  media surface against the Z.ai endpoints:
  - `generate_image` (CogView / GLM-Image, `/images/generations`)
  - `generate_speech` (GLM-TTS, `/audio/speech`)
  - `transcribe_audio` (GLM-ASR, `/audio/transcriptions`)
  - `ocr` (GLM-OCR layout parsing, `/layout_parsing`)
  - `generate_video` + `retrieve_video_result` (CogVideoX,
    `/videos/generations` with async task polling)
  - `web_search` (Z.ai Web Search API, `/web_search`)
- **cardctl**: new `ZaiAdapter` (OpenAI-compatible `/models` listing) and
  `zai.toml` enrichment overlay registered in the model-card tool, with
  `glm`/`zhipu`/`zhipuai`/`bigmodel` aliases; media/generation model ids are
  filtered out of chat cards.
- **Packaging**: new `llmcore[zai]` extra (`openai` + `httpx`), included in
  `llmcore[all]`.
- **Pricing**: GLM model cards and the `zai.toml` enrichment now carry
  verified USD per-1M-token pricing (input/output/cached-input) from the
  official Z.ai pricing page, plus per-unit reference notes for image/video/
  web-search services.
- **Provider registration**: `zai` registered in `ProviderManager` with
  `glm`, `zhipu`, `zhipuai`, and `bigmodel` aliases.
- **Model cards**: builtin cards for `glm-5.2`, `glm-5.1`, `glm-4.7`,
  `glm-4.6v` (vision), and `embedding-3`.
- **Tests**: 57-test offline suite for the Z.ai provider.

## v0.50.0

### Added — June 2026 agent, context, provider, and observability rollup

- **Deepgram voice/audio provider**: native SDK integration for speech-to-text,
  text-to-speech, Flux streaming, Voice Agent, text intelligence, token grants,
  Deepgram model cards, docs, examples, and offline tests.
- **Per-call usage accounting**: `LLMCore.chat_with_usage()` and `ChatUsage`
  expose prompt/completion/total token counts for transient calls without
  requiring session persistence.
- **Search providers**: the optional `llmcore.search` subsystem now includes
  Bright Data, Serper.dev, SerpApi, and Semantic Scholar, with provider-neutral
  result models and manager/facade wiring.
- **Token counting**: native provider fallback paths and OpenAI token counting
  now route through llmcore's shared model-aware token counters.
- **Agent execution**: typed plan-step specs, structured plan-step tool
  execution, loaded-tool validation, activity-protocol routing to loaded tools,
  runtime permission metadata, and preserved resumed history/pending action
  snapshots.
- **Context and memory**: objective-aware compression, semantic citation
  provenance, structured tool-result summaries, typed citation source handling,
  and external backend consolidation hooks.
- **Observability**: semantic retrieval events, context diagnostics after agent
  runs, context failure diagnostics, phase token summaries, iteration summaries,
  and ecosystem federation telemetry.
- **HITL auditability**: OWASP metadata and audit reporting for dangerous action
  patterns.

### Changed

- Bumped package and documentation metadata to `0.50.0`.
- Removed noisy import-time optional SDK warnings by lazy-loading
  `sentence-transformers` and `google-genai` only when their providers are
  instantiated.
- Updated `SingleAgent` environment config loading to use the current unified
  `confy.Config` path instead of the deprecated `load_agents_config(config_path=...)`
  compatibility path.

## v0.49.14

### Added — Deepgram voice/audio provider (STT, TTS, Flux, Voice Agent, Text Intelligence)

Adds a complete, native-SDK **Deepgram** provider — llmcore's first real-time
**voice/audio** provider. Deepgram is fundamentally different from the
text-completion LLM providers: its primary surfaces are WebSocket streams for
speech-to-text (STT), text-to-speech (TTS), and a bidirectional **Voice Agent**
(STT → LLM → TTS over one socket). There is no text chat-completion surface, so
`chat_completion` raises a clear, actionable `ProviderError` (HTTP 400,
non-retryable) and callers use the media methods instead. Tested against
`deepgram-sdk` v7.3.1.

- **New provider `DeepgramProvider`** (`llmcore/providers/deepgram_provider.py`),
  registered in `ProviderManager.PROVIDER_MAP` as `"deepgram"`. Follows the
  Gemini native-SDK template: lazy SDK import with an availability flag, an
  `ImportError` (surfaced as *"install llmcore[deepgram]"*) when the SDK is
  absent, and `ConfigError` for bad config. Wraps the official async-first,
  WebSocket-native `deepgram-sdk` (v7.x).
- **Batch media**: `transcribe_audio` (bytes / file path, or a remote `url=`) →
  `TranscriptionResult`; `generate_speech` (Aura voices) → `SpeechResult`.
- **Streaming**: `transcribe_stream` / `open_transcription_socket` (live STT),
  `transcribe_stream_flux` / `open_flux_socket` (Flux, listen.v2, turn-aware),
  and a dual-mode `stream_speech` / `open_speech_socket` (a string → REST
  streaming; an async iterable of text → a TTS WebSocket). The one-call
  streaming helpers fan microphone audio in and stream events/audio out
  concurrently, always close the socket on completion, and surface producer
  errors as `ProviderError`.
- **Voice Agent**: `open_voice_agent` (manual `DeepgramVoiceAgentSession`) and
  `run_voice_agent` (high-level driver that auto-answers `FunctionCallRequest`s
  via a `function_handler` and exposes an `on_event` hook). Runtime steering:
  `inject_user_message` / `inject_agent_message` / `update_prompt` /
  `update_think` / `update_speak` / `respond_to_function_call` / `keepalive`.
  **No system prompt is ever defaulted** — callers pass `prompt=` explicitly (or
  inject upstream), honouring the ecosystem "no hardcoded prompt" invariant.
- **Text intelligence**: `analyze_text` (read.v1) → `TextAnalysisResult`
  (summary / topics / sentiment / intents).
- **Token auth & account**: `grant_token` (short-lived access token for
  browsers/clients) and `get_projects`; the full management API remains
  available via the `provider.client` escape hatch.
- **New provider-neutral models** (`llmcore/models_multimodal.py`):
  `StreamEventType`, `TranscriptionStreamEvent`, `VoiceAgentEventType`,
  `VoiceAgentFunctionCall`, `VoiceAgentEvent`, and `TextAnalysisResult`.
- **Configuration**: a fully-documented `[providers.deepgram]` block in
  `config/default_config.toml` wires every capability (auth, transport,
  default models, `[stt]`/`[stt.streaming]`, `[flux]`, `[tts]`/`[tts.streaming]`,
  and `[agent]` with the SDK's `provider`-nested `listen`/`think`/`speak`
  shape and top-level `audio`).
- **Model cards**: 11 cards under `model_cards/default_cards/deepgram/` (STT:
  `nova-3`, `nova-3-medical`, `nova-2`, `nova-2-phonecall`, `whisper-large`,
  `flux-general-en`; TTS: `aura-2-thalia-en`, `aura-2-andromeda-en`,
  `aura-2-apollo-en`, `aura-asteria-en`, `aura-luna-en`), generated by
  `tools/generate_deepgram_cards.py`. Each card records Deepgram's published
  pay-as-you-go rates (per audio-minute STT / per-character TTS) in
  `provider_extension.pricing` with units, source URL, and capture date; the
  token-centric `pricing` field stays `null` (tokens are not the billing unit).
- **Packaging**: new optional extra `deepgram = ["deepgram-sdk>=7.0.0",
  "websockets>=12.0"]` (also folded into `all`).
- **Tokens / context**: Deepgram bills per audio-minute (STT) / per-character
  (TTS), so `count_tokens` returns a documented character-count heuristic and
  `get_max_context_length` returns a configurable nominal value
  (`fallback_context_length`, default 2000). These are **not** billing units.
- **Tests**: 54 new tests across `tests/providers/test_deepgram_provider.py`,
  `test_deepgram_streaming.py`, and `test_deepgram_agent.py` (fake
  clients/sockets; no network). Full provider suite: 553 passed, 2 skipped.
- **Docs & examples**: `docs/Deepgram_provider_usage.md` plus seven runnable
  `examples/deepgram_*.py` scripts.

The existing public API is unchanged; this is purely additive.

### Added — Per-call token usage surface (`LLMCore.chat_with_usage`)

Adds a **usage-returning** companion to `chat()` so that *callers* can meter
token consumption per call without enabling session persistence. This is the
foundational dependency for external usage/quota systems built on top of
llmcore (e.g. Convergence's metering bridge): previously the prompt/completion
token counts llmcore computes internally were only persisted when
`save_session=True`, and were never returned to the caller of a transient
`chat()` call.

- **New method `LLMCore.chat_with_usage(message, *, ...) -> tuple[str, ChatUsage]`**
  (`llmcore/api.py`). Non-streaming only; mirrors `chat()`'s full keyword
  signature. It runs the *exact* same code path as `chat()` (provider
  resolution, context preparation, the provider call, and llmcore's own
  prompt/completion token counting) and additionally returns the per-call
  token usage. The existing `chat() -> str` contract is **unchanged** — this is
  purely additive and opt-in.
- **New public value object `ChatUsage`** (`llmcore/usage.py`, exported from
  `llmcore`). A frozen dataclass carrying `prompt_tokens` / `completion_tokens`
  / `total_tokens` / `provider` / `model`, with `tokens_in` / `tokens_out`
  read-only aliases (so it is a drop-in for either naming convention) and an
  `is_available` flag. When usage cannot be determined every count is `None`,
  letting downstream meters degrade to a no-op rather than recording a
  zero-token event.
- **Concurrency-safe & residue-free.** Usage is read back via the existing
  per-session introspection cache (`get_last_interaction_context_info`) under a
  *call-local* session id, so concurrent calls never read each other's usage.
  When the caller passes no `session_id`, an ephemeral one is synthesised and
  its transient caches are dropped on return.
- **Tests** (`tests/api/test_chat_with_usage.py`) — `ChatUsage` value
  semantics, the signature/protocol contract, and offline end-to-end behaviour
  against an injected fake provider (no network).
- **Docs** — `docs/USAGE_chat_with_usage.md`.

## v0.49.13

### Added — Semantic Scholar search provider (`llmcore.search`)

Adds **Semantic Scholar** (https://www.semanticscholar.org/product/api) as a
first-class **search provider**, joining Bright Data, Serper.dev and SerpApi in
the optional `llmcore.search` subsystem. Semantic Scholar is a free, AI-powered
academic search engine over 200M+ papers; the provider wraps all three public S2
APIs (Academic Graph, Recommendations, Datasets), which share a host
(`https://api.semanticscholar.org`) under different path prefixes.

- **New provider `SemanticScholarSearchProvider`**
  (`llmcore/search/providers/semanticscholar_provider.py`) — native `httpx`
  client (no vendor SDK), advertising `web_search` and `batch_search`.
  Highlights:
  - **Optional API key / keyless by default.** The S2 key is optional; the
    provider operates against the shared public pool when no key is set (a
    missing key is **not** an error — unlike the other providers). When present,
    the key is sent via the `x-api-key` header and resolved from `api_key` /
    `token`, `api_key_env_var`, or `SEMANTIC_SCHOLAR_API_KEY` (with `S2_API_KEY`
    fallback); never logged.
  - **Four search flavors via `search_type`:** `relevance` (default,
    `/paper/search`), `bulk` (`/paper/search/bulk`, with continuation `token`),
    `match` (`/paper/search/match`, single best title match), and `snippet`
    (`/snippet/search`, text passages for RAG — `item.description` is the
    passage). `count` → `limit` (clamped per endpoint: 100 / 1000 / 1); all S2
    filters (`year`, `publicationDateOrYear`, `venue`, `fieldsOfStudy`,
    `publicationTypes`, `minCitationCount`, `openAccessPdf`, `sort`, `token`, …)
    pass through verbatim (`openAccessPdf` handled as a valueless presence flag).
    `country` / `language` / `device` / `engine` / `mode` are accepted but
    ignored (academic search is not geolocated and is always synchronous). Full
    payload preserved on `WebSearchResult.raw`.
  - **Mandatory exponential backoff** on `429` / `5xx` (S2 requires it), plus an
    optional proactive `min_request_interval` request spacer and a conservative
    default batch concurrency of 1.
  - **Client-side `batch_search`** fan-out (S2 has no multi-query endpoint),
    returning one ordered result per input query.
  - **Rich provider-specific methods** (not shoehorned into the cross-provider
    `discover`/`dataset_search` contracts, which don't fit S2's item-to-item
    recommender or bulk-corpus workflow): `paper`, `paper_batch` (≤500 ids),
    `paper_citations`, `paper_references`, `paper_authors`, `paper_match`,
    `autocomplete`, `snippet_search`, `author`, `author_batch`, `author_papers`,
    `author_search`, `recommend_papers`, `recommend_from_examples`, and the
    Datasets helpers `list_releases`, `get_release`, `get_dataset`,
    `get_dataset_diffs`.
  - **Free-ish `health_check()`** via a minimal autocomplete probe (S2 has no
    quota endpoint).
- **Registry & exports:** registered in `SEARCH_PROVIDER_MAP` and
  `_SEARCH_PROVIDER_ENV_DEFAULTS` (`semanticscholar`, aliases `semantic_scholar`
  / `semantic-scholar` / `s2` → `SEMANTIC_SCHOLAR_API_KEY`); exported from
  `llmcore.search` and the top-level `llmcore` package.
- **Configuration:** a `[search_providers.semanticscholar]` block added to
  `default_config.toml` **commented out** — because the provider loads keyless,
  an uncommented block would auto-load and break the "search is empty unless
  configured" invariant; it is a one-line opt-in (no key required). Keys:
  `api_key_env_var`, `base_url`, `default_search_type`, `default_fields`,
  `timeout`, `max_retries`, `max_concurrency`, `min_request_interval`,
  `ssl_verify`.
- **confy-curator schema** (`tools/llmcore.confy-schema.json`): added a
  "Search Provider: Semantic Scholar" section (order 49) for the wizard.
- **Packaging:** new `semanticscholar` extra (`pip install
  "llmcore[semanticscholar]"`); added to the `all` extra.
- **Tests:** `tests/search/test_semanticscholar_provider.py` — 61 `respx`-based
  unit tests (no network) covering keyless vs keyed auth (header presence), the
  four search flavors and per-endpoint clamps, filter pass-through &
  `openAccessPdf` flag, snippet/citation/reference/author normalization, POST
  batch & recommendations bodies, the Datasets helpers, retry-on-429/5xx +
  transport retry, 401/403 raises, the health check, client-side batch fan-out,
  and manager wiring (keyless + `s2` alias + keyed). Full search suite: 199
  passed, 0 regressions.
- **Docs:** `docs/Search_providers_usage.md` (new §11 Semantic Scholar) and
  the search providers design rationale (glance row, capability matrix, config
  reference, tradeoffs) updated; new `examples/semanticscholar_search_example.py`.

> `cardctl` is intentionally **not** extended for Semantic Scholar — it manages
> *LLM model cards* (token pricing/context), which do not apply to a free,
> per-request academic API. See `tools/cardctl/BRIGHTDATA_SKIP_RATIONALE.md`. No
> public APIs, schemas, or existing provider behavior changed; the addition is
> fully backward-compatible.

## v0.49.12

### Added — SerpApi search provider (`llmcore.search`)

Adds **SerpApi** (https://serpapi.com) as a first-class **search provider**,
joining Bright Data and Serper.dev in the optional `llmcore.search` subsystem.
SerpApi is a real-time *meta-SERP* API: a single endpoint scrapes 100+ search
engines/verticals selected with one `engine` parameter.

- **New provider `SerpApiSearchProvider`** (`llmcore/search/providers/serpapi_provider.py`)
  — native `httpx` client (no vendor SDK), advertising `web_search` and
  `batch_search`. Highlights:
  - **100+ engines** via a free-form `engine` argument (`google`, `bing`,
    `baidu`, `duckduckgo`, `yahoo`, `yandex`, `google_news`, `google_images`,
    `google_shopping`, `google_scholar`, `youtube`, `amazon`, `ebay`, `walmart`,
    `google_maps`, …). Engine is **not** an enum (SerpApi adds engines often); a
    `KNOWN_ENGINES` set drives only a soft debug warning, never rejection.
  - **Engine-aware mapping & normalization:** `query` → the engine's query field
    (`q`/`query`/`p`/`text`/`search_query`/`term`/`_nkw`/`find_desc`), `count` →
    `num` (best-effort), `country` → `gl`, `language` → `hl`, `device` →
    `device`; the primary result array is resolved per engine
    (`organic_results`/`news_results`/`images_results`/`video_results`/
    `shopping_results`/`local_results`/…). The full payload is preserved on
    `WebSearchResult.raw`.
  - **Auth via `api_key` query parameter** (not a header). Key resolved from
    `api_key`/`token`, `api_key_env_var`, or `SERPAPI_API_KEY` (with `SERPAPI_KEY`
    / `SERP_API_KEY` SDK-convention fallbacks); always redacted from logs.
  - **Async mode** (`mode="async"`): submits with `async=true`, then polls the
    Search Archive (`GET /searches/{id}`) until `Success`/`Error`
    (`no_cache` is dropped automatically as it is incompatible with async).
  - **Client-side `batch_search`:** SerpApi has no server-side batch endpoint, so
    queries are run concurrently bounded by `max_concurrency` (one credit each),
    returning one ordered `WebSearchResult` per input.
  - **Provider-specific helpers:** `search()` (raw `/search` pass-through),
    `search_archive(id)`, `account()` and `locations()`.
  - **Free `health_check()`** via the Account API (`/account.json`) — consumes
    **zero** search credits (unlike Serper).
  - Passes through every other SerpApi request parameter verbatim (`location`,
    `uule`, `lat`/`lon`, `google_domain`, `tbm`, `tbs`, `safe`, `start`,
    `no_cache`, `output`, `json_restrictor`, `zero_trace`, …).
- **Registry & exports:** registered in `SEARCH_PROVIDER_MAP` and
  `_SEARCH_PROVIDER_ENV_DEFAULTS` (`serpapi`, aliases `serp_api` /
  `serpapi_search` → `SERPAPI_API_KEY`); exported from `llmcore.search` and the
  top-level `llmcore` package.
- **Configuration:** new commented `[search_providers.serpapi]` block in
  `default_config.toml` (`api_key_env_var`, `base_url`, `default_engine`,
  `default_output`, `no_cache`, `zero_trace`, `json_restrictor`, `timeout`,
  `max_retries`, `poll_interval`, `poll_timeout`, `max_concurrency`,
  `ssl_verify`).
- **confy-curator schema** (`tools/llmcore.confy-schema.json`): added a
  "Search Provider: SerpApi" section (order 48) for the configuration wizard.
- **Packaging:** new `serpapi` extra (`pip install "llmcore[serpapi]"`); added to
  the `all` extra.
- **Tests:** `tests/search/test_serpapi_provider.py` — 47 `respx`-based unit
  tests (no network) covering param mapping/auth, engine-specific query fields,
  vertical normalization, async submit+poll (incl. timeout/error), client-side
  batch fan-out, archive/account/locations, retries, the free health check, and
  manager wiring.
- **Docs:** `docs/Search_providers_usage.md` and
  the search providers design rationale updated with a SerpApi section, capability
  matrix row and config reference; new `examples/serpapi_search_example.py`.

> `cardctl` is intentionally **not** extended for SerpApi — it manages *LLM model
> cards* (token pricing/context), which do not apply to a per-credit SERP API.
> See `tools/cardctl/BRIGHTDATA_SKIP_RATIONALE.md`. No public APIs, schemas, or
> existing provider behavior changed; the addition is fully backward-compatible.

## v0.49.11

### Added — Web/Data Search Providers (`llmcore.search`)

A new, **optional** subsystem that adds web/data **search** providers alongside
the existing LLM providers, usable "just like" LLM providers (config‑driven,
discovered through a manager, accessed via a uniform interface). The first
provider is **Bright Data**.

- **New package `llmcore.search`:**
  - `BaseSearchProvider` (ABC) + `SearchCapability` enum — the search‑side
    analogue of `BaseProvider`. Optional capability methods default to
    `NotImplementedError` (same idiom as the LLM provider's optional modalities).
  - `SearchProviderManager` — mirrors `ProviderManager`, but **optional**: loads
    zero or more providers from `[search_providers]` and never fails when the
    section is absent. Auto‑adopts a lone provider as the default.
  - Provider‑agnostic result models: `WebSearchResult`/`SearchItem`,
    `ScrapeResult`, `DiscoverResult`/`DiscoverItem`,
    `DatasetInfo`/`DatasetField`/`DatasetMetadata`/`DatasetSnapshot`
    (all with `to_dict()`/`to_json()`/`elapsed_ms()`).
  - `BrightDataSearchProvider` — native `httpx` client (no vendor SDK). Supports
    SERP web search (sync + async), Web Unlocker scraping, the Discover API
    (AI‑ranked), and the Dataset Marketplace (filter → snapshot → download), plus
    a connectivity `health_check()`.
- **`LLMCore` API:** `web_search()`, `scrape_url()`, `discover()`,
  `list_datasets()`, `get_dataset_metadata()`, `dataset_search()`,
  `get_search_provider()`, `get_available_search_providers()`. A
  `_search_provider_manager` is initialized after the LLM `ProviderManager`,
  closed in `close()`, and rebuilt on `reload_config()`;
  `set_raw_payload_logging()` now also propagates to search providers.
- **New exception:** `SearchProviderError` (search‑side analogue of
  `ProviderError`, with optional `status_code`).
- **Configuration:** new `llmcore.default_search_provider` key and a
  `[search_providers.brightdata]` section in `default_config.toml`
  (token via `BRIGHTDATA_API_TOKEN`; `serp_zone` / `unlocker_zone`;
  `default_engine`, `timeout`, `poll_interval`, `poll_timeout`, `max_retries`,
  `ssl_verify`). Zones are **not** auto‑created.
- **confy‑curator schema** (`tools/llmcore.confy-schema.json`): added a
  "Search Provider: Bright Data" section and a "Default Search Provider" field
  to Core Settings (validated against confy‑curator's `SchemaModel`).
- **Packaging:** new optional extra `brightdata = ["httpx>=0.27.0"]`, also
  included in the `all` extra.
- **Docs & examples:** `docs/search/README.md`, `docs/search/USAGE.md`,
  `examples/brightdata_search_example.py`.
- **Tests:** `tests/search/` (62 tests) — `respx`‑based provider tests asserting
  exact endpoints/payloads/headers, model tests, manager tests, and an
  `LLMCore`‑level wiring test.

### Notes / non‑goals

- **`cardctl` intentionally not extended.** Bright Data has no token‑priced
  "models"; adding it to the model‑card registry would be a category error. See
  `tools/cardctl/BRIGHTDATA_SKIP_RATIONALE.md`.
- **Backward compatible.** The subsystem is additive and optional; deployments
  that do not configure `[search_providers]` are unaffected. The search methods
  raise a clear `ConfigError` only if called with no provider configured.


