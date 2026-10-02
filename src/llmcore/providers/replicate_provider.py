# src/llmcore/providers/replicate_provider.py
"""Replicate media provider for the LLMCore media subsystem.

Replicate hosts tens of thousands of community models behind **one** prediction
API. The media spec is explicit that this must be *one generic adapter plus
model-schema descriptors*, **not a class per model** — a catalog that large
cannot be enumerated in a library, and any hardcoded field mapping would be
stale within a release.

How a generic adapter is possible
----------------------------------

Every Replicate model publishes an OpenAPI schema describing its own inputs and
outputs. ``black-forest-labs/flux-schnell`` requires ``prompt`` and returns an
array of URIs; ``openai/whisper`` requires ``audio`` and returns an object with
a ``transcription`` field. So rather than guessing that "the prompt field is
called prompt", this adapter **reads the schema** and maps llmcore's canonical
protocol arguments onto whatever that model actually calls them.

That is the whole design. It means a model llmcore has never heard of works, and
a model that renames ``image`` to ``input_image`` next week keeps working.

Lifecycle
---------

``POST /v1/predictions`` (or ``/v1/models/{owner}/{name}/predictions`` for
official models) returns ``starting`` and a set of URLs. llmcore uses **the URLs
Replicate hands back** rather than rebuilding routes it does not own — the same
lesson fal taught in M5. Webhooks are supported natively, so the adapter opts
into llmcore's receiver and takes a per-job callback URL.

Transport (selectable via ``backend``)
--------------------------------------

* ``"httpx"`` — direct REST against ``api.replicate.com``. **The default.**
* ``"sdk"`` — the official ``replicate`` package.

References:
  - https://replicate.com/docs/reference/http
  - Vendor SDK: replicate-python (v1.0.7)
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

try:
    import httpx

    httpx_available = True
except ImportError:  # pragma: no cover
    httpx_available = False
    httpx = None  # type: ignore

try:
    import replicate as replicate_sdk

    replicate_sdk_available = True
except ImportError:
    replicate_sdk_available = False
    replicate_sdk = None  # type: ignore

from ..exceptions import ConfigError, ProviderError
from ..models import Message, ModelDetails, Tool
from .base import BaseProvider, ContextPayload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..media.models import MediaCapability, MediaExecution, MediaJob, MediaRef

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.replicate.com"
_API_KEY_ENV_VARS: tuple[str, ...] = ("REPLICATE_API_TOKEN", "REPLICATE_API_KEY")

#: Conservative per-capability defaults. Deliberately small — Replicate's
#: catalog is enormous and moves constantly, so these are a starting point, not
#: a catalog. Override in ``[providers.replicate.models]`` or per call.
_DEFAULT_MODELS: dict[str, str] = {
    "image_generate": "black-forest-labs/flux-schnell",
    "image_edit": "black-forest-labs/flux-kontext-pro",
    "image_upscale": "nightmareai/real-esrgan",
    "video_generate": "minimax/video-01",
    "asr": "openai/whisper",
    "tts": "jaaari/kokoro-82m",
    "music": "meta/musicgen",
}

#: Replicate prediction states → llmcore job statuses.
_STATUS_MAP: dict[str, str] = {
    "starting": "queued",
    "processing": "running",
    "succeeded": "succeeded",
    "failed": "failed",
    "canceled": "canceled",
}

#: Canonical llmcore argument → the field names models actually use for it,
#: in preference order. The model's own schema decides which one exists; this
#: is only the candidate list, so an unfamiliar model still resolves.
_INPUT_ALIASES: dict[str, tuple[str, ...]] = {
    "prompt": ("prompt", "text", "description", "text_prompt", "input_text"),
    "negative_prompt": ("negative_prompt", "neg_prompt"),
    "image": ("image", "input_image", "image_url", "img", "image_path", "first_frame_image"),
    "audio": ("audio", "audio_file", "audio_url", "audio_path", "input_audio"),
    "video": ("video", "video_url", "input_video", "video_path"),
    "duration_seconds": ("duration", "duration_seconds", "length", "seconds"),
    "seed": ("seed",),
    "width": ("width",),
    "height": ("height",),
    "n": ("num_outputs", "num_images", "n"),
    "scale": ("scale", "upscale_factor", "scale_factor"),
    "voice": ("voice", "speaker", "voice_id"),
    "language": ("language", "language_code"),
}

#: Output keys that carry text rather than a file reference.
_TEXT_KEYS: tuple[str, ...] = ("transcription", "text", "output_text", "caption", "transcript")


class ReplicateProvider(BaseProvider):
    """Replicate media provider — one adapter for the whole catalog.

    Media-only by design: Replicate serves model predictions, not chat
    completions, so :meth:`chat_completion` raises rather than pretending.

    Configuration keys (under ``[providers.replicate]``):

    - ``api_key`` / ``api_key_env_var`` — falls back to ``REPLICATE_API_TOKEN``.
    - ``backend`` — ``"httpx"`` (default) or ``"sdk"``.
    - ``base_url`` — override the REST root.
    - ``timeout`` — HTTP timeout in seconds (default 300).
    - ``models`` — per-capability model overrides (``owner/name`` or
      ``owner/name:version``).
    - ``use_schema`` — resolve input field names from each model's published
      schema (default ``True``). Disable only to force literal field names.
    """

    _MEDIA_CAPABILITIES: frozenset[str] = frozenset(_DEFAULT_MODELS)

    #: Replicate takes a `webhook` field per prediction, so the router may
    #: offer this adapter a per-job callback URL.
    accepts_webhook_url: bool = True

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """Initialize the Replicate provider.

        Raises:
            ConfigError: If no transport is installed or no API token is found.
        """
        super().__init__(config, log_raw_payloads)

        if not (httpx_available or replicate_sdk_available):
            raise ConfigError(
                "The Replicate provider requires 'httpx' (direct REST, preferred) or the "
                "'replicate' SDK. Install with: pip install llmcore[replicate]"
            )

        api_key = config.get("api_key")
        if not api_key:
            env_var = config.get("api_key_env_var")
            candidates = (env_var, *_API_KEY_ENV_VARS) if env_var else _API_KEY_ENV_VARS
            for name in candidates:
                if name and os.environ.get(name):
                    api_key = os.environ[name]
                    break
        if not api_key:
            raise ConfigError(
                "Replicate API token not found. Set REPLICATE_API_TOKEN or configure "
                "providers.replicate.api_key / api_key_env_var. Create one at "
                "https://replicate.com/account/api-tokens."
            )
        self._api_key = str(api_key)

        self._base_url = str(config.get("base_url", _BASE_URL)).rstrip("/")
        self._timeout = float(config.get("timeout", 300))
        self._use_schema = bool(config.get("use_schema", True))
        self._models: dict[str, str] = {
            **_DEFAULT_MODELS,
            **{str(k): str(v) for k, v in (config.get("models") or {}).items()},
        }
        self.default_model = config.get("default_model") or self._models["image_generate"]

        self._backend = self._resolve_backend(config.get("backend"))
        self._http: Any = None
        self._sdk: Any = None
        if self._backend == "sdk":
            self._sdk = replicate_sdk.Client(api_token=self._api_key)

        #: model reference → its published Input schema properties.
        self._schema_cache: dict[str, dict[str, Any]] = {}
        #: model reference → its latest version id, learned from the same
        #: lookup. Community models can only be run by version.
        self._version_cache: dict[str, str] = {}

        logger.debug(
            "Replicate provider initialized (backend=%s, schema=%s).",
            self._backend,
            self._use_schema,
        )

    @staticmethod
    def _resolve_backend(requested: str | None) -> str:
        """Resolve the transport, preferring direct REST."""
        available = {"httpx": httpx_available, "sdk": replicate_sdk_available}
        req = (requested or "auto").lower()
        if req not in ("auto", "httpx", "sdk"):
            logger.warning("Unknown Replicate backend '%s'; using auto-detection.", req)
            req = "auto"
        if req != "auto" and available.get(req):
            return req
        if req != "auto":
            logger.warning("Requested Replicate backend '%s' is unavailable; falling back.", req)
        for backend in ("httpx", "sdk"):
            if available[backend]:
                return backend
        raise ConfigError("No usable Replicate transport is installed.")

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _get_http(self) -> Any:
        """Return the lazily-built HTTP client."""
        if not httpx_available:
            raise ProviderError(self.get_name(), "The 'httpx' package is required for Replicate.")
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
        return self._http

    def _raise_status(self, status: int, body: str, context: str) -> None:
        """Map a Replicate HTTP error onto a :class:`ProviderError`."""
        logger.error("Replicate error %s on %s: %s", status, context, body)
        if status in (401, 403):
            raise ProviderError(
                self.get_name(),
                f"Replicate authentication failed. Check REPLICATE_API_TOKEN. Error: {body}",
                model_name=context,
                status_code=status,
            )
        if status == 402:
            raise ProviderError(
                self.get_name(),
                f"Replicate rejected '{context}' for billing reasons (the token is valid). "
                f"Check the account's spend limit or payment method. Error: {body}",
                model_name=context,
                status_code=status,
                retryable=False,
            )
        if status == 404:
            raise ProviderError(
                self.get_name(),
                f"Replicate model or prediction '{context}' not found. Model references are "
                f"'owner/name' or 'owner/name:version'. Error: {body}",
                model_name=context,
                status_code=status,
            )
        if status == 422:
            raise ProviderError(
                self.get_name(),
                f"Replicate rejected the input for '{context}'. The model's schema may use "
                f"different field names; pass them explicitly as keyword arguments. "
                f"Error: {body}",
                model_name=context,
                status_code=status,
            )
        if status == 429:
            # Replicate reports a precise retry_after, and tightens the limit
            # to a burst of 1 while an account holds under $5 of credit — worth
            # saying out loud, because it looks like a bug until you know.
            raise ProviderError(
                self.get_name(),
                f"Replicate rate limit reached. Note that accounts under $5 of credit are "
                f"throttled to ~6 prediction creations per minute with a burst of 1. "
                f"Error: {body}",
                model_name=context,
                status_code=status,
                retryable=True,
            )
        raise ProviderError(
            self.get_name(),
            f"Replicate API error ({status}): {body}",
            model_name=context,
            status_code=status,
            retryable=status >= 500,
        )

    # ------------------------------------------------------------------
    # BaseProvider surface (Replicate is media-only)
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        """Return the configured instance name (default: ``"replicate"``)."""
        return self._provider_instance_name or "replicate"

    async def chat_completion(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Not supported: this adapter targets Replicate's media models.

        Raises:
            ProviderError: Always, with a pointer to the media surface.
        """
        raise ProviderError(
            self.get_name(),
            "The Replicate adapter serves generative-media predictions and has no "
            "chat-completions API. Use llm.media instead.",
            status_code=400,
            retryable=False,
        )

    async def get_models_details(self) -> list[ModelDetails]:
        """Describe the configured per-capability models.

        Replicate's catalog has tens of thousands of models and no meaningful
        way to enumerate them for this purpose, so this reports what *this
        instance* is configured to call rather than pretending to list them.
        """
        seen: dict[str, str] = {}
        for capability, model in self._models.items():
            seen.setdefault(model, capability)
        return [
            ModelDetails(
                id=model,
                provider_name=self.get_name(),
                context_length=0,
                supports_streaming=False,
                supports_tools=False,
                model_type="media",
                metadata={"capability": capability, "prediction": True},
            )
            for model, capability in sorted(seen.items())
        ]

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """Parameters are per model and published in each model's own schema."""
        return {}

    def get_max_context_length(self, model: str | None = None) -> int:
        """Replicate media models have no token context window."""
        return 0

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Tokens are not a meaningful unit for Replicate media predictions."""
        return 0

    async def count_message_tokens(
        self, messages: list[Message], model: str | None = None
    ) -> int:
        """Tokens are not a meaningful unit for Replicate media predictions."""
        return 0

    def extract_response_content(self, response: dict[str, Any]) -> str:
        """Return any text a prediction produced."""
        output = response.get("output")
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            for key in _TEXT_KEYS:
                if isinstance(output.get(key), str):
                    return str(output[key])
        return ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        """Replicate predictions are not token streams here."""
        return ""

    # ------------------------------------------------------------------
    # Media capability declaration
    # ------------------------------------------------------------------

    def media_capabilities(self) -> "frozenset[MediaCapability]":
        """Capabilities this instance is configured to serve."""
        from ..media.models import MediaCapability

        return frozenset(MediaCapability(c) for c in self._models)

    def media_execution(
        self, capability: MediaCapability, model: str | None = None
    ) -> MediaExecution:
        """Every Replicate capability is a prediction, so always a job."""
        from ..media.models import MediaExecution

        return MediaExecution.ASYNC_JOB

    def _model_for(self, capability: str, model: str | None) -> str:
        """Resolve the model reference for *capability*."""
        return model or self._models.get(capability) or self.default_model

    # ------------------------------------------------------------------
    # Model schema descriptors
    # ------------------------------------------------------------------

    async def get_input_schema(self, model: str) -> dict[str, Any]:
        """Return the published Input properties for *model*, cached.

        A failed lookup returns ``{}`` rather than raising: the schema is an
        optimization for field naming, and losing it should degrade the mapping
        to its default guesses rather than fail a submission outright.
        """
        reference = model.split(":")[0]
        if reference in self._schema_cache:
            return self._schema_cache[reference]

        try:
            resp = await self._get_http().get(f"/v1/models/{reference}")
            if resp.status_code >= 400:
                raise ProviderError(
                    self.get_name(), f"model lookup failed ({resp.status_code})"
                )
            version = resp.json().get("latest_version") or {}
            if version.get("id"):
                self._version_cache[reference] = str(version["id"])
            schemas = (version.get("openapi_schema") or {}).get("components", {}).get(
                "schemas", {}
            )
            properties = dict((schemas.get("Input") or {}).get("properties") or {})
        except Exception as e:  # schema is an optimization, never a gate
            logger.warning("Replicate schema lookup failed for %s: %s", reference, e)
            properties = {}

        self._schema_cache[reference] = properties
        return properties

    async def _map_inputs(
        self, model: str, canonical: dict[str, Any], extra: dict[str, Any]
    ) -> dict[str, Any]:
        """Map canonical protocol arguments onto *model*'s real field names.

        This is what makes one adapter serve the whole catalog. For each
        canonical argument the model's schema is consulted for the first alias
        it actually declares; with no schema, the first alias is used as a
        best guess. Explicit keyword arguments always win, because the caller
        knows their model better than this mapping does.
        """
        schema = await self.get_input_schema(model) if self._use_schema else {}
        mapped: dict[str, Any] = {}

        for name, value in canonical.items():
            if value is None:
                continue
            aliases = _INPUT_ALIASES.get(name, (name,))
            field = next((a for a in aliases if a in schema), None) if schema else None
            if field is None:
                # No schema, or the model declares none of the aliases. Fall
                # back to the canonical spelling and let the API say no; a 422
                # naming the field is far more useful than a silent drop.
                field = aliases[0]
            mapped[field] = value

        mapped.update({k: v for k, v in extra.items() if v is not None})
        return mapped

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    async def _submit_prediction(
        self, model: str, inputs: dict[str, Any], webhook_url: str | None
    ) -> dict[str, Any]:
        """Create a prediction for *model*."""
        body: dict[str, Any] = {"input": inputs}
        if webhook_url:
            body["webhook"] = webhook_url
            body["webhook_events_filter"] = ["completed"]

        owner_name = model
        path = f"/v1/models/{owner_name}/predictions"
        if ":" in model:
            owner_name, version = model.split(":", 1)
            body["version"] = version
            path = "/v1/predictions"
        elif self._use_schema:
            # Replicate has two creation routes and the reference does not say
            # which one a model answers on: *official* models run unversioned
            # at /v1/models/{owner}/{name}/predictions, while *community*
            # models 404 there and must be run by version at /v1/predictions.
            #
            # Discovering that by trying the first and falling back on a 404
            # costs two creation requests, which trips Replicate's burst limit
            # (1 request while an account holds under $5 of credit) and turns a
            # working call into a 429. So resolve the version from the model
            # lookup we already make for the input schema — a GET that does not
            # count against prediction-creation limits — and use the one route
            # that works for both kinds.
            version = await self._latest_version(owner_name)
            if version:
                body["version"] = version
                path = "/v1/predictions"

        if self._backend == "sdk":
            prediction = await self._sdk.predictions.async_create(
                model=owner_name, input=inputs, webhook=webhook_url or None
            )
            dump = getattr(prediction, "dict", None) or getattr(prediction, "model_dump", None)
            return dict(dump()) if callable(dump) else dict(prediction)

        try:
            resp = await self._get_http().post(path, json=body)
        except httpx.HTTPError as e:
            raise ProviderError(
                self.get_name(), f"Replicate transport error: {e}", model_name=model,
                retryable=True,
            ) from e

        # Last resort: a model that publishes no version, run unversioned and
        # rejected. One extra attempt is worth it here because the alternative
        # is failing a call that would have worked.
        if resp.status_code == 404 and "version" not in body:
            version = await self._latest_version(owner_name)
            if version:
                body["version"] = version
                resp = await self._get_http().post("/v1/predictions", json=body)

        if resp.status_code >= 400:
            self._raise_status(resp.status_code, resp.text, model)
        return dict(resp.json())

    async def _latest_version(self, reference: str) -> str | None:
        """Return the newest version id for *reference*, if it publishes one."""
        if reference not in self._version_cache:
            await self.get_input_schema(reference)
        return self._version_cache.get(reference)

    async def _submit_job(
        self,
        capability: str,
        model: str | None,
        canonical: dict[str, Any],
        extra: dict[str, Any],
        webhook_url: str | None = None,
    ) -> MediaJob:
        """Submit a prediction and wrap it in a tracked :class:`MediaJob`."""
        from ..media.models import MediaCapability, MediaJob, MediaJobStatus

        reference = self._model_for(capability, model)
        webhook_url = extra.pop("webhook_url", None) or webhook_url
        inputs = await self._map_inputs(reference, canonical, extra)
        prediction = await self._submit_prediction(reference, inputs, webhook_url)

        urls = prediction.get("urls") or {}
        job = MediaJob(
            capability=MediaCapability(capability),
            provider=self.get_name(),
            model=reference,
            status=self._status_of(prediction),
            provider_job_id=prediction.get("id"),
            poll_url=urls.get("get"),
            provider_metadata={
                "prediction": prediction,
                "cancel_url": urls.get("cancel"),
                "inputs": inputs,
            },
        )
        if job.status is MediaJobStatus.SUCCEEDED:
            return self._apply_prediction(job, prediction)
        return job

    @staticmethod
    def _status_of(prediction: dict[str, Any]) -> Any:
        """Map a prediction's state onto a job status."""
        from ..media.models import MediaJobStatus

        mapped = _STATUS_MAP.get(str(prediction.get("status") or "").lower(), "running")
        return MediaJobStatus(mapped)

    async def poll_media_job(self, job: MediaJob) -> MediaJob:
        """Refresh *job* against the predictions API."""
        from ..media.models import MediaJobStatus

        if job.is_terminal:
            return job
        prediction_id = job.provider_job_id
        if not prediction_id:
            job.status = MediaJobStatus.FAILED
            job.error = "Lost the Replicate prediction id; the job cannot be polled."
            return job

        # Prefer the URL Replicate handed back over one rebuilt here.
        url = job.poll_url or f"/v1/predictions/{prediction_id}"
        if self._backend == "sdk":
            prediction = await self._sdk.predictions.async_get(prediction_id)
            dump = getattr(prediction, "dict", None) or getattr(prediction, "model_dump", None)
            prediction = dict(dump()) if callable(dump) else dict(prediction)
        else:
            resp = await self._get_http().get(url)
            if resp.status_code >= 400:
                self._raise_status(resp.status_code, resp.text, prediction_id)
            prediction = dict(resp.json())

        return self._apply_prediction(job, prediction)

    async def cancel_media_job(self, job: MediaJob) -> MediaJob:
        """Cancel *job*. Replicate stops the prediction and stops billing it."""
        from ..media.models import MediaJobStatus

        if job.is_terminal:
            return job
        prediction_id = job.provider_job_id
        if not prediction_id:
            return job

        if self._backend == "sdk":
            await self._sdk.predictions.async_cancel(prediction_id)
            job.status = MediaJobStatus.CANCELED
            job.touch()
            return job

        url = job.provider_metadata.get("cancel_url") or (
            f"/v1/predictions/{prediction_id}/cancel"
        )
        resp = await self._get_http().post(url)
        if resp.status_code >= 400:
            self._raise_status(resp.status_code, resp.text, prediction_id)
        job.status = MediaJobStatus.CANCELED
        job.touch()
        return job

    async def apply_webhook_payload(
        self, job: MediaJob, payload: dict[str, Any]
    ) -> MediaJob:
        """Fold a Replicate webhook delivery into *job*.

        Replicate posts the whole prediction object, so the delivery is the same
        shape a poll returns. A payload whose ``id`` does not match is not
        trusted — anyone who learns a callback URL can POST to it, and a wrong
        artifact is worse than a slow one.
        """
        payload_id = payload.get("id")
        if payload_id and job.provider_job_id and payload_id != job.provider_job_id:
            logger.warning(
                "Replicate webhook id %s does not match job %s; polling instead.",
                payload_id,
                job.id,
            )
            return await self.poll_media_job(job)
        if not payload.get("status"):
            return await self.poll_media_job(job)
        return self._apply_prediction(job, payload)

    def _apply_prediction(self, job: MediaJob, prediction: dict[str, Any]) -> MediaJob:
        """Fold a prediction object into *job*."""
        from ..media.models import MediaJobStatus, MediaUsage

        job.status = self._status_of(prediction)
        job.provider_metadata["prediction"] = prediction
        job.touch()

        if prediction.get("error"):
            job.status = MediaJobStatus.FAILED
            job.error = str(prediction["error"])
            return job
        if job.status is not MediaJobStatus.SUCCEEDED:
            return job

        job.artifacts = self._artifacts_from_output(
            prediction.get("output"),
            job.capability,
            (prediction.get("input") or {}).get("output_format"),
        )
        job.progress = 1.0
        metrics = prediction.get("metrics") or {}
        job.usage = MediaUsage(
            provider=self.get_name(),
            model=job.model,
            basis="per_request",
            compute_seconds=metrics.get("predict_time"),
            raw=metrics,
        )
        return job

    def _artifacts_from_output(
        self, output: Any, capability: Any, requested_format: str | None = None
    ) -> list[Any]:
        """Turn a prediction's output into artifacts.

        Replicate model outputs are schema-defined and therefore varied: a URI
        string, a list of URI strings, or an object mixing files and text.
        Rather than special-casing models, this walks the shapes the schemas
        actually produce. An unrecognised shape yields no artifacts instead of
        raising — the untouched payload is always kept on the job.
        """
        from ..media.models import MediaArtifact, MediaKind

        kind = self._kind_for(capability)
        artifacts: list[MediaArtifact] = []

        def _add_uri(value: str, forced: Any = None) -> None:
            artifacts.append(
                MediaArtifact(
                    kind=forced or kind,
                    uri=value,
                    mime_type=self._mime_for(value, requested_format),
                )
            )

        if isinstance(output, str):
            if output.startswith(("http://", "https://", "data:")):
                _add_uri(output)
            else:
                artifacts.append(
                    MediaArtifact(kind=MediaKind.TEXT, text=output, mime_type="text/plain")
                )
        elif isinstance(output, list):
            for item in output:
                if isinstance(item, str) and item.startswith(("http://", "https://", "data:")):
                    _add_uri(item)
                elif isinstance(item, dict):
                    artifacts.extend(
                        self._artifacts_from_output(item, capability, requested_format)
                    )
        elif isinstance(output, dict):
            for key in _TEXT_KEYS:
                value = output.get(key)
                if isinstance(value, str) and value:
                    artifacts.append(
                        MediaArtifact(
                            kind=MediaKind.TEXT,
                            text=value,
                            mime_type="text/plain",
                            provider_metadata={"field": key},
                        )
                    )
                    break
            for key, value in output.items():
                if key in _TEXT_KEYS:
                    continue
                if isinstance(value, str) and value.startswith(("http://", "https://")):
                    artifacts.append(
                        MediaArtifact(
                            kind=kind,
                            uri=value,
                            mime_type=self._mime_for(value, requested_format),
                            provider_metadata={"field": key},
                        )
                    )
        return artifacts

    @staticmethod
    def _kind_for(capability: Any) -> Any:
        """Infer the artifact kind a capability produces."""
        from ..media.models import MediaCapability, MediaKind

        value = getattr(capability, "value", str(capability))
        if value.startswith("image"):
            return MediaKind.IMAGE
        if value.startswith("video"):
            return MediaKind.VIDEO
        if value == "asr":
            return MediaKind.TEXT
        if value in {"tts", "music", "sfx"}:
            return MediaKind.AUDIO
        return MediaKind.IMAGE if capability is MediaCapability.OCR else MediaKind.AUDIO

    @staticmethod
    def _mime_for(uri: str, requested_format: str | None = None) -> str | None:
        """Infer a MIME type for *uri*.

        Replicate's output URLs often carry no file extension. Rather than
        guessing a format, fall back to the ``output_format`` we *asked* for —
        which is grounded in the request rather than invented. Still ``None``
        when neither is available: an honest unknown beats a plausible lie,
        since callers key decode paths off this.
        """
        import mimetypes

        guessed, _ = mimetypes.guess_type(uri.split("?")[0])
        if guessed:
            return guessed
        if requested_format:
            guessed, _ = mimetypes.guess_type(f"x.{str(requested_format).lstrip('.')}")
            return guessed
        return None

    async def _ref_to_input(self, ref: MediaRef | None) -> str | None:
        """Render *ref* as something Replicate can read.

        Remote refs pass straight through; local bytes become a data URI, which
        Replicate accepts for file inputs and which avoids standing up an upload
        path for what are usually small files.
        """
        if ref is None:
            return None
        if ref.is_remote:
            return ref.url
        import base64

        data = ref.read_bytes()
        mime = ref.mime_type or "application/octet-stream"
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"

    # ------------------------------------------------------------------
    # Capability methods
    # ------------------------------------------------------------------

    async def generate_image_media(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int | None = None,
        size: str | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> MediaJob:
        """Generate images from *prompt*."""
        width = height = None
        if size and "x" in size:
            try:
                width, height = (int(p) for p in size.split("x", 1))
            except ValueError:
                width = height = None
        return await self._submit_job(
            "image_generate",
            model,
            {"prompt": prompt, "n": n, "seed": seed, "width": width, "height": height},
            kwargs,
        )

    async def edit_image_media(
        self,
        prompt: str,
        *,
        image: MediaRef | None = None,
        mask: MediaRef | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> MediaJob:
        """Edit *image* according to *prompt*."""
        return await self._submit_job(
            "image_edit",
            model,
            {"prompt": prompt, "image": await self._ref_to_input(image)},
            kwargs,
        )

    async def upscale_image_media(
        self,
        *,
        image: MediaRef,
        model: str | None = None,
        scale: float | None = None,
        **kwargs: Any,
    ) -> MediaJob:
        """Upscale *image*."""
        return await self._submit_job(
            "image_upscale",
            model,
            {"image": await self._ref_to_input(image), "scale": scale},
            kwargs,
        )

    async def generate_video_media(
        self,
        prompt: str,
        *,
        model: str | None = None,
        image: MediaRef | None = None,
        duration_seconds: float | None = None,
        **kwargs: Any,
    ) -> MediaJob:
        """Generate video from *prompt*, optionally conditioned on an image."""
        return await self._submit_job(
            "video_generate",
            model,
            {
                "prompt": prompt,
                "image": await self._ref_to_input(image),
                "duration_seconds": duration_seconds,
            },
            kwargs,
        )

    async def transcribe_media(
        self,
        *,
        audio: MediaRef,
        model: str | None = None,
        language: str | None = None,
        diarize: bool | None = None,
        timestamps: bool | None = None,
        **kwargs: Any,
    ) -> MediaJob:
        """Transcribe *audio*."""
        return await self._submit_job(
            "asr",
            model,
            {"audio": await self._ref_to_input(audio), "language": language},
            kwargs,
        )

    async def synthesize_speech_media(
        self,
        text: str,
        *,
        model: str | None = None,
        voice: str | None = None,
        audio_format: str | None = None,
        sample_rate_hz: int | None = None,
        speed: float | None = None,
        **kwargs: Any,
    ) -> MediaJob:
        """Synthesize *text* into speech."""
        return await self._submit_job(
            "tts", model, {"prompt": text, "voice": voice}, kwargs
        )

    async def generate_music_media(
        self,
        prompt: str,
        *,
        model: str | None = None,
        duration_seconds: float | None = None,
        **kwargs: Any,
    ) -> MediaJob:
        """Generate music from *prompt*."""
        return await self._submit_job(
            "music", model, {"prompt": prompt, "duration_seconds": duration_seconds}, kwargs
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release the HTTP client."""
        if self._http is not None:
            try:
                await self._http.aclose()
            finally:
                self._http = None
        self._sdk = None
