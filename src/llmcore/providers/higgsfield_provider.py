# src/llmcore/providers/higgsfield_provider.py
"""Higgsfield media provider for the LLMCore media subsystem.

Higgsfield serves generative image and video models (its own Soul family, plus
hosted Kling and MiniMax Hailuo video) behind one async request API. The shape
is close to fal's — the model *is* the endpoint path, every call is a queued
request, and the API hands back the URLs to track it — so this adapter follows
the fal pattern deliberately rather than inventing a second one.

What is different, and how it maps
----------------------------------

* **Credentials are a pair.** The scheme is ``Authorization: Key {id}:{secret}``,
  not a single opaque token. ``HIGGSFIELD_API_KEY`` is expected to already hold
  the joined ``id:secret`` form; ``api_key_id`` / ``api_key_secret`` can be
  configured separately when an operator keeps them apart.
* **``nsfw`` is its own terminal state**, alongside ``failed``. Both end the job,
  but one is a content-policy refusal and the other is a malfunction, and a
  caller retrying blindly on "failed" should not retry a refusal. The adapter
  maps both to ``FAILED`` while recording which it was.
* **No model catalog endpoint exists.** Model paths come from the console and
  change, so llmcore ships a small per-capability default set and makes it
  configurable rather than pretending to enumerate a catalog.
* **``403`` means out of credits, not bad auth.** A valid key on an empty
  account answers ``403 not_enough_credits``; reporting that as an
  authentication failure would send a caller after a problem they do not have.

Transport (selectable via ``backend``)
--------------------------------------

* ``"httpx"`` — direct REST against ``api.higgsfield.ai``. **The default.**
* ``"sdk"`` — the official ``higgsfield-client`` package, whose ``AsyncClient``
  covers the whole generation lifecycle (``submit`` / ``status`` / ``result`` /
  ``cancel``), so this is a genuine fallback rather than a partial one.

References:
  - https://docs.higgsfield.ai/docs
  - https://docs.higgsfield.ai/docs/openapi.json
  - SDK clone: /av/avalon/xrepos/higgsfield-client (0.1.0, aefd1ca)
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
    import higgsfield_client

    higgsfield_sdk_available = True
except ImportError:
    higgsfield_sdk_available = False
    higgsfield_client = None  # type: ignore

from ..exceptions import ConfigError, ProviderError
from ..models import Message, ModelDetails, Tool
from .base import BaseProvider, ContextPayload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..media.models import MediaCapability, MediaExecution, MediaJob, MediaRef

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.higgsfield.ai"

#: Environment variables checked, in order, for the credential pair.
_API_KEY_ENV_VARS: tuple[str, ...] = ("HIGGSFIELD_API_KEY", "HIGGSFIELD_KEY")

#: Per-capability default model paths, from the published OpenAPI document.
#: Deliberately small: Higgsfield has no catalog endpoint and the console's
#: model list moves, so these are a starting point rather than a catalog.
#: Override per capability in ``[providers.higgsfield.models]`` or per call.
_DEFAULT_MODELS: dict[str, str] = {
    "image_generate": "higgsfield-ai/soul/standard",
    "video_generate": "minimax/hailuo-2.3/standard/text-to-video",
}

#: Model paths used when a video request supplies a conditioning image.
#: Higgsfield splits text-to-video and image-to-video into separate endpoints,
#: so the adapter picks the right one rather than sending an image to a
#: text-only path and getting a 422.
_IMAGE_TO_VIDEO_MODELS: dict[str, str] = {
    "minimax/hailuo-2.3/standard/text-to-video": (
        "minimax/hailuo-2.3/standard/image-to-video"
    ),
    "kling-video/v2.5-turbo/pro/text-to-video": (
        "kling-video/v2.5-turbo/pro/image-to-video"
    ),
    "kling-video/v2.5-turbo/standard/text-to-video": (
        "kling-video/v2.5-turbo/standard/image-to-video"
    ),
}

#: Higgsfield request states → llmcore job statuses.
_STATUS_MAP: dict[str, str] = {
    "queued": "queued",
    "in_progress": "running",
    "completed": "succeeded",
    "failed": "failed",
    "canceled": "canceled",
    # A content-policy refusal, not a malfunction. Terminal either way, but the
    # distinction is preserved in provider_metadata so a caller does not retry
    # a refusal as though it were a transient error.
    "nsfw": "failed",
}

#: ``higgsfield_client`` signals state by the *class* it returns rather than by
#: a field, so the class name is the only thing to map on.
_SDK_STATUS_CLASSES: dict[str, str] = {
    "Queued": "queued",
    "InProgress": "in_progress",
    "Completed": "completed",
    "Failed": "failed",
    "NSFW": "nsfw",
    "Cancelled": "canceled",
    "Canceled": "canceled",
}

#: Result keys that carry files, mapped to the artifact kind they produce.
_RESULT_KEYS: tuple[tuple[str, str], ...] = (
    ("images", "image"),
    ("image", "image"),
    ("video", "video"),
    ("videos", "video"),
    ("audio", "audio"),
    ("audios", "audio"),
)


class HiggsfieldProvider(BaseProvider):
    """Higgsfield generative image and video provider.

    Media-only by design: Higgsfield serves generative media, not chat
    completions, so :meth:`chat_completion` raises rather than pretending.

    Configuration keys (under ``[providers.higgsfield]``):

    - ``api_key`` — the joined ``id:secret`` credential, or set
      ``HIGGSFIELD_API_KEY``.
    - ``api_key_id`` / ``api_key_secret`` — the halves, when kept apart.
    - ``backend`` — ``"httpx"`` (default) or ``"sdk"``.
    - ``base_url`` — override the REST root.
    - ``timeout`` — HTTP timeout in seconds (default 300).
    - ``webhook_url`` — static callback; a per-job one is preferred.
    - ``models`` — per-capability model-path overrides.
    """

    _MEDIA_CAPABILITIES: frozenset[str] = frozenset(_DEFAULT_MODELS)

    #: Higgsfield accepts a per-request webhook, so the router may offer one.
    accepts_webhook_url: bool = True

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """Initialize the Higgsfield provider.

        Raises:
            ConfigError: If no transport is installed or no credential is found.
        """
        super().__init__(config, log_raw_payloads)

        if not (httpx_available or higgsfield_sdk_available):
            raise ConfigError(
                "The Higgsfield provider requires 'httpx' (direct REST, preferred) or "
                "the 'higgsfield-client' SDK. Install with: pip install llmcore[higgsfield]"
            )

        api_key = config.get("api_key")
        key_id = config.get("api_key_id")
        key_secret = config.get("api_key_secret")
        if not api_key and key_id and key_secret:
            api_key = f"{key_id}:{key_secret}"
        if not api_key:
            env_var = config.get("api_key_env_var")
            candidates = (env_var, *_API_KEY_ENV_VARS) if env_var else _API_KEY_ENV_VARS
            for name in candidates:
                if name and os.environ.get(name):
                    api_key = os.environ[name]
                    break
        if not api_key:
            raise ConfigError(
                "Higgsfield API key not found. Set HIGGSFIELD_API_KEY to the "
                "'<key-id>:<key-secret>' pair, or configure "
                "providers.higgsfield.api_key (or api_key_id + api_key_secret). "
                "Create credentials at https://console.higgsfield.ai."
            )
        self._api_key = str(api_key)
        if ":" not in self._api_key:
            # Not fatal — the API decides — but this is the single most likely
            # misconfiguration, and a 401 much later is a poor way to learn it.
            logger.warning(
                "The Higgsfield credential does not contain ':'. The API expects "
                "'Key <key-id>:<key-secret>'; a single token will be rejected as "
                "invalid credentials."
            )

        self._base_url = str(config.get("base_url", _BASE_URL)).rstrip("/")
        self._timeout = float(config.get("timeout", 300))
        self._webhook_url = config.get("webhook_url") or None
        self._models: dict[str, str] = {
            **_DEFAULT_MODELS,
            **{str(k): str(v) for k, v in (config.get("models") or {}).items()},
        }
        self.default_model = config.get("default_model") or self._models["image_generate"]

        self._backend = self._resolve_backend(config.get("backend"))
        self._http: Any = None
        self._sdk: Any = None
        if self._backend == "sdk":
            self._sdk = higgsfield_client.AsyncClient(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )

        logger.debug(
            "Higgsfield provider initialized (backend=%s, base_url=%s).",
            self._backend,
            self._base_url,
        )

    @staticmethod
    def _resolve_backend(requested: str | None) -> str:
        """Resolve the transport, preferring direct REST."""
        available = {"httpx": httpx_available, "sdk": higgsfield_sdk_available}
        req = (requested or "auto").lower()
        if req not in ("auto", "httpx", "sdk"):
            logger.warning("Unknown Higgsfield backend '%s'; using auto-detection.", req)
            req = "auto"
        if req != "auto" and available.get(req):
            return req
        if req != "auto":
            logger.warning("Requested Higgsfield backend '%s' is unavailable; falling back.", req)
        for backend in ("httpx", "sdk"):
            if available[backend]:
                return backend
        raise ConfigError("No usable Higgsfield transport is installed.")

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _get_http(self) -> Any:
        """Return the lazily-built HTTP client."""
        if not httpx_available:
            raise ProviderError(
                self.get_name(), "The 'httpx' package is required for Higgsfield."
            )
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Key {self._api_key}"},
                timeout=self._timeout,
            )
        return self._http

    def _raise_status(self, status: int, body: str, context: str) -> None:
        """Map a Higgsfield HTTP failure onto a :class:`ProviderError`.

        Raises:
            ProviderError: Always.
        """
        logger.error("Higgsfield error %s on %s: %s", status, context, body)
        lowered = body.lower()

        # Verified live: a valid credential on an account with no balance
        # answers 403 not_enough_credits. Calling that an auth failure would
        # send the caller after a problem they do not have.
        if status in (402, 403) or "not_enough_credits" in lowered:
            raise ProviderError(
                self.get_name(),
                f"Higgsfield rejected '{context}' for billing reasons: the account is "
                f"out of credits (the credential is valid). Top up at "
                f"https://console.higgsfield.ai. Error: {body}",
                model_name=context,
                status_code=status,
                retryable=False,
            )
        if status == 401:
            raise ProviderError(
                self.get_name(),
                f"Higgsfield authentication failed. The API expects "
                f"'Key <key-id>:<key-secret>' — check HIGGSFIELD_API_KEY holds both "
                f"halves. Error: {body}",
                model_name=context,
                status_code=status,
            )
        if status == 404:
            raise ProviderError(
                self.get_name(),
                f"Higgsfield could not find '{context}'. Model ids are endpoint paths "
                f"(e.g. 'higgsfield-ai/soul/standard'); browse them at "
                f"https://console.higgsfield.ai. Error: {body}",
                model_name=context,
                status_code=status,
            )
        if status == 422:
            raise ProviderError(
                self.get_name(),
                f"Higgsfield rejected the parameters for '{context}'. Fields differ per "
                f"model; pass model-specific ones as keyword arguments. Error: {body}",
                model_name=context,
                status_code=status,
            )
        if status == 429:
            raise ProviderError(
                self.get_name(),
                f"Higgsfield rate limit reached. Error: {body}",
                model_name=context,
                status_code=status,
                retryable=True,
            )
        raise ProviderError(
            self.get_name(),
            f"Higgsfield API error ({status}): {body}",
            model_name=context,
            status_code=status,
            retryable=status >= 500,
        )

    def _model_for(self, capability: str, model: str | None) -> str:
        """Resolve the endpoint path for *capability*."""
        return model or self._models.get(capability) or self.default_model

    # ------------------------------------------------------------------
    # BaseProvider surface (Higgsfield is media-only)
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        """Return the configured instance name (default: ``"higgsfield"``)."""
        return self._provider_instance_name or "higgsfield"

    async def chat_completion(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Not supported: Higgsfield serves generative media, not chat.

        Raises:
            ProviderError: Always, with a pointer to the media surface.
        """
        raise ProviderError(
            self.get_name(),
            "Higgsfield is a generative-media provider and has no chat-completions "
            "API. Use llm.media.images / llm.media.video instead.",
            status_code=400,
            retryable=False,
        )

    async def get_models_details(self) -> list[ModelDetails]:
        """Describe the configured model paths.

        Higgsfield publishes no catalog endpoint, so this reports what *this
        instance* is configured to call rather than inventing a listing.
        """
        seen: dict[str, str] = {}
        for capability, path in self._models.items():
            seen.setdefault(path, capability)
        return [
            ModelDetails(
                id=path,
                provider_name=self.get_name(),
                context_length=0,
                supports_streaming=False,
                supports_tools=False,
                model_type="media",
                metadata={"capability": capability, "async_request": True},
            )
            for path, capability in sorted(seen.items())
        ]

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """Higgsfield parameters are per model, so none are advertised."""
        return {}

    def get_max_context_length(self, model: str | None = None) -> int:
        """Higgsfield media models have no token context window."""
        return 0

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Tokens are not a meaningful unit for Higgsfield media models."""
        return 0

    async def count_message_tokens(
        self, messages: list[Message], model: str | None = None
    ) -> int:
        """Tokens are not a meaningful unit for Higgsfield media models."""
        return 0

    def extract_response_content(self, response: dict[str, Any]) -> str:
        """Higgsfield returns media, not text."""
        return ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        """Higgsfield has no token streaming."""
        return ""

    # ------------------------------------------------------------------
    # Media capability declaration
    # ------------------------------------------------------------------

    def media_capabilities(self) -> "frozenset[MediaCapability]":
        """Capabilities this instance is configured to serve."""
        from ..media.models import MediaCapability

        return frozenset(MediaCapability(c) for c in self._models if c in _DEFAULT_MODELS)

    def media_execution(
        self, capability: "MediaCapability", model: str | None = None
    ) -> "MediaExecution":
        """Every Higgsfield capability is an async request."""
        from ..media.models import MediaExecution

        return MediaExecution.ASYNC_JOB

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    async def _submit(
        self, path: str, payload: dict[str, Any], webhook_url: str | None
    ) -> dict[str, Any]:
        """Submit a generation request and return the raw response."""
        body = dict(payload)
        hook = webhook_url or self._webhook_url
        if hook:
            body["webhook_url"] = hook

        if self._backend == "sdk":
            controller = await self._sdk.submit(
                application=path, arguments=payload, webhook_url=hook
            )
            # Normalized into the same shape the REST response has, so
            # everything downstream is transport-independent.
            return {
                "request_id": controller.request_id,
                "status": "queued",
                "status_url": getattr(controller, "status_url", None),
                "cancel_url": getattr(controller, "cancel_url", None),
            }

        try:
            resp = await self._get_http().post(f"/{path}", json=body)
        except httpx.HTTPError as e:
            raise ProviderError(
                self.get_name(),
                f"Higgsfield transport error: {e}",
                model_name=path,
                retryable=True,
            ) from e
        if resp.status_code >= 400:
            self._raise_status(resp.status_code, resp.text, path)
        return dict(resp.json())

    async def _submit_job(
        self,
        capability: str,
        model: str | None,
        payload: dict[str, Any],
        webhook_url: str | None = None,
    ) -> "MediaJob":
        """Submit a request and wrap it in a tracked :class:`MediaJob`."""
        from ..media.models import MediaCapability, MediaJob

        path = self._model_for(capability, model)
        webhook_url = payload.pop("webhook_url", None) or webhook_url
        clean = {k: v for k, v in payload.items() if v is not None}
        submission = await self._submit(path, clean, webhook_url)

        job = MediaJob(
            capability=MediaCapability(capability),
            provider=self.get_name(),
            model=path,
            status=self._status_of(submission),
            provider_job_id=submission.get("request_id"),
            # Use the URLs Higgsfield returns rather than rebuilding routes it
            # owns — the lesson fal taught when its queue turned out to be
            # namespaced by application rather than by model path.
            poll_url=submission.get("status_url"),
            provider_metadata={
                "path": path,
                "submission": submission,
                "cancel_url": submission.get("cancel_url"),
            },
        )
        return self._apply_result(job, submission)

    @staticmethod
    def _status_of(payload: dict[str, Any]) -> Any:
        """Map a request state onto a job status."""
        from ..media.models import MediaJobStatus

        raw = str(payload.get("status") or "").lower()
        return MediaJobStatus(_STATUS_MAP.get(raw, "running"))

    async def poll_media_job(self, job: "MediaJob") -> "MediaJob":
        """Refresh *job* against the request status endpoint."""
        from ..media.models import MediaJobStatus

        if job.is_terminal:
            return job
        request_id = job.provider_job_id
        if not request_id:
            job.status = MediaJobStatus.FAILED
            job.error = "Lost the Higgsfield request id; the job cannot be polled."
            return job

        if self._backend == "sdk":
            return self._apply_result(job, await self._sdk_status(request_id))

        url = job.poll_url or f"/requests/{request_id}/status"
        resp = await self._get_http().get(url)
        if resp.status_code >= 400:
            self._raise_status(resp.status_code, resp.text, request_id)
        return self._apply_result(job, dict(resp.json()))

    async def cancel_media_job(self, job: "MediaJob") -> "MediaJob":
        """Cancel *job*."""
        from ..media.models import MediaJobStatus

        if job.is_terminal:
            return job
        request_id = job.provider_job_id
        if not request_id:
            return job

        if self._backend == "sdk":
            await self._sdk.cancel(request_id)
            job.status = MediaJobStatus.CANCELED
            job.touch()
            return job

        url = job.provider_metadata.get("cancel_url") or f"/requests/{request_id}/cancel"
        resp = await self._get_http().post(url)
        if resp.status_code >= 400 and resp.status_code != 404:
            self._raise_status(resp.status_code, resp.text, request_id)
        job.status = MediaJobStatus.CANCELED
        job.touch()
        return job

    async def apply_webhook_payload(
        self, job: "MediaJob", payload: dict[str, Any]
    ) -> "MediaJob":
        """Fold a Higgsfield webhook delivery into *job*.

        Higgsfield posts the same ``RequestStatus`` shape a poll returns. A
        payload whose ``request_id`` does not match is not trusted: anyone who
        learns a callback URL can POST to it, and a wrong artifact is worse than
        a slow one.
        """
        incoming = payload.get("request_id")
        if incoming and job.provider_job_id and incoming != job.provider_job_id:
            logger.warning(
                "Higgsfield webhook request_id %s does not match job %s; polling instead.",
                incoming,
                job.id,
            )
            return await self.poll_media_job(job)
        if not payload.get("status"):
            return await self.poll_media_job(job)
        return self._apply_result(job, payload)

    async def _sdk_status(self, request_id: str) -> dict[str, Any]:
        """Return a REST-shaped status payload via the SDK.

        The SDK reports state by returning a different ``Status`` subclass
        rather than a field, and fetches the output through a separate
        ``result`` call, so both are normalized into the single dict shape the
        REST path produces. Keeping one payload shape is what lets
        :meth:`_apply_result` stay transport-independent.
        """
        status = await self._sdk.status(request_id)
        state = _SDK_STATUS_CLASSES.get(type(status).__name__, "in_progress")
        payload: dict[str, Any] = {"request_id": request_id, "status": state}

        if state == "completed":
            # Only a completed request has output to fetch; asking earlier would
            # raise or return nothing useful.
            result = await self._sdk.result(request_id)
            if isinstance(result, dict):
                payload.update(result)
        error = getattr(status, "error", None)
        if error:
            payload["error"] = str(error)
        return payload

    def _apply_result(self, job: "MediaJob", payload: dict[str, Any]) -> "MediaJob":
        """Fold a ``RequestStatus`` payload into *job*."""
        from ..media.models import MediaJobStatus, MediaUsage

        raw_state = str(payload.get("status") or "").lower()
        job.status = self._status_of(payload)
        job.provider_metadata["last_status"] = payload
        if raw_state:
            job.provider_metadata["raw_status"] = raw_state
        job.touch()

        if raw_state == "nsfw":
            # Terminal like a failure, but a content refusal rather than a
            # malfunction. Said plainly so a caller does not retry it.
            job.status = MediaJobStatus.FAILED
            job.error = (
                "Higgsfield refused this request under its content policy "
                "(status 'nsfw'). This is not a transient error; retrying the same "
                "prompt will be refused again."
            )
            return job
        if payload.get("error"):
            job.status = MediaJobStatus.FAILED
            job.error = str(payload["error"])
            return job
        if job.status is not MediaJobStatus.SUCCEEDED:
            return job

        job.artifacts = self._artifacts_from_result(payload)
        job.progress = 1.0
        job.usage = MediaUsage(
            provider=self.get_name(), model=job.model, basis="per_request"
        )
        return job

    def _artifacts_from_result(self, payload: dict[str, Any]) -> list[Any]:
        """Extract artifacts from a completed payload.

        Walks the documented result keys (``images``, ``video``, ``audio`` and
        their plurals). An unrecognised shape yields no artifacts rather than
        raising — the untouched payload stays on the job either way.
        """
        from ..media.models import MediaArtifact, MediaKind

        artifacts: list[MediaArtifact] = []
        results = payload.get("results") if isinstance(payload.get("results"), dict) else payload

        for key, kind_name in _RESULT_KEYS:
            value = results.get(key)
            if value is None:
                continue
            entries = value if isinstance(value, list) else [value]
            for entry in entries:
                url = entry.get("url") if isinstance(entry, dict) else entry
                if not isinstance(url, str) or not url:
                    continue
                extra = (
                    {k: v for k, v in entry.items() if k != "url"}
                    if isinstance(entry, dict)
                    else {}
                )
                artifacts.append(
                    MediaArtifact(
                        kind=MediaKind(kind_name),
                        uri=url,
                        mime_type=self._mime_for(url),
                        width=extra.get("width"),
                        height=extra.get("height"),
                        duration_seconds=extra.get("duration"),
                        provider_metadata={"field": key, **extra},
                    )
                )
        return artifacts

    @staticmethod
    def _mime_for(uri: str) -> str | None:
        """Guess a MIME type from a URI extension, or ``None``."""
        import mimetypes

        guessed, _ = mimetypes.guess_type(uri.split("?")[0])
        return guessed

    async def _ref_to_url(self, ref: "MediaRef | None") -> str | None:
        """Render *ref* as a URL Higgsfield can fetch.

        Remote refs pass through. Local bytes become a data URI: Higgsfield's
        generation endpoints take URLs, and the published client's upload helper
        covers its agent surface rather than these endpoints.
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
        aspect_ratio: str | None = None,
        **kwargs: Any,
    ) -> "MediaJob":
        """Generate images from *prompt*."""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "num_images": n,
            "aspect_ratio": aspect_ratio,
            **kwargs,
        }
        if size and "resolution" not in payload:
            payload["resolution"] = size
        return await self._submit_job("image_generate", model, payload)

    async def generate_video_media(
        self,
        prompt: str,
        *,
        model: str | None = None,
        image: "MediaRef | None" = None,
        duration_seconds: float | None = None,
        negative_prompt: str | None = None,
        **kwargs: Any,
    ) -> "MediaJob":
        """Generate video from *prompt*, optionally conditioned on an image.

        Higgsfield splits text-to-video and image-to-video into separate
        endpoints, so supplying *image* switches to the matching path rather
        than posting an image to a text-only endpoint and getting a 422.
        """
        path = self._model_for("video_generate", model)
        image_url = await self._ref_to_url(image)
        if image_url:
            mapped = _IMAGE_TO_VIDEO_MODELS.get(path)
            if mapped:
                logger.debug(
                    "Higgsfield: image supplied, routing to the image-to-video path %s.",
                    mapped,
                )
                path = mapped
            else:
                logger.warning(
                    "Higgsfield model '%s' has no known image-to-video counterpart; "
                    "sending the image to it as-is. If it rejects image_url, set the "
                    "image-to-video path explicitly with model=.",
                    path,
                )

        payload: dict[str, Any] = {
            "prompt": prompt,
            "image_url": image_url,
            "duration": duration_seconds,
            "negative_prompt": negative_prompt,
            **kwargs,
        }
        return await self._submit_job("video_generate", path, payload)

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
