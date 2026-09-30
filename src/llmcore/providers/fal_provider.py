# src/llmcore/providers/fal_provider.py
"""fal (fal.ai) media provider for the LLMCore media subsystem.

fal is a **model marketplace**: every model is its own HTTP endpoint behind one
durable queue, so a single adapter reaches image, video, audio, SFX and
frame-interpolation families that would otherwise need a vendor integration
each.  That breadth is why the media spec makes fal the *provider-neutrality
test*: its lifecycle is meaningfully different from a first-party vendor's, so
if the abstraction bends here, the abstraction is wrong rather than the adapter.

What is different about fal, and how it maps
--------------------------------------------

* **The model IS the endpoint.**  ``fal-ai/flux/schnell``, ``fal-ai/film/video``
  and so on are paths, not names in a catalog.  llmcore therefore keeps no
  large hardcoded model table — fal's gallery moves constantly.  Per-capability
  defaults live in ``[providers.fal.models]`` and any call can override with
  ``model=``.
* **Everything is a queue submission.**  Even image generation, which is
  request/response on OpenAI, is ``IN_QUEUE → IN_PROGRESS → COMPLETED`` here.
  The adapter returns :class:`~llmcore.media.MediaJob` for every capability and
  reports ``ASYNC_JOB`` execution, so callers are never surprised.
* **Cancellation is a request, not a guarantee.**  ``PUT .../cancel`` answers
  ``202 CANCELLATION_REQUESTED``, and fal states the request may still complete
  if a runner already picked it up.  The adapter records that honestly rather
  than reporting a cancellation that may not have happened.
* **Inputs are URLs.**  fal models take file *URLs*, so local bytes are uploaded
  to the fal CDN first; a remote ``MediaRef`` is passed straight through and
  never round-trips through this process.

Transport (selectable via ``backend``)
--------------------------------------

* ``"httpx"`` — direct REST against ``queue.fal.run``.  **The default**: the
  queue protocol is four endpoints, and this keeps the vendor SDK off the
  critical path.
* ``"sdk"`` — the official ``fal-client`` package.

References:
  - https://fal.ai/docs/llms.txt
  - https://fal.ai/docs/documentation/model-apis/inference/queue
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
    import fal_client

    fal_sdk_available = True
except ImportError:
    fal_sdk_available = False
    fal_client = None  # type: ignore

from ..exceptions import ConfigError, ProviderError
from ..models import Message, ModelDetails, Tool
from .base import BaseProvider, ContextPayload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from ..media.models import MediaCapability, MediaExecution, MediaJob, MediaRef

logger = logging.getLogger(__name__)

#: Queue API root. Every model is a path under this host.
_QUEUE_BASE_URL = "https://queue.fal.run"

#: Synchronous root, used only for the CDN upload helper's sibling endpoints.
_REST_BASE_URL = "https://fal.run"
# Storage lives on its own host: posting to fal.run makes the router read
# "storage/upload" as an owner/app pair and answer 404.
_STORAGE_BASE_URL = "https://rest.fal.ai"
_CDN_BASE_URL = "https://v3.fal.media"

#: Environment variables checked, in order, for the fal credential.  fal's own
#: convention is ``FAL_KEY``; ``FAL_API_KEY`` is accepted because that is the
#: spelling operators commonly use.
_API_KEY_ENV_VARS: tuple[str, ...] = ("FAL_KEY", "FAL_API_KEY")

#: Conservative per-capability default endpoints.  Deliberately small: fal's
#: gallery changes weekly, so these are a starting point, not a catalog.
#: Override per capability in ``[providers.fal.models]`` or per call.
_DEFAULT_MODELS: dict[str, str] = {
    "image_generate": "fal-ai/flux/schnell",
    "image_edit": "fal-ai/flux-pro/kontext",
    "image_upscale": "fal-ai/clarity-upscaler",
    "video_generate": "fal-ai/minimax-video",
    "video_interpolate": "fal-ai/film",
    "sfx": "fal-ai/mmaudio-v2",
    "music": "fal-ai/stable-audio",
    "tts": "fal-ai/kokoro",
    "asr": "fal-ai/whisper",
}

#: fal queue states → llmcore job statuses.
_STATUS_MAP: dict[str, str] = {
    "IN_QUEUE": "queued",
    "IN_PROGRESS": "running",
    "COMPLETED": "succeeded",
}


class FalProvider(BaseProvider):
    """fal.ai media provider.

    Media-only by design: fal serves generative media models, not chat
    completions, so :meth:`chat_completion` raises rather than pretending.
    The real surface is ``llm.media`` (see ``docs/MEDIA_SUBSYSTEM_SPEC.md``).

    Configuration keys (under ``[providers.fal]``):

    - ``api_key`` / ``api_key_env_var`` — credential; falls back to ``FAL_KEY``
      then ``FAL_API_KEY``.
    - ``backend`` — ``"httpx"`` (default) or ``"sdk"``.
    - ``base_url`` — override the queue root.
    - ``timeout`` — HTTP timeout in seconds (default 300).
    - ``webhook_url`` — when set, submissions register this callback so results
      can be delivered without polling.
    - ``[providers.fal.models]`` — per-capability endpoint overrides.
    """

    _MEDIA_CAPABILITIES: frozenset[str] = frozenset(_DEFAULT_MODELS)

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """Initialize the fal provider.

        Args:
            config: Provider configuration from ``[providers.fal]``.
            log_raw_payloads: Whether to log raw request/response payloads.

        Raises:
            ConfigError: If no transport is installed or no API key is found.
        """
        super().__init__(config, log_raw_payloads)

        if not (httpx_available or fal_sdk_available):
            raise ConfigError(
                "The fal provider requires 'httpx' (direct REST, preferred) or the "
                "'fal-client' SDK. Install with: pip install llmcore[fal]"
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
                "fal API key not found. Set FAL_KEY (or FAL_API_KEY) or configure "
                "providers.fal.api_key / api_key_env_var. Create a key at "
                "https://fal.ai/dashboard/keys."
            )
        self._api_key = str(api_key)

        self._base_url = str(config.get("base_url", _QUEUE_BASE_URL)).rstrip("/")
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
            self._sdk = fal_client.AsyncClient(key=self._api_key)

        logger.debug(
            "fal provider initialized (backend=%s, base_url=%s, webhook=%s).",
            self._backend,
            self._base_url,
            bool(self._webhook_url),
        )

    @staticmethod
    def _resolve_backend(requested: str | None) -> str:
        """Resolve the transport, preferring direct REST.

        The queue protocol is four endpoints, so the SDK earns its place only
        when a caller asks for it explicitly.
        """
        available = {"httpx": httpx_available, "sdk": fal_sdk_available}
        req = (requested or "auto").lower()
        if req not in ("auto", "httpx", "sdk"):
            logger.warning("Unknown fal backend '%s'; using auto-detection.", req)
            req = "auto"
        if req != "auto" and available.get(req):
            return req
        if req != "auto":
            logger.warning("Requested fal backend '%s' is unavailable; falling back.", req)
        for backend in ("httpx", "sdk"):
            if available[backend]:
                return backend
        raise ConfigError("No usable fal transport is installed.")

    # ------------------------------------------------------------------
    # BaseProvider surface (fal is media-only)
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        """Return the configured instance name (default: ``"fal"``)."""
        return self._provider_instance_name or "fal"

    async def chat_completion(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Not supported: fal serves generative media, not chat completions.

        Raises:
            ProviderError: Always, with a pointer to the media surface.
        """
        raise ProviderError(
            self.get_name(),
            "fal is a generative-media provider and has no chat-completions API. "
            "Use llm.media.images / .video / .audio instead.",
            status_code=400,
            retryable=False,
        )

    async def get_models_details(self) -> list[ModelDetails]:
        """Describe the configured per-capability endpoints.

        fal's gallery has thousands of models and no stable list endpoint for
        them, so this reports what *this instance* is configured to call rather
        than pretending to enumerate the marketplace.
        """
        seen: dict[str, str] = {}
        for capability, endpoint in self._models.items():
            seen.setdefault(endpoint, capability)
        return [
            ModelDetails(
                id=endpoint,
                provider_name=self.get_name(),
                context_length=0,
                supports_streaming=False,
                supports_tools=False,
                model_type="media",
                metadata={"capability": capability, "queue": True},
            )
            for endpoint, capability in sorted(seen.items())
        ]

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """fal parameters are per-model, so no fixed schema is advertised."""
        return {}

    def get_max_context_length(self, model: str | None = None) -> int:
        """fal media models have no token context window."""
        return 0

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Tokens are not a meaningful unit for fal media models."""
        return 0

    async def count_message_tokens(
        self, messages: list[Message], model: str | None = None
    ) -> int:
        """Tokens are not a meaningful unit for fal media models."""
        return 0

    def extract_response_content(self, response: dict[str, Any]) -> str:
        """Return any text fal produced (some models emit transcripts)."""
        return str(response.get("text") or "")

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        """fal has no token streaming."""
        return ""

    # ------------------------------------------------------------------
    # Media subsystem adapter
    # ------------------------------------------------------------------

    def media_capabilities(self) -> "frozenset[MediaCapability]":
        """Capabilities this instance is configured to serve."""
        from ..media.models import MediaCapability

        return frozenset(MediaCapability(c) for c in self._models)

    def media_execution(
        self, capability: "MediaCapability", model: str | None = None
    ) -> "MediaExecution":
        """Every fal capability is a queue submission, including images."""
        from ..media.models import MediaExecution

        return MediaExecution.ASYNC_JOB

    def _endpoint_for(self, capability: str, model: str | None) -> str:
        """Resolve the fal endpoint path for a capability."""
        return model or self._models.get(capability) or self.default_model

    # --- transport ---

    def _get_http(self) -> Any:
        """Return (lazily creating) the queue HTTP client."""
        if not httpx_available:
            raise ProviderError(self.get_name(), "The 'httpx' package is required for fal.")
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Key {self._api_key}"},
                timeout=self._timeout,
            )
        return self._http

    def _raise_status(self, status: int, body: str, endpoint: str) -> None:
        """Map an HTTP failure onto a ProviderError.

        Raises:
            ProviderError: Always.
        """
        logger.error("fal error %s on %s: %s", status, endpoint, body)
        if status in (401, 403):
            raise ProviderError(
                self.get_name(),
                f"fal authentication failed. Check FAL_KEY / FAL_API_KEY. Error: {body}",
                model_name=endpoint,
                status_code=status,
            )
        if status == 404:
            raise ProviderError(
                self.get_name(),
                f"fal endpoint '{endpoint}' not found. Model ids are endpoint paths "
                f"(e.g. 'fal-ai/flux/schnell'); check the gallery at https://fal.ai/models. "
                f"Error: {body}",
                model_name=endpoint,
                status_code=status,
            )
        if status == 429:
            raise ProviderError(
                self.get_name(),
                f"fal rate limit / concurrency limit reached. Error: {body}",
                model_name=endpoint,
                status_code=status,
            )
        raise ProviderError(
            self.get_name(), f"fal API error ({status}): {body}", model_name=endpoint,
            status_code=status,
        )

    async def _submit(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit *payload* to *endpoint*'s queue and return the submission."""
        if self._backend == "sdk":
            handle = await self._sdk.submit(endpoint, arguments=payload)
            return {"request_id": handle.request_id}

        params = {"fal_webhook": self._webhook_url} if self._webhook_url else None
        client = self._get_http()
        try:
            resp = await client.post(f"/{endpoint}", json=payload, params=params)
        except httpx.HTTPError as e:
            raise ProviderError(
                self.get_name(), f"fal transport error: {e}", model_name=endpoint, retryable=True
            ) from e
        if resp.status_code >= 400:
            self._raise_status(resp.status_code, resp.text, endpoint)
        return resp.json()

    @staticmethod
    def _queue_path(endpoint: str) -> str:
        """Return the *app*-scoped queue path for *endpoint*.

        fal namespaces queue requests by application, not by the full model
        path: a request submitted to ``fal-ai/flux/schnell`` is tracked under
        ``fal-ai/flux``, and using the full path returns ``405``. Only the first
        two segments (``owner/app``) address the queue.
        """
        parts = [seg for seg in endpoint.split("/") if seg]
        return "/".join(parts[:2]) if len(parts) > 2 else endpoint

    def _queue_url(self, endpoint: str, request_id: str, suffix: str = "") -> str:
        """Build a queue URL for *request_id*, used when fal gave us none."""
        return f"/{self._queue_path(endpoint)}/requests/{request_id}{suffix}"

    async def _fetch_status(
        self, endpoint: str, request_id: str, url: str | None = None
    ) -> dict[str, Any]:
        """Return the raw queue status for *request_id*.

        *url* is the ``status_url`` fal returned at submission. It is preferred
        over anything reconstructed here, because fal owns the shape of its own
        queue routes; the reconstruction is only a fallback.
        """
        if self._backend == "sdk":
            status = await self._sdk.status(endpoint, request_id)
            name = type(status).__name__.upper()
            mapped = {"QUEUED": "IN_QUEUE", "INPROGRESS": "IN_PROGRESS", "COMPLETED": "COMPLETED"}
            return {
                "status": mapped.get(name, "IN_PROGRESS"),
                "queue_position": getattr(status, "position", None),
            }
        resp = await self._get_http().get(url or self._queue_url(endpoint, request_id, "/status"))
        if resp.status_code >= 400:
            self._raise_status(resp.status_code, resp.text, endpoint)
        return resp.json()

    async def _fetch_result(
        self, endpoint: str, request_id: str, url: str | None = None
    ) -> dict[str, Any]:
        """Return the completed output payload for *request_id*.

        *url* is the ``response_url`` fal returned at submission; see
        :meth:`_fetch_status` for why it wins over a reconstructed path.
        """
        if self._backend == "sdk":
            return await self._sdk.result(endpoint, request_id)
        resp = await self._get_http().get(url or self._queue_url(endpoint, request_id))
        if resp.status_code >= 400:
            self._raise_status(resp.status_code, resp.text, endpoint)
        return resp.json()

    async def _upload(self, ref: "MediaRef") -> str:
        """Return a URL fal can fetch for *ref*.

        A remote ref is passed straight through; local bytes are uploaded to
        fal storage, because fal models take URLs rather than inline payloads.

        Two storage backends are tried in the same order the official client
        uses: the token-authenticated CDN v3 upload first, then the older
        signed-URL flow. Accounts do not all have both — ``fal-cdn-v3`` is the
        current default while ``gcs`` answers ``400 Invalid storage type`` on
        newer accounts — so a single hard-coded path would work for some users
        and not others.
        """
        if ref.is_remote:
            return ref.url or ""
        data = ref.read_bytes()
        content_type = ref.mime_type or "application/octet-stream"
        file_name = ref.filename or "upload.bin"

        if self._backend == "sdk":
            return await self._sdk.upload(data, content_type)

        errors: list[str] = []
        for attempt in (self._upload_via_cdn_v3, self._upload_via_signed_url):
            try:
                return await attempt(data, content_type, file_name)
            except ProviderError as e:
                errors.append(f"{attempt.__name__}: {e}")
                logger.warning("fal upload via %s failed: %s", attempt.__name__, e)
            except Exception as e:  # every cause is reported below
                errors.append(f"{attempt.__name__}: {type(e).__name__}: {e}")
                logger.warning("fal upload via %s failed: %s", attempt.__name__, e)
        raise ProviderError(
            self.get_name(),
            "fal upload failed on every storage backend. " + " | ".join(errors),
            retryable=True,
        )

    async def _upload_via_cdn_v3(self, data: bytes, content_type: str, file_name: str) -> str:
        """Upload through the fal CDN v3 token flow (the current default)."""
        token_resp = await self._get_http().post(
            f"{_STORAGE_BASE_URL}/storage/auth/token",
            params={"storage_type": "fal-cdn-v3"},
            json={},
        )
        if token_resp.status_code >= 400:
            self._raise_status(token_resp.status_code, token_resp.text, "storage/auth/token")
        token = token_resp.json()

        headers = {
            "Content-Type": content_type,
            "X-Fal-File-Name": file_name,
            "Authorization": f"{token.get('token_type', 'Bearer')} {token['token']}",
        }
        base = token.get("base_url") or _CDN_BASE_URL
        async with httpx.AsyncClient(timeout=self._timeout) as cdn:
            resp = await cdn.post(f"{base}/files/upload", content=data, headers=headers)
            if resp.status_code >= 400:
                self._raise_status(resp.status_code, resp.text, "files/upload")
            return str(resp.json()["access_url"])

    async def _upload_via_signed_url(self, data: bytes, content_type: str, file_name: str) -> str:
        """Upload through the older initiate-then-PUT signed URL flow."""
        init = await self._get_http().post(
            f"{_STORAGE_BASE_URL}/storage/upload/initiate",
            params={"storage_type": "gcs"},
            json={"content_type": content_type, "file_name": file_name},
        )
        if init.status_code >= 400:
            self._raise_status(init.status_code, init.text, "storage/upload")
        payload = init.json()
        async with httpx.AsyncClient(timeout=self._timeout) as put_client:
            put = await put_client.put(
                payload["upload_url"], content=data, headers={"Content-Type": content_type}
            )
            put.raise_for_status()
        return str(payload["file_url"])

    # --- job plumbing ---

    async def _submit_job(
        self, capability: str, model: str | None, payload: dict[str, Any]
    ) -> "MediaJob":
        """Submit a queue request and wrap it in a tracked :class:`MediaJob`."""
        from ..media.models import MediaCapability, MediaJob, MediaJobStatus

        endpoint = self._endpoint_for(capability, model)
        clean = {k: v for k, v in payload.items() if v is not None}
        submission = await self._submit(endpoint, clean)

        job = MediaJob(
            capability=MediaCapability(capability),
            provider=self.get_name(),
            model=endpoint,
            status=MediaJobStatus.QUEUED,
            provider_job_id=submission.get("request_id"),
            poll_url=submission.get("status_url"),
            queue_position=submission.get("queue_position"),
            provider_metadata={
                "endpoint": endpoint,
                "submission": submission,
                "response_url": submission.get("response_url"),
                "cancel_url": submission.get("cancel_url"),
            },
        )
        return job

    async def poll_media_job(self, job: "MediaJob") -> "MediaJob":
        """Refresh *job* against the fal queue."""
        from ..media.models import MediaJobStatus

        if job.is_terminal:
            return job
        endpoint = job.provider_metadata.get("endpoint") or job.model
        request_id = job.provider_job_id
        if not request_id:
            job.status = MediaJobStatus.FAILED
            job.error = "Lost the fal request id; the job cannot be polled."
            return job

        status = await self._fetch_status(endpoint, request_id, job.poll_url)
        raw_state = str(status.get("status") or "").upper()
        job.queue_position = status.get("queue_position")
        job.provider_metadata["last_status"] = status
        job.touch()

        if status.get("error"):
            job.status = MediaJobStatus.FAILED
            job.error = str(status["error"])
            return job

        mapped = _STATUS_MAP.get(raw_state)
        if mapped != "succeeded":
            job.status = (
                MediaJobStatus.QUEUED if mapped == "queued" else MediaJobStatus.RUNNING
            )
            return job

        result = await self._fetch_result(
            endpoint, request_id, job.provider_metadata.get("response_url")
        )
        return self._apply_result(job, result, status)

    async def cancel_media_job(self, job: "MediaJob") -> "MediaJob":
        """Request cancellation of *job*.

        fal answers ``202 CANCELLATION_REQUESTED`` and states the request may
        still complete if a runner already picked it up, so the job is marked
        cancelled only when fal confirms it was not already finished — and the
        caveat is recorded in ``provider_metadata`` rather than hidden.
        """
        from ..media.models import MediaJobStatus

        if job.is_terminal:
            return job
        endpoint = job.provider_metadata.get("endpoint") or job.model
        request_id = job.provider_job_id
        if not request_id:
            return job

        if self._backend == "sdk":
            await self._sdk.cancel(endpoint, request_id)
            job.status = MediaJobStatus.CANCELED
            job.provider_metadata["cancellation"] = "requested"
            job.touch()
            return job

        cancel_url = job.provider_metadata.get("cancel_url") or self._queue_url(
            endpoint, request_id, "/cancel"
        )
        resp = await self._get_http().put(cancel_url)
        body = resp.text
        if resp.status_code == 400 and "ALREADY_COMPLETED" in body:
            # Not an error: the work finished before the cancel landed.
            job.provider_metadata["cancellation"] = "already_completed"
            return await self.poll_media_job(job)
        if resp.status_code >= 400 and resp.status_code != 404:
            self._raise_status(resp.status_code, body, endpoint)

        job.status = MediaJobStatus.CANCELED
        job.provider_metadata["cancellation"] = "requested"
        job.provider_metadata["cancellation_note"] = (
            "fal accepted the cancellation, but a request already being processed "
            "may still complete and be billed."
        )
        job.touch()
        return job

    def _apply_result(
        self, job: "MediaJob", result: dict[str, Any], status: dict[str, Any] | None = None
    ) -> "MediaJob":
        """Fold a completed fal payload into *job*."""
        from ..media.models import MediaJobStatus, MediaUsage

        artifacts = self._artifacts_from_result(result)
        job.artifacts = artifacts
        job.progress = 1.0
        job.status = MediaJobStatus.SUCCEEDED
        job.provider_metadata["result"] = result
        metrics = (status or {}).get("metrics") or {}
        job.usage = MediaUsage(
            provider=self.get_name(),
            model=job.model,
            basis="per_request",
            compute_seconds=metrics.get("inference_time"),
            raw=metrics,
        )
        job.touch()
        return job

    def _artifacts_from_result(self, result: dict[str, Any]) -> list[Any]:
        """Extract artifacts from a fal output payload.

        Output shapes are per-model — ``images``, ``image``, ``video``,
        ``audio``, ``audio_url``, ``text`` — so this walks the known keys rather
        than assuming one schema, which is what a marketplace requires.
        """
        from ..media.models import MediaArtifact, MediaKind

        artifacts: list[MediaArtifact] = []

        def _add(entry: Any, kind: MediaKind) -> None:
            if isinstance(entry, str):
                artifacts.append(MediaArtifact(kind=kind, uri=entry))
                return
            if not isinstance(entry, dict):
                return
            url = entry.get("url")
            if not url:
                return
            artifacts.append(
                MediaArtifact(
                    kind=kind,
                    uri=url,
                    mime_type=entry.get("content_type"),
                    width=entry.get("width"),
                    height=entry.get("height"),
                    duration_seconds=entry.get("duration"),
                    provider_metadata={
                        k: v for k, v in entry.items() if k not in {"url", "content_type"}
                    },
                )
            )

        for key, kind in (
            ("images", MediaKind.IMAGE),
            ("image", MediaKind.IMAGE),
            ("video", MediaKind.VIDEO),
            ("videos", MediaKind.VIDEO),
            ("audio", MediaKind.AUDIO),
            ("audio_url", MediaKind.AUDIO),
            ("audio_file", MediaKind.AUDIO),
        ):
            value = result.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                for entry in value:
                    _add(entry, kind)
            else:
                _add(value, kind)

        text = result.get("text")
        if text:
            artifacts.append(
                MediaArtifact(kind=MediaKind.TEXT, text=str(text), mime_type="text/plain")
            )
        return artifacts

    # --- capability methods ---

    async def generate_image_media(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        reference_images: "Sequence[MediaRef] | None" = None,
        **kwargs: Any,
    ) -> "MediaJob":
        """Generate images. Returns a job: fal queues every request."""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "num_images": n,
            "image_size": size,
            "seed": seed,
            "negative_prompt": negative_prompt,
            **kwargs,
        }
        if reference_images:
            payload["image_url"] = await self._upload(reference_images[0])
        return await self._submit_job("image_generate", model, payload)

    async def edit_image_media(
        self,
        prompt: str,
        *,
        image: "MediaRef",
        mask: "MediaRef | None" = None,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        **kwargs: Any,
    ) -> "MediaJob":
        """Edit an image."""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "image_url": await self._upload(image),
            "num_images": n,
            "image_size": size,
            **kwargs,
        }
        if mask is not None:
            payload["mask_url"] = await self._upload(mask)
        return await self._submit_job("image_edit", model, payload)

    async def upscale_image_media(
        self,
        *,
        image: "MediaRef",
        model: str | None = None,
        scale: float | None = None,
        **kwargs: Any,
    ) -> "MediaJob":
        """Upscale an image."""
        payload = {"image_url": await self._upload(image), "upscale_factor": scale, **kwargs}
        return await self._submit_job("image_upscale", model, payload)

    async def generate_video_media(
        self,
        prompt: str,
        *,
        model: str | None = None,
        first_frame: "MediaRef | None" = None,
        last_frame: "MediaRef | None" = None,
        reference_images: "Sequence[MediaRef] | None" = None,
        duration_seconds: float | None = None,
        resolution: str | None = None,
        aspect_ratio: str | None = None,
        fps: float | None = None,
        with_audio: bool | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> "MediaJob":
        """Generate a video."""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "duration": duration_seconds,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "seed": seed,
            **kwargs,
        }
        if first_frame is not None:
            payload["image_url"] = await self._upload(first_frame)
        if last_frame is not None:
            payload["end_image_url"] = await self._upload(last_frame)
        if with_audio is not None:
            payload["generate_audio"] = with_audio
        return await self._submit_job("video_generate", model, payload)

    async def edit_video_media(
        self, prompt: str, *, video: "MediaRef", model: str | None = None, **kwargs: Any
    ) -> "MediaJob":
        """Edit a video."""
        payload = {"prompt": prompt, "video_url": await self._upload(video), **kwargs}
        return await self._submit_job("video_generate", model, payload)

    async def interpolate_video_media(
        self,
        *,
        video: "MediaRef | None" = None,
        frames: "Sequence[MediaRef] | None" = None,
        model: str | None = None,
        target_fps: float | None = None,
        **kwargs: Any,
    ) -> "MediaJob":
        """Interpolate frames (e.g. FILM) to raise FPS or blend between frames.

        Distinct from a generative first/last-frame transition: this fills
        between frames that already exist.
        """
        payload: dict[str, Any] = {"fps": target_fps, **kwargs}
        if video is not None:
            payload["video_url"] = await self._upload(video)
        if frames:
            urls = [await self._upload(frame) for frame in frames]
            # FILM, the default endpoint, takes an explicit pair rather than a
            # list: it interpolates *between* two stills. Anything in between is
            # what it generates, so only the endpoints are inputs.
            payload.setdefault("start_image_url", urls[0])
            payload.setdefault("end_image_url", urls[-1])
            if len(urls) > 2:
                logger.warning(
                    "fal interpolation uses the first and last of the %d frames given; "
                    "the intermediate frames are what the model generates.",
                    len(urls),
                )
        return await self._submit_job("video_interpolate", model, payload)

    async def generate_sfx_media(
        self,
        prompt: str | None = None,
        *,
        model: str | None = None,
        video: "MediaRef | None" = None,
        duration_seconds: float | None = None,
        **kwargs: Any,
    ) -> "MediaJob":
        """Generate sound effects, optionally conditioned on a video (foley)."""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "duration": duration_seconds,
            **kwargs,
        }
        if video is not None:
            payload["video_url"] = await self._upload(video)
        return await self._submit_job("sfx", model, payload)

    async def generate_music_media(
        self,
        prompt: str,
        *,
        model: str | None = None,
        duration_seconds: float | None = None,
        **kwargs: Any,
    ) -> "MediaJob":
        """Generate music."""
        payload = {"prompt": prompt, "seconds_total": duration_seconds, **kwargs}
        return await self._submit_job("music", model, payload)

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
    ) -> "MediaJob":
        """Synthesize speech."""
        payload = {"prompt": text, "voice": voice, "speed": speed, **kwargs}
        return await self._submit_job("tts", model, payload)

    async def transcribe_media(
        self,
        *,
        audio: "MediaRef",
        model: str | None = None,
        language: str | None = None,
        diarize: bool | None = None,
        timestamps: bool | None = None,
        **kwargs: Any,
    ) -> "MediaJob":
        """Transcribe audio."""
        payload: dict[str, Any] = {
            "audio_url": await self._upload(audio),
            "language": language,
            "diarize": diarize,
            **kwargs,
        }
        return await self._submit_job("asr", model, payload)

    async def close(self) -> None:
        """Release the HTTP / SDK clients (best effort)."""
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception as e:
                logger.error("Error closing fal HTTP client: %s", e)
            self._http = None
        self._sdk = None
        logger.info("FalProvider closed.")
