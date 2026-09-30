# src/llmcore/media/manager.py
"""The media subsystem entry point.

:class:`MediaManager` is the sibling of
:class:`~llmcore.providers.manager.ProviderManager` and
:class:`~llmcore.search.manager.SearchProviderManager`: it discovers media
adapters, resolves capabilities to providers, and owns the job manager and
artifact store.

**Adapters are the chat providers.**  Rather than a parallel
``[media.providers.*]`` credential tree, the manager scans the already-loaded
``[providers.*]`` instances and treats any that implements
:class:`~llmcore.media.protocols.MediaCapableProvider` as a media adapter.  One
credential per vendor, one place to configure it, and the capability matrix
already tolerates providers that cannot chat (``deepgram``, ``typesafe``).
``[media.routing]`` then expresses preference order per capability.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..exceptions import MediaCapabilityError
from .artifacts import ArtifactStore, MaterializePolicy, default_fetcher
from .jobs import JobPolicy, MediaJobManager
from .models import MediaCapability, MediaJob, MediaResult
from .protocols import CAPABILITY_PROTOCOLS, MediaCapableProvider
from .routers import AudioRouter, ImageRouter, VideoRouter

logger = logging.getLogger(__name__)

__all__ = ["MediaManager"]

#: Built-in preference order per capability, used when ``[media.routing]`` is
#: silent.  Sourced from the workload recommendations in the provider survey
#: (see ``docs/MEDIA_SUBSYSTEM_SPEC.md`` §2.7); a provider that is not
#: configured is skipped, so these are hints rather than requirements.
DEFAULT_ROUTING: dict[MediaCapability, tuple[str, ...]] = {
    MediaCapability.TTS: ("elevenlabs", "openai", "deepgram", "zai", "mistral", "deepinfra"),
    MediaCapability.TTS_STREAM: ("elevenlabs", "deepgram", "openai"),
    MediaCapability.ASR: ("deepgram", "openai", "elevenlabs", "mistral", "zai", "friendli"),
    MediaCapability.ASR_STREAM: ("deepgram", "openai", "elevenlabs"),
    MediaCapability.VOICE_AGENT: ("deepgram", "openai"),
    MediaCapability.MUSIC: ("elevenlabs", "fal", "replicate"),
    MediaCapability.SFX: ("elevenlabs", "fal", "replicate"),
    MediaCapability.IMAGE_GENERATE: ("openai", "gemini", "fal", "zai", "replicate", "deepinfra"),
    MediaCapability.IMAGE_EDIT: ("openai", "gemini", "bfl", "fal"),
    MediaCapability.IMAGE_UPSCALE: ("fal", "replicate", "luma"),
    MediaCapability.OCR: ("mistral", "zai", "gemini"),
    MediaCapability.VIDEO_GENERATE: ("gemini", "fal", "replicate", "luma", "zai"),
    MediaCapability.VIDEO_EDIT: ("luma", "fal", "replicate"),
    MediaCapability.VIDEO_INTERPOLATE: ("fal", "replicate"),
}


class MediaManager:
    """Discovers media adapters and routes capability requests to them.

    Args:
        adapters: Mapping of provider instance name to media adapter.
        routing: Per-capability preference order, overriding :data:`DEFAULT_ROUTING`.
        artifact_store: Store for produced assets; a default is built when omitted.
        job_policy: Polling/timeout policy for long-running jobs.
    """

    def __init__(
        self,
        adapters: dict[str, MediaCapableProvider] | None = None,
        *,
        routing: dict[MediaCapability, tuple[str, ...]] | None = None,
        artifact_store: ArtifactStore | None = None,
        job_policy: JobPolicy | None = None,
    ) -> None:
        self._adapters: dict[str, MediaCapableProvider] = dict(adapters or {})
        self._routing: dict[MediaCapability, tuple[str, ...]] = {
            **DEFAULT_ROUTING,
            **(routing or {}),
        }
        self.artifacts = artifact_store or ArtifactStore()
        self.jobs = MediaJobManager(self._adapters.get, policy=job_policy)

        self.images = ImageRouter(self)
        self.audio = AudioRouter(self)
        self.video = VideoRouter(self)

    # ------------------------------------------------------------------
    # Construction from config
    # ------------------------------------------------------------------

    @classmethod
    def from_provider_manager(
        cls,
        provider_manager: Any,
        config_get: Callable[[str, Any], Any] | None = None,
    ) -> MediaManager:
        """Build a manager from the loaded chat providers.

        Any provider instance implementing
        :class:`~llmcore.media.protocols.MediaCapableProvider` becomes a media
        adapter. Providers that only implement the legacy
        :class:`~llmcore.providers.base.BaseProvider` media methods are *not*
        adapters — they keep working through their own methods, and gain routing
        when they are migrated.

        Args:
            provider_manager: The loaded :class:`ProviderManager`.
            config_get: A ``config.get(key, default)`` accessor; defaults are
                used when omitted.

        Returns:
            A configured manager. Never raises on an absent ``[media]`` section.
        """
        get = config_get or (lambda _k, d=None: d)

        adapters: dict[str, MediaCapableProvider] = {}
        for name in provider_manager.get_available_providers():
            try:
                provider = provider_manager.get_provider(name)
            except Exception as e:
                logger.debug("Skipping provider '%s' during media discovery: %s", name, e)
                continue
            if isinstance(provider, MediaCapableProvider):
                adapters[name] = provider

        routing = cls._routing_from_config(get)

        store = ArtifactStore(
            get("media.artifact_path", "~/.llmcore/media"),
            policy=_coerce_policy(get("media.artifact_materialize", MaterializePolicy.ON_EXPIRY)),
            fetcher=_safe_default_fetcher(),
        )

        manager = cls(
            adapters,
            routing=routing,
            artifact_store=store,
            job_policy=JobPolicy.from_config(get),
        )
        logger.debug(
            "MediaManager initialized with %d adapter(s): %s",
            len(adapters),
            ", ".join(sorted(adapters)) or "none",
        )
        return manager

    @staticmethod
    def _routing_from_config(
        get: Callable[[str, Any], Any],
    ) -> dict[MediaCapability, tuple[str, ...]]:
        """Read ``[media.routing]`` into a capability→preference map."""
        section = get("media.routing", {}) or {}
        if not isinstance(section, dict):
            logger.warning("[media.routing] is not a table; ignoring it.")
            return {}
        routing: dict[MediaCapability, tuple[str, ...]] = {}
        for key, value in section.items():
            try:
                capability = MediaCapability(str(key).lower())
            except ValueError:
                logger.warning("Unknown media capability '%s' in [media.routing]; ignoring.", key)
                continue
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, (list, tuple)):
                logger.warning("[media.routing].%s must be a list of provider names.", key)
                continue
            routing[capability] = tuple(str(v).lower() for v in value)
        return routing

    # ------------------------------------------------------------------
    # Adapter registry
    # ------------------------------------------------------------------

    def register_adapter(self, name: str, adapter: MediaCapableProvider) -> None:
        """Register (or replace) a media adapter under *name*."""
        self._adapters[name.lower()] = adapter
        logger.debug("Registered media adapter '%s'.", name)

    def unregister_adapter(self, name: str) -> None:
        """Remove the adapter registered under *name*, if present."""
        if self._adapters.pop(name.lower(), None) is not None:
            logger.debug("Unregistered media adapter '%s'.", name)

    @property
    def adapter_names(self) -> list[str]:
        """Names of every registered media adapter."""
        return sorted(self._adapters)

    def has_adapters(self) -> bool:
        """Whether any media adapter is available."""
        return bool(self._adapters)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def capabilities(self) -> dict[MediaCapability, list[str]]:
        """Return every available capability mapped to the providers offering it."""
        out: dict[MediaCapability, list[str]] = {}
        for name, adapter in self._adapters.items():
            for capability in self._capabilities_of(adapter):
                out.setdefault(capability, []).append(name)
        return {c: sorted(v) for c, v in sorted(out.items())}

    def who_can(self, capability: MediaCapability | str) -> list[str]:
        """Return providers that can serve *capability*, in preference order."""
        cap = MediaCapability(capability)
        available = {
            name
            for name, adapter in self._adapters.items()
            if cap in self._capabilities_of(adapter)
        }
        preferred = [p for p in self._routing.get(cap, ()) if p in available]
        rest = sorted(available - set(preferred))
        return preferred + rest

    def _capabilities_of(self, adapter: MediaCapableProvider) -> frozenset[MediaCapability]:
        """Capabilities *adapter* declares, filtered by protocol conformance.

        A declared capability whose protocol the adapter does not actually
        implement is dropped with a warning: better a missing capability than a
        confident ``AttributeError`` at call time.
        """
        try:
            declared = adapter.media_capabilities()
        except Exception as e:
            logger.warning(
                "Adapter '%s' failed to report capabilities: %s", _name_of(adapter), e
            )
            return frozenset()

        usable: set[MediaCapability] = set()
        for capability in declared:
            protocol = CAPABILITY_PROTOCOLS.get(capability)
            if protocol is None or isinstance(adapter, protocol):
                usable.add(capability)
            else:
                logger.warning(
                    "Adapter '%s' declares %s but does not implement %s; skipping it.",
                    _name_of(adapter),
                    capability,
                    protocol.__name__,
                )
        return frozenset(usable)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        capability: MediaCapability | str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> MediaCapableProvider:
        """Choose the adapter that will serve *capability*.

        Resolution order (spec §2.7):

        1. Explicit ``provider`` — used, or an error if it lacks the capability.
        2. Explicit ``model`` — the owning provider, resolved from model cards.
        3. ``[media.routing]`` preference for the capability.
        4. Built-in :data:`DEFAULT_ROUTING` preference.
        5. Any remaining adapter that can do it.

        Raises:
            MediaCapabilityError: When nothing can serve it, naming the
                providers that could if they were configured.
        """
        cap = MediaCapability(capability)

        if provider:
            adapter = self._adapters.get(provider.lower())
            if adapter is None:
                raise MediaCapabilityError(
                    f"Media provider '{provider}' is not configured.",
                    capability=cap,
                    candidates=self.adapter_names,
                )
            if cap not in self._capabilities_of(adapter):
                raise MediaCapabilityError(
                    f"Media provider '{provider}' does not support this capability.",
                    capability=cap,
                    candidates=self.who_can(cap),
                )
            return adapter

        if model:
            owner = self._provider_for_model(model, cap)
            if owner is not None:
                return self._adapters[owner]

        candidates = self.who_can(cap)
        if not candidates:
            raise MediaCapabilityError(
                "No configured media provider offers this capability.",
                capability=cap,
                candidates=list(self._routing.get(cap, ())),
            )
        return self._adapters[candidates[0]]

    def _provider_for_model(self, model: str, capability: MediaCapability) -> str | None:
        """Resolve *model* to a configured adapter via the model-card registry."""
        try:
            from ..model_cards.registry import get_model_card_registry

            registry = get_model_card_registry()
        except Exception:
            return None

        for name, adapter in self._adapters.items():
            if capability not in self._capabilities_of(adapter):
                continue
            try:
                if registry.get(name, model) is not None:
                    return name
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _finalize(self, outcome: MediaResult | MediaJob) -> MediaResult | MediaJob:
        """Track jobs so a submitted job is never lost to a dropped reference."""
        if isinstance(outcome, MediaJob):
            return self.jobs.track(outcome)
        return outcome

    async def wait(
        self, job: MediaJob, *, timeout: float | None = None
    ) -> MediaResult:
        """Wait for *job* and return its result.

        Convenience over ``manager.jobs.wait(...)`` plus ``job.to_result()``,
        which is what almost every caller wants.
        """
        finished = await self.jobs.wait(job, timeout=timeout)
        return finished.to_result()

    async def close(self) -> None:
        """Release media resources. Adapters are owned by ``ProviderManager``.

        Active jobs are deliberately **not** cancelled: a long video generation
        is already paid for, and killing it on shutdown would waste it. Callers
        that want cancellation call ``manager.jobs.cancel_all()`` explicitly.
        """
        active = self.jobs.list(active_only=True)
        if active:
            logger.info(
                "MediaManager closing with %d active job(s) left running: %s",
                len(active),
                ", ".join(j.id for j in active),
            )


def _name_of(adapter: Any) -> str:
    """Best-effort adapter name for log messages."""
    try:
        return adapter.get_name()
    except Exception:
        return type(adapter).__name__


def _coerce_policy(value: Any) -> MaterializePolicy:
    """Coerce a config value to a materialize policy, defaulting safely."""
    try:
        return MaterializePolicy(str(value))
    except ValueError:
        logger.warning("Unknown media.artifact_materialize '%s'; using 'on_expiry'.", value)
        return MaterializePolicy.ON_EXPIRY


def _safe_default_fetcher() -> Any:
    """Return an httpx fetcher when available, else ``None``.

    Materialization then raises a clear error only if it is actually attempted,
    rather than making httpx a hard requirement of the subsystem.
    """
    try:
        return default_fetcher()
    except Exception:
        return None
