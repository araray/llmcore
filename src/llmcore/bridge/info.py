"""ServerInfo / capability assembly for ControlService.GetInfo (spec §15).

Clients negotiate on connect: verify ``contract_version`` and required
``capabilities`` before issuing calls. In v1 the Tier-0 inference + catalog +
control surface is live; Tier-2 audio is *designed but not implemented*, so its
capability flags are absent (the bridge returns UNIMPLEMENTED for AudioService).
"""

from __future__ import annotations

from typing import Iterable

from . import BRIDGE_VERSION, CONTRACT_VERSION
from ._generated.llmcore.v1 import control_pb2
from .facade import LLMCoreFacade

__all__ = [
    "audio_capable",
    "build_server_info",
    "capabilities_for",
    "sessions_capable",
    "vector_capable",
]

# The live-audio surface is Deepgram-specific; the one-shot methods live on
# BaseProvider and so cannot distinguish an audio-capable deployment.
_AUDIO_STREAMING_METHODS = ("transcribe_stream", "stream_speech", "run_voice_agent")


def audio_capable(facade: LLMCoreFacade) -> bool:
    """Whether the bridge should advertise / enable Tier-2 audio for ``facade``.

    Resolution order:

    1. If the facade exposes a ``supports_audio()`` hook (the test ``FakeFacade``
       does), use it verbatim — this is the explicit, deterministic gate.
    2. Otherwise (a real ``LLMCore``), probe configured providers and report
       ``True`` iff at least one exposes the live-audio surface
       (``transcribe_stream`` / ``stream_speech`` / ``run_voice_agent``).

    Any failure is treated as "not capable": capability detection must never
    break ``GetInfo``.
    """
    hook = getattr(facade, "supports_audio", None)
    if callable(hook):
        try:
            return bool(hook())
        except Exception:
            return False
    get = getattr(facade, "get_provider", None)
    if get is None:
        return False
    try:
        names = list(facade.get_available_providers())
    except Exception:
        names = []
    for name in names or [None]:
        try:
            provider = get(name)
        except Exception:
            continue
        if any(hasattr(provider, m) for m in _AUDIO_STREAMING_METHODS):
            return True
    return False


def sessions_capable(facade: LLMCoreFacade) -> bool:
    """Whether the bridge should advertise Tier-1 sessions for ``facade``.

    Resolution order (mirrors :func:`audio_capable`):

    1. If the facade exposes a ``supports_sessions()`` hook (the test
       ``FakeFacade`` does, env-gated), use it verbatim — the explicit,
       deterministic gate that keeps the Tier-0 capability set byte-identical
       for the existing conformance suites until a test opts in.
    2. Otherwise (a real ``LLMCore``), report capable iff it exposes the
       ``create_session`` coroutine, which it always does.

    Detection never raises — any failure means "not capable".
    """
    hook = getattr(facade, "supports_sessions", None)
    if callable(hook):
        try:
            return bool(hook())
        except Exception:
            return False
    return callable(getattr(facade, "create_session", None))


def vector_capable(facade: LLMCoreFacade) -> bool:
    """Whether the bridge should advertise the Tier-1 vector/RAG surface.

    Same resolution as :func:`sessions_capable`: a ``supports_vector()`` hook
    wins when present (env-gated in the test ``FakeFacade``); otherwise a real
    ``LLMCore`` is capable iff it exposes ``search_vector_store``.
    """
    hook = getattr(facade, "supports_vector", None)
    if callable(hook):
        try:
            return bool(hook())
        except Exception:
            return False
    return callable(getattr(facade, "search_vector_store", None))


def _llmcore_version() -> str:
    try:  # pragma: no cover - trivial
        import llmcore

        return getattr(llmcore, "__version__", "unknown")
    except Exception:
        return "unknown"


def capabilities_for(
    transports: Iterable[str],
    *,
    audio: bool = False,
    sessions: bool = False,
    vector: bool = False,
) -> list[str]:
    """Return the capability flag list advertised for ``transports``.

    Tier-0 is always present. ``chat.tool_calls`` is intentionally *omitted*
    (provisional until pinned against the real provider response, spec §5.2).
    Tier-1 (``tier1.*``) flags are appended for the ``sessions`` and ``vector``
    families independently; Tier-2 (``tier2.*``) flags only when ``audio`` is
    enabled.
    """
    caps = ["tier0", "inference.chat", "inference.chat_stream", "inference.count_tokens",
            "inference.estimate_cost", "catalog.providers", "catalog.models", "control.info"]
    for t in transports:
        caps.append(f"transport.{t}")
    if vector:
        # Tier-1 vector store & RAG collections (phase B4).
        caps.extend([
            "tier1.vector",
            "vector.add_documents",
            "vector.search",
            "vector.list_vector_collections",
            "vector.list_rag_collections",
            "vector.get_rag_collection_info",
            "vector.delete_rag_collection",
        ])
    if sessions:
        # Tier-1 umbrella + the sessions/context-items surface (phase B4).
        caps.extend([
            "tier1.sessions",
            "sessions.create",
            "sessions.get",
            "sessions.list",
            "sessions.delete",
            "sessions.update_name",
            "sessions.fork",
            "sessions.clone",
            "sessions.delete_messages",
            "sessions.get_messages_by_range",
            "sessions.add_context_item",
            "sessions.get_context_item",
            "sessions.remove_context_item",
            # Context presets share the sessions/context-management family.
            "presets.save",
            "presets.get",
            "presets.list",
            "presets.delete",
        ])
    if audio:
        # Tier-2 umbrella + the live streaming RPCs and the one-shot (unary)
        # RPCs, all gated on facade.supports_audio() / LLMCORE_BRIDGE_FAKE_AUDIO.
        caps.extend([
            "tier2.audio",
            "audio.transcribe_stream",
            "audio.synthesize_stream",
            "audio.voice_agent",
            "audio.synthesize",
            "audio.transcribe",
            "audio.generate_image",
            "audio.ocr",
            "audio.analyze_text",
        ])
    return caps


def build_server_info(
    facade: LLMCoreFacade, transports: Iterable[str]
) -> control_pb2.ServerInfo:
    """Build the ``ServerInfo`` handshake message.

    Args:
        facade: The live facade (queried for available providers).
        transports: Enabled transports, e.g. ``("grpc", "http")``.

    Returns:
        A populated ``llmcore.v1.ServerInfo``.
    """
    transports = list(transports)
    audio = audio_capable(facade)
    sessions = sessions_capable(facade)
    vector = vector_capable(facade)
    tiers = ["T0"]
    if sessions or vector:
        tiers.append("T1")
    if audio:
        tiers.append("T2")
    info = control_pb2.ServerInfo(
        llmcore_version=_llmcore_version(),
        bridge_version=BRIDGE_VERSION,
        contract_version=CONTRACT_VERSION,
        transports=transports,
        capabilities=capabilities_for(
            transports, audio=audio, sessions=sessions, vector=vector
        ),
        tiers=tiers,
    )
    try:
        providers = facade.get_available_providers() or []
    except Exception:
        providers = []
    for name in providers:
        info.providers.append(control_pb2.ProviderInfo(name=str(name), available=True))
    return info
