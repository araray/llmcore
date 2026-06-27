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

__all__ = ["audio_capable", "build_server_info", "capabilities_for"]

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


def _llmcore_version() -> str:
    try:  # pragma: no cover - trivial
        import llmcore

        return getattr(llmcore, "__version__", "unknown")
    except Exception:
        return "unknown"


def capabilities_for(transports: Iterable[str], *, audio: bool = False) -> list[str]:
    """Return the capability flag list advertised for ``transports``.

    Tier-0 is always present. ``chat.tool_calls`` is intentionally *omitted*
    (provisional until pinned against the real provider response, spec §5.2).
    Tier-2 (``tier2.*``) flags are appended only when ``audio`` is enabled.
    """
    caps = ["tier0", "inference.chat", "inference.chat_stream", "inference.count_tokens",
            "inference.estimate_cost", "catalog.providers", "catalog.models", "control.info"]
    for t in transports:
        caps.append(f"transport.{t}")
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
    info = control_pb2.ServerInfo(
        llmcore_version=_llmcore_version(),
        bridge_version=BRIDGE_VERSION,
        contract_version=CONTRACT_VERSION,
        transports=transports,
        capabilities=capabilities_for(transports, audio=audio),
        tiers=["T0", "T2"] if audio else ["T0"],
    )
    try:
        providers = facade.get_available_providers() or []
    except Exception:
        providers = []
    for name in providers:
        info.providers.append(control_pb2.ProviderInfo(name=str(name), available=True))
    return info
