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

__all__ = ["build_server_info", "capabilities_for"]


def _llmcore_version() -> str:
    try:  # pragma: no cover - trivial
        import llmcore

        return getattr(llmcore, "__version__", "unknown")
    except Exception:
        return "unknown"


def capabilities_for(transports: Iterable[str]) -> list[str]:
    """Return the capability flag list advertised for ``transports``.

    Tier-0 is always present. ``chat.tool_calls`` is intentionally *omitted*
    (provisional until pinned against the real provider response, spec §5.2).
    Tier-2 (``tier2.*``) flags are omitted until phase B3.
    """
    caps = ["tier0", "inference.chat", "inference.chat_stream", "inference.count_tokens",
            "inference.estimate_cost", "catalog.providers", "catalog.models", "control.info"]
    for t in transports:
        caps.append(f"transport.{t}")
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
    info = control_pb2.ServerInfo(
        llmcore_version=_llmcore_version(),
        bridge_version=BRIDGE_VERSION,
        contract_version=CONTRACT_VERSION,
        transports=transports,
        capabilities=capabilities_for(transports),
        tiers=["T0"],
    )
    try:
        providers = facade.get_available_providers() or []
    except Exception:
        providers = []
    for name in providers:
        info.providers.append(control_pb2.ProviderInfo(name=str(name), available=True))
    return info
