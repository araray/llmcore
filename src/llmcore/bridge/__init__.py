"""llmcore.bridge — out-of-process bridge exposing the LLMCore facade to
non-Python clients over gRPC (primary) and HTTP+SSE (secondary).

This subpackage is the Python half of the language-bindings effort. It is a
**thin transport-agnostic adapter** over the existing public ``LLMCore`` facade:
it performs decode -> call -> encode and nothing more. No retrieval/RAG/routing
logic lives here — that remains in ``llmcore`` proper.

Install with the optional extra::

    pip install "llmcore[bridge]"

Run::

    llmcore-bridge serve --transport grpc,http --uds /run/llmcore/bridge.sock

See ``bindings/`` for the wire contract (``proto/llmcore/v1``) and the
contract-vs-reality CI guard (``bindings/scripts/contract_guard.py``).
"""

from __future__ import annotations

#: Bridge component version. Independent of ``llmcore.__version__``; advertised
#: via ``ControlService.GetInfo`` (ServerInfo.bridge_version).
BRIDGE_VERSION = "0.1.0"

#: Wire contract version (proto package ``llmcore.v1``).
CONTRACT_VERSION = "llmcore.v1"

__all__ = ["BRIDGE_VERSION", "CONTRACT_VERSION"]
