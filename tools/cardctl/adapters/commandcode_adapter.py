# tools/cardctl/adapters/commandcode_adapter.py
"""Command Code model-discovery adapter for cardctl.

Command Code's ``GET /provider/v1/models`` is public, so ``cardctl generate
commandcode`` needs no key. It is also thin. Each entry has ``id``, ``name``,
``context_length`` and ``supported_endpoints``, and nothing else: no prices,
no max output, no capability flags.

The rest is filled from sources that actually state it:

- **Prices** come from ``enrichments/commandcode.toml``, transcribed from
  Command Code's own pricing page. Nothing from third-party aggregators.
- **Capabilities and max output** are inherited from llmcore's card for the
  same model at its original vendor when one exists:
  ``claude-opus-5-5`` from ``anthropic/claude-opus-5-5``,
  ``moonshotai/Kimi-K3`` from ``kimi/kimi-k3``, and so on. The gateway passes
  requests through, so what the model can do is what the origin card says.
- **Tool calling** is set for every chat model. Command Code is a coding agent
  and drives every model it lists through tool calls, which its docs say it
  validates and repairs on every model it serves.
- Without a sibling card, vision and reasoning are set from the model-family
  rules the official pi provider uses (Claude, GPT and Gemini take images).
  Anything else is left off rather than guessed.

``routes`` (the endpoints) are recorded in ``provider_extension`` because
llmcore's provider routes from them when the live catalog is unreachable.

The two Jev decision models are served on ``/systemone`` but missing from
``/models``, so they are added here as curated entries.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from .base import BaseAdapter, NormalizedModel

logger = logging.getLogger(__name__)

MODELS_URL = "https://api.commandcode.ai/provider/v1/models"

#: Catalog org prefix -> llmcore card providers to inherit from, in order.
_ORG_SIBLINGS: dict[str, tuple[str, ...]] = {
    "google": ("google", "gemini"),
    "deepseek": ("deepseek",),
    "moonshotai": ("kimi",),
    "zai-org": ("zai", "friendli"),
    "z-ai": ("zai",),
    "qwen": ("qwen", "friendli", "deepinfra"),
    "xai": ("xai",),
    "mistral": ("mistral",),
    "minimaxai": ("friendli", "deepinfra"),
    "nvidia": ("deepinfra", "friendli"),
}

#: Families that take image input, from the official pi provider's defaults.
_VISION_FAMILIES = (re.compile(r"^claude-"), re.compile(r"^gpt-"), re.compile(r"^google/"))
#: Model names that say they take images.
_VISION_NAME = re.compile(r"(vision|-vl\b|-vl-|omni)", re.IGNORECASE)
#: Families that reason, from the same defaults.
_REASONING_FAMILIES = _VISION_FAMILIES

#: Served on /systemone, absent from /models. From Command Code's pricing page.
_JEV_MODELS: tuple[dict[str, Any], ...] = (
    {"id": "typesafe/jev", "name": "Jev", "context_length": 32768, "supported_endpoints": ["/systemone"]},
    {
        "id": "typesafe/jev-1.14.0",
        "name": "Jev 1.14.0",
        "context_length": 32768,
        "supported_endpoints": ["/systemone"],
    },
)


def _registry() -> Any:
    """llmcore's card registry, for inheriting from sibling cards."""
    try:
        from llmcore.model_cards import get_model_card_registry

        return get_model_card_registry()
    except Exception as exc:  # pragma: no cover - llmcore is always importable here
        logger.warning("Command Code: card registry unavailable (%s); no inheritance.", exc)
        return None


def sibling_candidates(model_id: str) -> list[tuple[str, str]]:
    """``(provider, model_id)`` pairs that may describe the same model."""
    if model_id.startswith("claude-"):
        return [("anthropic", model_id)]
    if model_id.startswith("gpt-"):
        return [("openai", model_id)]
    if "/" not in model_id:
        return []
    org, name = model_id.split("/", 1)
    out: list[tuple[str, str]] = []
    for provider in _ORG_SIBLINGS.get(org.lower(), ()):
        # Friendli and DeepInfra key cards by the full org/model id.
        out.append((provider, model_id if provider in ("friendli", "deepinfra") else name))
    return out


def states_capabilities(card: Any) -> bool:
    """Whether ``card`` actually states what its model can do.

    Several generated vendor cards carry the placeholder block, with every flag
    false except streaming. A Claude card claiming no tool use is a gap in the
    data, not a fact about Claude. Copying it would spread the gap, so a sibling
    counts only when it states at least one real capability.
    """
    caps = getattr(card, "capabilities", None)
    if caps is None:
        return False
    return any(
        bool(getattr(caps, flag, False))
        for flag in ("tool_use", "function_calling", "vision", "reasoning", "json_mode")
    )


def find_sibling(registry: Any, model_id: str) -> Any:
    """The first sibling card for ``model_id``, or ``None``."""
    if registry is None:
        return None
    for provider, candidate in sibling_candidates(model_id):
        try:
            card = registry.get(provider, candidate)
        except Exception:
            card = None
        if card is not None:
            return card
    return None


class CommandCodeAdapter(BaseAdapter):
    """Discovers Command Code models from the public gateway catalog."""

    provider_name = "commandcode"
    requires_api_key = False
    api_key_env_var = "CMD_API_KEY"
    base_url = "https://api.commandcode.ai/provider/v1"

    async def fetch_models(self) -> list[NormalizedModel]:
        """Fetch and normalize the catalog.

        Raises:
            RuntimeError: If the catalog cannot be read. A partial card set is
                worse than none, because a missing model looks identical to a
                retired one and regenerating would delete its card.
        """
        url = f"{self.base_url.rstrip('/')}/models"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Command Code model discovery failed: {exc}") from exc

        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(
                "Command Code returned no `data` list of models; refusing to generate "
                "cards from a shape this adapter does not recognise."
            )

        registry = _registry()
        seen = {str(r.get("id")) for r in rows if isinstance(r, dict)}
        rows = list(rows) + [dict(j) for j in _JEV_MODELS if j["id"] not in seen]
        out = [m for m in (self.normalize(row, registry) for row in rows) if m is not None]
        inherited = sum(1 for m in out if m.raw_api_data.get("_extension", {}).get("inherits_from"))
        logger.info(
            "Command Code: normalized %d of %d entries (%d inherit capabilities from a "
            "vendor card).",
            len(out),
            len(rows),
            inherited,
        )
        return out

    def normalize(self, row: Any, registry: Any = None) -> NormalizedModel | None:
        """One catalog row to a :class:`NormalizedModel`."""
        if not isinstance(row, dict) or not row.get("id"):
            return None
        model_id = str(row["id"])
        endpoints = [str(e) for e in (row.get("supported_endpoints") or [])]
        is_decision = "/systemone" in endpoints or model_id.startswith("typesafe/")
        context = row.get("context_length")

        model = NormalizedModel(
            model_id=model_id,
            provider=self.provider_name,
            display_name=str(row.get("name") or model_id),
            # Jev answers typed questions with probabilities, never text, so it
            # is not something a chat router should ever select.
            model_type="other" if is_decision else "chat",
            context_length=int(context) if isinstance(context, int) and context > 0 else None,
            # `created` is deliberately not read: the gateway stamps every entry
            # with the time of the response, so it is not a release date.
            owned_by="command-code",
            supports_streaming=not is_decision,
            supports_tools=not is_decision,
            raw_api_data=dict(row),
        )
        route = (
            "/systemone"
            if is_decision
            else "/messages"
            if "/messages" in endpoints
            else "/chat/completions"
        )
        model.description = (
            f"{model.display_name} via Command Code's Provider API ({route})."
            if not is_decision
            else None
        )
        extension: dict[str, Any] = {"routes": endpoints}
        if "/" in model_id:
            extension["upstream_org"] = model_id.split("/", 1)[0]

        sibling = None if is_decision else find_sibling(registry, model_id)
        if sibling is not None:
            ctx = sibling.context
            if ctx is not None and ctx.max_output_tokens:
                model.max_output_tokens = int(ctx.max_output_tokens)
            arch = getattr(sibling, "architecture", None)
            if arch is not None and getattr(arch, "family", None):
                model.architecture_family = arch.family
            extension["inherits_from"] = f"{sibling.provider}/{sibling.model_id}"

        if sibling is not None and states_capabilities(sibling):
            caps = sibling.capabilities
            model.supports_vision = bool(caps.vision)
            model.supports_reasoning = bool(caps.reasoning)
            model.supports_json_mode = bool(caps.json_mode)
            model.supports_structured_output = bool(caps.structured_output)
            model.supports_audio_input = bool(caps.audio_input)
            model.supports_video_input = bool(caps.video_input)
            extension["capabilities_from"] = extension["inherits_from"]
        elif not is_decision:
            model.supports_vision = any(p.search(model_id) for p in _VISION_FAMILIES) or bool(
                _VISION_NAME.search(model_id)
            )
            model.supports_reasoning = any(p.search(model_id) for p in _REASONING_FAMILIES)
            extension["capabilities_from"] = "family-defaults"

        if model_id.endswith(":free"):
            model.tags.append("free")
        model.raw_api_data["_extension"] = extension
        return model
