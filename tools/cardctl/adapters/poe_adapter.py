# tools/cardctl/adapters/poe_adapter.py
"""Poe model discovery adapter.

Uses the Poe bot settings endpoint to discover available models.
The Poe API uses OpenAI-compatible chat completions but has its own
model listing mechanism.

Context Length Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~
``/v1/models`` now reports ``context_window.context_length`` and
``max_output_tokens`` (live, 2026-11-16), plus ``reasoning``,
``parameters``, ``supported_features`` and ``supported_endpoints``. Older
fallbacks remain for bots without them:

1. API field ``context_window.context_length`` / ``context_length``.
2. Parse the model description for ``Context window: Nk`` patterns.
3. Model family heuristic (maps known model families to upstream limits).
4. Conservative fallback: 128,000 tokens.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from .base import BaseAdapter, NormalizedModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context length: description parsing
# ---------------------------------------------------------------------------

# Patterns to extract context window from Poe model descriptions.
# Descriptions often contain text like:
#   "Context window: 205k tokens"
#   "Context Window: 1,000,000"
#   "Supported context length: 198k tokens"
#   "128k context window"
_CTX_PATTERNS: list[re.Pattern[str]] = [
    # "Context window: 205k tokens" / "Context Window: 1,000,000"
    re.compile(
        r"[Cc]ontext\s+[Ww]indow[:\s]+([0-9][0-9,]*\.?\d*)\s*([kKmM])?\s*(?:tokens?)?",
    ),
    # "Context length: 198k" / "Supported context length: 198k tokens"
    re.compile(
        r"[Cc]ontext\s+[Ll]ength[:\s]+([0-9][0-9,]*\.?\d*)\s*([kKmM])?\s*(?:tokens?)?",
    ),
    # "128k context window" / "1M token context"
    re.compile(
        r"([0-9][0-9,]*\.?\d*)\s*([kKmM])\s*(?:tokens?\s+)?context",
    ),
]


def _parse_context_from_description(description: str) -> int | None:
    """Extract context window size from a Poe model description.

    Tries multiple regex patterns and returns the first match, normalized
    to an integer token count.  Returns None if no pattern matches.
    """
    for pattern in _CTX_PATTERNS:
        match = pattern.search(description)
        if match:
            raw_number = match.group(1).replace(",", "")
            suffix = (match.group(2) or "").lower()
            try:
                value = float(raw_number)
            except ValueError:
                continue
            if suffix == "k":
                return int(value * 1_000)
            elif suffix == "m":
                return int(value * 1_000_000)
            else:
                return int(value)
    return None


# ---------------------------------------------------------------------------
# Context length: model family heuristic
# ---------------------------------------------------------------------------


def _guess_context_length(model_id: str) -> int | None:
    """Heuristic context length based on known model families.

    Since Poe doesn't expose context length in its ``/v1/models`` response,
    we map well-known bot name prefixes to their upstream provider's context
    limits.

    Args:
        model_id: The Poe bot name.

    Returns:
        Best-guess context length in tokens.
    """
    lower = model_id.lower()

    # GPT-5.x family
    if "gpt-5" in lower:
        return 400_000

    # GPT-4.1 family (1M context)
    if "gpt-4.1" in lower:
        return 1_048_576

    # GPT-4o family
    if "gpt-4o" in lower:
        return 128_000

    # Claude families (all 200K input)
    if "claude" in lower:
        return 200_000

    # Gemini 2.5+ and 3.x (1M context)
    if "gemini" in lower:
        if any(v in lower for v in ("3", "2.5", "2.0")):
            return 1_000_000
        return 128_000

    # GLM family (200K context)
    if "glm" in lower:
        return 202_752

    # Qwen family
    if "qwen" in lower:
        # qwen3.5+ Plus/Max variants have 1M
        if any(v in lower for v in ("plus", "max", "omni")):
            return 1_000_000
        # qwen3 coder variants: 256K
        if "coder" in lower:
            return 262_144
        # General qwen3.5+: 262K
        if any(v in lower for v in ("3.5", "3.6")):
            return 262_144
        return 128_000

    # Kimi/Moonshot (256K)
    if "kimi" in lower:
        return 262_144

    # DeepSeek (128K standard)
    if "deepseek" in lower:
        return 128_000

    # MiniMax M2 family (200K)
    if "minimax" in lower:
        return 200_000

    # Grok family
    if "grok" in lower:
        if any(v in lower for v in ("4.1", "4-fast")):
            return 2_000_000
        return 131_072

    # o-series reasoning models
    if lower.startswith(("o1", "o3", "o4")):
        return 200_000

    # Llama
    if "llama" in lower:
        return 128_000

    # Mistral / Mixtral
    if "mistral" in lower or "mixtral" in lower:
        if "small-4" in lower:
            return 256_000
        return 128_000

    # Mimo
    if "mimo" in lower:
        if "pro" in lower:
            return 1_000_000
        return 262_144

    # Seed (ByteDance)
    if "seed" in lower and "seed-2" in lower:
        return 256_000

    # Gemma 4 (262K)
    if "gemma-4" in lower or "gemma4" in lower:
        return 262_144
    if "gemma" in lower:
        return 128_000

    # Image/Video/Audio generation bots (context doesn't apply)
    _media_prefixes = (
        "sora",
        "veo",
        "dall-e",
        "imagen",
        "flux",
        "sdxl",
        "stable",
        "kling",
        "hailuo",
        "runway",
        "pika",
        "pixverse",
        "wan",
        "seedance",
        "seedream",
        "seededit",
        "hidream",
        "hunyuan",
        "mochi",
        "ltx",
        "ray2",
        "vidu",
        "luma",
        "dreamina",
        "ideogram",
        "recraft",
        "bria",
        "clarity",
        "omnihuman",
        "liveportrait",
        "sketch-to",
        "remove-background",
        "topaz",
        "trellis",
        "nano-banana",
        "z-image",
        "elevenlabs",
        "lyria",
        "orpheus",
        "cartesia",
        "unreal-speech",
        "stable-audio",
        "sonic",
        "deepgram",
        "whisper",
    )
    if any(t in lower for t in _media_prefixes):
        # A media endpoint has no text-token input window. The old 4,096 here
        # was a number with no referent: nothing publishes it, and a caller
        # reading it would believe a prompt budget exists where none does.
        return None

    # Unknown. Deliberately not a "conservative default": the previous
    # 128,000 made every unrecognised model claim a window it was never
    # told it had, and a plausible number is read as authoritative in a way
    # an absent one is not. ``None`` means the card carries no context
    # block, which llmcore reads as unknown.
    return None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class PoeAdapter(BaseAdapter):
    provider_name = "poe"
    api_key_env_var = "POE_API_KEY"
    base_url = "https://api.poe.com/v1/models"

    async def fetch_models(self) -> list[NormalizedModel]:
        self.check_api_key()
        api_key = self.get_api_key()

        headers = {"Authorization": f"Bearer {api_key}"}
        result: list[NormalizedModel] = []

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            try:
                resp = await client.get(self.base_url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                models_list = data.get("data", [])
            except Exception as e:
                logger.warning("Poe model listing failed: %s", e)
                return []

        for m in models_list:
            normalized = self._parse_model(m)
            if normalized:
                result.append(normalized)

        logger.info("Fetched %d models from poe", len(result))
        return result

    def _parse_model(self, m: dict[str, Any]) -> NormalizedModel | None:
        model_id = m.get("id", "")
        if not model_id:
            return None

        description = m.get("description", "") or ""
        architecture = m.get("architecture", {}) or {}
        input_mods = architecture.get("input_modalities", []) or []
        output_mods = architecture.get("output_modalities", []) or []
        window = m.get("context_window") or {}
        features = m.get("supported_features") or []
        endpoints = m.get("supported_endpoints") or []

        # Video/image/audio bots are bots too; type them by what they output.
        model_type = "chat"
        if "text" not in output_mods and output_mods:
            if "video" in output_mods or "/v1/videos" in endpoints:
                model_type = "video-generation"
            elif "image" in output_mods:
                model_type = "image-generation"
            elif "audio" in output_mods:
                model_type = "tts"

        # Resolve context length: API → description → heuristic
        context_length = window.get("context_length") or m.get("context_length")
        if not context_length:
            context_length = _parse_context_from_description(description)
        if not context_length:
            context_length = _guess_context_length(model_id)

        created = m.get("created")
        if isinstance(created, (int, float)) and created > 10**11:
            created = int(created) // 1000  # Poe reports milliseconds
        normalized = NormalizedModel(
            model_id=model_id,
            provider="poe",
            display_name=(m.get("metadata") or {}).get("display_name") or m.get("name") or model_id,
            description=description,
            model_type=model_type,
            context_length=context_length if model_type == "chat" else None,
            max_output_tokens=window.get("max_output_tokens"),
            supports_streaming=True,
            supports_vision="image" in input_mods,
            supports_audio_input="audio" in input_mods,
            supports_tools="tools" in features,
            supports_reasoning=bool(m.get("reasoning")),
            owned_by=m.get("owned_by"),
            architecture_family=m.get("owned_by"),
            created_timestamp=created,
            raw_api_data=m,
        )

        return normalized
