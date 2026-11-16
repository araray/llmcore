# tests/media/test_gemini_music_and_decisions.py
"""Gemini Lyria music and gemini-decision-1 decisions (live-verified 2026-11-16).

Live: lyria-3.5 returned ~1.5 MB of audio/mpeg plus a text part (section
markers); gemini-decision-1 answered noul/choice/score in TypeSafe's exact
shape (urgent 0.977, team billing, anger 1.63), with 0 output tokens.
"""

from __future__ import annotations

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("google.genai")

from llmcore.media.models import MediaCapability  # noqa: E402
from llmcore.providers.gemini_provider import GeminiProvider  # noqa: E402
from llmcore.providers.typesafe_provider import Box, Choice, Noul, Score  # noqa: E402


@pytest.fixture
def provider() -> GeminiProvider:
    p = GeminiProvider({"api_key": "k"})
    p._client = MagicMock()
    return p


async def test_lyria_music(provider):
    assert MediaCapability.MUSIC in provider.media_capabilities()
    response = NS(candidates=[NS(content=NS(parts=[
        NS(inline_data=None, text="<instrumental>"),
        NS(inline_data=NS(data=b"ID3mp3", mime_type="audio/mpeg"), text=None),
    ]))], usage_metadata=NS(prompt_token_count=13, candidates_token_count=302))
    provider._client.aio.models.generate_content = AsyncMock(return_value=response)
    result = await provider.generate_music_media("calm piano", duration_seconds=10)
    call = provider._client.aio.models.generate_content.await_args.kwargs
    assert call["model"] == "lyria-3.5" and "about 10 seconds" in call["contents"]
    assert result.artifact.data == b"ID3mp3" and result.artifact.mime_type == "audio/mpeg"
    assert result.artifact.provider_metadata == {"lyrics": "<instrumental>"}


async def test_decisions_share_typesafe_types(provider):
    response = MagicMock()
    response.model_dump.return_value = {
        "answers": {
            "urgent": {"type": "noul", "noul": 0.977, "confidence": 0.977},
            "team": {"type": "choice", "choice": "billing", "confidence": 0.99,
                     "probabilities": {"billing": 0.99, "tech": 0.01}},
            "anger": {"type": "score", "score": 1.63, "confidence": 0.63,
                      "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
                      "probabilities": {"0": 0.01, "1": 0.36, "2": 0.63}},
        },
        "usage": {"total_input_tokens": 199, "total_output_tokens": 0},
    }
    provider._client.aio.decisions.create = AsyncMock(return_value=response)
    result = await provider.system_one("charged twice!", {
        "urgent": Noul(instructions="Urgent?"),
        "team": Choice(instructions="Team?", criteria={"billing": None, "tech": "bugs"}),
        "anger": Score(instructions="Anger?", criteria=["calm", "annoyed", "furious"]),
    })
    sent = provider._client.aio.decisions.create.await_args.kwargs
    assert sent["model"] == "gemini-decision-1" and sent["input"] == "charged twice!"
    assert sent["questions"]["team"]["criteria"] == {"billing": "", "tech": "bugs"}  # Gemini needs strings
    assert result.nouls["urgent"].noul == 0.977 and result.choices["team"].choice == "billing"
    assert result.scores["anger"].legend[2] == "furious" and result.usage.input_tokens == 199


async def test_box_is_refused_before_any_request(provider):
    provider._client.aio.decisions.create = AsyncMock()
    with pytest.raises(ValueError, match="no 'box'"):
        await provider.create_decision("x", {"b": Box(instructions="where")})
    provider._client.aio.decisions.create.assert_not_awaited()
