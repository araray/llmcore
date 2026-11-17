# tests/providers/test_deepinfra_current_api.py
"""DeepInfra against the live catalogue (2026-11-16).

The /v1/models tags are short (embed, tts, stt, image-gen, video-gen, vlm,
vision, reasoning); none of the old long forms matched, so every model was
classified as chat with tools. openai/whisper-large (the old STT default)
answers 404. The openrouter-format feed moved to schema 2.4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("openai")

from llmcore.providers.deepinfra_provider import (  # noqa: E402
    DEFAULT_DEEPINFRA_STT_MODEL,
    DeepInfraProvider,
)


@pytest.mark.parametrize(
    ("tags", "model_type", "vision", "tools"),
    [(["embed"], "embedding", False, False), (["tts"], "tts", False, False),
     (["stt"], "stt", False, False), (["image-gen"], "image-generation", False, False),
     (["video-gen"], "text-to-video", False, False),
     (["chat", "vlm", "vision", "reasoning"], "chat", True, True)],
)
def test_live_tags_classify(tags, model_type, vision, tools):
    assert DeepInfraProvider._classify_model("org/model", tags) == (model_type, vision, tools)


def test_stt_default_is_served():
    assert DEFAULT_DEEPINFRA_STT_MODEL == "openai/whisper-large-v3-turbo"


def test_cardctl_reads_schema_24():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    from cardctl.adapters.deepinfra_adapter import DeepInfraAdapter

    entry = {
        "schema_version": "2.4", "id": "deepseek-ai/DeepSeek-V4-Pro", "name": "DeepSeek-V4-Pro",
        "input_modalities": [{"type": "text", "supported_inputs": {"max_context_length": {"value": 1048576}},
                              "pricing": [{"type": "prompt", "cost_usd": "0.0000013"},
                                          {"type": "cached_prompt", "cost_usd": "0.0000001"}]}],
        "output_modalities": [{"type": "text", "max_length": {"value": 393216},
                               "supported_parameters": {"tools": {"type": "boolean"}, "reasoning": {"type": "boolean"},
                                                        "structured_outputs": {"type": "boolean"}},
                               "pricing": [{"type": "completion", "cost_usd": "0.0000026"}]}],
    }
    model = DeepInfraAdapter.__new__(DeepInfraAdapter)._parse_model(entry)
    assert (model.context_length, model.max_output_tokens) == (1048576, 393216)
    assert model.supports_tools and model.supports_reasoning and model.supports_structured_output
    assert model.raw_api_data["_pricing"] == {"input": 1.3, "output": 2.6, "cached_input": 0.1}
    assert model.tags == ["text", "output:text"]
