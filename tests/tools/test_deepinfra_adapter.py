# tests/tools/test_deepinfra_adapter.py
"""Tests for the cardctl DeepInfra adapter (tools/cardctl/adapters/deepinfra_adapter.py).

Covers:
* registration in the adapter registry,
* ``_parse_model`` capability / model-type / pricing / extension mapping,
* the per-token -> per-million pricing conversion,
* end-to-end card building + validation (which also exercises the
  ``CardBuilder._build_pricing`` live-pricing fallback), and
* ``fetch_models`` against a mocked ``/openrouter/models`` endpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest
import respx

# Ensure repo root + src are importable regardless of how pytest is invoked.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.cardctl.adapters import get_adapter, list_providers  # noqa: E402
from tools.cardctl.adapters.deepinfra_adapter import (  # noqa: E402
    DeepInfraAdapter,
    _per_million,
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_registered_in_registry():
    assert "deepinfra" in list_providers()
    adapter = get_adapter("deepinfra")
    assert isinstance(adapter, DeepInfraAdapter)
    assert adapter.provider_name == "deepinfra"
    # /openrouter/models is public — generation must not require a key.
    assert adapter.requires_api_key is False
    assert adapter.base_url == "https://api.deepinfra.com"
    assert adapter.models_endpoint == "/openrouter/models"


# ---------------------------------------------------------------------------
# Pricing helper
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.0000002", 0.2),
        ("0.0000006", 0.6),
        (0.0000001, 0.1),
        (None, 0.0),
        ("not-a-number", 0.0),
    ],
)
def test_per_million(raw, expected):
    assert _per_million(raw) == expected


# ---------------------------------------------------------------------------
# _parse_model
# ---------------------------------------------------------------------------
@pytest.fixture
def adapter():
    return DeepInfraAdapter()


def test_parse_chat_vision_model(adapter):
    nm = adapter._parse_model(
        {
            "id": "Qwen/Qwen2.5-VL-32B-Instruct",
            "name": "Qwen2.5 VL 32B",
            "created": 1715300000,
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
            "supported_features": ["tools", "response_format", "structured_outputs"],
            "context_length": 128000,
            "max_output_length": 16384,
            "quantization": "bf16",
            "hugging_face_id": "Qwen/Qwen2.5-VL-32B-Instruct",
            "pricing": {
                "prompt": "0.0000002",
                "completion": "0.0000006",
                "input_cache_read": "0.0000001",
            },
            "datacenters": [{"country_code": "US"}, {"country_code": "FI"}],
        }
    )
    assert nm.model_type == "chat"
    assert nm.supports_vision is True
    assert nm.supports_tools is True
    assert nm.supports_json_mode is True
    assert nm.supports_structured_output is True
    assert nm.context_length == 128000
    assert nm.max_output_tokens == 16384
    assert nm.open_weights is True
    # Pricing converted to per-million USD and stashed for the builder.
    assert nm.raw_api_data["_pricing"] == {
        "input": 0.2,
        "output": 0.6,
        "cached_input": 0.1,
    }
    ext = nm.raw_api_data["_extension"]
    assert ext["hugging_face_id"] == "Qwen/Qwen2.5-VL-32B-Instruct"
    assert ext["quantization"] == "bf16"
    assert ext["datacenters"] == ["US", "FI"]
    assert ext["input_modalities"] == ["text", "image"]


def test_parse_image_model(adapter):
    nm = adapter._parse_model(
        {
            "id": "black-forest-labs/FLUX-1-schnell",
            "input_modalities": ["text"],
            "output_modalities": ["image"],
            "supported_features": [],
        }
    )
    assert nm.model_type == "image-generation"
    assert nm.supports_vision is False


def test_parse_tts_model(adapter):
    nm = adapter._parse_model(
        {
            "id": "hexgrad/Kokoro-82M",
            "input_modalities": ["text"],
            "output_modalities": ["audio"],
        }
    )
    assert nm.model_type == "tts"


def test_parse_deprecated_model(adapter):
    nm = adapter._parse_model(
        {
            "id": "old/model",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "deprecation_date": "2025-01-01",
        }
    )
    assert nm.is_deprecated is True
    assert nm.deprecation_date == "2025-01-01"


def test_parse_no_id_returns_none(adapter):
    assert adapter._parse_model({"name": "missing id"}) is None


# ---------------------------------------------------------------------------
# Builder integration (exercises CardBuilder live-pricing fallback)
# ---------------------------------------------------------------------------
def test_build_card_with_live_pricing(adapter):
    from tools.cardctl.core.builder import CardBuilder
    from tools.cardctl.core.enrichment import EnrichmentStore
    from tools.cardctl.core.validator import validate_card_dict

    nm = adapter._parse_model(
        {
            "id": "meta-llama/Llama-3.3-70B-Instruct",
            "name": "Llama 3.3 70B",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "supported_features": ["tools"],
            "context_length": 131072,
            "max_output_length": 16384,
            "pricing": {"prompt": "0.00000023", "completion": "0.0000004"},
        }
    )
    store = EnrichmentStore.load("deepinfra")
    card = CardBuilder("deepinfra", store).build(nm)

    res = validate_card_dict(card)
    assert res.valid, res.error
    assert card["provider"] == "deepinfra"
    assert card["model_type"] == "chat"
    # Live pricing flowed through the builder into the card.
    pm = card["pricing"]["per_million_tokens"]
    assert pm["input"] == 0.23
    assert pm["output"] == 0.4


# ---------------------------------------------------------------------------
# fetch_models (mocked /openrouter/models)
# ---------------------------------------------------------------------------
@respx.mock
async def test_fetch_models(adapter, monkeypatch):
    monkeypatch.delenv("DEEPINFRA_TOKEN", raising=False)
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    respx.get("https://api.deepinfra.com/openrouter/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "deepseek-ai/DeepSeek-V3",
                        "name": "DeepSeek V3",
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                        "supported_features": ["tools", "response_format"],
                        "context_length": 131072,
                        "max_output_length": 16384,
                        "pricing": {"prompt": "0.0000003", "completion": "0.0000005"},
                    },
                    {"name": "no-id-entry"},  # must be skipped
                ]
            },
        )
    )
    models = await adapter.fetch_models()
    assert len(models) == 1
    m = models[0]
    assert m.model_id == "deepseek-ai/DeepSeek-V3"
    assert m.supports_tools is True
    assert m.raw_api_data["_pricing"] == {"input": 0.3, "output": 0.5}


def test_get_api_key_alias(monkeypatch, adapter):
    monkeypatch.delenv("DEEPINFRA_TOKEN", raising=False)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "alias-key")
    assert adapter.get_api_key() == "alias-key"
