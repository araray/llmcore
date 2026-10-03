# tests/providers/test_unknown_context_in_listings.py
"""A model listing must be able to say it does not know the context window.

Two live failures prompted this, both found by checking rather than assuming.

**Deepgram's listing raised `ValidationError`.** `ModelDetails.context_length`
was `int` with a default of 4,096 -- the same fabrication the card generator
used to make, one layer up -- so passing an explicit `None` was refused. Once
the 144 Deepgram cards stopped claiming a 128,000-token window (their input is
audio), the provider fed `None` straight in and `get_models_details()` failed
on the first model.

**HuggingFace's listing raised `AttributeError`, and always had.** It called
`registry.get_provider_cards()`, which does not exist on the registry, and
behind that first failure sat three more: `card.context.get(...)` and
`card.architecture.get(...)` treat pydantic models as dicts, and
`card.capabilities.tool_calling` is not a field -- it is `tool_use`. Nothing
covered the method, so four bugs sat in one function indefinitely.
"""

from __future__ import annotations

import pytest

from llmcore.model_cards import get_model_card_registry
from llmcore.models import ModelDetails


class TestUnknownIsRepresentable:
    def test_none_is_accepted(self):
        details = ModelDetails(id="m", provider_name="p", context_length=None)
        assert details.context_length is None

    def test_the_default_is_unknown_not_4096(self):
        """4,096 was a number with no referent. A caller cannot tell a real
        4k model from one nobody described, which is the whole problem."""
        assert ModelDetails(id="m", provider_name="p").context_length is None

    def test_a_real_window_still_round_trips(self):
        assert ModelDetails(id="m", provider_name="p", context_length=200_000).context_length == 200_000


class TestDeepgramListing:
    def test_every_card_can_become_model_details(self):
        """The exact shape the provider builds, without needing its SDK."""
        registry = get_model_card_registry()
        summaries = registry.list_cards(provider="deepgram")
        if not summaries:
            pytest.skip("no packaged deepgram cards")

        built = [
            ModelDetails(
                id=s.model_id,
                provider_name="deepgram",
                display_name=s.display_name,
                context_length=s.context_length,
                model_type=s.model_type,
            )
            for s in summaries
        ]
        assert len(built) == len(summaries)
        # The point of the fix: most of them legitimately say nothing.
        assert any(d.context_length is None for d in built)


class TestHuggingFaceListing:
    @pytest.mark.asyncio
    async def test_it_lists_models_at_all(self):
        """It raised AttributeError on its first line before this."""
        from llmcore.providers.huggingface_provider import HuggingFaceProvider

        rows = await HuggingFaceProvider({"api_key": "test"}).get_models_details()
        assert len(rows) > 100

    @pytest.mark.asyncio
    async def test_it_reads_pydantic_fields_not_dict_keys(self):
        """`card.context.get("max_output_tokens")` and
        `card.architecture.get("family")` were AttributeErrors waiting behind
        the first one: both are models, not mappings."""
        from llmcore.providers.huggingface_provider import HuggingFaceProvider

        rows = await HuggingFaceProvider({"api_key": "test"}).get_models_details()
        # Nothing asserted about the values -- only that reaching them works
        # for every card, which is what the dict-style access prevented.
        assert all(
            d.max_output_tokens is None or isinstance(d.max_output_tokens, int)
            for d in rows
        )
        assert all(d.family is None or isinstance(d.family, str) for d in rows)

    @pytest.mark.asyncio
    async def test_tool_support_comes_from_a_field_that_exists(self):
        """`capabilities.tool_calling` is not a field. `tool_use` is."""
        from llmcore.providers.huggingface_provider import HuggingFaceProvider

        rows = await HuggingFaceProvider({"api_key": "test"}).get_models_details()
        assert all(isinstance(d.supports_tools, bool) for d in rows)

    @pytest.mark.asyncio
    async def test_a_card_with_no_window_lists_as_unknown(self):
        """83 of these repos are gated or have no config.json. Quoting a
        number for them would be inventing one."""
        from llmcore.providers.huggingface_provider import HuggingFaceProvider

        rows = await HuggingFaceProvider({"api_key": "test"}).get_models_details()
        assert any(d.context_length is None for d in rows)
        assert any(d.context_length for d in rows)


class TestConsumersHandleUnknown:
    @pytest.mark.asyncio
    async def test_memory_falls_through_to_the_provider_fallback(self):
        """`_get_precise_context_length` promises an int and callers size a
        context budget with it, so an unknown listing must not propagate."""
        from llmcore.memory.manager import MemoryManager

        class Provider:
            async def get_models_details(self):
                return [ModelDetails(id="m", provider_name="p", context_length=None)]

            def get_max_context_length(self, model=None):
                return 8192

        got = await MemoryManager._get_precise_context_length(
            object.__new__(MemoryManager), Provider(), "m"
        )
        assert got == 8192

    @pytest.mark.asyncio
    async def test_a_known_window_is_still_preferred(self):
        from llmcore.memory.manager import MemoryManager

        class Provider:
            async def get_models_details(self):
                return [ModelDetails(id="m", provider_name="p", context_length=262_144)]

            def get_max_context_length(self, model=None):
                return 8192

        got = await MemoryManager._get_precise_context_length(
            object.__new__(MemoryManager), Provider(), "m"
        )
        assert got == 262_144

    def test_the_grpc_bridge_sends_zero_for_unknown(self):
        """Proto int fields cannot carry None, and 0 is how this wire format
        already spells unset -- the media providers pass 0 for models with no
        token window at all."""
        pytest.importorskip("llmcore.bridge.catalog_pb2")
        from llmcore.bridge.core import _model_details_to_proto

        proto = _model_details_to_proto(
            ModelDetails(id="m", provider_name="p", context_length=None)
        )
        assert proto.context_length == 0
