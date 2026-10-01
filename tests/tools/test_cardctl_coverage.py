# tests/tools/test_cardctl_coverage.py
"""cardctl must cover every provider llmcore ships.

This file exists because of a concrete failure. The media providers (`fal`,
`elevenlabs`, `replicate`) and the Higgsfield provider were added to
``PROVIDER_MAP`` and shipped **without** cardctl adapters, and nothing
complained: ``generate`` only reports on the provider you name, and ``stats``
only sees providers that already have cards, so a provider with no adapter was
invisible to both.

:class:`TestEveryProviderHasAnAdapter` is the guard that makes that class of
omission fail loudly instead of silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cardctl.adapters import (  # noqa: E402
    _ADAPTER_REGISTRY,
    get_adapter,
    list_providers,
)
from tools.cardctl.adapters.base import NormalizedModel  # noqa: E402
from tools.cardctl.adapters.curated import CuratedAdapter, CuratedModel  # noqa: E402
from tools.cardctl.core.common import (  # noqa: E402
    canonical_provider_dir_name,
    cards_dir_for_provider,
)


def _canonical_providers() -> set[str]:
    from llmcore.providers.manager import _PROVIDER_INSTANCE_ALIASES, PROVIDER_MAP

    # Several tests inject doubles with `monkeypatch.setitem(PROVIDER_MAP, ...)`.
    # Teardown normally removes them, but a guard that reads the live registry
    # should not depend on that — so only classes that actually live in
    # llmcore.providers count. A leaked "fake" provider then cannot make this
    # report a missing adapter, while a real unadaptered provider still does.
    return {
        name
        for name, cls in PROVIDER_MAP.items()
        if name not in _PROVIDER_INSTANCE_ALIASES
        and getattr(cls, "__module__", "").startswith("llmcore.providers")
    }


class TestEveryProviderHasAnAdapter:
    """The regression guard for the gap that shipped three providers uncarded."""

    def test_no_registered_provider_is_missing_an_adapter(self):
        missing = sorted(_canonical_providers() - set(_ADAPTER_REGISTRY))
        assert missing == [], (
            f"These providers are in PROVIDER_MAP but have no cardctl adapter, so "
            f"their model cards can never be generated: {missing}. Add "
            f"tools/cardctl/adapters/<name>_adapter.py and register it in "
            f"tools/cardctl/adapters/__init__.py."
        )

    @pytest.mark.parametrize("provider", sorted(_canonical_providers()))
    def test_every_adapter_instantiates(self, provider):
        """A registry entry pointing at a broken import is as bad as no entry."""
        adapter = get_adapter(provider)
        assert adapter.provider_name
        assert isinstance(adapter.requires_api_key, bool)

    @pytest.mark.parametrize("provider", sorted(_canonical_providers()))
    def test_key_requiring_adapters_name_their_env_var(self, provider):
        adapter = get_adapter(provider)
        if adapter.requires_api_key:
            assert adapter.api_key_env_var, (
                f"{provider} requires a key but names no environment variable, so "
                f"the error a user gets cannot tell them what to set."
            )

    def test_the_media_providers_specifically_are_covered(self):
        """Named explicitly: these are the ones that were missed."""
        for provider in ("fal", "elevenlabs", "replicate", "higgsfield", "deepgram"):
            assert provider in _ADAPTER_REGISTRY


class TestCardDirectoryAliasing:
    """An alias must not create a second card tree."""

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("gemini", "google"),
            ("moonshot", "kimi"),
            ("friendliai", "friendli"),
            ("glm", "zai"),
            ("fal_ai", "fal"),
            ("eleven_labs", "elevenlabs"),
            ("jev", "typesafe"),
        ],
    )
    def test_aliases_resolve_to_the_canonical_directory(self, alias, canonical):
        """Found live: `cardctl generate gemini` built a duplicate gemini/ tree
        alongside google/, because the directory is named after whatever string
        the caller passed."""
        assert canonical_provider_dir_name(alias) == canonical
        assert cards_dir_for_provider(alias).name == canonical

    def test_a_canonical_name_is_unchanged(self):
        assert canonical_provider_dir_name("openai") == "openai"

    def test_resolution_is_case_insensitive(self):
        assert canonical_provider_dir_name("Gemini") == "google"

    def test_no_duplicate_card_trees_exist_on_disk(self):
        """The directories that an alias would have created must not be present."""
        root = cards_dir_for_provider("openai").parent
        for alias in ("gemini", "moonshot", "friendliai", "glm", "fal_ai"):
            assert not (root / alias).exists(), (
                f"{alias}/ exists as its own card tree; it should share the "
                f"canonical provider's directory."
            )


class TestCuratedAdapters:
    """Providers with no catalog endpoint declare a curated set instead."""

    CURATED = ("fal", "higgsfield", "replicate")

    @pytest.mark.parametrize("provider", CURATED)
    def test_curated_adapters_declare_models(self, provider):
        adapter = get_adapter(provider)
        assert isinstance(adapter, CuratedAdapter)
        assert adapter.curated_models, f"{provider} declares an empty curated set"

    @pytest.mark.parametrize("provider", CURATED)
    async def test_curated_sets_produce_normalized_models(self, provider):
        adapter = get_adapter(provider)
        # Replicate enriches over the network when a token is present; drop it so
        # this stays offline and deterministic.
        adapter._api_key = None
        import os

        saved = os.environ.pop(adapter.api_key_env_var, None)
        try:
            models = await adapter.fetch_models()
        finally:
            if saved is not None:
                os.environ[adapter.api_key_env_var] = saved

        assert models
        assert all(m.provider == adapter.provider_name for m in models)
        assert all("curated" in m.tags for m in models), (
            "a curated card must say so, or a stale entry looks like a discovery"
        )

    @pytest.mark.parametrize("provider", CURATED)
    async def test_every_curated_model_declares_a_capability(self, provider):
        """A media card claiming no capabilities reads as 'does nothing'."""
        adapter = get_adapter(provider)
        media_flags = [
            f for f in vars(NormalizedModel("x", "y")) if f.startswith("supports_")
        ]
        for entry in adapter.curated_models:
            assert entry.caps, f"{entry.model_id} declares no capability flags"
            for cap in entry.caps:
                assert f"supports_{cap}" in media_flags, (
                    f"{entry.model_id} declares unknown capability {cap!r}"
                )

    async def test_an_unknown_capability_fails_loudly(self):
        class Bad(CuratedAdapter):
            provider_name = "bad"
            curated_models = (CuratedModel(model_id="a/b", caps=("teleportation",)),)

        with pytest.raises(ValueError, match="unknown capability"):
            await Bad().fetch_models()

    async def test_an_empty_curated_set_yields_nothing(self):
        class Empty(CuratedAdapter):
            provider_name = "empty"

        assert await Empty().fetch_models() == []


class TestMediaCapabilityPlumbing:
    """Media flags must survive the whole path into a card."""

    def test_normalized_model_carries_media_flags(self):
        model = NormalizedModel("m", "p", supports_video_generation=True)
        assert model.supports_video_generation is True
        assert model.supports_music_generation is False

    def test_the_card_schema_accepts_them(self):
        from llmcore.model_cards.schema import ModelCapabilities, ModelType

        caps = ModelCapabilities(video_generation=True, transcription=True)
        assert caps.video_generation and caps.transcription
        assert ModelType.VIDEO_GENERATION.value == "video-generation"

    def test_the_builder_maps_them_onto_the_card(self):
        """Without this the media cards claimed no capabilities at all."""
        from tools.cardctl.core.builder import CardBuilder
        from tools.cardctl.core.enrichment import EnrichmentStore

        builder = CardBuilder("fal", EnrichmentStore.load("fal"))
        model = NormalizedModel(
            "fal-ai/film",
            "fal",
            model_type="video-generation",
            supports_video_interpolation=True,
        )
        card = builder.build(model)
        assert card["capabilities"]["video_interpolation"] is True
        assert card["capabilities"]["video_generation"] is False


class TestGeneratedMediaCards:
    """The cards actually on disk, as a reviewer would read them."""

    @pytest.mark.parametrize(
        ("provider", "expected_type"),
        [("fal", "image-generation"), ("higgsfield", "image-generation"),
         ("elevenlabs", "tts"), ("deepgram", "stt"), ("replicate", "image-generation")],
    )
    def test_cards_exist_and_declare_capabilities(self, provider, expected_type):
        import json

        card_dir = cards_dir_for_provider(provider)
        if not card_dir.is_dir():
            pytest.skip(f"no cards generated for {provider} in this checkout")
        cards = [json.loads(p.read_text()) for p in card_dir.glob("*.json")]
        assert cards, f"{provider} has a card directory but no cards"
        assert any(c.get("model_type") == expected_type for c in cards)
        # Every media card must claim at least one capability.
        for card in cards:
            assert any(card["capabilities"].values()), (
                f"{provider}/{card['model_id']} claims no capabilities at all"
            )

    def test_deepgram_cards_are_not_duplicated_per_language(self):
        """Found live: Deepgram lists one record per model AND language, so an
        ungrouped adapter wrote 553 records into 144 files."""
        import json

        card_dir = cards_dir_for_provider("deepgram")
        if not card_dir.is_dir():
            pytest.skip("no deepgram cards in this checkout")
        ids = [json.loads(p.read_text())["model_id"] for p in card_dir.glob("*.json")]
        assert len(ids) == len(set(ids))


class TestRegistryHygiene:
    def test_list_providers_is_sorted_and_unique(self):
        provs = list_providers()
        assert provs == sorted(set(provs))

    def test_an_unknown_provider_lists_the_alternatives(self):
        with pytest.raises(ValueError, match="Available:"):
            get_adapter("not-a-provider")
