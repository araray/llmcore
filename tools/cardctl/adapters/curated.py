# tools/cardctl/adapters/curated.py
"""Adapter base for providers that publish no model catalog.

Most providers answer ``GET /v1/models``. Several of the media providers llmcore
ships do not, and for a structural reason rather than an oversight:

* **fal** and **Higgsfield** address a model *by endpoint path*. There are
  thousands (fal) or a console-managed set (Higgsfield), and no listing route.
* **Replicate** has tens of thousands of community models; enumerating them
  would produce a card dump, not a catalog.
* **vLLM** serves whatever a given deployment loaded, so "the catalog" depends
  on a server that may not be running when cards are generated.

For these, a *curated* set is the honest answer: cards are emitted for the model
paths llmcore actually ships as per-capability defaults, marked as curated so
nobody mistakes them for a discovered catalog. A card that says "this is the
default llmcore will call, and here is what it does" is useful; a card that
implies llmcore enumerated a marketplace would be a lie.

Subclasses declare :attr:`CuratedAdapter.curated_models` and nothing else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .base import BaseAdapter, NormalizedModel

logger = logging.getLogger(__name__)

__all__ = ["CuratedAdapter", "CuratedModel"]


@dataclass(frozen=True)
class CuratedModel:
    """One hand-maintained model entry.

    Attributes:
        model_id: Provider-side id or endpoint path.
        model_type: Card model type (``image-generation``, ``video-generation``,
            ``tts``, ``stt``, ``media``, ...).
        capability: The llmcore ``MediaCapability`` this serves, when it maps to
            exactly one.
        display_name: Human-readable name.
        description: What it is and why it is the default.
        caps: Media capability flags to set, by ``NormalizedModel`` suffix —
            e.g. ``("image_generation",)``.
        tags: Card tags.
        owned_by: Upstream owner, when the host is a marketplace.
        notes: Extra raw data recorded on the card.
    """

    model_id: str
    model_type: str = "media"
    capability: str | None = None
    display_name: str | None = None
    description: str | None = None
    caps: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    owned_by: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


class CuratedAdapter(BaseAdapter):
    """Emit cards from a declared model set rather than from a catalog API.

    Requires no API key by default: there is nothing to call. Subclasses that
    *can* enrich entries from a live per-model endpoint (Replicate's schema
    route, for instance) override :meth:`enrich` and set
    ``requires_api_key``.
    """

    requires_api_key: bool = False

    #: The declared set. Override in subclasses.
    curated_models: tuple[CuratedModel, ...] = ()

    async def fetch_models(self) -> list[NormalizedModel]:
        """Return a normalized model for each curated entry."""
        if not self.curated_models:
            logger.warning(
                "Adapter for '%s' declares no curated models, so no cards will be "
                "written. This is almost certainly a mistake.",
                self.provider_name,
            )
            return []

        models: list[NormalizedModel] = []
        for entry in self.curated_models:
            model = NormalizedModel(
                model_id=entry.model_id,
                provider=self.provider_name,
                display_name=entry.display_name or entry.model_id.rsplit("/", 1)[-1],
                description=entry.description,
                model_type=entry.model_type,
                owned_by=entry.owned_by,
                # Curation is recorded on the card itself: a reader must be able
                # to tell this was declared by llmcore, not discovered from an
                # API, so a stale entry is debuggable rather than mysterious.
                tags=[*entry.tags, "curated"],
                raw_api_data={
                    "curated": True,
                    "source": f"llmcore {self.provider_name} provider defaults",
                    "llmcore_capability": entry.capability,
                    **entry.notes,
                },
            )
            for cap in entry.caps:
                attr = f"supports_{cap}"
                if not hasattr(model, attr):
                    raise ValueError(
                        f"{type(self).__name__} declares unknown capability {cap!r} "
                        f"for {entry.model_id!r}."
                    )
                setattr(model, attr, True)
            models.append(await self.enrich(model, entry))
        return models

    async def enrich(self, model: NormalizedModel, entry: CuratedModel) -> NormalizedModel:
        """Hook for adapters that can add live metadata. Identity by default."""
        return model
