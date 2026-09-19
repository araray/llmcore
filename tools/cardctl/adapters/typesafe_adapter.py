"""TypeSafe.ai model discovery adapter.

TypeSafe's ``GET /v1/models`` lists the *aliases* an account may send in the
``model`` field (``jev-latest``, ``jev-preview``) with a description and a
release date; it does not expose versioned ids, context limits, pricing, or
capabilities. The adapter therefore:

* collapses the listed aliases onto the versioned id they currently resolve
  to (``_ALIAS_TO_VERSION``; update it when Jev bumps), emitting one
  :class:`NormalizedModel` per versioned id with the aliases attached, and
* fills the static facts published on https://docs.typesafe.ai/models (context
  budget, pricing, rate limits, question types) so a regenerated card matches
  the hand-written ``default_cards/typesafe/jev-1.13.0.json``.

Unknown names (a future ``jev-2-latest`` etc.) become their own
``model_type="decision"`` entries so they are never silently dropped.

Canonical provider key: ``typesafe`` (default_cards/typesafe/). Also reachable
via the ``jev`` alias in the registry.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import BaseAdapter, NormalizedModel

logger = logging.getLogger(__name__)

#: Alias -> versioned id, as documented at https://docs.typesafe.ai/models.
#: ``/v1/systemone`` responses echo the versioned id in ``model``.
_ALIAS_TO_VERSION: dict[str, str] = {
    "jev-latest": "jev-1.13.0",
    "jev-preview": "jev-1.13.0",
}

#: Static facts per versioned id (docs.typesafe.ai/models, 2026-09).
_MODEL_FACTS: dict[str, dict[str, Any]] = {
    "jev-1.13.0": {
        "display_name": "Jev 1.13",
        "context_length": 65_536,
        "pricing": {"input": 0.042, "output": 0.0},
        "extension": {
            "endpoint": "/v1/systemone",
            "question_types": ["noul", "choice", "score"],
            "state_plus_longest_question_max_tokens": 32_768,
            "tokens_per_second_limit": 250_000,
            "requests_per_minute_limit": 1_200,
            "input_modalities": ["text"],
            "primary_language": "en",
            "output_tokens_billed": False,
        },
    },
}

_TAGS = ["decision", "classification", "scoring", "calibrated-probabilities", "text", "output:json"]


class TypeSafeAdapter(BaseAdapter):
    provider_name = "typesafe"
    api_key_env_var = "TYPESAFE_API_KEY"
    base_url = "https://api.typesafe.ai"
    models_endpoint = "/v1/models"

    async def fetch_models(self) -> list[NormalizedModel]:
        self.check_api_key()
        headers = {"Authorization": f"Bearer {self.get_api_key()}", "Accept": "application/json"}
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(f"{self.base_url}{self.models_endpoint}", headers=headers)
            resp.raise_for_status()
            data = resp.json()
        entries = data.get("models", []) if isinstance(data, dict) else []
        result = self._normalize_entries(entries)
        logger.info("Fetched %d models from typesafe (%d listed names)", len(result), len(entries))
        return result

    def _normalize_entries(self, entries: list[dict[str, Any]]) -> list[NormalizedModel]:
        """Collapse listed aliases onto versioned ids; keep unknown names as-is."""
        by_id: dict[str, NormalizedModel] = {}
        for entry in entries:
            name = entry.get("name") if isinstance(entry, dict) else None
            if not isinstance(name, str) or not name:
                continue
            version = _ALIAS_TO_VERSION.get(name)
            model_id = version or name
            model = by_id.get(model_id)
            if model is None:
                model = self._base_model(model_id)
                model._listed_description = False  # type: ignore[attr-defined]
                by_id[model_id] = model
            if version:
                if name not in model.aliases:
                    model.aliases.append(name)
                model.raw_api_data.setdefault("_listed", []).append(entry)
                # The stable alias carries the canonical description; any other
                # listed alias only fills a missing one.
                if entry.get("description") and (
                    name == "jev-latest" or not model._listed_description
                ):
                    model.description = entry["description"]
                    model._listed_description = True
            else:
                model.description = entry.get("description") or model.description
                model.raw_api_data["_listed"] = [entry]
            release = entry.get("release_date")
            if isinstance(release, str) and release:
                model.raw_api_data.setdefault("_release_dates", {})[name] = release
        return [by_id[k] for k in sorted(by_id)]

    def _base_model(self, model_id: str) -> NormalizedModel:
        facts = _MODEL_FACTS.get(model_id, {})
        family = model_id.split("-", 1)[0] or "jev"
        raw: dict[str, Any] = {}
        if facts.get("pricing"):
            raw["_pricing"] = dict(facts["pricing"])
        if facts.get("extension"):
            raw["_extension"] = dict(facts["extension"])
        return NormalizedModel(
            model_id=model_id,
            provider=self.provider_name,
            display_name=facts.get("display_name"),
            description=(
                "TypeSafe System One typed-judgment model (noul/choice/score answers "
                "with calibrated probabilities). Not a chat model."
            ),
            model_type="decision",
            context_length=facts.get("context_length", 65_536),
            supports_streaming=False,
            supports_tools=False,
            supports_structured_output=True,
            architecture_family=family,
            architecture_type="transformer",
            tags=list(_TAGS),
            raw_api_data=raw,
        )
