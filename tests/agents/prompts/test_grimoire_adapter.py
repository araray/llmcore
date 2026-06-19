from __future__ import annotations

from types import SimpleNamespace

import pytest

from llmcore.agents.prompts import GrimoirePromptRegistryAdapter, TemplateNotFoundError


class FakeGrimoire:
    def __init__(self) -> None:
        self.spells = {
            "agent/planning": SimpleNamespace(
                id="agent/planning",
                name="Planning",
                version="2.1.0",
                description="Plan a task",
                tags=["agent", "planning"],
                content_hash="abc123",
            )
        }
        self.calls = []

    def get_spell(self, spell_id: str):
        if spell_id not in self.spells:
            raise KeyError(spell_id)
        return self.spells[spell_id]

    def conjure_spell(self, spell_id: str, *, variables: dict, strict: bool):
        self.calls.append((spell_id, variables, strict))
        if spell_id not in self.spells:
            raise KeyError(spell_id)
        return SimpleNamespace(
            blocks=[
                SimpleNamespace(role=SimpleNamespace(value="SYSTEM"), content="Plan {{ goal }}"),
                SimpleNamespace(role=SimpleNamespace(value="USER"), content=f"Goal: {variables['goal']}"),
            ]
        )


def test_grimoire_adapter_renders_mapped_spell() -> None:
    grimoire = FakeGrimoire()
    adapter = GrimoirePromptRegistryAdapter(
        grimoire,
        template_map={"planning_prompt": "agent/planning"},
    )

    rendered = adapter.render("planning_prompt", {"goal": "ship tests"})

    assert rendered == "Plan {{ goal }}\n\nGoal: ship tests"
    assert grimoire.calls == [("agent/planning", {"goal": "ship tests"}, True)]


def test_grimoire_adapter_get_template_returns_active_version_facade() -> None:
    adapter = GrimoirePromptRegistryAdapter(
        FakeGrimoire(),
        template_map={"planning_prompt": "agent/planning"},
    )

    template = adapter.get_template("planning_prompt")

    assert template.id == "planning_prompt"
    assert template.name == "Planning"
    assert template.grimoire_id == "agent/planning"
    assert template.tags == ["agent", "planning"]
    assert template.active_version is not None
    assert template.active_version.version_number == 2
    assert template.active_version.content_hash == "abc123"
    assert template.active_version_id == template.active_version.id


def test_grimoire_adapter_records_usage_metrics_in_memory() -> None:
    adapter = GrimoirePromptRegistryAdapter(
        FakeGrimoire(),
        template_map={"planning_prompt": "agent/planning"},
    )
    version_id = adapter.get_template("planning_prompt").active_version_id
    assert version_id is not None

    adapter.record_use(version_id, success=True, tokens=100)
    adapter.record_use(version_id, success=False, tokens=50)

    metrics = adapter.get_metrics(version_id)
    assert metrics.total_uses == 2
    assert metrics.successful_uses == 1
    assert metrics.failed_uses == 1
    assert metrics.avg_tokens_used == 75


def test_grimoire_adapter_missing_spell_raises_template_error() -> None:
    adapter = GrimoirePromptRegistryAdapter(FakeGrimoire())

    with pytest.raises(TemplateNotFoundError):
        adapter.get_template("missing_prompt")


def test_grimoire_adapter_can_include_role_headers() -> None:
    adapter = GrimoirePromptRegistryAdapter(
        FakeGrimoire(),
        template_map={"planning_prompt": "agent/planning"},
        include_role_headers=True,
    )

    rendered = adapter.render("planning_prompt", {"goal": "ship tests"})

    assert rendered == "[SYSTEM]\nPlan {{ goal }}\n\n[USER]\nGoal: ship tests"
