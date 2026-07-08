# tests/agents/test_grimoire_pack.py
"""Control-plane pack tests (0.52.0).

The bundled ``llmcore-builtin`` grimoire pack is THE source of every agent
prompt. These tests are the completeness gate the fail-loud design relies on:

- every ``REQUIRED_SPELLS`` template renders via the adapter (this is the
  zero-config validation that startup skips for speed);
- the rendered cognitive prompts preserve the format contracts the phase
  parsers regex on (compared against the legacy f-string generators);
- layer overrides win and are attributed; broken overlays fail LOUDLY;
- the builtin tool runes cover the 5 default tools with implementation keys
  that exist in ``_IMPLEMENTATION_REGISTRY``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import ClassVar

import pytest

from llmcore.agents.prompts.grimoire_adapter import (
    DEFAULT_TEMPLATE_MAP,
    GrimoirePromptRegistryAdapter,
)
from llmcore.exceptions import ConfigError
from llmcore.grimoire_runtime import (
    BUNDLED_LAYER,
    REQUIRED_SPELLS,
    build_grimoire,
    build_prompt_registry,
    bundled_pack_path,
    validate_grimoire_startup,
)


@pytest.fixture(scope="module")
def bundled_adapter() -> GrimoirePromptRegistryAdapter:
    from grimoire import Grimoire

    return GrimoirePromptRegistryAdapter(Grimoire(bundled_pack_path()))


class TestPackCompleteness:
    def test_every_required_template_is_mapped(self):
        for required in REQUIRED_SPELLS:
            assert required.template_id in DEFAULT_TEMPLATE_MAP, (
                f"{required.template_id} missing from DEFAULT_TEMPLATE_MAP"
            )

    def test_every_required_template_renders(self, bundled_adapter):
        for required in REQUIRED_SPELLS:
            messages = bundled_adapter.render_messages(
                required.template_id, dict(required.sample_vars)
            )
            assert messages, required.template_id
            roles = [m["role"] for m in messages]
            assert all(r in ("system", "user", "assistant") for r in roles)
            if required.expects_system:
                assert "system" in roles, (
                    f"{required.template_id} expected a system block, got {roles}"
                )

    def test_get_template_metadata(self, bundled_adapter):
        template = bundled_adapter.get_template("thinking_prompt")
        assert template.grimoire_id == "llmcore/cognitive/think"
        assert template.active_version is not None

    def test_persona_spells_carry_definitions(self):
        from grimoire import Grimoire

        g = Grimoire(bundled_pack_path())
        for pid in ("assistant", "analyst", "developer", "researcher", "creative"):
            spell = g.get_spell(f"llmcore/persona/{pid}")
            persona = spell.attributes.get("persona")
            assert isinstance(persona, dict), pid
            assert persona.get("id") == pid
            assert persona.get("traits"), pid


class TestFormatContracts:
    """The rendered prompts must keep the labels the phase parsers regex on."""

    def test_plan_contract(self, bundled_adapter):
        text = bundled_adapter.render(
            "planning_prompt",
            {"goal": "G", "context": "C", "constraints": "K", "existing_plan_section": ""},
        )
        for label in ("PLAN:", "REASONING:", "RISKS:"):
            assert label in text
        # The phase's registry-backed message builder preserves the same
        # contract (the legacy f-string generator was deleted in 0.52.0).
        from llmcore.agents.cognitive.models import PlanInput
        from llmcore.agents.cognitive.phases.plan import _generate_planning_messages
        from llmcore.models import Role

        messages = _generate_planning_messages(
            PlanInput(goal="G", context="C", constraints="K"), bundled_adapter
        )
        assert messages[0].role == Role.SYSTEM
        user_text = "\n".join(m.content for m in messages if m.role == Role.USER)
        for label in ("PLAN:", "REASONING:", "RISKS:"):
            assert label in user_text

    def test_think_contract(self, bundled_adapter):
        text = bundled_adapter.render(
            "thinking_prompt",
            {
                "goal": "G",
                "current_step": "S",
                "history": "No previous actions.",
                "context": "",
                "tools": "- finish(answer)",
                "remaining_steps": "3",
            },
        )
        for label in ("Thought:", "Action:", "Action Input:", "Final Answer:"):
            assert label in text
        # Phase-2 convergence deltas: explicit finish-tool instruction, the
        # no-tool → finish-immediately rule, and the step-budget line.
        assert "call the `finish` tool" in text
        assert "call `finish` immediately" in text
        assert "You have 3 step(s) remaining" in text

    def test_think_remaining_steps_defaults_to_unlimited(self, bundled_adapter):
        text = bundled_adapter.render(
            "thinking_prompt",
            {
                "goal": "G",
                "current_step": "S",
                "history": "No previous actions.",
                "context": "",
                "tools": "- finish(answer)",
            },
        )
        assert "You have unlimited step(s) remaining" in text

    def test_validate_contract(self, bundled_adapter):
        text = bundled_adapter.render(
            "validation_prompt",
            {
                "goal": "G",
                "proposed_action": "x()",
                "reasoning": "r",
                "risk_tolerance": "medium",
            },
        )
        for label in ("APPROVED:", "CONFIDENCE:", "CONCERNS:", "SUGGESTIONS:"):
            assert label in text

    def test_reflect_contract(self, bundled_adapter):
        text = bundled_adapter.render(
            "reflection_prompt",
            {
                "goal": "G",
                "plan": "1. s",
                "current_step_display": "1. s",
                "last_action": "x()",
                "observation": "ok",
                "iteration": "1",
            },
        )
        for label in (
            "EVALUATION:",
            "PROGRESS:",
            "INSIGHTS:",
            "PLAN_UPDATE:",
            "STEP_COMPLETED:",
            "NEXT_FOCUS:",
        ):
            assert label in text

    def test_activity_final_answer_param_is_answer(self, bundled_adapter):
        """The executor requires parameters['answer'] — the pack documents
        `answer` (the legacy prompt wrongly said `r`/`result`)."""
        msgs = bundled_adapter.render_messages("activity_system", {})
        text = "\n".join(m["content"] for m in msgs)
        assert "answer (required)" in text
        assert "<answer>4</answer>" in text
        exec_msgs = bundled_adapter.render_messages(
            "activity_execute",
            {
                "goal": "G",
                "current_step": "S",
                "activities_section": "",
                "history_section": "",
                "context_section": "",
            },
        )
        exec_text = "\n".join(m["content"] for m in exec_msgs)
        assert "<answer>" in exec_text and "<r>" not in exec_text

    def test_goal_classifier_contract(self, bundled_adapter):
        text = bundled_adapter.render("goal_classifier", {"goal": "hi"})
        for label in ("COMPLEXITY:", "INTENT:", "CONFIDENCE:", "REQUIRES_TOOLS:", "MAX_ITERATIONS:"):
            assert label in text


class TestLayering:
    def _overlay(self, tmp_path: Path, body: str) -> Path:
        root = tmp_path / "overlay"
        (root / "spells").mkdir(parents=True)
        (root / "grimoire.yaml").write_text("name: user-overlay\nspell_paths: [spells/]\n")
        (root / "spells" / "think_override.spell.md").write_text(body)
        return root

    def test_user_layer_overrides_bundled(self, tmp_path):
        from grimoire import Grimoire

        overlay = self._overlay(
            tmp_path,
            textwrap.dedent(
                """\
                ---
                id: llmcore/cognitive/think
                name: Custom think
                variables:
                  goal: {type: string, required: true}
                  current_step: {type: string, required: false, default: ""}
                  history: {type: multiline, required: false, default: ""}
                  context: {type: multiline, required: false, default: ""}
                  tools: {type: multiline, required: false, default: ""}
                  remaining_steps: {type: string, required: false, default: ""}
                ---

                # SYSTEM
                CUSTOM SYSTEM

                # USER
                CUSTOM THINK for {{ goal }}

                Thought: / Action: / Action Input: / Final Answer:
                """
            ),
        )
        g = Grimoire.layered(
            [(BUNDLED_LAYER, bundled_pack_path(), False), ("user", overlay, False)]
        )
        adapter = GrimoirePromptRegistryAdapter(g)
        msgs = adapter.render_messages("thinking_prompt", {"goal": "X"})
        assert msgs[0]["content"] == "CUSTOM SYSTEM"
        assert "CUSTOM THINK for X" in msgs[1]["content"]
        assert g.resolve_layer("llmcore/cognitive/think", "spell") == "user"

    def test_broken_overlay_fails_strict_load(self, tmp_path):
        overlay = self._overlay(tmp_path, "not: [valid frontmatter")

        class Cfg:
            extra_pack_paths: ClassVar[list[str]] = []
            admin_repo_path = ""
            user_repo_paths: ClassVar[list[str]] = [str(overlay)]
            prompt_map: ClassVar[dict[str, str]] = {}
            strict = True
            validate_on_startup = True

        with pytest.raises(ConfigError, match="failed to load"):
            build_grimoire(Cfg())

    def test_override_breaking_required_template_fails_validation(self, tmp_path):
        """An overlay spell shadowing a required id but demanding an unknown
        REQUIRED variable must abort startup naming the template."""
        overlay = self._overlay(
            tmp_path,
            textwrap.dedent(
                """\
                ---
                id: llmcore/cognitive/think
                name: Broken think
                variables:
                  nonexistent_variable: {type: string, required: true}
                ---

                # USER
                {{ nonexistent_variable }}
                """
            ),
        )

        class Cfg:
            extra_pack_paths: ClassVar[list[str]] = []
            admin_repo_path = ""
            user_repo_paths: ClassVar[list[str]] = [str(overlay)]
            prompt_map: ClassVar[dict[str, str]] = {}
            strict = True
            validate_on_startup = True

            class metrics:
                enabled = False
                path = ""

        cfg = Cfg()
        g = build_grimoire(cfg)
        registry = build_prompt_registry(g, cfg)
        with pytest.raises(ConfigError, match="thinking_prompt"):
            validate_grimoire_startup(g, registry, cfg, overlays_present=True)


class TestBuiltinToolRunes:
    def test_five_builtins_covered_with_registered_impl_keys(self):
        from grimoire import Grimoire

        from llmcore.agents.tools import _IMPLEMENTATION_REGISTRY

        g = Grimoire(bundled_pack_path())
        expected = {
            "finish": "llmcore.tools.flow.finish",
            "human_approval": "llmcore.tools.flow.human_approval",
            "semantic_search": "llmcore.tools.search.semantic",
            "episodic_search": "llmcore.tools.search.episodic",
            "calculator": "llmcore.tools.calculation.calculator",
        }
        found: dict[str, str] = {}
        for rune in g.repo.list_runes(tags=["llmcore.builtin"]):
            for cmd in rune.commands:
                key = rune.mappings.get(f"llmcore.implementation_key.{cmd.name}")
                if key:
                    found[cmd.name] = key
        assert found == expected
        for key in expected.values():
            assert key in _IMPLEMENTATION_REGISTRY, key

    def test_finish_rune_has_answer_schema(self):
        from grimoire import Grimoire
        from grimoire.runes.schema import command_parameters_schema

        g = Grimoire(bundled_pack_path())
        rune = g.get_rune("llmcore/flow")
        finish = rune.get_command("finish")
        schema = command_parameters_schema(finish)
        assert schema["properties"]["answer"]["type"] == "string"
        assert "answer" in schema.get("required", [])
