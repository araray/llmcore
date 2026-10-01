# tests/routing/test_manager.py
"""The routing manager: the five layers composing.

These are the spec's phase gates, written as tests. The ones that matter most:

* a 429 fails over, a 400 does not, an auth failure is not retried (T2);
* an oversized prompt routes to a larger window instead of failing (T3);
* a magic string never reaches a provider (T5);
* PII in a prompt provably does not reach a remote target (T8).
"""

from __future__ import annotations

import pytest

from llmcore.exceptions import (
    ConfigError,
    ContextLengthError,
    NoTargetAvailableError,
    PromptBlockedError,
    ProviderError,
)
from llmcore.routing.manager import RoutingManager
from llmcore.routing.models import FailureKind, RoutingRequest, SelectionStrategy, Target, Verdict
from llmcore.routing.verifiers import register_verifier

from .conftest import FakeProvider, FakeProviderManager

BASIC = """
[routing]
default_pool = "main"
max_attempts = 3
[routing.pools.main]
targets = ["alpha:big", "beta:mid", "gamma:small"]
strategy = "priority"
[routing.pools.cheap]
targets = ["gamma:small"]
[routing.pools.local_only]
targets = ["ollama:llama3.3:70b"]
[routing.lanes]
trivial = "pool:cheap"
private = "pool:local_only"
[routing.lanes.deep]
target = "alpha:big"
params = { effort = "max" }
description = "Hard reasoning"
[routing.classifier]
chain = ["hint", "magic_string", "heuristic"]
[routing.profiles.frugal]
effort = "minimal"
max_tokens = 512
"""


def manager(provider_manager, config, **kwargs) -> RoutingManager:
    return RoutingManager(provider_manager, config=config, **kwargs)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestLoading:
    def test_pools_lanes_and_profiles_load(self, provider_manager, make_config):
        mgr = manager(provider_manager, make_config(BASIC))
        assert mgr.pools()["main"] == ["alpha:big", "beta:mid", "gamma:small"]
        assert mgr.lanes() == {
            "trivial": "pool:cheap",
            "private": "pool:local_only",
            "deep": "alpha:big",
        }
        assert mgr.profiles()["frugal"]["effort"] == "minimal"

    def test_one_malformed_pool_does_not_cost_the_others(self, provider_manager, make_config):
        config = make_config(
            """
[routing.pools.good]
targets = ["alpha:big"]
[routing.pools.bad]
targets = []
"""
        )
        assert set(manager(provider_manager, config).pools()) == {"good"}

    @pytest.mark.asyncio
    async def test_an_unconfigured_pool_names_the_configured_ones(
        self, provider_manager, make_config
    ):
        mgr = manager(provider_manager, make_config(BASIC))
        with pytest.raises(ConfigError, match="Configured pools"):
            await mgr.plan(RoutingRequest(prompt="x"), pool="nonexistent")

    @pytest.mark.asyncio
    async def test_no_routing_config_at_all_still_works(
        self, provider_manager, make_config, runner
    ):
        """Routing must be invisible until someone configures it."""
        mgr = manager(provider_manager, make_config("[llmcore]\n"))
        result = await mgr.execute(RoutingRequest(prompt="hi"), runner)
        assert result.value == "ok"


# ---------------------------------------------------------------------------
# Failure handling — the T2 gate
# ---------------------------------------------------------------------------


class TestFailover:
    @pytest.mark.asyncio
    async def test_the_first_healthy_target_serves(self, provider_manager, make_config, runner):
        result = await manager(provider_manager, make_config(BASIC)).execute(
            RoutingRequest(prompt="hello there"), runner
        )
        assert result.target.key == "alpha:big"
        assert not result.failed_over

    @pytest.mark.asyncio
    async def test_a_429_fails_over(self, make_config, runner):
        providers = {
            "alpha": FakeProvider("alpha", [ProviderError("alpha", "Rate limited", status_code=429)]),
            "beta": FakeProvider("beta"),
            "gamma": FakeProvider("gamma"),
        }
        result = await manager(FakeProviderManager(providers), make_config(BASIC)).execute(
            RoutingRequest(prompt="hi"), runner
        )
        assert result.target.key == "beta:mid"
        assert [outcome.failure for outcome in result.attempts] == [FailureKind.RATE_LIMIT, None]

    @pytest.mark.asyncio
    async def test_a_400_does_not_fail_over(self, make_config, runner):
        """It will fail everywhere, so trying peers only multiplies the error
        -- and bills for it."""
        providers = {
            "alpha": FakeProvider("alpha", [ProviderError("alpha", "Bad parameter", status_code=400)]),
            "beta": FakeProvider("beta"),
            "gamma": FakeProvider("gamma"),
        }
        with pytest.raises(ProviderError, match="Bad parameter"):
            await manager(FakeProviderManager(providers), make_config(BASIC)).execute(
                RoutingRequest(prompt="hi"), runner
            )
        assert providers["beta"].calls == 0

    @pytest.mark.asyncio
    async def test_an_auth_failure_is_not_retried_and_stays_benched(self, make_config, runner):
        """A bad key does not heal, so it is tried once per process, not once
        per request."""
        providers = {
            "alpha": FakeProvider("alpha", [ProviderError("alpha", "Invalid key", status_code=401)]),
            "beta": FakeProvider("beta"),
            "gamma": FakeProvider("gamma"),
        }
        mgr = manager(FakeProviderManager(providers), make_config(BASIC))
        first = await mgr.execute(RoutingRequest(prompt="a"), runner)
        second = await mgr.execute(RoutingRequest(prompt="b"), runner)
        assert first.target.key == second.target.key == "beta:mid"
        assert providers["alpha"].calls == 1

    @pytest.mark.asyncio
    async def test_a_5xx_retries_the_same_target_before_moving(self, make_config, runner):
        """Usually one bad node behind a load balancer; moving would throw away
        a warm prompt cache for nothing."""
        providers = {
            "alpha": FakeProvider(
                "alpha", [ProviderError("alpha", "Internal error", status_code=500), "ok"]
            ),
            "beta": FakeProvider("beta"),
            "gamma": FakeProvider("gamma"),
        }
        result = await manager(FakeProviderManager(providers), make_config(BASIC)).execute(
            RoutingRequest(prompt="hi"), runner
        )
        assert result.target.key == "alpha:big"
        assert providers["alpha"].calls == 2 and providers["beta"].calls == 0

    @pytest.mark.asyncio
    async def test_a_refusal_does_not_fail_over_by_default(self, make_config, runner):
        """Retrying a refusal elsewhere is 'shop until someone says yes'."""
        providers = {
            "alpha": FakeProvider("alpha", [ProviderError("alpha", "content policy violation")]),
            "beta": FakeProvider("beta"),
            "gamma": FakeProvider("gamma"),
        }
        with pytest.raises(ProviderError):
            await manager(FakeProviderManager(providers), make_config(BASIC)).execute(
                RoutingRequest(prompt="hi"), runner
            )
        assert providers["beta"].calls == 0

    @pytest.mark.asyncio
    async def test_a_refusal_fails_over_when_configured_to(self, make_config, runner):
        providers = {
            "alpha": FakeProvider("alpha", [ProviderError("alpha", "content policy violation")]),
            "beta": FakeProvider("beta"),
            "gamma": FakeProvider("gamma"),
        }
        mgr = manager(FakeProviderManager(providers), make_config(BASIC))
        result = await mgr.execute(
            RoutingRequest(prompt="hi"),
            runner,
            settings=mgr.settings.override(on_refusal="failover"),
        )
        assert result.target.key == "beta:mid"

    @pytest.mark.asyncio
    async def test_max_attempts_caps_the_spend(self, make_config, runner):
        """An unbounded walk on a vendor-wide outage turns one failed request
        into a bill."""
        err = ProviderError("x", "Rate limited", status_code=429)
        providers = {
            "alpha": FakeProvider("alpha", [err]),
            "beta": FakeProvider("beta", [err]),
            "gamma": FakeProvider("gamma", [err]),
        }
        mgr = manager(FakeProviderManager(providers), make_config(BASIC))
        with pytest.raises(NoTargetAvailableError):
            await mgr.execute(
                RoutingRequest(prompt="hi"), runner, settings=mgr.settings.override(max_attempts=2)
            )
        assert sum(p.calls for p in providers.values()) == 2

    @pytest.mark.asyncio
    async def test_exhaustion_says_why_each_candidate_was_skipped(self, make_config, runner):
        providers = {
            "alpha": FakeProvider("alpha", [ProviderError("a", "Rate limited", status_code=429)]),
            "beta": FakeProvider("beta", [ProviderError("b", "not_enough_credits", status_code=403)]),
            "gamma": FakeProvider("gamma", [TimeoutError("too slow")]),
        }
        mgr = manager(FakeProviderManager(providers), make_config(BASIC))
        with pytest.raises(NoTargetAvailableError) as excinfo:
            await mgr.execute(RoutingRequest(prompt="hi"), runner)
        reasons = dict(excinfo.value.candidates)
        assert "rate_limit" in reasons["alpha:big"]
        assert "insufficient_credit" in reasons["beta:mid"]
        assert "timeout" in reasons["gamma:small"]

    @pytest.mark.asyncio
    async def test_cooldowns_differ_by_failure_kind(self, make_config, runner):
        """An empty wallet will not fix itself in seconds; a 429 will."""
        providers = {
            "alpha": FakeProvider("alpha", [ProviderError("a", "Rate limited", status_code=429)]),
            "beta": FakeProvider("beta", [ProviderError("b", "not_enough_credits", status_code=403)]),
            "gamma": FakeProvider("gamma"),
        }
        mgr = manager(FakeProviderManager(providers), make_config(BASIC))
        await mgr.execute(RoutingRequest(prompt="hi"), runner)
        health = mgr.health()
        assert 0 < health["alpha:big"]["cooldown_remaining"] <= 30
        assert health["beta:mid"]["cooldown_remaining"] > 300

    @pytest.mark.asyncio
    async def test_a_zero_balance_is_recorded_from_the_failure(self, make_config, runner):
        """Better evidence than any balance endpoint: the vendor just said so."""
        providers = {
            "alpha": FakeProvider("alpha", [ProviderError("a", "not_enough_credits", status_code=403)]),
            "beta": FakeProvider("beta"),
            "gamma": FakeProvider("gamma"),
        }
        mgr = manager(FakeProviderManager(providers), make_config(BASIC))
        await mgr.execute(RoutingRequest(prompt="hi"), runner)
        assert mgr.health()["alpha:big"]["balance"]["amount"] == 0.0


# ---------------------------------------------------------------------------
# Context overflow — the T3 gate
# ---------------------------------------------------------------------------


class TestContextOverflow:
    CONTEXT = """
[routing]
default_pool = "mixed"
[routing.pools.mixed]
targets = ["openai:gpt-4o", "anthropic:claude-opus-4-1"]
"""

    @pytest.mark.asyncio
    async def test_a_prompt_too_large_skips_the_small_window(self, make_config, runner):
        """llmcore knows every model's window from its card, so this costs no
        request at all."""
        providers = {"openai": FakeProvider("openai"), "anthropic": FakeProvider("anthropic")}
        mgr = manager(FakeProviderManager(providers), make_config(self.CONTEXT))
        result = await mgr.execute(RoutingRequest(prompt="x" * 640_000), runner)
        assert result.target.key == "anthropic:claude-opus-4-1"
        assert providers["openai"].calls == 0

    @pytest.mark.asyncio
    async def test_a_reported_overflow_reroutes_to_a_larger_window(self, make_config, runner):
        """The provider's own number is used, not an estimate -- an estimate
        that was too small would route to another target that also cannot hold
        it, turning one clear error into several."""
        providers = {
            "openai": FakeProvider(
                "openai",
                [ContextLengthError(model_name="gpt-4o", limit=128_000, actual=150_000)],
            ),
            "anthropic": FakeProvider("anthropic"),
        }
        mgr = manager(FakeProviderManager(providers), make_config(self.CONTEXT))
        result = await mgr.execute(RoutingRequest(prompt="short but secretly huge"), runner)
        assert result.target.key == "anthropic:claude-opus-4-1"

    @pytest.mark.asyncio
    async def test_an_overflow_does_not_bench_the_target(self, make_config, runner):
        """Someone else's oversized prompt must not cost a healthy endpoint
        its place in the rotation."""
        providers = {
            "openai": FakeProvider(
                "openai",
                [ContextLengthError(model_name="gpt-4o", limit=128_000, actual=150_000), "ok"],
            ),
            "anthropic": FakeProvider("anthropic"),
        }
        mgr = manager(FakeProviderManager(providers), make_config(self.CONTEXT))
        await mgr.execute(RoutingRequest(prompt="huge"), runner)
        assert mgr.health()["openai:gpt-4o"]["cooldown_remaining"] == 0
        result = await mgr.execute(RoutingRequest(prompt="small"), runner)
        assert result.target.key == "openai:gpt-4o"


# ---------------------------------------------------------------------------
# Lanes and parameters
# ---------------------------------------------------------------------------


class TestLanesAndParams:
    @pytest.mark.asyncio
    async def test_a_classifier_routes_to_a_lane(self, provider_manager, make_config):
        mgr = manager(provider_manager, make_config(BASIC))
        plan = await mgr.plan(RoutingRequest(prompt="Translate 'hi' to French."))
        assert plan.lane == "trivial" and plan.pool == "cheap"

    @pytest.mark.asyncio
    async def test_an_explicit_lane_skips_classification(self, provider_manager, make_config):
        mgr = manager(provider_manager, make_config(BASIC))
        plan = await mgr.plan(RoutingRequest(prompt="Translate 'hi'."), lane="deep")
        assert plan.chosen.key == "alpha:big" and plan.classification is None

    @pytest.mark.asyncio
    async def test_a_pinned_target_bypasses_routing_entirely(self, provider_manager, make_config):
        """The documented way to reproduce a result."""
        mgr = manager(provider_manager, make_config(BASIC))
        plan = await mgr.plan(
            RoutingRequest(prompt="Translate 'hi'."), target="xai:grok-4.1?effort=high"
        )
        assert plan.chosen.spec() == "xai:grok-4.1?effort=high"
        assert plan.pool is None and plan.lane is None

    @pytest.mark.asyncio
    async def test_a_lane_that_is_not_configured_falls_back_with_a_note(
        self, provider_manager, make_config
    ):
        mgr = manager(provider_manager, make_config(BASIC))
        plan = await mgr.plan(RoutingRequest(prompt="x"), lane="nonexistent")
        assert plan.pool == "main"
        assert any("not configured" in note for note in plan.notes)

    @pytest.mark.asyncio
    async def test_params_layer_target_then_lane_then_profile_then_call(
        self, provider_manager, make_config
    ):
        """Spec 6.1, in order. The per-call value has to win or the API is a
        lie."""
        captured: dict[str, object] = {}

        async def capture(provider, target, params):
            captured.update(params)
            return "ok"

        mgr = manager(provider_manager, make_config(BASIC))
        await mgr.execute(
            RoutingRequest(prompt="x", hints={"lane": "deep"}),
            capture,
            profile="frugal",
            call_params={"max_tokens": 99},
        )
        # lane says effort=max, profile overrides to minimal, call wins on max_tokens
        assert captured == {"effort": "minimal", "max_tokens": 99}

    @pytest.mark.asyncio
    async def test_an_unknown_profile_is_ignored_with_a_warning(
        self, provider_manager, make_config, caplog
    ):
        mgr = manager(provider_manager, make_config(BASIC))
        plan = await mgr.plan(RoutingRequest(prompt="x"), lane="deep", profile="nope")
        assert plan.chosen.params == {"effort": "max"}
        assert "not configured" in caplog.text

    @pytest.mark.asyncio
    async def test_a_magic_string_never_reaches_a_provider(self, provider_manager, make_config):
        """T5's gate. A marker that leaks is llmcore's internals turning up in
        someone's context window, and a harness would quote it back."""
        seen: dict[str, str] = {}

        async def capture(provider, target, params):
            return "ok"

        mgr = manager(provider_manager, make_config(BASIC))
        request = RoutingRequest(prompt="[[lane:deep]] Explain this proof.")
        final, results = await mgr.transform_chain().apply(request, Target.parse("alpha:big"))
        assert "[[lane:deep]]" not in final.prompt
        assert final.prompt == "Explain this proof."

    @pytest.mark.asyncio
    async def test_the_strip_transform_is_installed_automatically(
        self, provider_manager, make_config
    ):
        """So a user cannot configure the leak by forgetting it."""
        mgr = manager(provider_manager, make_config(BASIC))
        assert "strip_magic" in mgr.transform_chain().names()


# ---------------------------------------------------------------------------
# Privacy — the T8 gate
# ---------------------------------------------------------------------------


PII_CONFIG = """
[routing]
default_pool = "remote"
[routing.pools.remote]
targets = ["alpha:big"]
[routing.pools.local_only]
targets = ["ollama:llama3.3:70b"]
[routing.transforms]
chain = ["pii"]
[routing.transforms.pii]
on_detect = "constrain"
pool = "local_only"
redact = true
"""


class TestPrivacyPath:
    @pytest.mark.asyncio
    async def test_pii_provably_does_not_reach_a_remote_target(
        self, providers, make_config, runner
    ):
        """The spec's T8 gate, and the reason `constrain` exists: redaction is
        a mitigation, routing is a guarantee."""
        provider_manager = FakeProviderManager(providers)
        mgr = manager(provider_manager, make_config(PII_CONFIG))
        result = await mgr.execute(
            RoutingRequest(prompt="Email john.doe@example.com about the invoice."), runner
        )
        assert result.target.key == "ollama:llama3.3:70b"
        assert providers["alpha"].calls == 0

    @pytest.mark.asyncio
    async def test_a_clean_prompt_still_goes_to_the_remote_pool(
        self, providers, make_config, runner
    ):
        mgr = manager(FakeProviderManager(providers), make_config(PII_CONFIG))
        result = await mgr.execute(RoutingRequest(prompt="What is 2 + 2?"), runner)
        assert result.target.key == "alpha:big"

    @pytest.mark.asyncio
    async def test_a_missing_local_pool_blocks_rather_than_leaking(
        self, providers, make_config, runner
    ):
        """Degrading to 'send it anyway' would be the one outcome the user was
        trying to prevent."""
        config = make_config(PII_CONFIG.replace('pool = "local_only"', 'pool = "nope"'))
        mgr = manager(FakeProviderManager(providers), config)
        with pytest.raises(PromptBlockedError):
            await mgr.execute(RoutingRequest(prompt="mail a@b.com"), runner)
        assert providers["alpha"].calls == 0

    @pytest.mark.asyncio
    async def test_block_reports_hashed_findings_only(self, providers, make_config, runner):
        """An exception message containing the identifiers it found would be
        the leak the feature exists to prevent."""
        config = make_config(PII_CONFIG.replace('on_detect = "constrain"', 'on_detect = "block"'))
        mgr = manager(FakeProviderManager(providers), config)
        with pytest.raises(PromptBlockedError) as excinfo:
            await mgr.execute(RoutingRequest(prompt="card 4111 1111 1111 1111"), runner)
        assert "4111" not in str(excinfo.value)
        assert excinfo.value.findings and all(len(f[2]) == 16 for f in excinfo.value.findings)


# ---------------------------------------------------------------------------
# Cascade — the T7 gate
# ---------------------------------------------------------------------------


CASCADE_CONFIG = """
[routing]
default_pool = "cheap"
[routing.pools.cheap]
targets = ["gamma:small"]
[routing.pools.strong]
targets = ["alpha:big"]
[routing.cascade]
enabled = true
[routing.cascade.rungs.default]
rungs = ["pool:strong"]
verifier = "stub"
"""


class _StubVerifier:
    name = "stub"
    cost_hint = "free"

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.index = 0

    async def verify(self, request, response):
        verdict = self.verdicts[min(self.index, len(self.verdicts) - 1)]
        self.index += 1
        return verdict


@pytest.fixture
def stub_verdicts():
    def install(*verdicts):
        register_verifier("stub", lambda **kw: _StubVerifier(verdicts), replace=True)

    return install


class TestCascade:
    @pytest.fixture
    def pair(self):
        providers = {
            "gamma": FakeProvider("gamma", ["weak answer"]),
            "alpha": FakeProvider("alpha", ["strong answer"]),
        }
        return providers, FakeProviderManager(providers)

    @pytest.mark.asyncio
    async def test_an_insufficient_answer_escalates(
        self, pair, make_config, runner, stub_verdicts
    ):
        providers, provider_manager = pair
        stub_verdicts(Verdict(sufficient=False, score=0.2), Verdict(sufficient=True, score=0.95))
        result = await manager(provider_manager, make_config(CASCADE_CONFIG)).execute(
            RoutingRequest(prompt="hard"), runner
        )
        assert result.target.key == "alpha:big"
        assert result.value == "strong answer" and result.rungs_used == 2

    @pytest.mark.asyncio
    async def test_a_sufficient_answer_costs_nothing_more(
        self, pair, make_config, runner, stub_verdicts
    ):
        providers, provider_manager = pair
        stub_verdicts(Verdict(sufficient=True, score=0.9))
        result = await manager(provider_manager, make_config(CASCADE_CONFIG)).execute(
            RoutingRequest(prompt="easy"), runner
        )
        assert result.target.key == "gamma:small" and providers["alpha"].calls == 0

    @pytest.mark.asyncio
    async def test_an_unjudgeable_answer_is_accepted_by_default(
        self, pair, make_config, runner, stub_verdicts
    ):
        """Escalating on every unjudgeable answer inverts the saving the
        cascade exists for."""
        providers, provider_manager = pair
        stub_verdicts(Verdict(sufficient=None, rationale="judge unavailable"))
        result = await manager(provider_manager, make_config(CASCADE_CONFIG)).execute(
            RoutingRequest(prompt="x"), runner
        )
        assert result.target.key == "gamma:small" and providers["alpha"].calls == 0

    @pytest.mark.asyncio
    async def test_an_unjudgeable_answer_escalates_when_configured_to(
        self, pair, make_config, runner, stub_verdicts
    ):
        providers, provider_manager = pair
        stub_verdicts(Verdict(sufficient=None), Verdict(sufficient=True, score=1.0))
        mgr = manager(provider_manager, make_config(CASCADE_CONFIG))
        result = await mgr.execute(
            RoutingRequest(prompt="x"),
            runner,
            settings=mgr.settings.override(on_unknown_verdict="escalate"),
        )
        assert result.target.key == "alpha:big"

    @pytest.mark.asyncio
    async def test_cascade_is_off_unless_enabled(self, pair, make_config, runner, stub_verdicts):
        """It trades latency and an extra call, which is wrong for interactive
        use."""
        providers, provider_manager = pair
        stub_verdicts(Verdict(sufficient=False, score=0.0))
        config = make_config(CASCADE_CONFIG.replace("enabled = true", "enabled = false"))
        result = await manager(provider_manager, config).execute(
            RoutingRequest(prompt="x"), runner
        )
        assert result.target.key == "gamma:small" and result.verdict is None

    @pytest.mark.asyncio
    async def test_max_rungs_caps_escalation(self, make_config, runner, stub_verdicts):
        providers = {
            "gamma": FakeProvider("gamma", ["weak"]),
            "beta": FakeProvider("beta", ["middling"]),
            "alpha": FakeProvider("alpha", ["strong"]),
        }
        config = make_config(
            CASCADE_CONFIG.replace(
                'rungs = ["pool:strong"]', 'rungs = ["beta:mid", "alpha:big"]'
            )
        )
        stub_verdicts(Verdict(sufficient=False, score=0.1))
        mgr = manager(FakeProviderManager(providers), config)
        await mgr.execute(
            RoutingRequest(prompt="x"), runner, settings=mgr.settings.override(cascade_max_rungs=1)
        )
        assert providers["alpha"].calls == 0, "the second rung must not be reached"

    @pytest.mark.asyncio
    async def test_a_failing_rung_keeps_the_previous_answer(
        self, make_config, runner, stub_verdicts
    ):
        """An escalation that errors should not lose the answer already in
        hand."""
        providers = {
            "gamma": FakeProvider("gamma", ["weak but real"]),
            "alpha": FakeProvider("alpha", [ProviderError("alpha", "down", status_code=503)]),
        }
        stub_verdicts(Verdict(sufficient=False, score=0.1))
        result = await manager(
            FakeProviderManager(providers), make_config(CASCADE_CONFIG)
        ).execute(RoutingRequest(prompt="x"), runner)
        assert result.value == "weak but real"


# ---------------------------------------------------------------------------
# Session affinity — the T10 gate
# ---------------------------------------------------------------------------


class TestSessionAffinity:
    @pytest.mark.asyncio
    async def test_a_session_keeps_its_target(self, make_config, runner):
        """Changing model mid-conversation drops the cached prefix and
        invalidates preserved reasoning."""
        providers = {
            "alpha": FakeProvider("alpha"),
            "beta": FakeProvider("beta"),
            "gamma": FakeProvider("gamma"),
        }
        config = make_config(
            BASIC.replace('strategy = "priority"', 'strategy = "round_robin"')
        )
        mgr = manager(FakeProviderManager(providers), config)
        keys = [
            (await mgr.execute(RoutingRequest(prompt="hi", session_id="s1"), runner)).target.key
            for _ in range(3)
        ]
        assert len(set(keys)) == 1, f"affinity broke: {keys}"

    @pytest.mark.asyncio
    async def test_the_pin_moves_when_the_target_becomes_unusable(self, make_config, runner):
        providers = {
            "alpha": FakeProvider("alpha", ["ok", ProviderError("a", "bad key", status_code=401)]),
            "beta": FakeProvider("beta"),
            "gamma": FakeProvider("gamma"),
        }
        mgr = manager(FakeProviderManager(providers), make_config(BASIC))
        first = await mgr.execute(RoutingRequest(prompt="a", session_id="s1"), runner)
        second = await mgr.execute(RoutingRequest(prompt="b", session_id="s1"), runner)
        assert first.target.key == "alpha:big"
        assert second.target.key == "beta:mid"

    @pytest.mark.asyncio
    async def test_affinity_none_distributes(self, make_config, runner):
        providers = {
            "alpha": FakeProvider("alpha"),
            "beta": FakeProvider("beta"),
            "gamma": FakeProvider("gamma"),
        }
        config = make_config(
            BASIC.replace('strategy = "priority"', 'strategy = "round_robin"\naffinity = "none"')
        )
        mgr = manager(FakeProviderManager(providers), config)
        keys = [
            (await mgr.execute(RoutingRequest(prompt="hi", session_id="s1"), runner)).target.key
            for _ in range(3)
        ]
        assert len(set(keys)) == 3

    @pytest.mark.asyncio
    async def test_a_request_without_a_session_is_unaffected(self, make_config, runner):
        providers = {
            "alpha": FakeProvider("alpha"),
            "beta": FakeProvider("beta"),
            "gamma": FakeProvider("gamma"),
        }
        config = make_config(BASIC.replace('strategy = "priority"', 'strategy = "round_robin"'))
        mgr = manager(FakeProviderManager(providers), config)
        keys = [(await mgr.execute(RoutingRequest(prompt="hi"), runner)).target.key for _ in range(3)]
        assert len(set(keys)) == 3


# ---------------------------------------------------------------------------
# explain()
# ---------------------------------------------------------------------------


class TestExplain:
    @pytest.mark.asyncio
    async def test_a_plan_names_the_chosen_target_and_the_rejects(
        self, make_config, runner
    ):
        """Opaque routing is a support burden: the first question anyone asks
        is 'why did it pick that?'."""
        providers = {
            "alpha": FakeProvider("alpha", [ProviderError("a", "Rate limited", status_code=429)]),
            "beta": FakeProvider("beta"),
            "gamma": FakeProvider("gamma"),
        }
        mgr = manager(FakeProviderManager(providers), make_config(BASIC))
        await mgr.execute(RoutingRequest(prompt="hi"), runner)
        plan = await mgr.plan(RoutingRequest(prompt="hello there"))
        assert plan.chosen.key == "beta:mid"
        skipped = next(c for c in plan.candidates if not c.eligible)
        assert "cooling down" in skipped.reason

    @pytest.mark.asyncio
    async def test_planning_makes_no_call(self, provider_manager, providers, make_config):
        mgr = manager(provider_manager, make_config(BASIC))
        await mgr.plan(RoutingRequest(prompt="Translate 'hi'."))
        assert all(provider.calls == 0 for provider in providers.values())

    @pytest.mark.asyncio
    async def test_a_plan_summary_reads_in_one_line(self, provider_manager, make_config):
        mgr = manager(provider_manager, make_config(BASIC))
        plan = await mgr.plan(RoutingRequest(prompt="Translate 'hi'."))
        summary = plan.summary()
        assert "trivial" in summary and "gamma:small" in summary


# ---------------------------------------------------------------------------
# Balances
# ---------------------------------------------------------------------------


class TestBalanceProbing:
    @pytest.mark.asyncio
    async def test_a_provider_without_an_endpoint_is_reported_as_unknown(
        self, provider_manager, make_config
    ):
        """'We asked and nobody knows' and 'we did not ask' are different
        facts."""
        mgr = manager(provider_manager, make_config(BASIC))
        balances = await mgr.probe_balances()
        assert balances["alpha:big"]["known"] is False
        assert "no balance endpoint" in balances["alpha:big"]["error"]

    @pytest.mark.asyncio
    async def test_a_reported_balance_feeds_selection(self, providers, make_config):
        from llmcore.routing.models import Balance

        async def balance():
            return Balance(amount=123.0)

        providers["beta"].remaining_balance = balance
        mgr = manager(FakeProviderManager(providers), make_config(BASIC))
        balances = await mgr.probe_balances()
        assert balances["beta:mid"] == {"known": True, "amount": 123.0, "unit": "usd"}
        assert mgr.health()["beta:mid"]["balance"]["amount"] == 123.0
