# tests/routing/test_targets_and_settings.py
"""Routing layers 1 and 2: targets, autoprovisioning, settings layering, health.

Two principles are load-bearing here and each gets explicit coverage:

* **Config is a warm-up, not a cage.** Every knob resolves config → env →
  per-request, and the request wins. A setting that could only be set in config
  would freeze the runtime, which is the thing this design exists to avoid.
* **A failure's *kind* is a fact; what routing does about it is policy.** 429,
  an empty wallet and an over-long prompt must not behave alike.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llmcore.exceptions import ConfigError, ContextLengthError, ProviderError
from llmcore.routing import (
    Balance,
    FailureKind,
    Outcome,
    RoutingRequest,
    SelectionStrategy,
    Target,
    TargetHealth,
    classify_failure,
)
from llmcore.routing.settings import (
    ClassifierBias,
    RefusalPolicy,
    RoutingSettings,
    UnknownVerdictPolicy,
)

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Target spec strings
# ---------------------------------------------------------------------------


class TestTargetParsing:
    @pytest.mark.parametrize(
        ("spec", "provider", "model", "instance", "params"),
        [
            ("openai:gpt-5.4", "openai", "gpt-5.4", None, {}),
            ("openai", "openai", None, None, {}),
            ("anthropic:claude-opus-5-5?effort=high", "anthropic", "claude-opus-5-5", None,
             {"effort": "high"}),
            # Model names routinely contain ':' and '/'. Splitting on the LAST
            # colon would mangle both of these.
            ("ollama:llama3.3:70b", "ollama", "llama3.3:70b", None, {}),
            ("vllm:Qwen/Qwen3-30B#my-box", "vllm", "Qwen/Qwen3-30B", "my-box", {}),
            ("replicate:black-forest-labs/flux-schnell", "replicate",
             "black-forest-labs/flux-schnell", None, {}),
            ("fal:fal-ai/flux/schnell", "fal", "fal-ai/flux/schnell", None, {}),
        ],
    )
    def test_parses(self, spec, provider, model, instance, params):
        target = Target.parse(spec)
        assert target.provider == provider
        assert target.model == model
        assert target.instance == instance
        assert target.params == params

    def test_params_are_coerced(self):
        target = Target.parse(
            "openai:gpt-5.4?temperature=0.2&max_tokens=512&stream=false&seed=7&stop=none"
        )
        assert target.params["temperature"] == pytest.approx(0.2)
        assert target.params["max_tokens"] == 512
        assert target.params["stream"] is False
        assert target.params["seed"] == 7
        assert target.params["stop"] is None

    def test_unrecognisable_values_stay_strings(self):
        """Provider params are vendor-defined; guessing a type is worse than
        passing the caller's text through."""
        target = Target.parse("openai:gpt-5.4?reasoning_effort=xhigh&user=abc-123")
        assert target.params["reasoning_effort"] == "xhigh"
        assert target.params["user"] == "abc-123"

    @pytest.mark.parametrize(
        "spec",
        [
            "openai:gpt-5.4",
            "ollama:llama3.3:70b",
            "vllm:Qwen/Qwen3-30B#my-box",
            "anthropic:claude-opus-5-5?effort=high",
        ],
    )
    def test_round_trips(self, spec):
        assert Target.parse(Target.parse(spec).spec()) == Target.parse(spec)

    @pytest.mark.parametrize("spec", ["", "   ", ":model", "?effort=high"])
    def test_rejects_unusable_specs(self, spec):
        with pytest.raises(ValueError):
            Target.parse(spec)

    def test_parsing_a_target_is_idempotent(self):
        target = Target.parse("openai:gpt-5.4")
        assert Target.parse(target) is target

    def test_with_params_merges_and_ignores_none(self):
        target = Target.parse("openai:gpt-5.4?effort=low")
        merged = target.with_params(effort="high", temperature=None, seed=1)
        assert merged.params == {"effort": "high", "seed": 1}
        assert target.params == {"effort": "low"}, "the original must be untouched"

    def test_key_excludes_params(self):
        """Health is per endpoint: the same model behind the same credential
        shares a rate limit and a balance regardless of requested effort."""
        a = Target.parse("openai:gpt-5.4?effort=low")
        b = Target.parse("openai:gpt-5.4?effort=max")
        assert a.key == b.key == "openai:gpt-5.4"

    def test_key_uses_the_pinned_instance(self):
        assert Target.parse("vllm:m#box-a").key != Target.parse("vllm:m#box-b").key


# ---------------------------------------------------------------------------
# Settings layering — config is a warm-up, not a cage
# ---------------------------------------------------------------------------


class TestSettingsLayering:
    def test_defaults_are_conservative(self):
        s = RoutingSettings()
        assert s.cascade_enabled is False, "cascade costs an extra call; opt in"
        assert s.on_refusal is RefusalPolicy.FAIL
        assert s.bias is ClassifierBias.QUALITY
        assert s.autoprovision is True
        assert s.default_strategy is SelectionStrategy.PRIORITY

    def test_config_sets_every_knob(self):
        values = {
            "routing.enabled": False,
            "routing.autoprovision": False,
            "routing.default_pool": "main",
            "routing.default_strategy": "lowest_cost",
            "routing.max_attempts": 7,
            "routing.on_refusal": "failover",
            "routing.cascade.enabled": True,
            "routing.cascade.threshold": 0.9,
            "routing.cascade.on_unknown": "escalate",
            "routing.classifier.chain": ["hint", "heuristic"],
            "routing.classifier.bias": "cost",
            "routing.transforms.chain": ["pii"],
        }
        s = RoutingSettings.from_config(lambda k, d=None: values.get(k, d))
        assert s.enabled is False and s.autoprovision is False
        assert s.default_pool == "main"
        assert s.default_strategy is SelectionStrategy.LOWEST_COST
        assert s.max_attempts == 7
        assert s.on_refusal is RefusalPolicy.FAILOVER
        assert s.cascade_enabled is True
        assert s.cascade_threshold == pytest.approx(0.9)
        assert s.on_unknown_verdict is UnknownVerdictPolicy.ESCALATE
        assert s.classifier_chain == ("hint", "heuristic")
        assert s.bias is ClassifierBias.COST
        assert s.transforms == ("pii",)

    def test_unknown_config_values_fall_back_rather_than_raising(self):
        """A typo in a config file should not make llmcore refuse to start."""
        s = RoutingSettings.from_config(
            lambda k, d=None: {"routing.default_strategy": "teleport"}.get(k, d)
        )
        assert s.default_strategy is SelectionStrategy.PRIORITY

    def test_the_request_overrides_config(self):
        s = RoutingSettings.from_config(
            lambda k, d=None: {"routing.max_attempts": 2, "routing.on_refusal": "fail"}.get(k, d)
        )
        out = s.override(max_attempts=9, on_refusal="failover")
        assert out.max_attempts == 9
        assert out.on_refusal is RefusalPolicy.FAILOVER
        assert s.max_attempts == 2, "the original snapshot must be immutable"

    def test_overrides_accept_strings(self):
        """Overrides arrive from CLI flags, env and the bridge as text."""
        s = RoutingSettings().override(
            max_attempts="5", cascade_enabled="true", cascade_threshold="0.8",
            default_strategy="lowest_latency", classifier_chain="hint,heuristic",
        )
        assert s.max_attempts == 5
        assert s.cascade_enabled is True
        assert s.cascade_threshold == pytest.approx(0.8)
        assert s.default_strategy is SelectionStrategy.LOWEST_LATENCY
        assert s.classifier_chain == ("hint", "heuristic")

    def test_none_overrides_are_ignored(self):
        """So chat() can forward a signature full of optional kwargs verbatim."""
        s = RoutingSettings().override(max_attempts=None, on_refusal=None)
        assert s == RoutingSettings()

    def test_a_misspelled_override_raises(self):
        """Silently dropping it would be a routing bug nobody could see."""
        with pytest.raises(TypeError, match="Unknown routing override"):
            RoutingSettings().override(max_attemps=3)

    def test_cooldowns_are_configurable(self):
        s = RoutingSettings.from_config(
            lambda k, d=None: {"routing.cooldowns": {"rate_limit": 1.5, "auth": None}}.get(k, d)
        )
        assert s.cooldown_for(FailureKind.RATE_LIMIT) == pytest.approx(1.5)
        assert s.cooldown_for(FailureKind.AUTH) is None
        # Unlisted kinds keep their defaults.
        assert s.cooldown_for(FailureKind.TIMEOUT) == pytest.approx(15.0)


class TestRefusalPolicyIsAChoice:
    """Whether to fail over on a refusal is config, not law."""

    def test_default_does_not_failover(self):
        assert RoutingSettings().should_failover(FailureKind.REFUSAL) is False

    def test_config_can_turn_it_on(self):
        s = RoutingSettings.from_config(
            lambda k, d=None: {"routing.on_refusal": "failover"}.get(k, d)
        )
        assert s.should_failover(FailureKind.REFUSAL) is True

    def test_a_request_can_turn_it_on(self):
        assert (
            RoutingSettings().override(on_refusal="failover")
            .should_failover(FailureKind.REFUSAL)
            is True
        )

    def test_a_malformed_request_never_fails_over(self):
        """Not policy — a 400 is malformed everywhere, so trying peers only
        multiplies the same error."""
        for s in (RoutingSettings(), RoutingSettings().override(on_refusal="failover")):
            assert s.should_failover(FailureKind.BAD_REQUEST) is False


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


class TestFailureClassification:
    def test_context_length_is_its_own_kind(self):
        kind, _ = classify_failure(
            ContextLengthError(model_name="m", limit=100, actual=200, message="too long")
        )
        assert kind is FailureKind.CONTEXT_LENGTH

    @pytest.mark.parametrize(
        ("message", "status", "expected"),
        [
            ("Rate limit reached", 429, FailureKind.RATE_LIMIT),
            ("not_enough_credits", 403, FailureKind.INSUFFICIENT_CREDIT),
            ("Your credit balance is too low", 400, FailureKind.INSUFFICIENT_CREDIT),
            ("paid_plan_required", 403, FailureKind.INSUFFICIENT_CREDIT),
            ("payment required", 402, FailureKind.INSUFFICIENT_CREDIT),
            ("exceeded your current quota", 429, FailureKind.INSUFFICIENT_CREDIT),
            ("Invalid credentials", 401, FailureKind.AUTH),
            ("maximum context length exceeded", 400, FailureKind.CONTEXT_LENGTH),
            ("Bad parameter", 400, FailureKind.BAD_REQUEST),
            ("Internal server error", 500, FailureKind.SERVER),
            ("Service unavailable", 503, FailureKind.SERVER),
            ("content policy violation", None, FailureKind.REFUSAL),
            ("request timed out", None, FailureKind.TIMEOUT),
        ],
    )
    def test_real_provider_messages(self, message, status, expected):
        """Every string here was observed from a provider llmcore ships."""
        exc = ProviderError("p", message, status_code=status)
        kind, _ = classify_failure(exc)
        assert kind is expected

    def test_billing_outranks_auth(self):
        """A valid key on an empty account answers 403; calling that an auth
        failure would bench the target for the whole process over something a
        top-up fixes."""
        kind, _ = classify_failure(
            ProviderError("p", "not_enough_credits", status_code=403)
        )
        assert kind is FailureKind.INSUFFICIENT_CREDIT

    def test_retry_after_is_surfaced(self):
        exc = ProviderError("p", "slow down", status_code=429)
        exc.retry_after = 42
        _, retry_after = classify_failure(exc)
        assert retry_after == pytest.approx(42.0)

    def test_timeouterror_is_recognised(self):
        kind, _ = classify_failure(TimeoutError("nope"))
        assert kind is FailureKind.TIMEOUT


class TestFailureSemantics:
    def test_only_bad_request_is_pointless_to_retry(self):
        assert FailureKind.BAD_REQUEST.failover_is_pointless
        for kind in (FailureKind.RATE_LIMIT, FailureKind.REFUSAL, FailureKind.SERVER):
            assert not kind.failover_is_pointless

    @pytest.mark.parametrize(
        ("kind", "affects"),
        [
            (FailureKind.RATE_LIMIT, True),
            (FailureKind.INSUFFICIENT_CREDIT, True),
            (FailureKind.TIMEOUT, True),
            (FailureKind.SERVER, True),
            # Properties of the request, not the target.
            (FailureKind.CONTEXT_LENGTH, False),
            (FailureKind.BAD_REQUEST, False),
            (FailureKind.REFUSAL, False),
        ],
    )
    def test_which_failures_penalise_the_target(self, kind, affects):
        assert kind.affects_health is affects

    def test_only_transient_server_errors_retry_the_same_target(self):
        assert FailureKind.SERVER.retry_same
        assert FailureKind.UNKNOWN.retry_same
        assert not FailureKind.RATE_LIMIT.retry_same


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestTargetHealth:
    def test_a_fresh_target_is_available(self):
        assert TargetHealth("t").is_available(NOW)

    def test_rate_limit_cools_down_briefly(self):
        h = TargetHealth("t")
        h.record(Outcome("t", ok=False, failure=FailureKind.RATE_LIMIT), now=NOW)
        assert not h.is_available(NOW)
        assert h.is_available(NOW + timedelta(seconds=25))

    def test_insufficient_credit_cools_down_for_minutes(self):
        """It will not fix itself in seconds — a top-up is a human action."""
        h = TargetHealth("t")
        h.record(Outcome("t", ok=False, failure=FailureKind.INSUFFICIENT_CREDIT), now=NOW)
        assert not h.is_available(NOW + timedelta(seconds=60))
        assert h.is_available(NOW + timedelta(minutes=11))

    def test_insufficient_credit_records_a_zero_balance(self):
        """Better evidence than any balance endpoint: the vendor just said so."""
        h = TargetHealth("t")
        h.record(Outcome("t", ok=False, failure=FailureKind.INSUFFICIENT_CREDIT), now=NOW)
        assert h.balance is not None and h.balance.amount == 0.0

    def test_auth_failure_is_permanent_for_the_process(self):
        h = TargetHealth("t")
        h.record(Outcome("t", ok=False, failure=FailureKind.AUTH), now=NOW)
        assert h.unusable
        assert not h.is_available(NOW + timedelta(days=1))
        assert h.cooldown_remaining(NOW) == float("inf")

    def test_retry_after_overrides_the_default_cooldown(self):
        h = TargetHealth("t")
        h.record(
            Outcome("t", ok=False, failure=FailureKind.RATE_LIMIT, retry_after_seconds=120),
            now=NOW,
        )
        assert not h.is_available(NOW + timedelta(seconds=119))
        assert h.is_available(NOW + timedelta(seconds=121))

    def test_context_length_does_not_bench_the_target(self):
        """Someone else's oversized prompt must not cost a healthy endpoint its
        place in the rotation."""
        h = TargetHealth("t")
        h.record(Outcome("t", ok=False, failure=FailureKind.CONTEXT_LENGTH), now=NOW)
        assert h.is_available(NOW)
        assert h.cooldown_until is None

    def test_a_success_clears_a_cooldown(self):
        h = TargetHealth("t")
        h.record(Outcome("t", ok=False, failure=FailureKind.RATE_LIMIT), now=NOW)
        h.record(Outcome("t", ok=True, latency_seconds=0.4), now=NOW)
        assert h.is_available(NOW)
        assert h.consecutive_failures == 0

    def test_latency_is_smoothed(self):
        h = TargetHealth("t")
        h.record(Outcome("t", ok=True, latency_seconds=1.0), now=NOW)
        assert h.ewma_latency_seconds == pytest.approx(1.0)
        h.record(Outcome("t", ok=True, latency_seconds=2.0), now=NOW)
        # One slow call must not evict a target that is usually fast.
        assert 1.0 < h.ewma_latency_seconds < 2.0

    def test_configured_cooldowns_are_honoured(self):
        h = TargetHealth("t")
        h.record(
            Outcome("t", ok=False, failure=FailureKind.RATE_LIMIT),
            cooldowns={FailureKind.RATE_LIMIT: 1.0},
            now=NOW,
        )
        assert h.is_available(NOW + timedelta(seconds=2))


class TestBalance:
    def test_unknown_is_not_zero(self):
        """Ranking an unknown balance as broke would demote good providers."""
        assert Balance(amount=None).is_known is False
        assert Balance(amount=0.0).is_known is True


class TestRoutingRequest:
    def test_token_estimate_counts_the_whole_request(self):
        r = RoutingRequest(
            prompt="a" * 400,
            system="b" * 400,
            messages=({"role": "user", "content": "c" * 400},),
        )
        assert r.approx_tokens == pytest.approx(300, abs=5)

    def test_non_string_content_is_skipped(self):
        """Multimodal parts are lists; a heuristic must not crash on them."""
        r = RoutingRequest(prompt="hi", messages=({"role": "user", "content": [{"x": 1}]},))
        assert r.approx_tokens >= 0


# ---------------------------------------------------------------------------
# Dynamic targets — config must not be an allow-list
# ---------------------------------------------------------------------------


class TestTargetResolution:
    @pytest.fixture
    def manager(self, monkeypatch):
        from llmcore.providers.manager import ProviderManager

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mgr = ProviderManager.__new__(ProviderManager)
        mgr._providers = {}
        mgr._ephemeral_instances = set()
        mgr._default_provider_name = "openai"
        mgr._config = None
        mgr._log_raw_payloads = False
        return mgr

    def test_an_unknown_provider_lists_the_known_types(self, manager):
        with pytest.raises(ConfigError, match="Known provider types"):
            manager.resolve_target("nonsense:model")

    def test_a_bad_instance_pin_is_a_config_error(self, manager):
        """An explicit pin that does not exist is not an invitation to build
        something similar."""
        with pytest.raises(ConfigError, match="pins instance"):
            manager.resolve_target("openai:gpt-5.4#no-such-instance")

    def test_autoprovision_can_be_disabled(self, manager):
        with pytest.raises(ConfigError, match="autoprovisioning is disabled"):
            manager.resolve_target("openai:gpt-5.4", autoprovision=False)

    def test_a_missing_credential_names_what_to_set(self, manager, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
            manager.resolve_target("anthropic:claude-opus-5-5")

    def test_credential_discovery_tries_every_spelling(self):
        from llmcore.providers.manager import ProviderManager

        _, tried = ProviderManager._discover_credential("gemini", None)
        assert "GOOGLE_API_KEY" in tried and "GEMINI_API_KEY" in tried

    def test_compat_providers_get_their_own_env_var(self):
        from llmcore.providers.manager import (
            _OPENAI_COMPATIBLE_DEFAULTS,
            ProviderManager,
        )

        _, tried = ProviderManager._discover_credential(
            "xai", _OPENAI_COMPATIBLE_DEFAULTS["xai"]
        )
        assert tried[0] == "XAI_API_KEY"
