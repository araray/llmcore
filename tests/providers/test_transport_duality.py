# tests/providers/test_transport_duality.py
"""Every provider should talk to its vendor two ways where that is possible.

llmcore's standing rule is **prefer calling the API directly, and fall back to
the vendor's Python SDK where one exists**. Direct calls keep llmcore working
when an SDK lags the API or pins awkward dependencies; the SDK fallback absorbs
per-vendor request shaping that would otherwise have to be reimplemented here.

This file is the audit, encoded. A provider that offers only one transport must
appear in :data:`SINGLE_TRANSPORT_REASONS` with a reason, so the choice is
*recorded* rather than accidental — the failure mode this guards against is a
provider quietly shipping SDK-only and nobody noticing, which is exactly what
happened with the Higgsfield ``sdk`` backend that was advertised, never
instantiated, and never called.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import ClassVar

import pytest

from llmcore.providers.manager import _PROVIDER_INSTANCE_ALIASES, PROVIDER_MAP

PROVIDER_SRC = Path(__file__).resolve().parents[2] / "src" / "llmcore" / "providers"


def _canonical() -> list[str]:
    return sorted(k for k in PROVIDER_MAP if k not in _PROVIDER_INSTANCE_ALIASES)


def _module_source(provider: str) -> str:
    """Return the source of *provider*'s class and every base it inherits from.

    Reading only the provider's own module is wrong: ``deepinfra`` and ``vllm``
    subclass ``OpenAIProvider`` and inherit its transport selector, so they have
    dual transport without a line of their own about it. Walking the MRO is what
    makes the audit see that — the first version of this helper did not, and
    classified both as single-transport.
    """
    chunks: list[str] = []
    for klass in PROVIDER_MAP[provider].__mro__:
        module = getattr(klass, "__module__", "")
        if not module.startswith("llmcore.providers"):
            continue
        path = PROVIDER_SRC / f"{module.rsplit('.', 1)[-1]}.py"
        if path.is_file():
            chunks.append(path.read_text())
    return "\n".join(chunks)


#: Providers that legitimately have one transport, and why. Anything not listed
#: here is expected to offer both.
SINGLE_TRANSPORT_REASONS: dict[str, str] = {
    # --- No *official* vendor SDK exists ----------------------------------
    #
    # Checked against PyPI rather than assumed, because an earlier version of
    # this list claimed TypeSafe published no SDK when `typesafe-sdk` 0.7.2 is
    # official and was already cloned in the vendor repos. The packages named
    # below are the ones that turn up in a search and are *not* from the vendor.
    "deepseek": (
        "OpenAI-compatible API; DeepSeek publishes no Python SDK. The PyPI "
        "names `deepseek` (Deskpai.com) and `deepseek-sdk` (Sifat Hasan) are "
        "third-party, and DeepSeek's own docs direct users to the openai SDK."
    ),
    "kimi": (
        "OpenAI-compatible API; Moonshot publishes no Python SDK. `kimi-sdk` on "
        "PyPI names no author or repository and is not identifiably official."
    ),
}

#: Markers that indicate a module can select between transports at runtime.
_SELECTOR_PATTERNS = (
    r"_resolve_backend",
    r'config\.get\(\s*"backend"',
    r'config\.get\(\s*"transport"',
    r'config\.get\(\s*"media_backend"',
    r"_use_sdk_for",
)


def _has_transport_selector(source: str) -> bool:
    return any(re.search(p, source) for p in _SELECTOR_PATTERNS)


def _has_direct_http(source: str) -> bool:
    return bool(re.search(r"httpx\.AsyncClient\(", source))


class TestTransportDualityIsRecorded:
    """The audit itself: no provider is single-transport by accident."""

    @pytest.mark.parametrize("provider", _canonical())
    def test_single_transport_providers_declare_a_reason(self, provider):
        source = _module_source(provider)
        if _has_transport_selector(source):
            return  # dual (or better) — nothing to justify
        assert provider in SINGLE_TRANSPORT_REASONS, (
            f"'{provider}' offers only one transport and is not listed in "
            f"SINGLE_TRANSPORT_REASONS. llmcore's rule is to call the API "
            f"directly and fall back to the vendor SDK where one exists — so "
            f"either add the second transport, or record here why there is only "
            f"one."
        )

    def test_the_reason_list_has_no_stale_entries(self):
        """A provider that gained a second transport must leave the exemption
        list, or the list stops meaning anything."""
        stale = [
            provider
            for provider in SINGLE_TRANSPORT_REASONS
            if provider in PROVIDER_MAP and _has_transport_selector(_module_source(provider))
        ]
        assert stale == [], (
            f"These providers now have a transport selector but are still listed "
            f"as single-transport: {stale}. Remove them from "
            f"SINGLE_TRANSPORT_REASONS."
        )

    def test_every_exemption_names_a_real_provider(self):
        unknown = sorted(set(SINGLE_TRANSPORT_REASONS) - set(PROVIDER_MAP))
        assert unknown == [], f"exemptions for providers that do not exist: {unknown}"

    def test_gaps_are_labelled_as_gaps(self):
        """An exemption is either 'no SDK exists' or an acknowledged GAP. Being
        explicit stops a temporary gap from reading like a design decision."""
        for provider, reason in SINGLE_TRANSPORT_REASONS.items():
            acceptable = reason.startswith("GAP:") or "no Python SDK" in reason
            assert acceptable, (
                f"{provider}: the reason should either state that no SDK exists or "
                f"be marked 'GAP:' so it is tracked. Got: {reason!r}"
            )


class TestDualTransportProvidersActuallyUseBoth:
    """A declared backend that is never called is worse than no backend.

    The Higgsfield provider shipped with ``backend = "sdk"`` accepted by its
    resolver, an SDK import, and an ``self._sdk`` attribute that was never
    assigned and never used — so selecting it silently did nothing. These tests
    make that shape fail.
    """

    DUAL = ("fal", "elevenlabs", "replicate", "higgsfield", "typesafe", "mistral")

    #: Providers whose *direct* path was added alongside an existing SDK path.
    #: They are checked for a direct client and a selector, but not for
    #: "direct is the default" — their SDKs own realtime sockets, ADC token
    #: exchange, prompt-caching headers and retry policy, so the SDK remains the
    #: default and direct is opt-in. That is a documented deviation rather than
    #: an oversight, which is why it is listed here explicitly.
    SDK_DEFAULT: ClassVar[tuple[str, ...]] = (
        "anthropic", "gemini", "deepgram", "ollama",
    )

    @pytest.mark.parametrize("provider", DUAL)
    def test_the_sdk_client_is_instantiated(self, provider):
        source = _module_source(provider)
        # Accepts both `self._sdk = vendor_module.Client(...)` and
        # `self._sdk = ImportedClient(...)`; the first version of this pattern
        # only matched the dotted form and reported a false negative.
        assert re.search(r"self\._sdk\s*=\s*_?[A-Za-z]\w*[.(]", source), (
            f"{provider} declares an SDK backend but never constructs a client, "
            f"so selecting it would silently fall through to the other transport."
        )

    @pytest.mark.parametrize("provider", DUAL)
    def test_the_sdk_client_is_actually_called(self, provider):
        source = _module_source(provider)
        calls = re.findall(r"self\._sdk\.(\w+)", source)
        assert calls, f"{provider} constructs an SDK client but never calls it."

    @pytest.mark.parametrize("provider", DUAL)
    def test_both_transports_are_reachable(self, provider):
        source = _module_source(provider)
        assert _has_direct_http(source), f"{provider} has no direct HTTP client"
        assert _has_transport_selector(source), f"{provider} cannot select a transport"

    @pytest.mark.parametrize("provider", DUAL)
    def test_direct_is_the_default(self, provider):
        """The house rule is direct-first. Hugging Face is the one documented
        exception and is not in this list."""
        source = _module_source(provider)
        match = re.search(r'for backend in \("(\w+)",', source)
        assert match, f"{provider}: could not determine transport preference order"
        assert match.group(1) == "httpx", (
            f"{provider} prefers {match.group(1)!r} over direct REST without being "
            f"the documented Hugging Face exception."
        )


class TestSdkDefaultProvidersStillHaveBothPaths:
    """Providers where the SDK stays the default must still offer direct REST."""

    @pytest.mark.parametrize(
        "provider", TestDualTransportProvidersActuallyUseBoth.SDK_DEFAULT
    )
    def test_a_direct_http_client_exists(self, provider):
        assert _has_direct_http(_module_source(provider)), (
            f"{provider} is listed as having a direct path but builds no "
            f"httpx client."
        )

    @pytest.mark.parametrize(
        "provider", TestDualTransportProvidersActuallyUseBoth.SDK_DEFAULT
    )
    def test_the_transport_is_selectable(self, provider):
        assert _has_transport_selector(_module_source(provider))

    @pytest.mark.parametrize(
        "provider", TestDualTransportProvidersActuallyUseBoth.SDK_DEFAULT
    )
    def test_the_direct_path_maps_its_own_errors(self, provider):
        """A direct path that raised raw httpx errors would make the two
        transports report the same condition differently."""
        assert "_raise_direct_status" in _module_source(provider), (
            f"{provider}'s direct path has no error mapper, so failures would "
            f"not match the SDK path's exceptions."
        )


class TestSdkExtrasAreDeclared:
    """An SDK fallback nobody can install is not a fallback."""

    SDK_EXTRAS: ClassVar[dict[str, str]] = {
        "fal": "fal-client",
        "elevenlabs": "elevenlabs",
        "replicate": "replicate",
        "higgsfield": "higgsfield-client",
        "typesafe": "typesafe-sdk",
        "mistral": "mistralai",
    }

    @pytest.mark.parametrize(("provider", "package"), sorted(SDK_EXTRAS.items()))
    def test_the_sdk_is_in_its_extra(self, provider, package):
        pyproject = (PROVIDER_SRC.parents[2] / "pyproject.toml").read_text()
        match = re.search(rf"^{provider} = \[(.+?)\]", pyproject, re.M | re.S)
        assert match, f"no [{provider}] extra declared in pyproject.toml"
        assert package in match.group(1), (
            f"the {provider} extra does not include {package!r}, so "
            f"backend = 'sdk' is unreachable for anyone installing "
            f"llmcore[{provider}]."
        )


class TestHiggsfieldSdkPath:
    """Covers the specific regression: an advertised-but-dead SDK backend."""

    def test_selecting_the_sdk_builds_a_client(self):
        from llmcore.providers.higgsfield_provider import HiggsfieldProvider

        provider = HiggsfieldProvider(
            {"api_key": "id:secret", "_instance_name": "h", "backend": "sdk"}
        )
        assert provider._backend == "sdk"
        assert provider._sdk is not None, "the SDK backend must build a client"
        assert type(provider._sdk).__name__ == "AsyncClient"

    def test_the_sdk_status_helper_normalizes_state(self):
        """The SDK signals state by returning a different class, not a field."""
        from llmcore.providers.higgsfield_provider import _SDK_STATUS_CLASSES

        for name in ("Queued", "InProgress", "Completed", "Failed", "NSFW", "Cancelled"):
            assert name in _SDK_STATUS_CLASSES

    def test_the_sdk_signature_matches_what_the_adapter_calls(self):
        """Pinned because the adapter was written against 0.1.0 and the
        installed package is 0.2.0."""
        higgsfield_client = pytest.importorskip("higgsfield_client")
        submit = inspect.signature(higgsfield_client.AsyncClient.submit)
        assert {"application", "arguments", "webhook_url"} <= set(submit.parameters)
        for method in ("status", "result", "cancel"):
            params = inspect.signature(
                getattr(higgsfield_client.AsyncClient, method)
            ).parameters
            assert "request_id" in params
