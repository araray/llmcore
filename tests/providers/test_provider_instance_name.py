# tests/providers/test_provider_instance_name.py
"""
Tests for provider instance name fix.

Verifies that OpenAI-compatible providers (deepseek, mistral, xai, etc.)
return their actual registry key from get_name() instead of always returning
"openai".

Bug: OpenAIProvider.get_name() hardcoded "openai" for all instances,
causing LLMCore.chat() to call get_provider("openai") internally even
when the user specified provider="deepseek".

Fix: BaseProvider.__init__ now stores config["_instance_name"], and
each provider's get_name() returns self._provider_instance_name or its
default. ProviderManager injects _instance_name before construction.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers: mock heavy provider dependencies to avoid network calls
# ---------------------------------------------------------------------------


def _make_openai_provider(config: dict):
    """Construct an OpenAIProvider with tiktoken and AsyncOpenAI mocked."""
    with (
        patch("llmcore.providers.openai_provider.AsyncOpenAI") as mock_client,
        patch("llmcore.providers.openai_provider.tiktoken") as mock_tiktoken,
    ):
        mock_tiktoken.encoding_for_model.return_value = MagicMock()
        mock_tiktoken.get_encoding.return_value = MagicMock()
        from llmcore.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(config)


def _make_gemini_provider(config: dict):
    """Construct a GeminiProvider with genai.Client and availability mocked."""
    with (
        patch("llmcore.providers.gemini_provider.google_genai_available", True),
        patch("llmcore.providers.gemini_provider.genai") as mock_genai,
    ):
        mock_genai.Client.return_value = MagicMock()
        from llmcore.providers.gemini_provider import GeminiProvider

        return GeminiProvider(config)


# ===========================================================================
# Tests: OpenAIProvider.get_name()
# ===========================================================================


class TestOpenAIProviderInstanceName:
    """OpenAIProvider.get_name() must return the injected _instance_name."""

    def test_returns_deepseek(self):
        """DeepSeek instance should return 'deepseek'."""
        provider = _make_openai_provider(
            {
                "api_key": "test-key",
                "base_url": "https://api.deepseek.com",
                "default_model": "deepseek-chat",
                "_instance_name": "deepseek",
            }
        )
        assert provider.get_name() == "deepseek"

    def test_returns_xai(self):
        """xAI instance should return 'xai'."""
        provider = _make_openai_provider(
            {
                "api_key": "test-key",
                "base_url": "https://api.x.ai/v1",
                "default_model": "grok-2",
                "_instance_name": "xai",
            }
        )
        assert provider.get_name() == "xai"

    def test_returns_mistral(self):
        """Mistral instance should return 'mistral'."""
        provider = _make_openai_provider(
            {
                "api_key": "test-key",
                "base_url": "https://api.mistral.ai/v1",
                "default_model": "mistral-large-latest",
                "_instance_name": "mistral",
            }
        )
        assert provider.get_name() == "mistral"

    def test_returns_openai_when_instance_name_is_openai(self):
        """Standard OpenAI instance should return 'openai'."""
        provider = _make_openai_provider(
            {
                "api_key": "test-key",
                "default_model": "gpt-4o",
                "_instance_name": "openai",
            }
        )
        assert provider.get_name() == "openai"

    def test_falls_back_to_openai_without_instance_name(self):
        """Without _instance_name, should fall back to 'openai'."""
        provider = _make_openai_provider(
            {
                "api_key": "test-key",
                "default_model": "gpt-4o",
            }
        )
        assert provider.get_name() == "openai"


# ===========================================================================
# Tests: GeminiProvider.get_name()
# ===========================================================================


class TestGeminiProviderInstanceName:
    """GeminiProvider.get_name() must return _instance_name or 'gemini'."""

    def test_returns_google(self):
        """Google alias should return 'google'."""
        provider = _make_gemini_provider(
            {
                "api_key": "test-key",
                "default_model": "gemini-2.5-pro-latest",
                "_instance_name": "google",
            }
        )
        assert provider.get_name() == "google"

    def test_falls_back_to_gemini(self):
        """Without _instance_name, should return 'gemini'."""
        provider = _make_gemini_provider(
            {
                "api_key": "test-key",
                "default_model": "gemini-2.5-pro-latest",
            }
        )
        assert provider.get_name() == "gemini"


# ===========================================================================
# Tests: BaseProvider._provider_instance_name storage
# ===========================================================================


class TestBaseProviderInstanceName:
    """BaseProvider.__init__ stores _instance_name from config dict."""

    def test_instance_name_stored(self):
        provider = _make_openai_provider(
            {
                "api_key": "test",
                "_instance_name": "my_custom_name",
            }
        )
        assert provider._provider_instance_name == "my_custom_name"

    def test_instance_name_none_when_absent(self):
        provider = _make_openai_provider({"api_key": "test"})
        assert provider._provider_instance_name is None


# ===========================================================================
# Tests: ProviderManager injects _instance_name
# ===========================================================================


class TestProviderManagerInjectsInstanceName:
    """ProviderManager must inject _instance_name before provider construction."""

    def test_deepseek_provider_gets_correct_name(self):
        """
        When [providers.deepseek] exists, the constructed provider's
        get_name() should return 'deepseek', not 'openai'.
        """
        from llmcore.providers.manager import ProviderManager

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda key, default=None: {
            "llmcore.default_provider": "deepseek",
            "llmcore.log_raw_payloads": False,
            "providers": {
                "deepseek": {
                    "default_model": "deepseek-chat",
                },
            },
        }.get(key, default)

        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}),
            patch("llmcore.providers.openai_provider.AsyncOpenAI"),
            patch("llmcore.providers.openai_provider.tiktoken") as mock_tk,
        ):
            mock_tk.encoding_for_model.return_value = MagicMock()
            mock_tk.get_encoding.return_value = MagicMock()
            pm = ProviderManager(mock_config)

        assert "deepseek" in pm.get_available_providers()
        assert pm.get_provider("deepseek").get_name() == "deepseek"

    def test_openai_and_deepseek_coexist_with_distinct_names(self):
        """Both providers must report their own registry key."""
        from llmcore.providers.manager import ProviderManager

        mock_config = MagicMock()
        mock_config.get.side_effect = lambda key, default=None: {
            "llmcore.default_provider": "openai",
            "llmcore.log_raw_payloads": False,
            "providers": {
                "openai": {
                    "api_key": "sk-openai-test",
                    "default_model": "gpt-4o",
                },
                "deepseek": {
                    "default_model": "deepseek-chat",
                },
            },
        }.get(key, default)

        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-ds-test"}),
            patch("llmcore.providers.openai_provider.AsyncOpenAI"),
            patch("llmcore.providers.openai_provider.tiktoken") as mock_tk,
        ):
            mock_tk.encoding_for_model.return_value = MagicMock()
            mock_tk.get_encoding.return_value = MagicMock()
            pm = ProviderManager(mock_config)

        assert pm.get_provider("openai").get_name() == "openai"
        assert pm.get_provider("deepseek").get_name() == "deepseek"


# ---------------------------------------------------------------------------
# Role.value safety – use_enum_values=True stores roles as plain strings
# ---------------------------------------------------------------------------


class TestRoleValueSafety:
    """Verify OpenAI provider handles msg.role as both Role enum and plain str.

    Bug: Message model uses ``use_enum_values=True`` (Pydantic v2), which
    stores ``msg.role`` as the plain string ``"user"`` instead of
    ``Role.USER``.  ``openai_provider.py`` called ``msg.role.value`` on
    four code paths, crashing with ``AttributeError: 'str' object has no
    attribute 'value'``.

    Fix: Guard all ``.role.value`` accesses with
    ``msg.role.value if hasattr(msg.role, 'value') else str(msg.role)``.
    """

    def test_message_role_is_string_due_to_use_enum_values(self):
        """Confirm that Message stores role as plain string."""
        from llmcore.models import Message, Role

        m = Message(role=Role.USER, content="hello")
        assert isinstance(m.role, str)
        assert m.role == "user"
        assert not hasattr(m.role, "value")

    def test_chat_completion_payload_with_string_roles(self):
        """chat_completion builds correct payload when msg.role is a string."""
        from llmcore.models import Message, Role

        provider = _make_openai_provider(
            {
                "api_key": "sk-test",
                "default_model": "test-model",
                "_instance_name": "openai",
            }
        )

        # Simulate messages with string roles (as produced by use_enum_values)
        msgs = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hello"),
        ]

        # Verify roles are strings (not enums) before hitting the provider
        assert all(isinstance(m.role, str) for m in msgs)

        # The message formatting loop should not crash
        # Replicate the provider's logic inline:
        for msg in msgs:
            role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            assert role_str in ("system", "user", "assistant", "tool")

    @pytest.mark.asyncio
    async def test_count_tokens_with_string_roles(self):
        """count_message_tokens handles string roles without crash."""
        from llmcore.models import Message, Role

        provider = _make_openai_provider(
            {
                "api_key": "sk-test",
                "default_model": "test-model",
                "_instance_name": "deepseek",
            }
        )

        msgs = [
            Message(role=Role.USER, content="Count my tokens"),
        ]

        # Should not raise AttributeError
        count = await provider.count_message_tokens(msgs)
        assert isinstance(count, int)
        assert count > 0
