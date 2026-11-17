"""Poe bots that reject system messages get the system text in the first user turn."""
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("openai")
from llmcore.models import Message  # noqa: E402
from llmcore.models import Role as LLMCoreRole  # noqa: E402
from llmcore.providers.poe_provider import PoeProvider, fold_system_into_user  # noqa: E402


def msg(role, content):
    return Message(role=role, content=content, session_id="s")


def roles(context):
    return [m.role if isinstance(m.role, str) else m.role.value for m in context]


def test_fold_moves_system_text_to_the_first_user_turn_only():
    context = [msg(LLMCoreRole.SYSTEM, "Be brief."), msg(LLMCoreRole.USER, "Hello"),
               msg(LLMCoreRole.ASSISTANT, "Hi"), msg(LLMCoreRole.USER, "Again")]
    folded = fold_system_into_user(context)
    assert roles(folded) == ["user", "assistant", "user"]
    assert folded[0].content.startswith("Instructions:\nBe brief.") and folded[0].content.endswith("Hello")
    assert folded[2].content == "Again"
    assert fold_system_into_user([msg(LLMCoreRole.USER, "x")])[0].content == "x"  # nothing to fold
    only = fold_system_into_user([msg(LLMCoreRole.SYSTEM, "Rules")])
    assert roles(only) == ["user"] and only[0].content == "Instructions:\nRules"


@pytest.mark.parametrize("bot,folds", [("mistral-medium", True), ("Mistral-Medium", True), ("GPT-5-mini", False)])
async def test_configured_bots_receive_a_folded_context(bot, folds):
    with patch("llmcore.providers.poe_provider.OpenAIProvider.__init__", return_value=None):
        provider = PoeProvider.__new__(PoeProvider)
        provider._backend = "openai"
        provider.default_model = bot
        provider._no_system_bots = {"mistral-medium"}
        provider._bot_parameters = lambda model, kwargs: kwargs
        sent = AsyncMock(return_value={"choices": []})
        with patch("llmcore.providers.poe_provider.OpenAIProvider.chat_completion", sent):
            await provider.chat_completion([msg(LLMCoreRole.SYSTEM, "Rules"), msg(LLMCoreRole.USER, "Q")], model=bot)
        context = sent.call_args.args[0]
        assert ("system" not in roles(context)) is folds
