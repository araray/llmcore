from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import llmcore.tokens as tokens
from llmcore.tokens import EstimateCounter, TiktokenCounter, count_tokens, get_counter


class FakeEncoding:
    def __init__(self, name: str, token_count: int = 3) -> None:
        self.name = name
        self.token_count = token_count

    def encode(self, text: str) -> list[str]:
        return [f"{self.name}:{i}" for i in range(self.token_count) if text]


class FakeTiktoken:
    def __init__(self) -> None:
        self.encoding_for_model_calls: list[str] = []
        self.get_encoding_calls: list[str] = []

    def encoding_for_model(self, model: str) -> FakeEncoding:
        self.encoding_for_model_calls.append(model)
        if model == "known-model":
            return FakeEncoding("model", token_count=5)
        raise KeyError(model)

    def get_encoding(self, name: str) -> FakeEncoding:
        self.get_encoding_calls.append(name)
        return FakeEncoding(name, token_count=7)


@pytest.fixture(autouse=True)
def reset_warning_state() -> None:
    tokens._warned_keys.clear()


class TestEstimateCounter:
    def test_counts_empty_as_zero(self) -> None:
        assert EstimateCounter().count("") == 0
        assert EstimateCounter().count(None) == 0

    def test_uses_ceiling_character_ratio(self) -> None:
        assert EstimateCounter().count("abcd") == 1
        assert EstimateCounter().count("abcde") == 2

    def test_rejects_invalid_ratio(self) -> None:
        with pytest.raises(ValueError, match="chars_per_token"):
            EstimateCounter(chars_per_token=0)


class TestCountTokens:
    def test_missing_tiktoken_falls_back_to_estimate(self, monkeypatch, caplog) -> None:
        monkeypatch.setitem(sys.modules, "tiktoken", None)

        with caplog.at_level("WARNING", logger="llmcore.tokens"):
            assert count_tokens("abcdefghi") == 3
            assert count_tokens("abcdefghi") == 3

        warnings = [r for r in caplog.records if "tiktoken is not available" in r.message]
        assert len(warnings) == 1
        assert isinstance(get_counter(), EstimateCounter)

    def test_known_model_uses_encoding_for_model(self, monkeypatch) -> None:
        fake = FakeTiktoken()
        monkeypatch.setitem(sys.modules, "tiktoken", fake)

        assert count_tokens("hello world", model="known-model") == 5
        assert fake.encoding_for_model_calls == ["known-model"]
        assert fake.get_encoding_calls == []

    def test_unknown_frontier_model_uses_o200k_base(self, monkeypatch, caplog) -> None:
        fake = FakeTiktoken()
        monkeypatch.setitem(sys.modules, "tiktoken", fake)

        with caplog.at_level("WARNING", logger="llmcore.tokens"):
            assert count_tokens("hello world", model="gpt-5-mini") == 7

        assert fake.encoding_for_model_calls == ["gpt-5-mini"]
        assert fake.get_encoding_calls == ["o200k_base"]
        assert any("gpt-5-mini" in r.message for r in caplog.records)

    def test_unknown_non_frontier_model_uses_cl100k_base(self, monkeypatch) -> None:
        fake = FakeTiktoken()
        monkeypatch.setitem(sys.modules, "tiktoken", fake)

        assert count_tokens("hello world", model="custom-model") == 7
        assert fake.get_encoding_calls == ["cl100k_base"]

    def test_no_model_uses_default_tiktoken_encoding(self, monkeypatch) -> None:
        fake = FakeTiktoken()
        monkeypatch.setitem(sys.modules, "tiktoken", fake)

        assert count_tokens("hello world") == 7
        assert fake.encoding_for_model_calls == []
        assert fake.get_encoding_calls == ["cl100k_base"]


class TestTiktokenCounter:
    def test_direct_construction_raises_when_missing(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "tiktoken", None)

        with pytest.raises(RuntimeError, match="tiktoken is not available"):
            TiktokenCounter()

    def test_direct_construction_uses_selected_encoding(self, monkeypatch) -> None:
        fake = SimpleNamespace(
            encoding_for_model=lambda model: FakeEncoding(model, token_count=4),
            get_encoding=lambda name: FakeEncoding(name, token_count=2),
        )
        monkeypatch.setitem(sys.modules, "tiktoken", fake)

        assert TiktokenCounter(model="known-model").count("hello") == 4
