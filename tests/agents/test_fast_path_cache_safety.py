"""The fast-path response cache must not serve another conversation's answer.

Grounded in 6,694 real harness prompts. 35.1% are exact duplicates, but of
the 483 repeated prompts **92% recur inside a single session** and **63%
produced outputs differing by more than 2x** -- the same text with a
different correct answer, because the conversation state moved. The most
repeated prompt of all was "continue... ensure you commit often" (78×).

So the cache had three defects, each demonstrable:

* the key was the prompt text alone, so one session's answer to "continue"
  was served to another's;
* lookup used Jaccard over word *sets*, which ignores order -- "delete the
  old file" and "the file delete old" matched perfectly;
* nothing stopped a state-dependent prompt being stored in the first place.
"""

from __future__ import annotations

import pytest

from llmcore.agents.learning.fast_path import (
    ACKNOWLEDGEMENTS,
    STATE_POINTER_OPENINGS,
    ResponseCache,
    is_cacheable,
)


class TestScopeIsolation:
    def test_one_conversation_cannot_read_anothers_entry(self):
        cache = ResponseCache()
        cache.set("what is 2+2", "4", scope="session-a")
        assert cache.get("what is 2+2", scope="session-a") == "4"
        assert cache.get("what is 2+2", scope="session-b") is None

    def test_the_same_scope_still_hits(self):
        cache = ResponseCache()
        cache.set("what is the capital of France", "Paris", scope="s")
        assert cache.get("what is the capital of France", scope="s") == "Paris"

    def test_normalisation_still_applies_within_a_scope(self):
        cache = ResponseCache()
        cache.set("What Is 2+2", "4", scope="s")
        assert cache.get("  what is 2+2  ", scope="s") == "4"

    def test_an_absent_scope_is_its_own_namespace(self):
        cache = ResponseCache()
        cache.set("q", "global", scope=None)
        assert cache.get("q", scope=None) == "global"
        assert cache.get("q", scope="s") is None

    def test_fuzzy_lookup_does_not_leak_across_scopes(self):
        cache = ResponseCache(similarity_threshold=0.5)
        cache.set("please delete the old file now", "done", scope="a")
        assert cache.get("please delete the old file", scope="b") is None


class TestStatePointersAreNeverCached:
    @pytest.mark.parametrize("query", [
        "continue",
        "Continue",
        "continue... ensure you commit often",   # the most repeated real prompt
        "continue working on the router",
        "proceed",
        "go on",
        "next",
        "resume where you left off",
        "keep going",
    ])
    def test_a_continuation_is_not_cacheable(self, query):
        assert is_cacheable(query) is False

    @pytest.mark.parametrize("query", ["ok", "OK", "yes", "thanks", "thank you",
                                       "got it", "perfect", "done"])
    def test_a_bare_acknowledgement_is_not_cacheable(self, query):
        assert is_cacheable(query) is False

    @pytest.mark.parametrize("query", [
        "what is the capital of France",
        "explain what a monad is",
        "convert 10 miles to kilometres",
        "okay so what does git rebase --onto actually do",  # not a bare ack
    ])
    def test_a_self_contained_question_is_cacheable(self, query):
        assert is_cacheable(query) is True

    def test_storing_a_state_pointer_is_refused_not_merely_unread(self):
        # Refusing at write time means the mistake cannot be made later by
        # a different reader.
        cache = ResponseCache()
        cache.set("continue", "stale answer", scope="s")
        assert cache.get("continue", scope="s") is None
        assert cache._cache == {}

    def test_an_empty_prompt_is_not_cacheable(self):
        assert is_cacheable("") is False
        assert is_cacheable("   ") is False
        assert is_cacheable("...") is False

    def test_the_vocabularies_are_not_empty(self):
        # A typo emptying either set would silently disable the guard.
        assert "continue" in STATE_POINTER_OPENINGS
        assert "ok" in ACKNOWLEDGEMENTS


class TestOrderSensitivity:
    def test_reordered_words_are_not_a_perfect_match(self):
        cache = ResponseCache(similarity_threshold=0.8)
        cache.set("delete the old file", "Deleted old_file.txt", scope="s")
        assert cache.get("the file delete old", scope="s") is None

    def test_the_identical_query_still_matches(self):
        cache = ResponseCache(similarity_threshold=0.8)
        cache.set("delete the old file", "Deleted old_file.txt", scope="s")
        assert cache.get("delete the old file", scope="s") == "Deleted old_file.txt"

    def test_exact_matching_is_the_default(self):
        # Measured: normalising real prompts raised the duplicate rate only
        # from 35.1% to 35.4%, so fuzzy matching buys almost nothing and
        # risks a wrong answer. It is opt-in.
        assert ResponseCache().similarity_threshold == 1.0

    def test_with_exact_matching_a_near_miss_misses(self):
        cache = ResponseCache()
        cache.set("what is the capital of France", "Paris", scope="s")
        assert cache.get("what is the capital of france?", scope="s") is None

    def test_a_close_paraphrase_can_still_hit_when_fuzzy_is_enabled(self):
        cache = ResponseCache(similarity_threshold=0.6)
        cache.set("show me the open pull requests", "…", scope="s")
        assert cache.get("show me the open pull requests now", scope="s") == "…"


class TestHousekeeping:
    def test_expired_entries_are_dropped(self):
        cache = ResponseCache(ttl_seconds=0.0)
        cache.set("what is 2+2", "4", scope="s")
        assert cache.get("what is 2+2", scope="s") is None

    def test_pruning_respects_max_entries(self):
        cache = ResponseCache(max_entries=5)
        for i in range(20):
            cache.set(f"what is {i} plus one", str(i + 1), scope="s")
        assert len(cache._cache) <= 5

    def test_clear_empties_every_scope(self):
        cache = ResponseCache()
        cache.set("what is 2+2", "4", scope="a")
        cache.set("what is 3+3", "6", scope="b")
        cache.clear()
        assert cache.get("what is 2+2", scope="a") is None
