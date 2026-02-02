# tests/embedding/test_cache.py
"""
Tests for the embedding cache implementation.

Tests cover:
- LRU cache basic operations and eviction
- Disk cache persistence and TTL
- Two-tier cache integration
- Batch operations
- Thread safety
- Statistics tracking

Reference: UNIFIED_IMPLEMENTATION_PLAN.md Phase 1, Task 1.2
"""

import threading
import time
from pathlib import Path
from typing import List

import pytest

from llmcore.embedding.cache import (
    DiskCache,
    EmbeddingCache,
    EmbeddingCacheConfig,
    LRUCache,
    create_embedding_cache,
)

# =============================================================================
# LRU CACHE TESTS
# =============================================================================


class TestLRUCache:
    """Tests for the LRU in-memory cache."""

    def test_basic_get_set(self) -> None:
        """Test basic get and set operations."""
        cache = LRUCache(maxsize=10)

        cache.set("key1", [1.0, 2.0, 3.0])
        cache.set("key2", [4.0, 5.0, 6.0])

        assert cache.get("key1") == [1.0, 2.0, 3.0]
        assert cache.get("key2") == [4.0, 5.0, 6.0]
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self) -> None:
        """Test that LRU eviction works correctly."""
        cache = LRUCache(maxsize=3)

        cache.set("a", [1.0])
        cache.set("b", [2.0])
        cache.set("c", [3.0])

        # All should be present
        assert cache.get("a") == [1.0]
        assert cache.get("b") == [2.0]
        assert cache.get("c") == [3.0]

        # Add a fourth item, should evict "a" (LRU after we accessed b and c last)
        # Actually after accessing a, b, c above, "a" is most recently used
        # Let me reconsider: after get("a"), get("b"), get("c"),
        # order is a, b, c so "a" was accessed first and should be evicted
        # But actually LRU moves item to end on get, so order after gets is a, b, c
        # meaning "a" is first and will be evicted... but we accessed a first, so
        # after all gets, order should be b, c, a... no wait.

        # Let me reset and be clearer:
        cache2 = LRUCache(maxsize=2)
        cache2.set("a", [1.0])
        cache2.set("b", [2.0])

        # Access "a" to make it recently used
        _ = cache2.get("a")  # Now order is b, a (b is LRU)

        # Add "c", should evict "b"
        cache2.set("c", [3.0])

        assert cache2.get("a") == [1.0]  # Still present
        assert cache2.get("c") == [3.0]  # New item
        assert cache2.get("b") is None  # Evicted

    def test_update_existing_key(self) -> None:
        """Test that updating an existing key moves it to end."""
        cache = LRUCache(maxsize=3)

        cache.set("a", [1.0])
        cache.set("b", [2.0])
        cache.set("c", [3.0])

        # Update "a" with new value
        cache.set("a", [1.5])

        # Add "d", should evict "b" (not "a" since it was just updated)
        cache.set("d", [4.0])

        assert cache.get("a") == [1.5]
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == [3.0]
        assert cache.get("d") == [4.0]

    def test_zero_maxsize_disabled(self) -> None:
        """Test that maxsize=0 disables the cache."""
        cache = LRUCache(maxsize=0)

        cache.set("key1", [1.0, 2.0])
        assert cache.get("key1") is None
        assert len(cache) == 0

    def test_stats_tracking(self) -> None:
        """Test cache statistics tracking."""
        cache = LRUCache(maxsize=10)

        cache.set("key1", [1.0])

        # Miss
        _ = cache.get("nonexistent")
        # Hit
        _ = cache.get("key1")
        # Another hit
        _ = cache.get("key1")

        stats = cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3, rel=0.01)
        assert stats["size"] == 1

    def test_clear(self) -> None:
        """Test clearing the cache."""
        cache = LRUCache(maxsize=10)

        cache.set("a", [1.0])
        cache.set("b", [2.0])
        _ = cache.get("a")  # Generate a hit

        cache.clear()

        assert len(cache) == 0
        assert cache.get("a") is None
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 1  # The get after clear

    def test_thread_safety(self) -> None:
        """Test thread safety of LRU cache operations."""
        cache = LRUCache(maxsize=1000)
        errors: List[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(100):
                    key = f"thread_{thread_id}_key_{i}"
                    cache.set(key, [float(i)])
                    _ = cache.get(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


# =============================================================================
# DISK CACHE TESTS
# =============================================================================


class TestDiskCache:
    """Tests for the SQLite disk cache."""

    def test_basic_get_set(self, tmp_path: Path) -> None:
        """Test basic get and set operations."""
        db_path = tmp_path / "test_cache.db"
        cache = DiskCache(str(db_path))

        cache.set("key1", [1.0, 2.0, 3.0], "model1", "provider1", "hash1")
        cache.set("key2", [4.0, 5.0, 6.0], "model2", "provider2", "hash2")

        assert cache.get("key1") == [1.0, 2.0, 3.0]
        assert cache.get("key2") == [4.0, 5.0, 6.0]
        assert cache.get("nonexistent") is None

        cache.close()

    def test_persistence(self, tmp_path: Path) -> None:
        """Test that cache persists across instances."""
        db_path = tmp_path / "persistent_cache.db"

        # Create cache and add data
        cache1 = DiskCache(str(db_path))
        cache1.set("persistent_key", [1.0, 2.0], "model", "provider", "hash")
        cache1.close()

        # Create new instance and verify data persists
        cache2 = DiskCache(str(db_path))
        assert cache2.get("persistent_key") == [1.0, 2.0]
        cache2.close()

    def test_ttl_expiration(self, tmp_path: Path) -> None:
        """Test TTL-based cache expiration."""
        db_path = tmp_path / "ttl_cache.db"
        cache = DiskCache(str(db_path), ttl_seconds=1)

        cache.set("expiring_key", [1.0, 2.0], "model", "provider", "hash")

        # Should be present immediately
        assert cache.get("expiring_key") == [1.0, 2.0]

        # Wait for TTL to expire
        time.sleep(1.5)

        # Should be expired and return None
        assert cache.get("expiring_key") is None

        cache.close()

    def test_max_entries_limit(self, tmp_path: Path) -> None:
        """Test max entries limit enforcement."""
        db_path = tmp_path / "limited_cache.db"
        cache = DiskCache(str(db_path), max_entries=5)

        # Add 10 entries
        for i in range(10):
            cache.set(f"key_{i}", [float(i)], "model", "provider", f"hash_{i}")

        # Should only have 5 entries
        stats = cache.stats
        assert stats["count"] == 5

        cache.close()

    def test_batch_get(self, tmp_path: Path) -> None:
        """Test batch get operations."""
        db_path = tmp_path / "batch_cache.db"
        cache = DiskCache(str(db_path))

        # Add some entries
        cache.set("key1", [1.0], "model", "provider", "hash1")
        cache.set("key2", [2.0], "model", "provider", "hash2")
        cache.set("key3", [3.0], "model", "provider", "hash3")

        # Batch get with mix of existing and non-existing
        results = cache.get_batch(["key1", "key2", "nonexistent", "key3"])

        assert results["key1"] == [1.0]
        assert results["key2"] == [2.0]
        assert results["key3"] == [3.0]
        assert "nonexistent" not in results

        cache.close()

    def test_cleanup_expired(self, tmp_path: Path) -> None:
        """Test cleanup of expired entries."""
        db_path = tmp_path / "cleanup_cache.db"
        cache = DiskCache(str(db_path), ttl_seconds=1)

        cache.set("key1", [1.0], "model", "provider", "hash1")
        cache.set("key2", [2.0], "model", "provider", "hash2")

        # Wait for expiration
        time.sleep(1.5)

        # Add a fresh entry
        cache.set("key3", [3.0], "model", "provider", "hash3")

        # Cleanup should remove expired entries
        deleted = cache.cleanup_expired()
        assert deleted == 2

        # Fresh entry should still be there
        assert cache.get("key3") == [3.0]

        cache.close()

    def test_stats(self, tmp_path: Path) -> None:
        """Test stats reporting."""
        db_path = tmp_path / "stats_cache.db"
        cache = DiskCache(str(db_path))

        cache.set("key1", [1.0, 2.0], "model", "provider", "hash1")
        cache.set("key2", [3.0, 4.0], "model", "provider", "hash2")

        stats = cache.stats

        assert stats["count"] == 2
        assert stats["size_bytes"] > 0
        assert stats["db_path"] == str(db_path)
        assert stats["oldest_entry"] is not None

        cache.close()


# =============================================================================
# EMBEDDING CACHE (TWO-TIER) TESTS
# =============================================================================


class TestEmbeddingCache:
    """Tests for the two-tier embedding cache."""

    def test_basic_operations(self, tmp_path: Path) -> None:
        """Test basic get/set operations."""
        cache = EmbeddingCache(
            memory_size=100,
            disk_path=str(tmp_path / "cache.db"),
        )

        cache.set("hello world", "text-embedding-3-small", "openai", [1.0, 2.0, 3.0])

        result = cache.get("hello world", "text-embedding-3-small", "openai")
        assert result == [1.0, 2.0, 3.0]

        # Different model should miss
        result2 = cache.get("hello world", "different-model", "openai")
        assert result2 is None

        cache.close()

    def test_memory_promotion(self, tmp_path: Path) -> None:
        """Test promotion from disk to memory on cache hit."""
        cache = EmbeddingCache(
            memory_size=100,
            disk_path=str(tmp_path / "cache.db"),
        )

        # Add to cache
        cache.set("test text", "model", "provider", [1.0, 2.0])

        # Clear memory cache to simulate restart
        cache._memory_cache.clear()

        # Get should hit disk and promote to memory
        result = cache.get("test text", "model", "provider")
        assert result == [1.0, 2.0]

        # Now should be in memory cache
        assert cache._memory_cache.get(
            EmbeddingCache._make_cache_key("test text", "model", "provider")
        ) == [1.0, 2.0]

        cache.close()

    def test_batch_operations(self, tmp_path: Path) -> None:
        """Test batch get and set operations."""
        cache = EmbeddingCache(
            memory_size=100,
            disk_path=str(tmp_path / "cache.db"),
        )

        texts = ["text1", "text2", "text3"]
        embeddings = [[1.0], [2.0], [3.0]]

        # Batch set
        cache.set_batch(texts, "model", "provider", embeddings)

        # Batch get
        results, missing = cache.get_batch(texts + ["text4"], "model", "provider")

        assert results[0] == [1.0]
        assert results[1] == [2.0]
        assert results[2] == [3.0]
        assert results[3] is None
        assert missing == [3]  # Index of "text4"

        cache.close()

    def test_disabled_cache(self) -> None:
        """Test that disabled cache returns None for all operations."""
        config = EmbeddingCacheConfig(enabled=False)
        cache = EmbeddingCache(config=config)

        cache.set("text", "model", "provider", [1.0, 2.0])
        result = cache.get("text", "model", "provider")

        assert result is None
        assert not cache.enabled

    def test_disk_disabled(self, tmp_path: Path) -> None:
        """Test cache with disk storage disabled."""
        cache = EmbeddingCache(
            memory_size=100,
            disk_enabled=False,
            disk_path=str(tmp_path / "should_not_exist.db"),
        )

        cache.set("text", "model", "provider", [1.0, 2.0])
        result = cache.get("text", "model", "provider")

        assert result == [1.0, 2.0]
        assert not (tmp_path / "should_not_exist.db").exists()

        cache.close()

    def test_stats(self, tmp_path: Path) -> None:
        """Test comprehensive statistics."""
        cache = EmbeddingCache(
            memory_size=100,
            disk_path=str(tmp_path / "cache.db"),
        )

        cache.set("text1", "model", "provider", [1.0])
        cache.get("text1", "model", "provider")  # Hit
        cache.get("nonexistent", "model", "provider")  # Miss

        stats = cache.stats

        assert stats["enabled"]
        assert stats["memory"]["hits"] == 1
        assert stats["memory"]["misses"] == 1
        assert stats["disk"]["count"] == 1

        cache.close()

    def test_create_from_config(self, tmp_path: Path) -> None:
        """Test factory function."""
        config = {
            "enabled": True,
            "memory_size": 500,
            "disk_enabled": True,
            "disk_path": str(tmp_path / "factory_cache.db"),
            "ttl_hours": 24,
        }

        cache = create_embedding_cache(config)

        assert cache.enabled
        assert cache._config.memory_size == 500
        assert cache._config.ttl_hours == 24

        cache.close()


# =============================================================================
# CACHE KEY TESTS
# =============================================================================


class TestCacheKey:
    """Tests for cache key generation."""

    def test_different_texts_different_keys(self) -> None:
        """Different texts should produce different keys."""
        key1 = EmbeddingCache._make_cache_key("text1", "model", "provider")
        key2 = EmbeddingCache._make_cache_key("text2", "model", "provider")

        assert key1 != key2

    def test_different_models_different_keys(self) -> None:
        """Same text with different models should produce different keys."""
        key1 = EmbeddingCache._make_cache_key("text", "model1", "provider")
        key2 = EmbeddingCache._make_cache_key("text", "model2", "provider")

        assert key1 != key2

    def test_different_providers_different_keys(self) -> None:
        """Same text and model with different providers should produce different keys."""
        key1 = EmbeddingCache._make_cache_key("text", "model", "provider1")
        key2 = EmbeddingCache._make_cache_key("text", "model", "provider2")

        assert key1 != key2

    def test_consistent_keys(self) -> None:
        """Same inputs should always produce same key."""
        key1 = EmbeddingCache._make_cache_key("text", "model", "provider")
        key2 = EmbeddingCache._make_cache_key("text", "model", "provider")

        assert key1 == key2

    def test_key_is_sha256_hex(self) -> None:
        """Key should be a 64-character hex string (SHA256)."""
        key = EmbeddingCache._make_cache_key("text", "model", "provider")

        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)
