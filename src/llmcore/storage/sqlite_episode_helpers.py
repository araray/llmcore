# src/llmcore/storage/sqlite_episode_helpers.py
"""
Helper functions for SQLite-based storage of agent Episode objects.

This module encapsulates the database interaction logic for adding and retrieving
agent episodes, which form the basis of an agent's episodic memory. These
functions are designed to be called by the main SqliteSessionStorage class.
"""

import json
import logging
from datetime import UTC, datetime

try:
    import aiosqlite
except ImportError:
    aiosqlite = None

from ..exceptions import StorageError
from ..models import Episode, EpisodeType

logger = logging.getLogger(__name__)


async def add_episode(conn: "aiosqlite.Connection", episode: Episode, episodes_table: str) -> None:
    """
    Adds a new episode to the episodic memory log for a session.

    Args:
        conn: The active aiosqlite database connection.
        episode: The Episode object to add.
        episodes_table: The name of the table storing episodes.

    Raises:
        StorageError: If an error occurs during database insertion.
    """
    try:
        episode_data_json = json.dumps(episode.data)
        await conn.execute(
            f"""
            INSERT INTO {episodes_table} (episode_id, session_id, timestamp, event_type, data)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                episode.episode_id,
                episode.session_id,
                episode.timestamp.isoformat(),
                str(episode.event_type),
                episode_data_json,
            ),
        )
        await conn.commit()
        logger.debug(
            f"Episode '{episode.episode_id}' for session '{episode.session_id}' saved to SQLite."
        )
    except aiosqlite.Error as e:
        logger.error(f"aiosqlite error saving episode '{episode.episode_id}': {e}")
        try:
            await conn.rollback()
        except Exception as rb_e:
            logger.error(f"Rollback failed for episode save: {rb_e}")
        raise StorageError(f"Database error saving episode '{episode.episode_id}': {e}")


async def get_episodes(
    conn: "aiosqlite.Connection",
    session_id: str,
    episodes_table: str,
    limit: int = 100,
    offset: int = 0,
) -> list[Episode]:
    """
    Retrieves a list of episodes for a given session, ordered by timestamp.

    Args:
        conn: The active aiosqlite database connection.
        session_id: The ID of the session to retrieve episodes for.
        episodes_table: The name of the table storing episodes.
        limit: The maximum number of episodes to return.
        offset: The number of episodes to skip (for pagination).

    Returns:
        A list of Episode objects.

    Raises:
        StorageError: If an error occurs during database query or data validation.
    """
    episodes: list[Episode] = []
    try:
        async with conn.execute(
            f"""
            SELECT * FROM {episodes_table}
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """,
            (session_id, limit, offset),
        ) as cursor:
            async for episode_row_data in cursor:
                episode_dict = dict(episode_row_data)
                try:
                    episode_dict["data"] = json.loads(episode_dict.get("data") or "{}")
                    episode_dict["event_type"] = EpisodeType(episode_dict["event_type"])
                    ts_str = episode_dict["timestamp"]
                    episode_dict["timestamp"] = (
                        datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts_str
                        else datetime.now(UTC)
                    )
                    episodes.append(Episode.model_validate(episode_dict))
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.warning(
                        f"Skipping invalid episode data for session {session_id}, episode_id {episode_dict.get('episode_id')}: {e}"
                    )

        logger.debug(
            f"Retrieved {len(episodes)} episodes for session '{session_id}' (offset={offset}, limit={limit})"
        )
        return episodes
    except aiosqlite.Error as e:
        raise StorageError(f"Database error retrieving episodes for session '{session_id}': {e}")
