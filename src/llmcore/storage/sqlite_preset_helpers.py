# src/llmcore/storage/sqlite_preset_helpers.py
"""
Helper functions for SQLite-based storage of ContextPreset objects.

This module encapsulates the database interaction logic for creating, reading,
updating, and deleting context presets and their associated items. These functions
are designed to be called by the main SqliteSessionStorage class, operating on
its database connection.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import aiosqlite
except ImportError:
    aiosqlite = None

from ..exceptions import StorageError
from ..models import ContextPreset, ContextPresetItem

logger = logging.getLogger(__name__)


async def save_context_preset(
    conn: "aiosqlite.Connection",
    preset: ContextPreset,
    presets_table: str,
    preset_items_table: str
) -> None:
    """Saves or updates a context preset and its items in SQLite."""
    preset_metadata_json = json.dumps(preset.metadata or {})
    try:
        await conn.execute("BEGIN;")
        await conn.execute(f"""
            INSERT OR REPLACE INTO {presets_table} (name, description, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (preset.name, preset.description, preset.created_at.isoformat(), preset.updated_at.isoformat(), preset_metadata_json))

        await conn.execute(f"DELETE FROM {preset_items_table} WHERE preset_name = ?", (preset.name,))
        if preset.items:
            items_data = [(item.item_id, preset.name, str(item.type), item.content,
                           item.source_identifier, json.dumps(item.metadata or {}))
                          for item in preset.items]
            await conn.executemany(f"""
                INSERT INTO {preset_items_table} (item_id, preset_name, type, content, source_identifier, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, items_data)
        await conn.commit()
        logger.info(f"Context preset '{preset.name}' with {len(preset.items)} items saved to SQLite.")
    except aiosqlite.Error as e:
        logger.error(f"aiosqlite error saving context preset '{preset.name}': {e}")
        try:
            await conn.rollback()
        except Exception as rb_e:
            logger.error(f"Rollback failed for preset save: {rb_e}")
        raise StorageError(f"Database error saving context preset '{preset.name}': {e}")


async def get_context_preset(
    conn: "aiosqlite.Connection",
    preset_name: str,
    presets_table: str,
    preset_items_table: str
) -> Optional[ContextPreset]:
    """Retrieves a specific context preset and its items by name from SQLite."""
    try:
        async with conn.execute(f"SELECT * FROM {presets_table} WHERE name = ?", (preset_name,)) as cursor:
            preset_row = await cursor.fetchone()
        if not preset_row:
            logger.debug(f"Context preset '{preset_name}' not found in SQLite.")
            return None

        preset_data = dict(preset_row)
        preset_data["metadata"] = json.loads(preset_data.get("metadata") or '{}')
        preset_data["created_at"] = datetime.fromisoformat(preset_data["created_at"].replace('Z', '+00:00'))
        preset_data["updated_at"] = datetime.fromisoformat(preset_data["updated_at"].replace('Z', '+00:00'))

        items: List[ContextPresetItem] = []
        async with conn.execute(f"SELECT * FROM {preset_items_table} WHERE preset_name = ?", (preset_name,)) as cursor:
            async for item_row_data in cursor:
                item_dict = dict(item_row_data)
                try:
                    item_dict["metadata"] = json.loads(item_dict.get("metadata") or '{}')
                    items.append(ContextPresetItem.model_validate(item_dict))
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid context_preset_item data for preset {preset_name}, item_id {item_dict.get('item_id')}: {e}")
        preset_data["items"] = items

        context_preset = ContextPreset.model_validate(preset_data)
        logger.debug(f"Context preset '{preset_name}' loaded from SQLite with {len(items)} items.")
        return context_preset
    except aiosqlite.Error as e:
        raise StorageError(f"Database error retrieving context preset '{preset_name}': {e}")


async def list_context_presets(
    conn: "aiosqlite.Connection",
    presets_table: str,
    preset_items_table: str
) -> List[Dict[str, Any]]:
    """Lists context preset metadata from SQLite, including item counts."""
    preset_metadata_list: List[Dict[str, Any]] = []
    try:
        async with conn.execute(f"""
            SELECT p.name, p.description, p.created_at, p.updated_at, p.metadata,
                   (SELECT COUNT(*) FROM {preset_items_table} pi WHERE pi.preset_name = p.name) as item_count
            FROM {presets_table} p ORDER BY p.updated_at DESC
        """) as cursor:
            async for row in cursor:
                data = dict(row)
                try:
                    data["metadata"] = json.loads(data.get("metadata") or '{}')
                except json.JSONDecodeError:
                    data["metadata"] = {}
                preset_metadata_list.append(data)
        logger.debug(f"Found {len(preset_metadata_list)} context presets in SQLite.")
        return preset_metadata_list
    except aiosqlite.Error as e:
        raise StorageError(f"Database error listing context presets: {e}")


async def delete_context_preset(
    conn: "aiosqlite.Connection",
    preset_name: str,
    presets_table: str
) -> bool:
    """Deletes a context preset and its items (due to CASCADE DELETE) from SQLite."""
    try:
        cursor = await conn.execute(f"DELETE FROM {presets_table} WHERE name = ?", (preset_name,))
        await conn.commit()
        if cursor.rowcount > 0:
            logger.info(f"Context preset '{preset_name}' and its items deleted from SQLite.")
            return True
        logger.warning(f"Attempted to delete non-existent context preset '{preset_name}'.")
        return False
    except aiosqlite.Error as e:
        try:
            await conn.rollback()
        except Exception as rb_e:
            logger.error(f"Rollback failed for preset delete: {rb_e}")
        raise StorageError(f"Database error deleting context preset '{preset_name}': {e}")


async def rename_context_preset(
    conn: "aiosqlite.Connection",
    old_name: str,
    new_name: str,
    presets_table: str,
    preset_items_table: str
) -> bool:
    """Renames a context preset in SQLite."""
    if old_name == new_name:
        return True

    try:
        ContextPreset(name=new_name, items=[])
    except ValueError as ve:
        logger.error(f"Invalid new preset name '{new_name}': {ve}")
        raise

    try:
        async with conn.execute(f"SELECT 1 FROM {presets_table} WHERE name = ?", (new_name,)) as cursor:
            if await cursor.fetchone():
                logger.warning(f"Cannot rename preset: new name '{new_name}' already exists.")
                return False

        old_preset = await get_context_preset(conn, old_name, presets_table, preset_items_table)
        if not old_preset:
            logger.warning(f"Cannot rename preset: old name '{old_name}' not found.")
            return False

        renamed_preset = ContextPreset(
            name=new_name,
            description=old_preset.description,
            items=old_preset.items,
            created_at=old_preset.created_at,
            updated_at=datetime.now(timezone.utc),
            metadata=old_preset.metadata
        )

        await conn.execute("BEGIN;")
        await save_context_preset(conn, renamed_preset, presets_table, preset_items_table)
        cursor = await conn.execute(f"DELETE FROM {presets_table} WHERE name = ?", (old_name,))
        if cursor.rowcount == 0:
            await conn.rollback()
            raise StorageError(f"Rename failed: old preset '{old_name}' disappeared during transaction.")

        await conn.commit()
        logger.info(f"Context preset '{old_name}' successfully renamed to '{new_name}'.")
        return True
    except (aiosqlite.Error, StorageError) as e:
        try:
            await conn.rollback()
        except Exception as rb_e:
            logger.error(f"Rollback failed for preset rename: {rb_e}")
        if isinstance(e, StorageError):
            raise
        raise StorageError(f"Database error renaming context preset: {e}")
