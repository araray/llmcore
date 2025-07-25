# llmcore/src/llmcore/task_master/tasks/provisioning.py
"""
Tenant provisioning tasks for the TaskMaster service.

This module contains arq tasks for creating and managing tenant-specific
database schemas, providing the foundation for multi-tenant data isolation.

UPDATED: Added tool management tables to tenant provisioning.
"""

import json
import logging
import os
from typing import Dict, Any, Optional

try:
    import psycopg
    from psycopg_pool import AsyncConnectionPool
    psycopg_available = True
except ImportError:
    psycopg = None
    AsyncConnectionPool = None
    psycopg_available = False

logger = logging.getLogger(__name__)


async def provision_tenant_task(
    ctx: Dict[str, Any],
    tenant_id: str,
    db_schema_name: str,
    database_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Arq task to provision a new tenant with its dedicated database schema.

    This task creates a new PostgreSQL schema for the tenant and migrates
    all necessary tables within that schema to provide complete data isolation.

    UPDATED: Now includes tool management tables for dynamic tool configuration.

    Args:
        ctx: Arq context containing shared resources
        tenant_id: UUID of the tenant to provision
        db_schema_name: Name of the schema to create (e.g., 'tenant_abc123')
        database_url: Optional database URL override

    Returns:
        Dictionary containing the provisioning result and status

    Raises:
        Exception: If tenant provisioning fails
    """
    if not psycopg_available:
        raise ImportError("psycopg library not available for tenant provisioning")

    logger.info(f"Starting tenant provisioning for tenant_id: {tenant_id}, schema: {db_schema_name}")

    # Get database URL from parameter, environment, or default
    db_url = (
        database_url or
        os.environ.get('LLMCORE_AUTH_DATABASE_URL') or
        os.environ.get('LLMCORE_STORAGE_SESSION_DB_URL') or
        'postgresql+asyncpg://postgres:password@localhost:5432/llmcore'
    )

    # Convert asyncpg URL to psycopg format if needed
    if 'asyncpg' in db_url:
        db_url = db_url.replace('postgresql+asyncpg://', 'postgresql://')

    try:
        # Create connection pool for this provisioning task
        pool = AsyncConnectionPool(conninfo=db_url, min_size=1, max_size=2)

        async with pool.connection() as conn:
            async with conn.transaction():
                # Step 1: Create the tenant schema
                logger.info(f"Creating schema: {db_schema_name}")
                await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {db_schema_name}")

                # Step 2: Create all tenant-specific tables within the schema
                logger.info(f"Creating tables in schema: {db_schema_name}")

                # Sessions table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {db_schema_name}.sessions (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        metadata JSONB DEFAULT '{{}}'::jsonb
                    )
                """)

                # Messages table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {db_schema_name}.messages (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES {db_schema_name}.sessions(id) ON DELETE CASCADE,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        tool_call_id TEXT,
                        tokens INTEGER,
                        metadata JSONB DEFAULT '{{}}'::jsonb
                    )
                """)

                # Create index for messages
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp
                    ON {db_schema_name}.messages (session_id, timestamp)
                """)

                # Context items table (renamed from session_context_items for clarity)
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {db_schema_name}.context_items (
                        id TEXT NOT NULL,
                        session_id TEXT NOT NULL REFERENCES {db_schema_name}.sessions(id) ON DELETE CASCADE,
                        item_type TEXT NOT NULL,
                        source_id TEXT,
                        content TEXT NOT NULL,
                        tokens INTEGER,
                        original_tokens INTEGER,
                        is_truncated BOOLEAN DEFAULT FALSE,
                        metadata JSONB DEFAULT '{{}}'::jsonb,
                        timestamp TIMESTAMPTZ NOT NULL,
                        PRIMARY KEY (session_id, id)
                    )
                """)

                # Context presets table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {db_schema_name}.context_presets (
                        name TEXT PRIMARY KEY,
                        description TEXT,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        metadata JSONB DEFAULT '{{}}'::jsonb
                    )
                """)

                # Context preset items table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {db_schema_name}.context_preset_items (
                        item_id TEXT NOT NULL,
                        preset_name TEXT NOT NULL REFERENCES {db_schema_name}.context_presets(name) ON DELETE CASCADE,
                        type TEXT NOT NULL,
                        content TEXT,
                        source_identifier TEXT,
                        metadata JSONB DEFAULT '{{}}'::jsonb,
                        PRIMARY KEY (preset_name, item_id)
                    )
                """)

                # Episodes table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {db_schema_name}.episodes (
                        episode_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES {db_schema_name}.sessions(id) ON DELETE CASCADE,
                        timestamp TIMESTAMPTZ NOT NULL,
                        event_type TEXT NOT NULL,
                        data JSONB NOT NULL
                    )
                """)

                # Create index for episodes
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_episodes_session_timestamp
                    ON {db_schema_name}.episodes (session_id, timestamp DESC)
                """)

                # Vector collections table (if using pgvector)
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {db_schema_name}.vector_collections (
                        id SERIAL PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        vector_dimension INTEGER NOT NULL,
                        description TEXT,
                        embedding_model_provider TEXT,
                        embedding_model_name TEXT,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB DEFAULT '{{}}'::jsonb
                    )
                """)

                # Vectors table (if using pgvector)
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {db_schema_name}.vectors (
                        id TEXT NOT NULL,
                        collection_name TEXT NOT NULL REFERENCES {db_schema_name}.vector_collections(name) ON DELETE CASCADE,
                        content TEXT,
                        embedding VECTOR(384),
                        metadata JSONB DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (id, collection_name)
                    )
                """)

                # NEW: Tools table for dynamic tool management
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {db_schema_name}.tools (
                        name TEXT PRIMARY KEY,
                        description TEXT NOT NULL,
                        parameters_schema JSONB NOT NULL,
                        implementation_key TEXT NOT NULL,
                        is_enabled BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # NEW: Toolkits table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {db_schema_name}.toolkits (
                        name TEXT PRIMARY KEY,
                        description TEXT,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # NEW: Toolkit tools junction table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {db_schema_name}.toolkit_tools (
                        toolkit_name TEXT NOT NULL REFERENCES {db_schema_name}.toolkits(name) ON DELETE CASCADE,
                        tool_name TEXT NOT NULL REFERENCES {db_schema_name}.tools(name) ON DELETE CASCADE,
                        PRIMARY KEY (toolkit_name, tool_name)
                    )
                """)

                # NEW: Create indexes for better query performance on tool tables
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_tools_enabled
                    ON {db_schema_name}.tools (is_enabled)
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_tools_implementation_key
                    ON {db_schema_name}.tools (implementation_key)
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_toolkit_tools_toolkit
                    ON {db_schema_name}.toolkit_tools (toolkit_name)
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_toolkit_tools_tool
                    ON {db_schema_name}.toolkit_tools (tool_name)
                """)

                # NEW: Insert default tools for new tenants
                logger.info(f"Inserting default tools for tenant: {tenant_id}")

                # Default tools that every tenant gets
                default_tools = [
                    {
                        'name': 'semantic_search',
                        'description': 'Search the knowledge base (Semantic Memory) for relevant information on a topic. Use this when you need factual information or documentation.',
                        'parameters_schema': {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to find relevant information"
                                },
                                "k": {
                                    "type": "integer",
                                    "description": "Number of results to retrieve (default: 3)",
                                    "default": 3
                                },
                                "collection": {
                                    "type": "string",
                                    "description": "Optional collection name to search in"
                                }
                            },
                            "required": ["query"]
                        },
                        'implementation_key': 'llmcore.tools.search.semantic'
                    },
                    {
                        'name': 'episodic_search',
                        'description': 'Search past experiences and interactions (Episodic Memory) to recall previous conversations, actions, or observations. Use this to remember what happened before.',
                        'parameters_schema': {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "What to search for in past experiences"
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of episodes to return (default: 10)",
                                    "default": 10
                                }
                            },
                            "required": ["query"]
                        },
                        'implementation_key': 'llmcore.tools.search.episodic'
                    },
                    {
                        'name': 'calculator',
                        'description': 'Perform mathematical calculations. Use this for arithmetic operations, calculations, or mathematical problem solving.',
                        'parameters_schema': {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "The mathematical expression to evaluate (e.g., '2 + 3 * 4', '(10 - 3) / 2')"
                                }
                            },
                            "required": ["expression"]
                        },
                        'implementation_key': 'llmcore.tools.calculation.calculator'
                    },
                    {
                        'name': 'finish',
                        'description': 'Use this tool when you have completed the task and have a final answer. This will end the agent\'s execution.',
                        'parameters_schema': {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                    "description": "The final answer or result of the task"
                                }
                            },
                            "required": ["answer"]
                        },
                        'implementation_key': 'llmcore.tools.flow.finish'
                    }
                ]

                # Insert default tools
                for tool in default_tools:
                    await conn.execute(f"""
                        INSERT INTO {db_schema_name}.tools
                        (name, description, parameters_schema, implementation_key)
                        VALUES (%s, %s, %s, %s)
                    """, (
                        tool['name'],
                        tool['description'],
                        json.dumps(tool['parameters_schema']),
                        tool['implementation_key']
                    ))

                # Create default toolkit with all basic tools
                await conn.execute(f"""
                    INSERT INTO {db_schema_name}.toolkits (name, description)
                    VALUES (%s, %s)
                """, (
                    'basic_tools',
                    'Essential tools for general agent operations including search, calculation, and task completion'
                ))

                # Add all default tools to the basic toolkit
                for tool in default_tools:
                    await conn.execute(f"""
                        INSERT INTO {db_schema_name}.toolkit_tools (toolkit_name, tool_name)
                        VALUES (%s, %s)
                    """, ('basic_tools', tool['name']))

        # Step 3: Update tenant status to 'active' in the main tenants table
        logger.info(f"Updating tenant status to active for: {tenant_id}")
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE tenants SET status = 'active' WHERE id = %s",
                (tenant_id,)
            )

        # Close the pool
        await pool.close()

        logger.info(f"Successfully provisioned tenant {tenant_id} with schema {db_schema_name}")

        return {
            "status": "success",
            "tenant_id": tenant_id,
            "db_schema_name": db_schema_name,
            "message": f"Tenant {tenant_id} successfully provisioned with schema {db_schema_name}",
            "tables_created": [
                "sessions", "messages", "context_items", "context_presets",
                "context_preset_items", "episodes", "vector_collections",
                "vectors", "tools", "toolkits", "toolkit_tools"
            ],
            "default_tools_created": len(default_tools),
            "default_toolkits_created": 1
        }

    except Exception as e:
        logger.error(f"Failed to provision tenant {tenant_id}: {e}", exc_info=True)

        # Attempt cleanup on failure
        try:
            pool = AsyncConnectionPool(conninfo=db_url, min_size=1, max_size=1)
            async with pool.connection() as conn:
                await conn.execute(f"DROP SCHEMA IF EXISTS {db_schema_name} CASCADE")
                await conn.execute(
                    "UPDATE tenants SET status = 'provisioning_failed' WHERE id = %s",
                    (tenant_id,)
                )
            await pool.close()
            logger.info(f"Cleaned up failed schema {db_schema_name} for tenant {tenant_id}")
        except Exception as cleanup_error:
            logger.error(f"Failed to cleanup after provisioning failure: {cleanup_error}")

        return {
            "status": "error",
            "tenant_id": tenant_id,
            "db_schema_name": db_schema_name,
            "message": f"Failed to provision tenant {tenant_id}: {str(e)}",
            "error": str(e)
        }
