-- ==============================================================================
-- LLMCore PostgreSQL Schema
-- ==============================================================================
-- This DDL creates all tables required for llmcore session storage.
--
-- Usage:
--   psql -h {{SERVER ADDRESS/IP}} -p {{SERVER PORT}} -U {{USERNAME}} -d {{DB NAME}} -f llmcore_postgres_schema.sql
--
-- Note: When using llmcore with storage.session.type = "postgres", these tables
-- are automatically created on initialization via _ensure_tables_exist().
-- This file is provided for reference or manual setup scenarios.
-- ==============================================================================

-- Sessions table: Chat session metadata
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Messages table: Individual messages within sessions
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tool_call_id TEXT,
    tokens INTEGER,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp
    ON messages (session_id, timestamp);

-- Context items table: Session-scoped context items
CREATE TABLE IF NOT EXISTS context_items (
    id TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    item_type TEXT NOT NULL,
    source_id TEXT,
    content TEXT NOT NULL,
    tokens INTEGER,
    original_tokens INTEGER,
    is_truncated BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (session_id, id)
);

-- Context presets table: Named preset configurations
CREATE TABLE IF NOT EXISTS context_presets (
    name TEXT PRIMARY KEY,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_context_presets_updated
    ON context_presets (updated_at);

-- Context preset items table: Items belonging to presets
CREATE TABLE IF NOT EXISTS context_preset_items (
    item_id TEXT NOT NULL,
    preset_name TEXT NOT NULL REFERENCES context_presets(name) ON DELETE CASCADE,
    type TEXT NOT NULL,
    content TEXT,
    source_identifier TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (preset_name, item_id)
);

-- Episodes table: Episodic memory for agent runs
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type TEXT NOT NULL,
    data JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_episodes_session_timestamp
    ON episodes (session_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_episodes_event_type
    ON episodes (event_type);

-- ==============================================================================
-- Verification query
-- ==============================================================================
-- Run this to verify all tables were created:
--
-- SELECT table_name FROM information_schema.tables
-- WHERE table_schema = 'public'
-- AND table_name IN ('sessions', 'messages', 'context_items',
--                    'context_presets', 'context_preset_items', 'episodes');
