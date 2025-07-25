-- Migration 003: Create tenant tool management tables
-- This migration creates the tables needed for dynamic tool and toolkit management
-- within each tenant's isolated schema.

-- Tools table: Stores tool definitions with their metadata and implementation keys
CREATE TABLE IF NOT EXISTS tools (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    parameters_schema JSONB NOT NULL,
    implementation_key TEXT NOT NULL,
    is_enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Toolkits table: Named collections of tools for organizational purposes
CREATE TABLE IF NOT EXISTS toolkits (
    name TEXT PRIMARY KEY,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Junction table: Many-to-many relationship between toolkits and tools
CREATE TABLE IF NOT EXISTS toolkit_tools (
    toolkit_name TEXT NOT NULL REFERENCES toolkits(name) ON DELETE CASCADE,
    tool_name TEXT NOT NULL REFERENCES tools(name) ON DELETE CASCADE,
    PRIMARY KEY (toolkit_name, tool_name)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_tools_enabled ON tools (is_enabled);
CREATE INDEX IF NOT EXISTS idx_tools_implementation_key ON tools (implementation_key);
CREATE INDEX IF NOT EXISTS idx_toolkit_tools_toolkit ON toolkit_tools (toolkit_name);
CREATE INDEX IF NOT EXISTS idx_toolkit_tools_tool ON toolkit_tools (tool_name);
