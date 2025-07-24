-- Migration script: 001_create_tenants_and_api_keys_tables.sql
-- Description: Creates the foundational tables for multi-tenant authentication

-- Create tenants table
CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    db_schema_name VARCHAR(63) NOT NULL UNIQUE, -- PostgreSQL identifier limit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(50) NOT NULL DEFAULT 'active'
);

-- Create API keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hashed_key TEXT NOT NULL UNIQUE,
    key_prefix VARCHAR(100) NOT NULL UNIQUE,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NULL,
    last_used_at TIMESTAMPTZ NULL
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);
CREATE INDEX IF NOT EXISTS idx_api_keys_tenant_id ON api_keys(tenant_id);
CREATE INDEX IF NOT EXISTS idx_tenants_status ON tenants(status);
CREATE INDEX IF NOT EXISTS idx_tenants_schema_name ON tenants(db_schema_name);

-- Add constraints
ALTER TABLE tenants ADD CONSTRAINT chk_tenant_status
    CHECK (status IN ('active', 'suspended', 'pending_provisioning', 'deleted'));

-- Add comments for documentation
COMMENT ON TABLE tenants IS 'Central registry for all tenants on the platform';
COMMENT ON TABLE api_keys IS 'Secure storage for API keys linked to tenants';
COMMENT ON COLUMN tenants.db_schema_name IS 'PostgreSQL schema name for tenant data isolation';
COMMENT ON COLUMN api_keys.hashed_key IS 'bcrypt-hashed API key for secure storage';
COMMENT ON COLUMN api_keys.key_prefix IS 'Non-sensitive prefix for key identification';
