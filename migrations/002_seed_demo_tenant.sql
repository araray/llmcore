-- Migration script: 002_seed_demo_tenant.sql
-- Description: Creates a demo tenant and API key for testing (REMOVE IN PRODUCTION)

-- Insert demo tenant
INSERT INTO tenants (id, name, db_schema_name, status)
VALUES (
    '550e8400-e29b-41d4-a716-446655440000'::UUID,
    'Demo Tenant',
    'tenant_demo',
    'active'
) ON CONFLICT (id) DO NOTHING;

-- Create schema for demo tenant
CREATE SCHEMA IF NOT EXISTS tenant_demo;

-- Insert demo API key (key: llmk_demo_secret123, hashed with bcrypt)
-- In production, use proper bcrypt hashing and secure key generation
INSERT INTO api_keys (id, hashed_key, key_prefix, tenant_id)
VALUES (
    '660e8400-e29b-41d4-a716-446655440001'::UUID,
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewBF1/oaKlhNX.S2', -- bcrypt hash of 'llmk_demo_secret123'
    'llmk_demo',
    '550e8400-e29b-41d4-a716-446655440000'::UUID
) ON CONFLICT (id) DO NOTHING;
