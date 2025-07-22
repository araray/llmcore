# tools/generate_api_key.py
"""
Utility script for generating API keys for llmcore tenants.

This script generates secure API keys and their corresponding database records
for use with the llmcore authentication system.
"""

import secrets
import string
import bcrypt
import asyncio
from uuid import uuid4
from datetime import datetime, timezone
from typing import Optional

import asyncpg


def generate_secure_key(prefix: str, secret_length: int = 32) -> tuple[str, str]:
    """
    Generate a secure API key with the specified prefix.

    Args:
        prefix: The tenant prefix for the key (e.g., 'demo', 'prod')
        secret_length: Length of the secret portion

    Returns:
        Tuple of (full_key, key_prefix)
    """
    # Generate secure random secret
    alphabet = string.ascii_letters + string.digits
    secret = ''.join(secrets.choice(alphabet) for _ in range(secret_length))

    # Construct full key
    full_key = f"llmk_{prefix}_{secret}"
    key_prefix = f"llmk_{prefix}"

    return full_key, key_prefix


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key using bcrypt.

    Args:
        api_key: The plaintext API key to hash

    Returns:
        The bcrypt hash as a string
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(api_key.encode('utf-8'), salt)
    return hashed.decode('utf-8')


async def create_tenant_and_key(
    database_url: str,
    tenant_name: str,
    tenant_prefix: str,
    schema_name: Optional[str] = None
) -> dict:
    """
    Create a new tenant and API key in the database.

    Args:
        database_url: PostgreSQL connection URL
        tenant_name: Human-readable name for the tenant
        tenant_prefix: Short prefix for the API key
        schema_name: Optional custom schema name (defaults to tenant_{prefix})

    Returns:
        Dictionary with tenant and API key information
    """
    if schema_name is None:
        schema_name = f"tenant_{tenant_prefix}"

    # Generate API key
    api_key, key_prefix = generate_secure_key(tenant_prefix)
    hashed_key = hash_api_key(api_key)

    # Generate UUIDs
    tenant_id = uuid4()
    api_key_id = uuid4()

    # Connect to database and create records
    conn = await asyncpg.connect(database_url)

    try:
        async with conn.transaction():
            # Create tenant
            await conn.execute("""
                INSERT INTO tenants (id, name, db_schema_name, created_at, status)
                VALUES ($1, $2, $3, $4, $5)
            """, tenant_id, tenant_name, schema_name, datetime.now(timezone.utc), 'active')

            # Create API key
            await conn.execute("""
                INSERT INTO api_keys (id, hashed_key, key_prefix, tenant_id, created_at)
                VALUES ($1, $2, $3, $4, $5)
            """, api_key_id, hashed_key, key_prefix, tenant_id, datetime.now(timezone.utc))

            # Create tenant schema
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')

            print(f"✅ Created tenant: {tenant_name}")
            print(f"   - Tenant ID: {tenant_id}")
            print(f"   - Schema: {schema_name}")
            print(f"   - API Key: {api_key}")
            print(f"   - Key Prefix: {key_prefix}")
            print("\n⚠️  IMPORTANT: Save the API key securely - it cannot be retrieved again!")

            return {
                'tenant_id': str(tenant_id),
                'tenant_name': tenant_name,
                'schema_name': schema_name,
                'api_key': api_key,
                'key_prefix': key_prefix,
                'api_key_id': str(api_key_id)
            }

    finally:
        await conn.close()


async def list_tenants(database_url: str):
    """
    List all tenants in the database.

    Args:
        database_url: PostgreSQL connection URL
    """
    conn = await asyncpg.connect(database_url)

    try:
        rows = await conn.fetch("""
            SELECT t.id, t.name, t.db_schema_name, t.status, t.created_at,
                   COUNT(ak.id) as api_key_count
            FROM tenants t
            LEFT JOIN api_keys ak ON t.id = ak.tenant_id
            GROUP BY t.id, t.name, t.db_schema_name, t.status, t.created_at
            ORDER BY t.created_at DESC
        """)

        print("📋 Tenants:")
        print("-" * 80)
        for row in rows:
            print(f"Name: {row['name']}")
            print(f"ID: {row['id']}")
            print(f"Schema: {row['db_schema_name']}")
            print(f"Status: {row['status']}")
            print(f"API Keys: {row['api_key_count']}")
            print(f"Created: {row['created_at']}")
            print("-" * 80)

    finally:
        await conn.close()


async def main():
    """Main CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(description='llmcore API Key Management')
    parser.add_argument('--db-url', required=True, help='PostgreSQL database URL')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Create tenant command
    create_parser = subparsers.add_parser('create', help='Create a new tenant and API key')
    create_parser.add_argument('--name', required=True, help='Tenant name')
    create_parser.add_argument('--prefix', required=True, help='API key prefix')
    create_parser.add_argument('--schema', help='Custom schema name (optional)')

    # List tenants command
    list_parser = subparsers.add_parser('list', help='List all tenants')

    # Generate key only command
    gen_parser = subparsers.add_parser('generate', help='Generate API key without database')
    gen_parser.add_argument('--prefix', required=True, help='API key prefix')

    args = parser.parse_args()

    if args.command == 'create':
        result = await create_tenant_and_key(
            args.db_url,
            args.name,
            args.prefix,
            args.schema
        )

    elif args.command == 'list':
        await list_tenants(args.db_url)

    elif args.command == 'generate':
        api_key, key_prefix = generate_secure_key(args.prefix)
        hashed_key = hash_api_key(api_key)

        print(f"Generated API Key: {api_key}")
        print(f"Key Prefix: {key_prefix}")
        print(f"Bcrypt Hash: {hashed_key}")
        print("\n⚠️  Remember to add this to your database manually!")

    else:
        parser.print_help()


if __name__ == '__main__':
    asyncio.run(main())


# Example usage:
# python tools/generate_api_key.py --db-url "postgresql://user:pass@localhost/llmcore" create --name "Test Tenant" --prefix "test"
# python tools/generate_api_key.py --db-url "postgresql://user:pass@localhost/llmcore" list
# python tools/generate_api_key.py generate --prefix "demo"
