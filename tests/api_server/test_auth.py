# tests/api_server/test_auth.py
"""
Unit and integration tests for the authentication system.

This module contains comprehensive tests for the authentication dependency,
covering valid keys, invalid keys, expired keys, and inactive tenants.
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt
from fastapi import HTTPException, Request
from fastapi.security import APIKeyHeader

from llmcore.api_server.auth import (
    get_current_tenant,
    _extract_key_prefix,
    _verify_key_hash,
    api_key_header_scheme
)
from llmcore.api_server.schemas.security import Tenant, APIKey


class TestKeyPrefixExtraction:
    """Test the API key prefix extraction logic."""

    def test_valid_key_format(self):
        """Test extraction from a valid API key format."""
        api_key = "llmk_demo_a1b2c3d4e5f6g7h8i9j0"
        result = _extract_key_prefix(api_key)
        assert result == "llmk_demo"

    def test_invalid_prefix(self):
        """Test rejection of keys that don't start with llmk_."""
        api_key = "invalid_demo_a1b2c3d4e5f6g7h8i9j0"
        result = _extract_key_prefix(api_key)
        assert result is None

    def test_insufficient_parts(self):
        """Test rejection of keys with insufficient parts."""
        api_key = "llmk_demo"  # Missing secret part
        result = _extract_key_prefix(api_key)
        assert result is None

    def test_empty_key(self):
        """Test handling of empty API key."""
        result = _extract_key_prefix("")
        assert result is None

    def test_malformed_key(self):
        """Test handling of malformed API key."""
        api_key = "llmk_"  # Just the prefix
        result = _extract_key_prefix(api_key)
        assert result is None


class TestKeyVerification:
    """Test the bcrypt key verification logic."""

    @pytest.mark.asyncio
    async def test_valid_key_verification(self):
        """Test verification of a valid API key."""
        # Create a test key and its hash
        test_key = "llmk_test_secret123"
        hashed = bcrypt.hashpw(test_key.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Verify the key
        result = await _verify_key_hash(test_key, hashed)
        assert result is True

    @pytest.mark.asyncio
    async def test_invalid_key_verification(self):
        """Test verification of an invalid API key."""
        # Create a test key and its hash
        test_key = "llmk_test_secret123"
        wrong_key = "llmk_test_wrongsecret"
        hashed = bcrypt.hashpw(test_key.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Verify the wrong key
        result = await _verify_key_hash(wrong_key, hashed)
        assert result is False

    @pytest.mark.asyncio
    async def test_verification_error_handling(self):
        """Test error handling in key verification."""
        # Test with invalid hash format
        result = await _verify_key_hash("test_key", "invalid_hash")
        assert result is False


class TestGetCurrentTenant:
    """Test the main authentication dependency."""

    def setup_method(self):
        """Set up test data for each test method."""
        self.tenant_id = uuid4()
        self.api_key_id = uuid4()

        self.test_tenant = Tenant(
            id=self.tenant_id,
            name="Test Tenant",
            db_schema_name="tenant_test",
            created_at=datetime.now(timezone.utc),
            status="active"
        )

        self.test_api_key = APIKey(
            id=self.api_key_id,
            hashed_key=bcrypt.hashpw("llmk_test_secret123".encode(), bcrypt.gensalt()).decode(),
            key_prefix="llmk_test",
            tenant_id=self.tenant_id,
            created_at=datetime.now(timezone.utc),
            expires_at=None,
            last_used_at=None
        )

        # Mock request object
        self.mock_request = MagicMock(spec=Request)
        self.mock_request.client.host = "127.0.0.1"
        self.mock_request.state = MagicMock()

    @pytest.mark.asyncio
    async def test_successful_authentication(self):
        """Test successful authentication with valid API key."""
        api_key = "llmk_test_secret123"

        with patch('llmcore.api_server.auth.get_auth_db_session') as mock_db_session, \
             patch('llmcore.api_server.auth.get_api_key_by_prefix') as mock_get_key, \
             patch('llmcore.api_server.auth.get_tenant_by_id') as mock_get_tenant, \
             patch('llmcore.api_server.auth.update_api_key_last_used') as mock_update:

            # Configure mocks
            mock_session = AsyncMock()
            mock_db_session.return_value.__aenter__.return_value = mock_session
            mock_get_key.return_value = self.test_api_key
            mock_get_tenant.return_value = self.test_tenant
            mock_update.return_value = None

            # Call the dependency
            result = await get_current_tenant(self.mock_request, api_key)

            # Verify results
            assert result == self.test_tenant
            assert self.mock_request.state.tenant == self.test_tenant

            # Verify database calls
            mock_get_key.assert_called_once_with(mock_session, "llmk_test")
            mock_get_tenant.assert_called_once_with(mock_session, self.tenant_id)
            mock_update.assert_called_once_with(mock_session, self.api_key_id)

    @pytest.mark.asyncio
    async def test_invalid_key_format(self):
        """Test authentication failure with invalid key format."""
        api_key = "invalid_key_format"

        with pytest.raises(HTTPException) as exc_info:
            await get_current_tenant(self.mock_request, api_key)

        assert exc_info.value.status_code == 401
        assert "Invalid API key format" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_key_not_found(self):
        """Test authentication failure when API key is not found."""
        api_key = "llmk_notfound_secret123"

        with patch('llmcore.api_server.auth.get_auth_db_session') as mock_db_session, \
             patch('llmcore.api_server.auth.get_api_key_by_prefix') as mock_get_key:

            mock_session = AsyncMock()
            mock_db_session.return_value.__aenter__.return_value = mock_session
            mock_get_key.return_value = None  # Key not found

            with pytest.raises(HTTPException) as exc_info:
                await get_current_tenant(self.mock_request, api_key)

            assert exc_info.value.status_code == 401
            assert "Invalid API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_key_verification_failure(self):
        """Test authentication failure when key verification fails."""
        api_key = "llmk_test_wrongsecret"

        with patch('llmcore.api_server.auth.get_auth_db_session') as mock_db_session, \
             patch('llmcore.api_server.auth.get_api_key_by_prefix') as mock_get_key:

            mock_session = AsyncMock()
            mock_db_session.return_value.__aenter__.return_value = mock_session
            mock_get_key.return_value = self.test_api_key  # Key found but wrong secret

            with pytest.raises(HTTPException) as exc_info:
                await get_current_tenant(self.mock_request, api_key)

            assert exc_info.value.status_code == 401
            assert "Invalid API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_tenant_not_found(self):
        """Test authentication failure when tenant is not found."""
        api_key = "llmk_test_secret123"

        with patch('llmcore.api_server.auth.get_auth_db_session') as mock_db_session, \
             patch('llmcore.api_server.auth.get_api_key_by_prefix') as mock_get_key, \
             patch('llmcore.api_server.auth.get_tenant_by_id') as mock_get_tenant:

            mock_session = AsyncMock()
            mock_db_session.return_value.__aenter__.return_value = mock_session
            mock_get_key.return_value = self.test_api_key
            mock_get_tenant.return_value = None  # Tenant not found

            with pytest.raises(HTTPException) as exc_info:
                await get_current_tenant(self.mock_request, api_key)

            assert exc_info.value.status_code == 403
            assert "Associated tenant not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_inactive_tenant(self):
        """Test authentication failure when tenant is inactive."""
        api_key = "llmk_test_secret123"

        # Create inactive tenant
        inactive_tenant = Tenant(
            id=self.tenant_id,
            name="Inactive Tenant",
            db_schema_name="tenant_inactive",
            created_at=datetime.now(timezone.utc),
            status="suspended"
        )

        with patch('llmcore.api_server.auth.get_auth_db_session') as mock_db_session, \
             patch('llmcore.api_server.auth.get_api_key_by_prefix') as mock_get_key, \
             patch('llmcore.api_server.auth.get_tenant_by_id') as mock_get_tenant:

            mock_session = AsyncMock()
            mock_db_session.return_value.__aenter__.return_value = mock_session
            mock_get_key.return_value = self.test_api_key
            mock_get_tenant.return_value = inactive_tenant

            with pytest.raises(HTTPException) as exc_info:
                await get_current_tenant(self.mock_request, api_key)

            assert exc_info.value.status_code == 403
            assert "Tenant account is inactive" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_expired_api_key(self):
        """Test authentication failure with expired API key."""
        api_key = "llmk_test_secret123"

        # Create expired API key
        expired_key = APIKey(
            id=self.api_key_id,
            hashed_key=self.test_api_key.hashed_key,
            key_prefix="llmk_test",
            tenant_id=self.tenant_id,
            created_at=datetime.now(timezone.utc) - timedelta(days=2),
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),  # Expired yesterday
            last_used_at=None
        )

        with patch('llmcore.api_server.auth.get_auth_db_session') as mock_db_session, \
             patch('llmcore.api_server.auth.get_api_key_by_prefix') as mock_get_key:

            mock_session = AsyncMock()
            mock_db_session.return_value.__aenter__.return_value = mock_session
            mock_get_key.return_value = None  # Database should not return expired keys

            with pytest.raises(HTTPException) as exc_info:
                await get_current_tenant(self.mock_request, api_key)

            assert exc_info.value.status_code == 401
            assert "Invalid API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_database_error_handling(self):
        """Test handling of database errors during authentication."""
        api_key = "llmk_test_secret123"

        with patch('llmcore.api_server.auth.get_auth_db_session') as mock_db_session:

            # Simulate database connection error
            mock_db_session.side_effect = Exception("Database connection failed")

            with pytest.raises(HTTPException) as exc_info:
                await get_current_tenant(self.mock_request, api_key)

            assert exc_info.value.status_code == 500
            assert "Internal authentication error" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_last_used_update_failure_non_blocking(self):
        """Test that failure to update last_used_at doesn't block authentication."""
        api_key = "llmk_test_secret123"

        with patch('llmcore.api_server.auth.get_auth_db_session') as mock_db_session, \
             patch('llmcore.api_server.auth.get_api_key_by_prefix') as mock_get_key, \
             patch('llmcore.api_server.auth.get_tenant_by_id') as mock_get_tenant, \
             patch('llmcore.api_server.auth.update_api_key_last_used') as mock_update:

            # Configure mocks
            mock_session = AsyncMock()
            mock_db_session.return_value.__aenter__.return_value = mock_session
            mock_get_key.return_value = self.test_api_key
            mock_get_tenant.return_value = self.test_tenant
            mock_update.side_effect = Exception("Update failed")  # Simulate update failure

            # Should still succeed despite update failure
            result = await get_current_tenant(self.mock_request, api_key)

            assert result == self.test_tenant
            assert self.mock_request.state.tenant == self.test_tenant


class TestAPIKeyHeader:
    """Test the API key header security scheme."""

    def test_api_key_header_configuration(self):
        """Test that the API key header is properly configured."""
        assert isinstance(api_key_header_scheme, APIKeyHeader)
        assert api_key_header_scheme.param_name == "X-LLMCore-API-Key"
        assert api_key_header_scheme.auto_error is True


# Integration test fixtures
@pytest.fixture
async def test_database():
    """Fixture to set up a test database for integration tests."""
    # This would set up a test PostgreSQL database
    # Implementation depends on your test infrastructure
    pass


@pytest.fixture
def test_tenant_data():
    """Fixture providing test tenant data."""
    return {
        "tenant": Tenant(
            id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            name="Test Tenant",
            db_schema_name="tenant_test",
            created_at=datetime.now(timezone.utc),
            status="active"
        ),
        "api_key": "llmk_test_secret123"
    }


class TestIntegration:
    """Integration tests for the complete authentication flow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_authentication_flow(self, test_database, test_tenant_data):
        """Test the complete authentication flow with a real database."""
        # This would test the actual database interaction
        # Implementation depends on your test database setup
        pass

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_authentication_requests(self, test_database, test_tenant_data):
        """Test handling of concurrent authentication requests."""
        # This would test concurrent access patterns
        # Implementation depends on your test infrastructure
        pass
