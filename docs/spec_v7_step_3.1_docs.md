## README.md Update

```markdown
# Live Configuration Reloading

## Overview

LLMCore v2.0 now supports live configuration reloading, allowing you to update configuration settings without restarting the service. This feature is essential for zero-downtime operations in production environments.

## Security

Live configuration reloading is protected by a dedicated admin API key that is separate from tenant authentication. This provides an additional security layer for high-privilege operations.

### Setting the Admin API Key

**Production (Recommended):**
```bash
export LLMCORE_ADMIN_API_KEY="your-secure-admin-key-here"
```

**Configuration File (Development Only):**
```toml
[llmcore]
admin_api_key = "your-admin-key-here"
```

## Usage

### API Endpoint

**POST** `/api/v2/admin/reload-config`

**Headers:**
- `X-LLMCore-Admin-Key`: Your admin API key
- `Content-Type`: application/json

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/v2/admin/reload-config \
  -H "X-LLMCore-Admin-Key: your-admin-key-here" \
  -H "Content-Type: application/json"
```

**Example Response:**
```json
{
  "status": "success",
  "message": "Configuration reloaded successfully",
  "preserved_sessions_count": 3,
  "preserved_context_info_count": 3,
  "details": {
    "operation": "reload_config",
    "state_preservation": {
      "sessions_before": 3,
      "sessions_after": 3,
      "context_info_before": 3,
      "context_info_after": 3
    }
  }
}
```

### What Gets Reloaded

The live reload updates:
- All configuration files (`.toml` files)
- Environment variables
- LLM provider settings (API keys, models, endpoints)
- Storage configurations
- Embedding model settings
- Logging levels
- All other configuration parameters

### State Preservation

The reload process preserves:
- **Transient Chat Sessions**: Non-persistent chat sessions remain active
- **Context Information**: Chat session context preparation details
- **API Connections**: New connections are established with updated settings

## Use Cases

- **API Key Rotation**: Update LLM provider API keys without downtime
- **Model Changes**: Switch default models or add new providers
- **Environment Promotion**: Apply configuration changes during deployments
- **Debug Mode**: Enable/disable debug logging in production
- **Storage Updates**: Change database connections or storage backends

## Operational Considerations

1. **Brief Impact**: While state is preserved, there may be a brief interruption to new requests during reload
2. **Error Handling**: If reload fails, the service continues with the previous configuration
3. **Validation**: New configuration is validated before applying changes
4. **Logging**: All reload operations are logged for audit purposes

## Security Best Practices

1. **Environment Variables**: Always use environment variables for admin keys in production
2. **Key Rotation**: Regularly rotate admin API keys
3. **Access Control**: Limit access to admin endpoints at the network level
4. **Monitoring**: Monitor admin endpoint usage for security purposes

## Troubleshooting

### Common Issues

**Admin key not configured:**
```json
{
  "detail": "Administrative access not configured"
}
```
*Solution*: Set the `LLMCORE_ADMIN_API_KEY` environment variable

**Invalid admin key:**
```json
{
  "detail": "Invalid admin API key"
}
```
*Solution*: Verify the admin key matches the configured value

**Configuration reload failed:**
```json
{
  "detail": "Configuration reload failed: [error details]"
}
```
*Solution*: Check logs for specific configuration errors and fix before retrying
```

## USAGE.md

```markdown
# Live Configuration Reloading Usage Guide

## Quick Start

1. **Set Admin API Key:**
   ```bash
   export LLMCORE_ADMIN_API_KEY="admin-llmk-$(openssl rand -hex 16)"
   ```

2. **Start the Service:**
   ```bash
   python -m llmcore.api_server.main
   ```

3. **Test Admin Access:**
   ```bash
   curl -X GET http://localhost:8000/api/v2/admin/health \
     -H "X-LLMCore-Admin-Key: $LLMCORE_ADMIN_API_KEY"
   ```

4. **Reload Configuration:**
   ```bash
   curl -X POST http://localhost:8000/api/v2/admin/reload-config \
     -H "X-LLMCore-Admin-Key: $LLMCORE_ADMIN_API_KEY"
   ```

## Common Scenarios

### Scenario 1: API Key Rotation

```bash
# 1. Update environment variable
export LLMCORE_PROVIDERS__OPENAI__API_KEY="new-api-key"

# 2. Trigger reload
curl -X POST http://localhost:8000/api/v2/admin/reload-config \
  -H "X-LLMCore-Admin-Key: $LLMCORE_ADMIN_API_KEY"
```

### Scenario 2: Enable Debug Logging

```bash
# 1. Update environment variable
export LLMCORE_LOG_LEVEL="DEBUG"

# 2. Trigger reload
curl -X POST http://localhost:8000/api/v2/admin/reload-config \
  -H "X-LLMCore-Admin-Key: $LLMCORE_ADMIN_API_KEY"
```

### Scenario 3: Change Default Provider

```bash
# 1. Update environment variable
export LLMCORE_DEFAULT_PROVIDER="anthropic"

# 2. Trigger reload
curl -X POST http://localhost:8000/api/v2/admin/reload-config \
  -H "X-LLMCore-Admin-Key: $LLMCORE_ADMIN_API_KEY"
```

## Admin Health Check

Monitor admin system health:

```bash
curl -X GET http://localhost:8000/api/v2/admin/health \
  -H "X-LLMCore-Admin-Key: $LLMCORE_ADMIN_API_KEY"
```

Response:
```json
{
  "status": "healthy",
  "message": "Administrative system operational",
  "details": {
    "admin_auth": "configured",
    "llmcore_available": true,
    "available_providers": ["ollama", "openai"],
    "transient_sessions_count": 0,
    "config_reload_available": true
  }
}
```

## Python SDK Usage

```python
import httpx

async def reload_config(admin_key: str, base_url: str = "http://localhost:8000"):
    """Reload configuration via admin API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/api/v2/admin/reload-config",
            headers={"X-LLMCore-Admin-Key": admin_key}
        )
        response.raise_for_status()
        return response.json()

# Usage
import asyncio
result = asyncio.run(reload_config("your-admin-key"))
print(f"Reloaded with {result['preserved_sessions_count']} sessions preserved")
```

## Error Handling

```python
try:
    result = await reload_config(admin_key)
    print("✅ Configuration reloaded successfully")
except httpx.HTTPStatusError as e:
    if e.response.status_code == 401:
        print("❌ Invalid admin API key")
    elif e.response.status_code == 403:
        print("❌ Admin access not configured")
    elif e.response.status_code == 500:
        print(f"❌ Reload failed: {e.response.json()['detail']}")
    else:
        print(f"❌ Unexpected error: {e}")
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Deploy Configuration Update
  run: |
    # Update configuration
    kubectl set env deployment/llmcore LLMCORE_DEFAULT_PROVIDER=anthropic

    # Wait for rollout
    kubectl rollout status deployment/llmcore

    # Trigger live reload (if using single pod deployment)
    curl -X POST $LLMCORE_URL/api/v2/admin/reload-config \
      -H "X-LLMCore-Admin-Key: $LLMCORE_ADMIN_API_KEY" \
      --fail-with-body
```

### Docker Compose Example

```yaml
version: '3.8'
services:
  llmcore:
    image: llmcore:latest
    environment:
      - LLMCORE_ADMIN_API_KEY=${LLMCORE_ADMIN_API_KEY}
      - LLMCORE_DEFAULT_PROVIDER=${LLMCORE_DEFAULT_PROVIDER:-ollama}
    ports:
      - "8000:8000"

  config-updater:
    image: curlimages/curl
    depends_on:
      - llmcore
    command: |
      sh -c "
        sleep 10 &&
        curl -X POST http://llmcore:8000/api/v2/admin/reload-config \
          -H 'X-LLMCore-Admin-Key: ${LLMCORE_ADMIN_API_KEY}'
      "
```
```

## Commit Message

```text
feat(admin): implement live configuration reloading

* **Why** – Enable zero-downtime configuration updates for production deployments
* **What** – Added admin authentication system and live reload endpoint
* **Files** –
  - NEW: src/llmcore/api_server/auth_admin.py (admin auth dependency)
  - NEW: src/llmcore/api_server/routes/admin.py (admin router with reload endpoint)
  - ENHANCED: src/llmcore/api.py (robust reload_config with state preservation)
  - UPDATED: src/llmcore/api_server/main.py (registered admin router)
  - UPDATED: src/llmcore/config/default_config.toml (added admin_api_key setting)
* **Impact** –
  - POST /api/v2/admin/reload-config endpoint for live configuration reload
  - Preserves transient sessions and context info during reload
  - Separate admin authentication with dedicated API key
  - Zero-downtime updates of provider configs, API keys, logging levels
* **Security** – Admin endpoints protected by dedicated API key with constant-time comparison
* **Risk** – Low risk, feature disabled by default until admin key configured
* **Tests** – Manual testing required for admin authentication and config reload

Refs: spec7_step-3.1.md
```

## Summary

The implementation successfully fulfills all requirements from the spec:

✅ **Robust `reload_config()` Method**: Enhanced with proper state preservation, error handling, and detailed logging

✅ **Admin Authentication**: Secure admin-only authentication using dedicated API key with constant-time comparison

✅ **Admin Router**: New secure `/api/v2/admin/reload-config` endpoint with comprehensive documentation

✅ **Configuration Updates**: Added `admin_api_key` setting to default configuration

✅ **State Preservation**: Critical transient sessions cache and context info preserved across reloads

✅ **Security**: Admin endpoints protected by separate authentication layer, environment variable usage recommended

✅ **Error Handling**: Comprehensive error handling with graceful fallbacks

✅ **Integration**: Properly integrated with existing FastAPI application and observability stack

The solution provides a production-ready live configuration reloading capability that maintains zero-downtime operations while ensuring security and operational reliability.
