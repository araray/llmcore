## Commit Message

```text
feat(agents): implement dynamic tool & toolkit management system

* **Why** – Transform static tool system into tenant-aware, database-driven architecture for runtime tool configuration and management
* **What** – Created dynamic tool loading, database schemas, API endpoints, and tenant provisioning integration
  - New migration script for tools/toolkits tables
  - New API endpoints (/api/v2/tools, /api/v2/toolkits) with full CRUD operations
  - Refactored ToolManager to load tools dynamically from database per-tenant
  - Updated AgentManager to integrate dynamic tool loading before each run
  - Added tool management tables to tenant provisioning process
  - Secure implementation registry prevents execution of arbitrary code
* **Impact** – Enables per-tenant tool customization, runtime tool management via API, and flexible agent capabilities without code deployments
* **Risk** – Comprehensive test coverage needed for new endpoints and database operations. Rollback involves reverting to static tool registration.

Refs: Phase 2, Step 2.1 - Dynamic Tool & Toolkit Management
```

## Implementation Summary

**Rationale**: The static tool system in v0.21.0 required code deployments for any tool changes, making it inflexible for production use. This implementation provides a secure, database-driven approach where:

**Pre-state**: Tools were hardcoded in Python and globally registered at startup
**Limitation**: No runtime tool management, no tenant-specific toolsets, required code changes for new tools
**Decision Path**: Database-driven definitions with secure implementation registry maintains security while enabling flexibility
**Post-state**: Tools are loaded dynamically per-tenant from database, manageable via API, with secure execution

## Key Changes Made:

### 1. **Database Schema** (`migrations/003_create_tenant_tool_management_tables.sql`)
- `tools` table: Stores tool definitions with implementation keys
- `toolkits` table: Named collections of tools
- `toolkit_tools` table: Many-to-many relationship
- Proper indexes for query performance

### 2. **Refactored ToolManager** (`agents/tools.py`)
- **Security boundary**: `_IMPLEMENTATION_REGISTRY` maps keys to actual functions
- **Dynamic loading**: `load_tools_for_run()` queries database per tenant/toolkit
- **Secure execution**: Implementation keys must exist in hardcoded registry
- **Dependency injection**: Tools receive required managers automatically

### 3. **New API Endpoints** (`api_server/routes/tools.py`)
- Full CRUD operations for tools and toolkits
- Tenant-scoped via database session dependency
- Validation ensures implementation keys are valid
- Utility endpoint lists available implementations

### 4. **Updated AgentManager** (`agents/manager.py`)
- Loads tools before each agent run via `load_tools_for_run()`
- Passes database session and enabled toolkits
- Maintains backward compatibility when no DB session provided

### 5. **Enhanced Tenant Provisioning** (`task_master/tasks/provisioning.py`)
- Creates tool management tables for each new tenant
- Inserts default tools (semantic_search, episodic_search, calculator, finish)
- Creates basic_tools toolkit with all default tools

### 6. **Integration Updates**
- Added tools router to main FastAPI app
- Updated imports in routes and models __init__.py files
- Enhanced agent task with database session support

## Security Considerations

The implementation maintains strict security through:
- **Secure Registry**: Only pre-approved functions in `_IMPLEMENTATION_REGISTRY` can be executed
- **Key Validation**: All implementation keys are validated against the secure registry
- **Tenant Isolation**: Database operations use tenant-scoped sessions
- **No Dynamic Code**: Database stores only metadata, never executable code

## Testing Recommendations

1. **Unit Tests**: Test tool CRUD operations, ToolManager loading, registry validation
2. **Integration Tests**: Test full agent runs with dynamic tools, tenant isolation
3. **Security Tests**: Verify invalid implementation keys are rejected
4. **Performance Tests**: Measure tool loading overhead vs static registration

This implementation successfully transforms the agent tooling system from static to dynamic while maintaining security and enabling flexible, tenant-specific tool management.
