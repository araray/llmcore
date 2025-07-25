Now let me create the database migration script to add the necessary columns for HITL support:

```sql
-- Migration: Add HITL support columns to agent_tasks table
-- File: migrations/004_add_hitl_support_to_agent_tasks.sql

-- Add HITL workflow support columns to agent_tasks table
ALTER TABLE agent_tasks
ADD COLUMN IF NOT EXISTS pending_action_data JSONB,
ADD COLUMN IF NOT EXISTS approval_prompt TEXT;

-- Add index for faster queries on pending approval tasks
CREATE INDEX IF NOT EXISTS idx_agent_tasks_pending_approval
ON agent_tasks(status) WHERE status = 'PENDING_APPROVAL';

-- Update the status constraint to include PENDING_APPROVAL if it exists
-- Note: This assumes there's a check constraint on status field
-- ALTER TABLE agent_tasks DROP CONSTRAINT IF EXISTS agent_tasks_status_check;
-- ALTER TABLE agent_tasks ADD CONSTRAINT agent_tasks_status_check
-- CHECK (status IN ('PENDING', 'QUEUED', 'RUNNING', 'SUCCESS', 'FAILURE', 'PENDING_APPROVAL'));

-- Add comments for documentation
COMMENT ON COLUMN agent_tasks.pending_action_data IS 'JSON data for HITL workflow: stores pending action or rejection info';
COMMENT ON COLUMN agent_tasks.approval_prompt IS 'Human-readable prompt for approval workflow';
```

Now let me provide the documentation updates:

## Documentation Updates

### API Reference Update (docs/api_reference.md)

```markdown
## Human-in-the-Loop (HITL) Workflow Endpoints

### GET /api/v2/tasks/pending_approval

Lists all tasks awaiting human approval for the current tenant.

**Response:**
```json
{
  "pending_tasks": [
    {
      "task_id": "agent_12345",
      "goal": "Send email to all users",
      "approval_prompt": "Approve sending the following email draft: ...",
      "pending_action": {
        "tool_name": "send_email",
        "arguments": {...},
        "requested_at": "2024-01-15T10:30:00Z"
      },
      "created_at": "2024-01-15T10:00:00Z",
      "updated_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total_count": 1,
  "tenant_id": "tenant-uuid"
}
```

### POST /api/v2/tasks/{task_id}/approve

Approves a pending task and resumes its execution.

**Response:**
```json
{
  "message": "Task agent_12345 has been approved and resumed.",
  "task_id": "agent_12345",
  "goal": "Send email to all users",
  "approval_prompt": "Approve sending the following email draft: ...",
  "new_job_id": "job_67890",
  "status": "approved_and_resumed"
}
```

### POST /api/v2/tasks/{task_id}/reject

Rejects a pending task with optional feedback.

**Request Body:**
```json
{
  "reason": "Email content needs revision - too promotional"
}
```

**Response:**
```json
{
  "message": "Task agent_12345 has been rejected and will receive feedback.",
  "task_id": "agent_12345",
  "goal": "Send email to all users",
  "rejection_reason": "Email content needs revision - too promotional",
  "new_job_id": "job_67891",
  "status": "rejected_with_feedback"
}
```
```

### Usage Guide Update (docs/usage.md)

```markdown
## Human-in-the-Loop (HITL) Workflows

The llmcore platform supports Human-in-the-Loop workflows for safe execution of sensitive actions. Agents can pause execution to request human approval before performing irreversible or high-stakes actions.

### Using the human_approval Tool

Agents have access to a special `human_approval` tool that pauses execution:

```python
# In your agent prompt or tool description:
# "Use human_approval tool before executing sensitive actions like sending emails,
#  modifying databases, or making financial transactions"

# Example tool call by agent:
{
  "name": "human_approval",
  "arguments": {
    "prompt": "Approve sending promotional email to 10,000 users?",
    "pending_action": {
      "name": "send_email",
      "arguments": {
        "recipients": "all_users",
        "subject": "Special Offer",
        "body": "..."
      }
    }
  }
}
```

### Managing Approval Workflows

1. **List pending approvals:**
```bash
curl -H "X-LLMCore-API-Key: your-key" \
  http://localhost:8000/api/v2/tasks/pending_approval
```

2. **Approve a task:**
```bash
curl -X POST -H "X-LLMCore-API-Key: your-key" \
  http://localhost:8000/api/v2/tasks/agent_12345/approve
```

3. **Reject a task with feedback:**
```bash
curl -X POST -H "X-LLMCore-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"reason": "Please revise the email content"}' \
  http://localhost:8000/api/v2/tasks/agent_12345/reject
```

### Agent Behavior

- **On Approval:** The agent executes the pending action and continues with its plan
- **On Rejection:** The agent receives the rejection reason as feedback and can adapt its approach
- **Timeout:** Tasks remain paused indefinitely until human action is taken
```

## Commit Message

```text
feat(hitl): implement Human-in-the-Loop workflow support

* **Why** – Enable safe execution of sensitive agent actions requiring human oversight
* **What** – Added human_approval tool, HITL endpoints, task resumption logic
* **Impact** – Agents can now pause for approval; operators can approve/reject via API
* **Risk** – Comprehensive HITL workflow with database state management and resumption

Files modified:
- src/llmcore/models.py: Added pending_action_data, approval_prompt to AgentTask
- src/llmcore/agents/tools.py: Added human_approval tool implementation
- src/llmcore/agents/manager.py: Added HITL pause/resume logic in agent loop
- src/llmcore/api_server/routes/tasks.py: Added approval/rejection endpoints
- src/llmcore/task_master/tasks/agent.py: Added task resumption support
- migrations/004_add_hitl_support_to_agent_tasks.sql: Database schema changes

Refs: spec7_step-2.3.md
```

**Rationale Block:**

**Pre-state:** Agents executed all tool calls automatically without human oversight, creating risk for sensitive or irreversible actions.

**Limitation:** No mechanism existed for human approval of agent actions, making the system unsuitable for production environments requiring human-in-the-loop workflows.

**Decision Path:** Implemented a special `human_approval` tool that signals the AgentManager to pause execution, store the pending action in the database with PENDING_APPROVAL status, and provide API endpoints for human operators to approve or reject the action. Task resumption logic handles both approval (executes stored action) and rejection (provides feedback to agent).

**Post-state:** Agents can now request human approval for sensitive actions, pause execution safely, and resume based on human decisions. The workflow is fully persistent and handles failures gracefully.

**Impact:** This enables safe deployment of autonomous agents in production environments where human oversight is required for high-stakes decisions.
