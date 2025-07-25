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
