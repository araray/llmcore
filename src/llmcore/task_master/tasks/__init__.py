# llmcore/src/llmcore/task_master/tasks/__init__.py
"""
TaskMaster tasks package initialization.

This module exports all available arq tasks for the TaskMaster worker system,
including data ingestion, agent execution, and tenant provisioning tasks.

UPDATED: Added provision_tenant_task for multi-tenant schema provisioning.
"""

from .ingestion import ingest_data_task
from .agent import run_agent_task
from .provisioning import provision_tenant_task  # NEW: Tenant provisioning task

__all__ = [
    "ingest_data_task",
    "run_agent_task",
    "provision_tenant_task",  # NEW: Export tenant provisioning task
]
