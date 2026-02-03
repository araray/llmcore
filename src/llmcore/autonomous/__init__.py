# src/llmcore/autonomous/__init__.py
"""
Autonomous Operation Module for llmcore.

Provides infrastructure for autonomous agent operation including:
- Goal management with LLM-powered decomposition
- Periodic task scheduling (heartbeat system)
- Human escalation framework
- System resource monitoring and constraint enforcement

Phase 1 (Foundation):
    - GoalManager, Goal, GoalStatus, GoalPriority, SuccessCriterion
    - EscalationManager, Escalation, EscalationLevel, EscalationReason

Phase 2 (Resource Management):
    - HeartbeatManager, HeartbeatTask
    - ResourceMonitor, ResourceConstraints, ResourceStatus

Example:
    from llmcore.autonomous import (
        GoalManager, GoalStore, SuccessCriterion,
        HeartbeatManager, HeartbeatTask,
        EscalationManager, EscalationLevel, EscalationReason,
        ResourceMonitor, ResourceConstraints,
    )
    from llmcore.config.autonomous_config import GoalsAutonomousConfig

    # Set up autonomous operation
    goals = GoalManager.from_config(GoalsAutonomousConfig())
    heartbeat = HeartbeatManager()
    escalation = EscalationManager()
    resources = ResourceMonitor()
"""

# Goal Management (Phase 1)
from .goals import (
    Goal,
    GoalManager,
    GoalPriority,
    GoalStatus,
    GoalStorageProtocol,
    GoalStore,
    SuccessCriterion,
)

# Escalation Framework (Phase 1)
from .escalation import (
    Escalation,
    EscalationLevel,
    EscalationManager,
    EscalationReason,
    NotificationHandler,
    callback_handler,
    file_handler,
    webhook_handler,
)

# Heartbeat System (Phase 2)
from .heartbeat import (
    HeartbeatManager,
    HeartbeatTask,
    heartbeat_task,
)

# Resource Monitoring (Phase 2)
from .resource import (
    ConstraintViolation,
    ResourceConstraints,
    ResourceMonitor,
    ResourceStatus,
    ResourceUsage,
)

__all__ = [
    # Goals
    "Goal",
    "GoalManager",
    "GoalPriority",
    "GoalStatus",
    "GoalStorageProtocol",
    "GoalStore",
    "SuccessCriterion",
    # Escalation
    "Escalation",
    "EscalationLevel",
    "EscalationManager",
    "EscalationReason",
    "NotificationHandler",
    "callback_handler",
    "file_handler",
    "webhook_handler",
    # Heartbeat
    "HeartbeatManager",
    "HeartbeatTask",
    "heartbeat_task",
    # Resource
    "ConstraintViolation",
    "ResourceConstraints",
    "ResourceMonitor",
    "ResourceStatus",
    "ResourceUsage",
]
