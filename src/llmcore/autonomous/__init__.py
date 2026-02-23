# src/llmcore/autonomous/__init__.py
"""
Autonomous Operation Module for llmcore.

Provides infrastructure for autonomous agent operation including:
- Goal management with LLM-powered decomposition
- Periodic task scheduling (heartbeat system)
- Human escalation framework
- System resource monitoring and constraint enforcement
- Skill loading system for dynamic knowledge injection

Phase 1 (Foundation):
    - GoalManager, Goal, GoalStatus, GoalPriority, SuccessCriterion
    - EscalationManager, Escalation, EscalationLevel, EscalationReason

Phase 2 (Resource Management):
    - HeartbeatManager, HeartbeatTask
    - ResourceMonitor, ResourceConstraints, ResourceStatus

Phase 3 (Context Intelligence):
    - SkillLoader, Skill, SkillMetadata

Example:
    from llmcore.autonomous import (
        GoalManager, GoalStore, SuccessCriterion,
        HeartbeatManager, HeartbeatTask,
        EscalationManager, EscalationLevel, EscalationReason,
        ResourceMonitor, ResourceConstraints,
        SkillLoader, Skill, SkillMetadata,
    )
    from llmcore.config.autonomous_config import GoalsAutonomousConfig, HeartbeatConfig

    # Set up autonomous operation
    goals = GoalManager.from_config(GoalsAutonomousConfig())
    heartbeat = HeartbeatManager.from_config(HeartbeatConfig())
    escalation = EscalationManager()
    resources = ResourceMonitor()
    skills = SkillLoader()
"""

# Goal Management (Phase 1)
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
from .goals import (
    Goal,
    GoalManager,
    GoalPriority,
    GoalStatus,
    GoalStorageProtocol,
    GoalStore,
    SuccessCriterion,
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

# Scheduler (Phase 2 — wraps HeartbeatManager)
from .scheduler import (
    AutonomousScheduler,
    ScheduledTask,
    TaskPriority,
)

# Skill Loading (Phase 3)
from .skills import (
    Skill,
    SkillLoader,
    SkillMetadata,
)

# State Persistence (Phase 1/3)
from .state import (
    AutonomousState,
    StateManager,
)

# Context redirect (Phase 3 — delegates to llmcore.context)
# NOTE: Not imported at package level to avoid circular imports.
# Use ``from llmcore.autonomous.context import ContextSynthesizer`` directly.

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
    # Skills
    "Skill",
    "SkillLoader",
    "SkillMetadata",
    # State
    "AutonomousState",
    "StateManager",
    # Scheduler
    "AutonomousScheduler",
    "ScheduledTask",
    "TaskPriority",
]
