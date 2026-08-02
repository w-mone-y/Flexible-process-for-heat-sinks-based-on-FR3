"""Non-blocking task-to-skill execution adapters."""

from .skill_executor import SkillExecutor
from .execution_monitor import ExecutionMonitor
from .skill_registry import (
    ActorSkillAdapter,
    PhysicalCompletionEvidence,
    SkillExecutionResult,
    SkillRegistry,
    TimedSkill,
)
from ..planning.task_models import TaskType
from .async_line_skills import build_physical_async_line_skill_registry


def build_async_line_skill_registry() -> SkillRegistry:
    """Build the deterministic logical executor for the shallow-U runtime.

    Physical robot and conveyor motion remains owned by the existing actors;
    the manufacturing runtime consumes these timed skills for scheduling and
    experiment accounting.  No retired turntable or gantry skill is installed.
    """

    registry = SkillRegistry()
    for task_type in TaskType:
        registry.register_factory(task_type, TimedSkill)
    return registry


__all__ = [
    "ActorSkillAdapter",
    "PhysicalCompletionEvidence",
    "ExecutionMonitor",
    "SkillExecutionResult",
    "SkillExecutor",
    "SkillRegistry",
    "TimedSkill",
    "build_async_line_skill_registry",
    "build_physical_async_line_skill_registry",
]
