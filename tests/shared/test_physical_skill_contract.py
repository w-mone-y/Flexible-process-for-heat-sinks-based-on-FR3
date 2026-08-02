from __future__ import annotations

from dataclasses import dataclass

from brazing_sim.execution import (
    PhysicalCompletionEvidence,
    SkillExecutionResult,
    SkillExecutor,
    SkillRegistry,
)
from brazing_sim.planning import ManufacturingTask, TaskType


def _task(task_id: str) -> ManufacturingTask:
    return ManufacturingTask(
        task_id=task_id,
        task_type=TaskType.INSTALL_FIN,
        order_id="ORDER",
        unit_id=f"{task_id}_UNIT",
        eligible_resources=["ARM1", "ARM3"],
        estimated_duration=1.0,
    )


@dataclass
class _FeedbackSkill:
    evidence: PhysicalCompletionEvidence | None = None
    task_id: str | None = None

    def start(self, task, resource_id, context, now) -> None:
        del resource_id, context, now
        self.task_id = task.task_id

    def update(self, now, dt) -> SkillExecutionResult:
        del dt
        return SkillExecutionResult.success(
            {"observed_at": now},
            completion_evidence=self.evidence,
        )

    def cancel(self, task_id: str) -> None:
        if self.task_id == task_id:
            self.task_id = None


def test_physical_skill_cannot_finish_without_measured_completion_evidence() -> None:
    instances: list[_FeedbackSkill] = []

    def create_skill() -> _FeedbackSkill:
        skill = _FeedbackSkill()
        instances.append(skill)
        return skill

    registry = SkillRegistry()
    registry.register_factory(
        TaskType.INSTALL_FIN,
        create_skill,
        requires_physical_evidence=True,
    )
    executor = SkillExecutor(registry)
    task = _task("INSTALL_01")
    executor.start_task(task, "ARM1", now=0.0)

    result = executor.update_task(0.1, now=0.1)[task.task_id]

    assert result.running
    assert result.metrics["completion_blocker"] == "WAITING_FOR_PHYSICAL_EVIDENCE"
    assert "ARM1" in executor.snapshot()

    instances[0].evidence = PhysicalCompletionEvidence(
        observed_at=0.2,
        source="mujoco:v2_scene_adapter",
        checks=("夹爪已松开", "翅片位姿已停稳"),
    )
    result = executor.update_task(0.1, now=0.2)[task.task_id]

    assert result.succeeded
    assert result.completion_evidence is not None
    assert executor.snapshot() == {}


def test_registry_factory_allows_same_task_type_on_two_resources() -> None:
    instances: list[_FeedbackSkill] = []

    def create_skill() -> _FeedbackSkill:
        skill = _FeedbackSkill(
            evidence=PhysicalCompletionEvidence(
                observed_at=0.1,
                source="mujoco:test",
                checks=("settled",),
            )
        )
        instances.append(skill)
        return skill

    registry = SkillRegistry()
    registry.register_factory(
        TaskType.INSTALL_FIN,
        create_skill,
        requires_physical_evidence=True,
    )
    executor = SkillExecutor(registry)

    executor.start_task(_task("INSTALL_ARM1"), "ARM1", now=0.0)
    executor.start_task(_task("INSTALL_ARM3"), "ARM3", now=0.0)

    assert len(instances) == 2
    assert set(executor.snapshot()) == {"ARM1", "ARM3"}
    results = executor.update_task(0.1, now=0.1)
    assert all(result.succeeded for result in results.values())
