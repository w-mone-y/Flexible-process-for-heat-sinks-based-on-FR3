"""Fixture configuration state and minimal changeover action derivation.

Before step D the builder emitted a fixed seven-step changeover chain for every
unit:

``REMOVE_PRESS → REMOVE_COMB → REMOVE_MOLD → FETCH_MOLD → INSTALL_MOLD → VERIFY_MOLD``

That chain ran even when the next unit needed exactly the same fixture, so three
units of one product paid the setup three times.  Here the chain is *derived*
from ``target − current``:

*   same configuration → **zero actions** (the family-batching win);
*   only the comb differs → comb actions only, the mold stays;
*   nothing installed yet (line cold) → install actions with no removals.

The action list is ordered by physical necessity: everything mounted on top must
come off before what is underneath, and installation reverses that order.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from ..planning.task_models import TaskType

# Mount order from the tray upwards.  Removal walks this list backwards.
_MOUNT_ORDER = ("mold", "comb", "press")

_REMOVE_TASKS: dict[str, TaskType] = {
    "press": TaskType.REMOVE_OLD_PRESS,
    "comb": TaskType.REMOVE_OLD_COMB,
    "mold": TaskType.REMOVE_OLD_MOLD,
}

_FETCH_TASKS: dict[str, TaskType] = {
    "mold": TaskType.FETCH_MOLD,
    "comb": TaskType.FETCH_COMB,
    "press": TaskType.FETCH_PRESS_MODULE,
}

_INSTALL_TASKS: dict[str, TaskType] = {
    "mold": TaskType.INSTALL_MOLD,
    "comb": TaskType.INSTALL_COMB,
    "press": TaskType.INSTALL_PRESS_MODULE,
}

_VERIFY_TASKS: dict[str, TaskType] = {
    "mold": TaskType.VERIFY_MOLD,
    "comb": TaskType.VERIFY_COMB,
}

_SLOT_LABELS_ZH: dict[str, str] = {
    "mold": "托盘模具",
    "comb": "梳齿模块",
    "press": "短压梁",
    "tool": "末端工具",
    "program": "作业程序",
}


@dataclass(frozen=True, slots=True)
class FixtureConfiguration:
    """What is currently set up on a station, as data.

    ``program`` is the parametric process program identifier.  It changes with
    the product even when the physical fixture does not, which is precisely the
    thing that used to require re-teaching and now does not: switching programs
    is a data operation with no gantry motion.
    """

    mold: str | None = None
    comb: str | None = None
    press: str | None = None
    tool: str | None = None
    program: str | None = None

    @property
    def is_empty(self) -> bool:
        """True when nothing is mounted (a cold line)."""

        return all(getattr(self, slot) is None for slot in _MOUNT_ORDER)

    def slot(self, name: str) -> str | None:
        if name not in _SLOT_LABELS_ZH:
            raise KeyError(f"unknown configuration slot: {name}")
        return getattr(self, name)

    def differing_slots(self, other: "FixtureConfiguration") -> tuple[str, ...]:
        """Slot names whose value differs, in deterministic mount order."""

        return tuple(
            name for name in (*_MOUNT_ORDER, "tool", "program") if self.slot(name) != other.slot(name)
        )

    def matches(self, other: "FixtureConfiguration") -> bool:
        """True when no physical changeover is required between the two."""

        return all(self.slot(name) == other.slot(name) for name in _MOUNT_ORDER)

    def with_slot(self, name: str, value: str | None) -> "FixtureConfiguration":
        if name not in _SLOT_LABELS_ZH:
            raise KeyError(f"unknown configuration slot: {name}")
        return replace(self, **{name: value})

    def as_dict(self) -> dict[str, Any]:
        return {name: self.slot(name) for name in (*_MOUNT_ORDER, "tool", "program")}

    def signature(self) -> str:
        """Stable key for setup-time matrices and family grouping."""

        return "|".join(str(self.slot(name) or "-") for name in _MOUNT_ORDER)


@dataclass(frozen=True, slots=True)
class ChangeoverAction:
    """One derived changeover step, ready to become a task graph node."""

    kind: str  # REMOVE / FETCH / INSTALL / VERIFY
    slot: str
    task_type: TaskType
    module_name: str | None
    description_zh: str

    @property
    def suffix(self) -> str:
        """Task-id suffix, kept compatible with the historical names."""

        legacy = {
            (TaskType.REMOVE_OLD_PRESS, "press"): "REMOVE_PRESS",
            (TaskType.REMOVE_OLD_COMB, "comb"): "REMOVE_COMB",
            (TaskType.REMOVE_OLD_MOLD, "mold"): "REMOVE_MOLD",
            (TaskType.FETCH_MOLD, "mold"): "FETCH_MOLD",
            (TaskType.INSTALL_MOLD, "mold"): "INSTALL_MOLD",
            (TaskType.VERIFY_MOLD, "mold"): "VERIFY_MOLD",
        }
        found = legacy.get((self.task_type, self.slot))
        return found or f"{self.kind}_{self.slot.upper()}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "slot": self.slot,
            "task_type": self.task_type.value,
            "module_name": self.module_name,
            "description_zh": self.description_zh,
        }


@dataclass(frozen=True, slots=True)
class ChangeoverPlan:
    """The minimal ordered action set taking ``source`` to ``target``."""

    source: FixtureConfiguration
    target: FixtureConfiguration
    actions: tuple[ChangeoverAction, ...]
    program_only: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.actions

    @property
    def changed_slots(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.slot for item in self.actions))

    def duration(self, durations: Mapping[TaskType, float]) -> float:
        """Total nominal changeover time under a duration table."""

        return sum(float(durations.get(item.task_type, 0.0)) for item in self.actions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.as_dict(),
            "target": self.target.as_dict(),
            "actions": [item.as_dict() for item in self.actions],
            "action_count": len(self.actions),
            "changed_slots": list(self.changed_slots),
            "program_only": self.program_only,
        }


def required_configuration(plan: Any) -> FixtureConfiguration:
    """Derive the fixture configuration a process plan needs.

    The mold and press follow the comb pitch in this cell, so they are named
    from the selected fixture module rather than configured separately.
    """

    module = plan.fixture_module
    pitch_mm = int(round(float(module.pitch_m) * 1000.0))
    return FixtureConfiguration(
        mold=f"mold_{pitch_mm}mm",
        comb=module.name,
        press="press_short_pair",
        tool=None,
        program=f"{plan.product.product_id}:{plan.recipe.name}",
    )


def plan_changeover(
    source: FixtureConfiguration,
    target: FixtureConfiguration,
    *,
    verify: bool = False,
) -> ChangeoverPlan:
    """Derive the minimal action set to reach ``target`` from ``source``.

    ``verify`` adds the high-reliability confirmation steps.  Removal proceeds
    top-down (press, comb, mold) and installation bottom-up, because a comb
    cannot be lifted out from under a fitted press.
    """

    differing = set(source.differing_slots(target)) & set(_MOUNT_ORDER)
    if not differing:
        # Nothing physical to do.  A program change is still recorded so the KPI
        # layer can distinguish "no changeover at all" from "data-only switch".
        program_only = source.program != target.program
        return ChangeoverPlan(source, target, (), program_only=program_only)

    # Any slot above a changing slot must be removed to reach it, even when its
    # own module is unchanged.
    lowest = min(_MOUNT_ORDER.index(name) for name in differing)
    disturbed = [name for name in _MOUNT_ORDER[lowest:] if source.slot(name) is not None]

    actions: list[ChangeoverAction] = []
    for name in reversed(disturbed):
        actions.append(
            ChangeoverAction(
                kind="REMOVE",
                slot=name,
                task_type=_REMOVE_TASKS[name],
                module_name=source.slot(name),
                description_zh=f"拆除{_SLOT_LABELS_ZH[name]}",
            )
        )

    for name in _MOUNT_ORDER[lowest:]:
        wanted = target.slot(name)
        if wanted is None:
            continue
        actions.append(
            ChangeoverAction(
                kind="FETCH",
                slot=name,
                task_type=_FETCH_TASKS[name],
                module_name=wanted,
                description_zh=f"龙门取出{_SLOT_LABELS_ZH[name]}",
            )
        )
        actions.append(
            ChangeoverAction(
                kind="INSTALL",
                slot=name,
                task_type=_INSTALL_TASKS[name],
                module_name=wanted,
                description_zh=f"安装{_SLOT_LABELS_ZH[name]}",
            )
        )
        if name in _VERIFY_TASKS and (verify or name == "mold"):
            actions.append(
                ChangeoverAction(
                    kind="VERIFY",
                    slot=name,
                    task_type=_VERIFY_TASKS[name],
                    module_name=wanted,
                    description_zh=f"验证{_SLOT_LABELS_ZH[name]}锁定",
                )
            )
    return ChangeoverPlan(source, target, tuple(actions))


def configuration_family(configurations: Iterable[FixtureConfiguration]) -> dict[str, int]:
    """Count how many configurations share each signature.

    Used by reporting to show why sequencing same-family orders together avoids
    setup work.
    """

    counts: dict[str, int] = {}
    for item in configurations:
        key = item.signature()
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "ChangeoverAction",
    "ChangeoverPlan",
    "FixtureConfiguration",
    "configuration_family",
    "plan_changeover",
    "required_configuration",
]
