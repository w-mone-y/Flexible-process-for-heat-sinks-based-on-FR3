"""Changeover KPIs and the three-tier baseline comparison.

The competition scores 换型时间缩短比例 (changeover-time reduction ratio), so the
numbers here are the ones a judge reads.  Three tiers are reported:

``MANUAL_TEACHING``
    Every product switch costs a human re-teaching window.  This is the
    denominator and must come from plant data (see :class:`TeachingBaseline`).

``AUTOMATIC_UNSORTED``
    Derived changeover actions, orders run in arrival order.  Isolates the
    benefit of automation alone.

``AUTOMATIC_FAMILY_BATCHED``
    Derived changeover actions plus setup-aware sequencing.  Isolates the
    additional benefit of letting the scheduler see setup cost.

Reporting both automatic tiers matters: it separates "we automated the motion"
from "we also scheduled it better", instead of claiming one number for both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..planning.task_models import TaskType
from .config_diff import FixtureConfiguration
from .setup_matrix import SetupTimeMatrix, TeachingBaseline, build_setup_matrix

CHANGEOVER_TASK_TYPES: frozenset[TaskType] = frozenset(
    {
        TaskType.REMOVE_OLD_PRESS,
        TaskType.REMOVE_OLD_COMB,
        TaskType.REMOVE_OLD_MOLD,
        TaskType.FETCH_MOLD,
        TaskType.INSTALL_MOLD,
        TaskType.VERIFY_MOLD,
        TaskType.FETCH_COMB,
        TaskType.INSTALL_COMB,
        TaskType.VERIFY_COMB,
        TaskType.FETCH_PRESS_MODULE,
        TaskType.INSTALL_PRESS_MODULE,
        TaskType.VERIFY_CHANGEOVER,
    }
)


@dataclass(frozen=True, slots=True)
class ChangeoverKpi:
    """The three headline changeover numbers plus their provenance."""

    changeover_seconds: float
    changeover_count: int
    changeover_action_count: int
    baseline_seconds: float
    baseline_source: str
    baseline_measured: bool
    productive_seconds: float = 0.0
    baseline_comparable: bool = True

    @property
    def changeover_ratio_vs_baseline(self) -> float:
        """Fraction of the manual baseline that is eliminated, in ``[0, 1]``.

        0.0 means no improvement; 0.9 means 90% of the manual re-teaching time
        is gone.
        """

        if self.baseline_seconds <= 0.0:
            return 0.0
        saved = self.baseline_seconds - self.changeover_seconds
        return max(0.0, min(1.0, saved / self.baseline_seconds))

    @property
    def changeover_share(self) -> float:
        """Changeover seconds as a fraction of total occupied time."""

        total = self.changeover_seconds + self.productive_seconds
        return 0.0 if total <= 0.0 else self.changeover_seconds / total

    def as_dict(self) -> dict[str, Any]:
        return {
            "changeover_seconds": self.changeover_seconds,
            "changeover_count": self.changeover_count,
            "changeover_action_count": self.changeover_action_count,
            "changeover_ratio_vs_baseline": self.changeover_ratio_vs_baseline,
            "changeover_share": self.changeover_share,
            "baseline_seconds": self.baseline_seconds,
            "baseline_source": self.baseline_source,
            "baseline_measured": self.baseline_measured,
            # Surfaced so a report cannot quietly present an estimate as data.
            "baseline_is_placeholder": not self.baseline_measured,
            "baseline_comparable": self.baseline_comparable,
        }


def collect_changeover_kpi(
    changeover_log: Iterable[Mapping[str, Any]],
    baseline: TeachingBaseline,
    *,
    productive_seconds: float = 0.0,
) -> ChangeoverKpi:
    """Aggregate a runtime's changeover log into the reportable KPI set.

    ``changeover_count`` counts *effective* changeovers (those that produced at
    least one action).  Units that needed no fixture change are deliberately not
    counted, because the manual baseline would have re-taught them too — that
    difference is the saving.
    """

    records = list(changeover_log)
    seconds = sum(float(item.get("nominal_seconds", 0.0)) for item in records)
    actions = sum(int(item.get("action_count", 0)) for item in records)
    effective = sum(1 for item in records if int(item.get("action_count", 0)) > 0)
    # The manual baseline pays a re-teaching window for every unit, including the
    # ones automation can skip.  Use the demo-scaled value so numerator and
    # denominator share this simulation's time base.
    baseline_seconds = baseline.demo_seconds_per_changeover * max(len(records), 1)
    return ChangeoverKpi(
        changeover_seconds=seconds,
        changeover_count=effective,
        changeover_action_count=actions,
        baseline_seconds=baseline_seconds,
        baseline_source=baseline.source,
        baseline_measured=baseline.measured,
        productive_seconds=float(productive_seconds),
        baseline_comparable=baseline.comparable,
    )


def is_changeover_task(task: Any) -> bool:
    """True for a fixture-changeover task, excluding post-braze teardown.

    ``REMOVE_OLD_COMB`` / ``REMOVE_OLD_PRESS`` serve two different purposes: they
    strip modules during a changeover, and they also release the finished part
    after brazing.  Only the former is setup time.  The builder tags real
    changeover nodes with ``changeover_slot``, and the teardown nodes carry
    ``after_brazing``, so the two are separable without guessing from task type.
    """

    payload = getattr(task, "payload", {}) or {}
    if payload.get("after_brazing"):
        return False
    if payload.get("changeover_slot"):
        return True
    # A changeover-only task type with no tag at all (legacy graphs).
    return task.task_type in CHANGEOVER_TASK_TYPES and task.route_phase == "CHANGEOVER"


def changeover_seconds_from_graph(
    graph: Iterable[Any],
    durations: Mapping[TaskType, float] | None = None,
) -> tuple[float, int]:
    """Measure changeover time directly from a task graph.

    An independent cross-check on the runtime log: the two must agree, and a
    mismatch means one of them is counting the wrong nodes.
    """

    total = 0.0
    count = 0
    for task in graph:
        if not is_changeover_task(task):
            continue
        count += 1
        if durations is None:
            total += float(task.estimated_duration)
        else:
            total += float(durations.get(task.task_type, task.estimated_duration))
    return total, count


@dataclass(frozen=True, slots=True)
class BaselineTier:
    """One row of the three-tier comparison."""

    name: str
    label_zh: str
    changeover_seconds: float
    changeover_count: int
    sequence: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label_zh": self.label_zh,
            "changeover_seconds": self.changeover_seconds,
            "changeover_count": self.changeover_count,
            "sequence": list(self.sequence),
        }


def compare_changeover_baselines(
    configurations: Sequence[FixtureConfiguration],
    durations: Mapping[TaskType, float],
    baseline: TeachingBaseline,
    *,
    matrix: SetupTimeMatrix | None = None,
) -> dict[str, Any]:
    """Build the three-tier comparison for one order queue.

    ``configurations`` is the arrival-order list of required fixture
    configurations, one entry per product unit.
    """

    items = list(configurations)
    if not items:
        return {"tiers": [], "improvements": {}, "unit_count": 0}

    setup = matrix or build_setup_matrix(items, durations)
    unsorted_seconds = setup.sequence_cost(items)
    sorted_seconds, sorted_order = setup.best_sequence_cost(items)

    def effective_count(sequence: Sequence[FixtureConfiguration]) -> int:
        count = 0
        current: FixtureConfiguration | None = None
        for target in sequence:
            if setup.setup_time(current, target) > 0.0:
                count += 1
            current = target
        return count

    # Demo-scaled so all three tiers are on the simulation's time base.
    manual_seconds = baseline.demo_seconds_per_changeover * len(items)
    tiers = [
        BaselineTier(
            name="MANUAL_TEACHING",
            label_zh="人工示教基线",
            changeover_seconds=manual_seconds,
            changeover_count=len(items),
            sequence=tuple(item.signature() for item in items),
        ),
        BaselineTier(
            name="AUTOMATIC_UNSORTED",
            label_zh="自动换型（按到达顺序）",
            changeover_seconds=unsorted_seconds,
            changeover_count=effective_count(items),
            sequence=tuple(item.signature() for item in items),
        ),
        BaselineTier(
            name="AUTOMATIC_FAMILY_BATCHED",
            label_zh="自动换型＋同族批量排序",
            changeover_seconds=sorted_seconds,
            changeover_count=effective_count(sorted_order),
            sequence=tuple(item.signature() for item in sorted_order),
        ),
    ]

    def ratio(value: float) -> float:
        if manual_seconds <= 0.0:
            return 0.0
        return max(0.0, min(1.0, (manual_seconds - value) / manual_seconds))

    return {
        "unit_count": len(items),
        "tiers": [item.as_dict() for item in tiers],
        "improvements": {
            # Versus the manual denominator — the competition's headline metric.
            "automation_ratio": ratio(unsorted_seconds),
            "automation_and_sequencing_ratio": ratio(sorted_seconds),
            # Sequencing alone, isolated from automation.
            "sequencing_only_ratio": (
                0.0
                if unsorted_seconds <= 0.0
                else max(0.0, (unsorted_seconds - sorted_seconds) / unsorted_seconds)
            ),
            "sequencing_saved_seconds": max(0.0, unsorted_seconds - sorted_seconds),
        },
        "baseline": baseline.as_dict(),
    }


__all__ = [
    "CHANGEOVER_TASK_TYPES",
    "BaselineTier",
    "ChangeoverKpi",
    "changeover_seconds_from_graph",
    "collect_changeover_kpi",
    "compare_changeover_baselines",
]
