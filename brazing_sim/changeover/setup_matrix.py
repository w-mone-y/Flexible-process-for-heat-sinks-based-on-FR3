"""Sequence-dependent setup times and the manual-teaching baseline.

Feeding ``setup_time[from][to]`` into the scheduling cost turns the problem from
FJSP into **FJSP-SDST** (flexible job shop with sequence-dependent setup times):
the cost of running an order depends on *which order ran before it*, so ordering
decisions and setup savings become coupled.  A scheduler that can see this cost
groups same-family orders together on its own.

The baseline deserves care.  The competition scores "换型时间缩短比例", which
needs a denominator.  Simulated automatic changeover is the numerator; the
denominator is how long a human takes to re-teach the cell.  That number must
come from the factory, not from us — :class:`TeachingBaseline` therefore carries
an explicit ``source`` string and refuses to pretend a guess is a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..planning.task_models import TaskType
from .config_diff import FixtureConfiguration, ChangeoverPlan, plan_changeover


@dataclass(frozen=True, slots=True)
class TeachingBaseline:
    """Manual re-teaching time used as the improvement denominator.

    **The time base matters.**  This cell runs on compressed demo durations (a
    full module swap is ~16 s, a furnace cycle 10 s), not real factory seconds.
    Dividing a real 30-minute re-teaching window by a 16-second demo changeover
    yields a meaningless "98.9% improvement" that is an artefact of unit
    mismatch.  ``demo_scale`` converts plant seconds into this simulation's time
    base, and :attr:`comparable` reports whether the comparison is legitimate.

    ``source`` records provenance.  ``measured`` is False for an estimate, and
    every report surfaces that rather than passing a guess off as plant data.
    """

    seconds_per_changeover: float
    source: str
    measured: bool = False
    notes_zh: str = ""
    # Demo seconds per real second for the same physical motion.
    demo_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.seconds_per_changeover <= 0.0:
            raise ValueError("人工示教基线必须为正数")
        if not str(self.source).strip():
            raise ValueError("人工示教基线必须声明数据来源")
        if self.demo_scale <= 0.0:
            raise ValueError("演示时间缩放系数必须为正数")

    @property
    def is_placeholder(self) -> bool:
        return not self.measured

    @property
    def demo_seconds_per_changeover(self) -> float:
        """The baseline expressed in this simulation's time base."""

        return self.seconds_per_changeover * self.demo_scale

    @property
    def comparable(self) -> bool:
        """True when the baseline shares the simulation's time base."""

        return self.measured or abs(self.demo_scale - 1.0) > 1e-9

    def as_dict(self) -> dict[str, Any]:
        return {
            "seconds_per_changeover": self.seconds_per_changeover,
            "demo_seconds_per_changeover": self.demo_seconds_per_changeover,
            "demo_scale": self.demo_scale,
            "source": self.source,
            "measured": self.measured,
            "is_placeholder": self.is_placeholder,
            "comparable": self.comparable,
            "notes_zh": self.notes_zh,
        }


# Placeholder until the factory visit (competition rules §9.1 grants an on-site
# survey, §9.2 a CTO-led mentor group).
#
# 30 min manual re-teaching per product switch is a conservative mid-range figure
# for a multi-station cell.  ``demo_scale`` maps it onto this simulation's clock:
# the automatic module swap takes ~16 demo seconds where the real motion is
# ~4 min, i.e. roughly 1/15 scale.  Without that factor the reported ratio
# measures the demo compression, not the automation.
#
# Replace BOTH the number and ``measured=True`` with plant data before publishing
# any changeover ratio.
PLACEHOLDER_TEACHING_BASELINE = TeachingBaseline(
    seconds_per_changeover=1800.0,
    source="占位值：未经现场验证的行业中位估计（30 分钟/次人工示教）",
    measured=False,
    demo_scale=1.0 / 15.0,
    notes_zh=(
        "赛题第九条第1、2款提供现场调研与 CTO 技术导师组答疑，务必用实测值替换本占位值。"
        "另需注意：本仿真使用压缩的演示节拍，直接用真实示教秒数作分母会得到虚高比例，"
        "因此基线先按 demo_scale 折算到仿真时基再比较。"
    ),
)


@dataclass(slots=True)
class SetupTimeMatrix:
    """``setup_time[from_signature][to_signature]`` in seconds."""

    durations: Mapping[TaskType, float]
    entries: dict[tuple[str, str], float] = field(default_factory=dict)
    plans: dict[tuple[str, str], ChangeoverPlan] = field(default_factory=dict)
    verify: bool = False

    def register(
        self,
        source: FixtureConfiguration,
        target: FixtureConfiguration,
    ) -> ChangeoverPlan:
        """Compute (and memoise) the changeover between two configurations."""

        key = (source.signature(), target.signature())
        found = self.plans.get(key)
        if found is not None:
            return found
        plan = plan_changeover(source, target, verify=self.verify)
        self.plans[key] = plan
        self.entries[key] = plan.duration(self.durations)
        return plan

    def setup_time(
        self,
        source: FixtureConfiguration | None,
        target: FixtureConfiguration,
    ) -> float:
        """Setup seconds required before ``target`` can run.

        ``source=None`` means a cold line, which pays the full installation.
        """

        origin = source if source is not None else FixtureConfiguration()
        self.register(origin, target)
        return self.entries[(origin.signature(), target.signature())]

    def sequence_cost(self, configurations: Iterable[FixtureConfiguration]) -> float:
        """Total setup time for one concrete production sequence."""

        total = 0.0
        current: FixtureConfiguration | None = None
        for target in configurations:
            total += self.setup_time(current, target)
            current = target
        return total

    def best_sequence_cost(
        self,
        configurations: Iterable[FixtureConfiguration],
    ) -> tuple[float, tuple[FixtureConfiguration, ...]]:
        """Setup cost of grouping identical configurations together.

        This is a greedy family-batching sequence, not a proven optimum: it
        starts from the cheapest first setup and then always continues with a
        remaining order of the same configuration when one exists.  That is
        exactly the behaviour a setup-aware scheduler exhibits, which makes it
        the right comparison point for the "sorted vs unsorted" experiment.
        """

        remaining = list(configurations)
        if not remaining:
            return 0.0, ()
        ordered: list[FixtureConfiguration] = []
        current: FixtureConfiguration | None = None
        total = 0.0
        while remaining:
            if current is not None:
                same = next(
                    (item for item in remaining if item.matches(current)),
                    None,
                )
                if same is not None:
                    remaining.remove(same)
                    ordered.append(same)
                    total += self.setup_time(current, same)
                    current = same
                    continue
            cheapest = min(remaining, key=lambda item: (self.setup_time(current, item), item.signature()))
            remaining.remove(cheapest)
            ordered.append(cheapest)
            total += self.setup_time(current, cheapest)
            current = cheapest
        return total, tuple(ordered)

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": [
                {"from": key[0], "to": key[1], "seconds": value}
                for key, value in sorted(self.entries.items())
            ],
            "signatures": sorted({key[0] for key in self.entries} | {key[1] for key in self.entries}),
            "verify": self.verify,
        }


def build_setup_matrix(
    configurations: Iterable[FixtureConfiguration],
    durations: Mapping[TaskType, float],
    *,
    verify: bool = False,
    include_cold_start: bool = True,
) -> SetupTimeMatrix:
    """Build the full pairwise setup matrix over the given configurations."""

    items = list(configurations)
    matrix = SetupTimeMatrix(durations=dict(durations), verify=verify)
    if include_cold_start:
        cold = FixtureConfiguration()
        for target in items:
            matrix.register(cold, target)
    for source in items:
        for target in items:
            matrix.register(source, target)
    return matrix


__all__ = [
    "PLACEHOLDER_TEACHING_BASELINE",
    "SetupTimeMatrix",
    "TeachingBaseline",
    "build_setup_matrix",
]
