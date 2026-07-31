"""Changeover modelling: configuration state, diffs and setup-time matrices.

Step D of the flexibility upgrade.  The physical changeover motions already
existed (gantry, mold/comb/press libraries, remove/fetch/install task types),
but they were emitted as an unconditional seven-step chain per unit and were
never measured.  This package adds the three missing pieces:

``config_state``
    A resource's fixture configuration as data — ``(mold, comb, press, tool,
    program)`` — so "what is currently set up" is a first-class value.

``config_diff``
    Target minus current, yielding the *minimal* action set.  Identical
    configurations produce zero actions, which is where family-batching savings
    physically come from.

``setup_matrix``
    Sequence-dependent setup times ``setup_time[from][to]``.  Feeding these into
    the scheduling cost turns the problem from FJSP into FJSP-SDST.
"""

from .config_diff import (
    ChangeoverAction,
    ChangeoverPlan,
    FixtureConfiguration,
    configuration_family,
    plan_changeover,
    required_configuration,
)
from .metrics import (
    CHANGEOVER_TASK_TYPES,
    BaselineTier,
    ChangeoverKpi,
    changeover_seconds_from_graph,
    collect_changeover_kpi,
    compare_changeover_baselines,
    is_changeover_task,
)
from .setup_matrix import (
    PLACEHOLDER_TEACHING_BASELINE,
    SetupTimeMatrix,
    TeachingBaseline,
    build_setup_matrix,
)

__all__ = [
    "CHANGEOVER_TASK_TYPES",
    "PLACEHOLDER_TEACHING_BASELINE",
    "BaselineTier",
    "ChangeoverAction",
    "ChangeoverKpi",
    "ChangeoverPlan",
    "FixtureConfiguration",
    "SetupTimeMatrix",
    "TeachingBaseline",
    "build_setup_matrix",
    "changeover_seconds_from_graph",
    "collect_changeover_kpi",
    "compare_changeover_baselines",
    "configuration_family",
    "is_changeover_task",
    "plan_changeover",
    "required_configuration",
]
