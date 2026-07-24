"""Fixed-vs-dynamic metric comparison."""

from __future__ import annotations

from typing import Any, Mapping


def _change(fixed: float, dynamic: float) -> dict[str, float | None]:
    absolute = dynamic - fixed
    percentage = None if abs(fixed) < 1e-12 else 100.0 * absolute / fixed
    return {"fixed": fixed, "dynamic": dynamic, "absolute_change": absolute, "percent_change": percentage}


def compare_experiments(
    fixed_metrics: Mapping[str, Any], dynamic_metrics: Mapping[str, Any]
) -> dict[str, dict[str, float | None]]:
    mapping = {
        "makespan": "makespan",
        "average_robot_utilization": "average_robot_utilization",
        "average_task_wait_seconds": "average_task_wait_seconds",
        "zone_conflict_count": "zone_conflict_count",
        "tool_change_count": "tool_change_count",
        "throughput": "throughput_per_sim_second",
        "recovery_rate": "recovery_rate",
        "on_time_rate": "on_time_rate",
    }
    return {
        label: _change(float(fixed_metrics.get(key, 0.0)), float(dynamic_metrics.get(key, 0.0)))
        for label, key in mapping.items()
    }


__all__ = ["compare_experiments"]
