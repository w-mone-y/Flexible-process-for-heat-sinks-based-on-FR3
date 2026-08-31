from __future__ import annotations

import pytest

from benchmarks.run_authority_comparison import compare_runs, parse_modes


def _run(mode: str, makespan: float, arm1_idle: float) -> dict[str, object]:
    return {
        "mode": mode,
        "completed": True,
        "simulation_seconds": makespan,
        "arm1_idle_s": arm1_idle,
        "throughput_per_sim_hour": 3 * 3600.0 / makespan,
    }


def test_authority_comparison_reports_actual_gain_against_operator_fallback() -> None:
    comparison = compare_runs(
        _run("AUTHORITY", 90.0, 12.0),
        _run("FALLBACK", 100.0, 20.0),
    )

    assert comparison["both_completed"] is True
    assert comparison["makespan_improvement_pct"] == pytest.approx(10.0)
    assert comparison["arm1_idle_improvement_pct"] == pytest.approx(40.0)


def test_authority_benchmark_mode_parser_is_explicit_and_deterministic() -> None:
    assert parse_modes("authority,fallback") == ("AUTHORITY", "FALLBACK")
    with pytest.raises(ValueError, match="AUTHORITY"):
        parse_modes("shadow")

