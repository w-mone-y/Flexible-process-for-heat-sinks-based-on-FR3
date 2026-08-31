from __future__ import annotations

from benchmarks.run_shadow_comparison import build_comparison


def test_shadow_comparison_reports_non_mutating_dispatch_boundary() -> None:
    report = build_comparison(("A",), time_limit_s=1.0, random_seed=3)

    assert report["dispatch_boundary"]["dispatch_mutated"] is False
    assert report["dispatch_boundary"]["scheduler_mode"] == "DYNAMIC_PRIORITY"
    assert report["shadow_schedule"]["status"] == "FEASIBLE"
    assert report["shadow_schedule"]["validation"]["valid"] is True
    assert report["shadow_schedule"]["reference_objective_value"] is not None

