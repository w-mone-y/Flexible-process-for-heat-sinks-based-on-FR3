from __future__ import annotations

import json

from brazing_sim.experiments.golden_suite import GoldenExperimentSuite


def test_golden_suite_runs_four_controlled_event_timed_comparisons(tmp_path) -> None:
    result = GoldenExperimentSuite(seed=42, step_seconds=0.05).run(tmp_path)

    assert result["schema_version"] == 1
    assert result["seed"] == 42
    assert len(result["groups"]) == 4
    assert {group["group_id"] for group in result["groups"]} == {
        "G1_SCHEDULER",
        "G2_INSTALL_RESOURCE",
        "G3_ORDER_SEQUENCE",
        "G4_RECOVERY_DISPOSITION",
    }
    for group in result["groups"]:
        assert len(group["runs"]) == 2
        assert all(run["complete"] for run in group["runs"])
        assert all(run["measurement_source"] == "SIMULATION_EVENT_TIMESTAMPS" for run in group["runs"])
        assert all(run["makespan_s"] > 0.0 for run in group["runs"])

    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload == result
    assert (tmp_path / "runs.csv").is_file()
    assert (tmp_path / "events.jsonl").is_file()
    assert (tmp_path / "summary.md").is_file()
