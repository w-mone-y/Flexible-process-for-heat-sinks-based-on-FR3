"""Isolated MuJoCo worker for one physical golden-experiment variant."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    from ..dual_line.application import V2BrazingApplication
    from ..dual_line.cli import parse_args

    args = parse_args(("--headless", "--fast", "--no-ui", "--max-sim-time", "260"))
    application = V2BrazingApplication(args)
    started = time.perf_counter()
    try:
        presets = tuple(str(value) for value in payload["presets"])
        group_id = str(payload["group_id"])
        variant_id = str(payload["variant_id"])
        for index, preset in enumerate(presets, start=1):
            application.runtime.submit_order(
                preset,
                order_id=f"{group_id}_{variant_id}_{index:02d}_{preset}",
                priority=100 - index,
            )
        application.scene.sync(application.runtime)
        state: dict[str, Any] = {}
        for _ in range(6_000):
            application.advance_frame()
            if application.runtime.complete:
                state = application.publish(viewer_running=False)
                break
        if not state:
            raise TimeoutError(f"{group_id}/{variant_id} physical V2 run did not terminate")
        makespan = max(
            float(unit["completed_at"]) for unit in state["units"] if unit.get("completed_at") is not None
        )
        experiment = state.get("experiment_metrics", {})
        completed = len(state.get("completed_orders", ()))
        return {
            "complete": bool(state.get("physical_execution_complete")),
            "makespan_s": makespan,
            "throughput_units_per_hour": 0.0 if makespan <= 0 else 3600.0 * completed / makespan,
            "average_robot_utilization": float(experiment.get("average_robot_utilization", 0.0)),
            "completed_units": completed,
            "recovery_rate": float(experiment.get("recovery_rate", 0.0)),
            "parallel_install_s": float(state.get("scheduled_parallel_install_seconds", 0.0)),
            "changeover_s": 0.0,
            "wall_seconds": time.perf_counter() - started,
            "events": list(application.runtime.events),
        }
    finally:
        application.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()
    payload = json.loads(args.payload)
    print(json.dumps(run(payload), ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
