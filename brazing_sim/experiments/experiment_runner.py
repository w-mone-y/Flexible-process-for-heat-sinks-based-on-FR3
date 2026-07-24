"""Reusable headless experiment runner used by CLI and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..flexible.models import ProcessPlan
from ..manufacturing_config import FaultScenario
from ..manufacturing_runtime import ManufacturingRuntime
from .baseline_comparison import compare_experiments
from .metrics_collector import MetricsCollector
from .report_exporter import ExperimentReporter


class ExperimentRunner:
    def __init__(self, *, step_seconds: float = 0.05, max_sim_time: float = 3600.0) -> None:
        self.step_seconds = float(step_seconds)
        self.max_sim_time = float(max_sim_time)

    def run(
        self,
        plans: Iterable[ProcessPlan],
        *,
        scheduler: str,
        scenario: FaultScenario | None = None,
        output_directory: str | Path | None = None,
    ) -> tuple[ManufacturingRuntime, dict]:
        runtime = ManufacturingRuntime(scheduler_mode=scheduler, flexible_cell=True)
        collector = MetricsCollector()
        runtime.events.subscribe(None, collector.handle_event)
        runtime.submit_plans(tuple(plans), now=0.0)
        runtime.set_fault_scenario(scenario)
        now = 0.0
        while now <= self.max_sim_time and not runtime.terminal:
            runtime.tick(now)
            now += self.step_seconds
        if not runtime.terminal:
            raise TimeoutError(f"experiment exceeded {self.max_sim_time}s simulation time")
        metrics = collector.calculate(runtime, now)
        if output_directory is not None:
            ExperimentReporter(output_directory).export(runtime, metrics)
        return runtime, metrics

    def compare(
        self,
        plans: Iterable[ProcessPlan],
        *,
        fixed_scenario: FaultScenario | None = None,
        dynamic_scenario: FaultScenario | None = None,
    ) -> dict:
        values = tuple(plans)
        _, fixed = self.run(values, scheduler="fixed", scenario=fixed_scenario)
        _, dynamic = self.run(values, scheduler="dynamic", scenario=dynamic_scenario)
        return compare_experiments(fixed, dynamic)


__all__ = ["ExperimentRunner"]
