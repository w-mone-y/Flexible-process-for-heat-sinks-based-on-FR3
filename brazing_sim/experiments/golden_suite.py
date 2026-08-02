"""Four reproducible golden comparisons for the flexibility competition.

Every reported duration is derived from terminal simulation events or unit
completion timestamps.  Scheduler estimates are retained only as inputs and
never used as measured output.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

from ..dual_line.runtime import DualLineRuntime
from ..dual_line.tray_flow import TrayOwner
from ..flexible import build_inline_plan
from ..manufacturing_runtime import ManufacturingRuntime
from .metrics_collector import MetricsCollector


@dataclass(frozen=True, slots=True)
class GoldenRun:
    group_id: str
    variant_id: str
    label_zh: str
    complete: bool
    makespan_s: float
    throughput_units_per_hour: float
    average_robot_utilization: float
    completed_units: int
    recovery_rate: float
    parallel_install_s: float
    changeover_s: float
    wall_seconds: float
    measurement_source: str = "SIMULATION_EVENT_TIMESTAMPS"


class _InstallResourceGate:
    """Experiment-only actor gate that fixes the S3 OR branch."""

    def __init__(self, resource: str) -> None:
        self.resource = str(resource).upper()

    def preferred_install_resource(self, unit_id: str) -> str:
        del unit_id
        return self.resource

    @staticmethod
    def tray_ready(tray_id: str, owner: TrayOwner) -> bool:
        del tray_id, owner
        return True

    @staticmethod
    def owner_available(owner: TrayOwner) -> bool:
        del owner
        return True

    @staticmethod
    def operation_complete(resource: str, unit_id: str, kind: str) -> bool:
        del resource, unit_id, kind
        return True

    @staticmethod
    def operation_start_allowed(resource: str, unit_id: str, kind: str) -> bool:
        del resource, unit_id, kind
        return True


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _comparison(first: GoldenRun, second: GoldenRun) -> dict[str, float | None]:
    absolute = second.makespan_s - first.makespan_s
    return {
        "baseline_makespan_s": first.makespan_s,
        "candidate_makespan_s": second.makespan_s,
        "absolute_change_s": absolute,
        "makespan_improvement_percent": (
            None
            if first.makespan_s <= 1.0e-12
            else 100.0 * (first.makespan_s - second.makespan_s) / first.makespan_s
        ),
        "baseline_throughput_units_per_hour": first.throughput_units_per_hour,
        "candidate_throughput_units_per_hour": second.throughput_units_per_hour,
    }


class GoldenExperimentSuite:
    """Run scheduler, resource, changeover and recovery comparisons."""

    def __init__(self, *, seed: int = 42, step_seconds: float = 0.05) -> None:
        if step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        self.seed = int(seed)
        self.step_seconds = float(step_seconds)
        self._events: list[dict[str, Any]] = []

    def _manufacturing_run(
        self,
        *,
        group_id: str,
        variant_id: str,
        label_zh: str,
        presets: Iterable[str],
        scheduler: str,
        max_wip: int,
        track_changeover: bool = False,
    ) -> GoldenRun:
        runtime = ManufacturingRuntime(
            scheduler_mode=scheduler,
            flexible_cell=True,
            max_wip_units=max_wip,
            track_changeover=track_changeover,
        )
        collector = MetricsCollector()
        runtime.events.subscribe(None, collector.handle_event)
        plans = tuple(
            build_inline_plan(
                preset=preset,
                order_id=f"{group_id}_{variant_id}_{index:02d}_{preset}",
                quantity=1,
                priority=10,
            )
            for index, preset in enumerate(presets, start=1)
        )
        runtime.submit_plans(plans, now=0.0)
        started = time.perf_counter()
        now = 0.0
        for _ in range(100_000):
            runtime.tick(now)
            if runtime.terminal:
                break
            now += self.step_seconds
        wall = time.perf_counter() - started
        if not runtime.terminal:
            raise TimeoutError(f"{group_id}/{variant_id} did not terminate")
        metrics = collector.calculate(runtime, now)
        completed = int(metrics["completed_units"])
        makespan = float(metrics["makespan"])
        for event in runtime.events.history:
            self._events.append({"group_id": group_id, "variant_id": variant_id, **event.as_dict()})
        return GoldenRun(
            group_id=group_id,
            variant_id=variant_id,
            label_zh=label_zh,
            complete=runtime.terminal and int(metrics["task_failed"]) == 0,
            makespan_s=makespan,
            throughput_units_per_hour=(0.0 if makespan <= 0 else 3600.0 * completed / makespan),
            average_robot_utilization=float(metrics["average_robot_utilization"]),
            completed_units=completed,
            recovery_rate=float(metrics["recovery_rate"]),
            parallel_install_s=float(runtime.snapshot(now).get("parallel_arm_seconds", 0.0)),
            changeover_s=float(metrics.get("changeover_seconds", 0.0)),
            wall_seconds=wall,
        )

    def _dual_line_run(
        self,
        *,
        group_id: str,
        variant_id: str,
        label_zh: str,
        presets: Iterable[str],
        install_resource: str | None = None,
        fault_type: str | None = None,
    ) -> GoldenRun:
        runtime = DualLineRuntime(fast=False)
        if install_resource is not None:
            runtime.set_execution_gate(_InstallResourceGate(install_resource))
        for index, preset in enumerate(presets, start=1):
            runtime.submit_order(
                preset,
                order_id=f"{group_id}_{variant_id}_{index:02d}_{preset}",
                priority=10,
            )
        if fault_type is not None:
            runtime.inject_fault(fault_type, target="fin_02")
        started = time.perf_counter()
        for _ in range(100_000):
            runtime.tick(self.step_seconds)
            if runtime.complete:
                break
        wall = time.perf_counter() - started
        if not runtime.complete:
            raise TimeoutError(f"{group_id}/{variant_id} did not terminate")
        finish_times = [
            float(unit.completed_at) for unit in runtime.units.values() if unit.completed_at is not None
        ]
        makespan = max(finish_times, default=float(runtime.sim_time))
        busy: dict[str, float] = {resource: 0.0 for resource in ("ARM1", "ARM2", "ARM3")}
        active: dict[str, float] = {}
        for event in runtime.events:
            resource = str(event.get("resource", ""))
            if event.get("type") == "OPERATION_STARTED" and resource in busy:
                active[resource] = float(event["time"])
            elif event.get("type") in {"OPERATION_COMPLETED", "OPERATION_CANCELLED"} and resource in busy:
                begin = active.pop(resource, None)
                if begin is not None:
                    busy[resource] += max(0.0, float(event["time"]) - begin)
            self._events.append({"group_id": group_id, "variant_id": variant_id, **event})
        completed = sum(unit.completed_at is not None for unit in runtime.units.values())
        fault_count = len(runtime.faults.faults)
        recovered = sum(record.recovered for record in runtime.faults.faults.values())
        return GoldenRun(
            group_id=group_id,
            variant_id=variant_id,
            label_zh=label_zh,
            complete=runtime.complete,
            makespan_s=makespan,
            throughput_units_per_hour=(0.0 if makespan <= 0 else 3600.0 * completed / makespan),
            average_robot_utilization=(0.0 if makespan <= 0 else sum(busy.values()) / (3.0 * makespan)),
            completed_units=completed,
            recovery_rate=(0.0 if fault_count == 0 else recovered / fault_count),
            parallel_install_s=float(runtime.scheduled_parallel_install_seconds),
            changeover_s=0.0,
            wall_seconds=wall,
        )

    def _physical_v2_run(
        self,
        *,
        group_id: str,
        variant_id: str,
        label_zh: str,
        presets: Iterable[str],
    ) -> GoldenRun:
        """Run the MuJoCo-backed V2 actor so changeover time is measured."""

        from ..dual_line.application import V2BrazingApplication
        from ..dual_line.cli import parse_args

        args = parse_args(("--headless", "--fast", "--no-ui", "--max-sim-time", "260"))
        application = V2BrazingApplication(args)
        started = time.perf_counter()
        try:
            values = tuple(presets)
            for index, preset in enumerate(values, start=1):
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
            for event in application.runtime.events:
                self._events.append({"group_id": group_id, "variant_id": variant_id, **event})
            experiment = state.get("experiment_metrics", {})
            completed = len(state.get("completed_orders", ()))
            return GoldenRun(
                group_id=group_id,
                variant_id=variant_id,
                label_zh=label_zh,
                complete=bool(state.get("physical_execution_complete")),
                makespan_s=makespan,
                throughput_units_per_hour=(0.0 if makespan <= 0 else 3600.0 * completed / makespan),
                average_robot_utilization=float(experiment.get("average_robot_utilization", 0.0)),
                completed_units=completed,
                recovery_rate=float(experiment.get("recovery_rate", 0.0)),
                parallel_install_s=float(state.get("scheduled_parallel_install_seconds", 0.0)),
                changeover_s=0.0,
                wall_seconds=time.perf_counter() - started,
            )
        finally:
            application.close()

    def _groups(self) -> list[dict[str, Any]]:
        fixed = self._manufacturing_run(
            group_id="G1_SCHEDULER",
            variant_id="FIXED",
            label_zh="固定顺序调度",
            presets=("A", "B", "C"),
            scheduler="fixed",
            max_wip=3,
        )
        dynamic = self._manufacturing_run(
            group_id="G1_SCHEDULER",
            variant_id="DYNAMIC",
            label_zh="动态优先级调度",
            presets=("A", "B", "C"),
            scheduler="dynamic",
            max_wip=3,
        )
        single = self._dual_line_run(
            group_id="G2_INSTALL_RESOURCE",
            variant_id="ARM1_ONLY",
            label_zh="仅 Arm1 安装翅片",
            presets=("A", "B", "C"),
            install_resource="ARM1",
        )
        dual = self._dual_line_run(
            group_id="G2_INSTALL_RESOURCE",
            variant_id="ARM1_ARM3",
            label_zh="Arm1 + Arm3 双安装支路",
            presets=("A", "B", "C"),
        )
        alternating = self._physical_v2_run(
            group_id="G3_ORDER_SEQUENCE",
            variant_id="ARRIVAL_ABA",
            label_zh="按到达顺序 ABA",
            presets=tuple("ABA"),
        )
        grouped = self._physical_v2_run(
            group_id="G3_ORDER_SEQUENCE",
            variant_id="FAMILY_AAB",
            label_zh="同族排序 AAB",
            presets=tuple("AAB"),
        )
        autonomous = self._dual_line_run(
            group_id="G4_RECOVERY_DISPOSITION",
            variant_id="AUTONOMOUS",
            label_zh="翅片偏位自主纠偏复检",
            presets=("A",),
            fault_type="FIN_POSE",
        )
        manual = self._dual_line_run(
            group_id="G4_RECOVERY_DISPOSITION",
            variant_id="MANUAL_10S",
            label_zh="同类几何异常人工处置 10 秒",
            presets=("A",),
            fault_type="FIN_GEOMETRY_FAILED",
        )
        definitions = (
            (
                "G1_SCHEDULER",
                "调度柔性：固定顺序 vs 动态优先级",
                fixed,
                dynamic,
                "订单、释放时刻、产品组合与资源配置相同，仅改变调度算法。",
            ),
            (
                "G2_INSTALL_RESOURCE",
                "资源柔性：单安装资源 vs 双安装资源",
                single,
                dual,
                "A/B/C 与炉批参数相同，仅改变 S3 可用安装资源。",
            ),
            (
                "G3_ORDER_SEQUENCE",
                "订单排序柔性：交错顺序 vs 同族集中",
                alternating,
                grouped,
                "产品数量、种类与配置规则相同，仅改变订单序列；不包含可见实体换型。",
            ),
            (
                "G4_RECOVERY_DISPOSITION",
                "扰动柔性：自主恢复 vs 人工处置",
                autonomous,
                manual,
                "同一 A 型订单、同一 fin_02 几何类异常；比较自主纠偏闭环与 10 秒人工处置。",
            ),
        )
        return [
            {
                "group_id": group_id,
                "title_zh": title,
                "controlled_variable_zh": control,
                "runs": [asdict(first), asdict(second)],
                "comparison": _comparison(first, second),
            }
            for group_id, title, first, second, control in definitions
        ]

    @staticmethod
    def _write_csv(path: Path, groups: list[dict[str, Any]]) -> None:
        rows = [run for group in groups for run in group["runs"]]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _summary(groups: list[dict[str, Any]]) -> str:
        lines = [
            "# 柔性制造四组黄金实验",
            "",
            "> 所有仿真时间均来自实际事件时间戳；墙钟仅用于复现性能，不与仿真节拍混用。",
            "",
            "| 组别 | 基线 | 候选 | 基线 makespan | 候选 makespan | 变化 |",
            "|---|---|---|---:|---:|---:|",
        ]
        for group in groups:
            first, second = group["runs"]
            improvement = group["comparison"]["makespan_improvement_percent"]
            change = "不可计算" if improvement is None else f"{improvement:+.2f}%"
            lines.append(
                f"| {group['title_zh']} | {first['label_zh']} | {second['label_zh']} | "
                f"{first['makespan_s']:.2f}s | {second['makespan_s']:.2f}s | {change} |"
            )
        lines.extend(
            (
                "",
                "## 口径",
                "",
                "- 每组只改变一个主因素；同组订单、种子、时步和停止条件保持一致。",
                "- Makespan 取最终产品单元完成事件，不使用任务预计时长。",
                "- 正百分比表示候选方案缩短 makespan；负值表示候选方案更慢。",
                "- G1 使用纯调度内核；G2/G3/G4 使用 V2 异步实体 actor 状态机。",
            )
        )
        return "\n".join(lines) + "\n"

    def run(self, output_directory: str | Path) -> dict[str, Any]:
        self._events.clear()
        output = Path(output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        groups = self._groups()
        root = Path(__file__).resolve().parents[2]
        result = {
            "schema_version": 1,
            "suite_id": "FLEXIBLE_LINE_GOLDEN_4",
            "seed": self.seed,
            "step_seconds": self.step_seconds,
            "git_commit": _git_commit(root),
            "measurement_basis": {
                "simulation_time": "terminal simulation event timestamps",
                "wall_time": "time.perf_counter around each local run",
                "estimate_substitution_forbidden": True,
            },
            "groups": groups,
        }
        (output / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._write_csv(output / "runs.csv", groups)
        with (output / "events.jsonl").open("w", encoding="utf-8") as stream:
            for event in self._events:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        (output / "suite_config.json").write_text(
            json.dumps(
                {"suite_id": result["suite_id"], "seed": self.seed, "step_seconds": self.step_seconds},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (output / "summary.md").write_text(self._summary(groups), encoding="utf-8")
        return result


__all__ = ["GoldenExperimentSuite", "GoldenRun"]
