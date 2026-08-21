#!/usr/bin/env python3
"""Run reproducible V2/V1/early-V1 order-efficiency comparisons."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import json
import platform
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class VersionSpec:
    version: str
    root: Path
    entrypoint: str


@dataclass
class Measurement:
    case: str
    version: str
    supported: bool
    completed: bool
    exit_code: int | None
    simulation_seconds: float | None
    wall_seconds: float | None
    units: int
    throughput_per_sim_hour: float | None
    parallel_install_seconds: float | None
    error: str
    command: list[str]


def _parse_snapshot(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    if start < 0:
        return {}
    return json.loads(stdout[start:])


def _simulation_seconds(snapshot: dict[str, Any], case: str, version: str) -> float | None:
    if version == "v2":
        value = snapshot.get("sim_time")
    elif case == "three_a":
        value = snapshot.get("batch", {}).get("elapsed_seconds")
    elif case in {"mixed_abc", "six_abcabc"}:
        completed = [
            order.get("completed_at")
            for order in snapshot.get("orders", [])
            if order.get("status") == "COMPLETED"
        ]
        value = max(completed) if completed else None
    else:
        value = snapshot.get("kpi", {}).get("order_elapsed")
    return None if value is None else float(value)


def _is_complete(snapshot: dict[str, Any], case: str, version: str, units: int) -> bool:
    if snapshot.get("last_error"):
        return False
    if version == "v2":
        return bool(snapshot.get("complete")) and len(snapshot.get("completed_orders", [])) == units
    if case == "three_a":
        batch = snapshot.get("batch", {})
        return batch.get("stage") == "COMPLETE" and batch.get("completed_units") == units
    if case in {"mixed_abc", "six_abcabc"}:
        orders = snapshot.get("orders", [])
        # V1 reports one aggregate row per submitted order file, while the
        # benchmark ``units`` count includes each row's quantity.
        return bool(orders) and all(order.get("status") == "COMPLETED" for order in orders)
    return snapshot.get("stage") in {"PASS", "COMPLETE"}


def _command(
    spec: VersionSpec,
    case: str,
    profile: str,
    queue_path: Path | None,
) -> tuple[list[str], int] | None:
    fast = ["--fast"] if profile == "fast" else []
    limit = ["--max-sim-time", "2000"]
    if spec.version == "v2":
        orders = {
            "single_a": "A",
            "three_a": "A,A,A",
            "mixed_abc": "A,B,C",
            "six_abcabc": "A,B,C,A,B,C",
        }[case]
        return [
            "python",
            spec.entrypoint,
            "--headless",
            "--orders",
            orders,
            *fast,
            *limit,
        ], {
            "single_a": 1,
            "three_a": 3,
            "mixed_abc": 3,
            "six_abcabc": 6,
        }[case]
    if spec.version == "v1-early" and case != "single_a":
        return None
    if case == "single_a":
        return [
            "python",
            spec.entrypoint,
            "--headless",
            "--order",
            "A",
            *fast,
            *limit,
        ], 1
    if case == "three_a":
        return [
            "python",
            spec.entrypoint,
            "--headless",
            "--batch",
            "A",
            *fast,
            *limit,
        ], 3
    assert queue_path is not None
    return [
        "python",
        spec.entrypoint,
        "--headless",
        "--orders-file",
        str(queue_path),
        *fast,
        *limit,
    ], (6 if case == "six_abcabc" else 3)


def _write_v1_mixed_queue(root: Path, target: Path, *, quantity: int = 1) -> None:
    order_files = [
        root / "config/orders/order_001.yaml",
        root / "config/orders/order_002.yaml",
        root / "config/orders/order_003.yaml",
    ]
    payload = {
        "schema_version": 1,
        "orders": [
            {
                "order_file": str(order_file),
                "quantity": quantity,
                "priority": 5,
            }
            for order_file in order_files
        ],
    }
    target.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _run_case(
    spec: VersionSpec,
    case: str,
    profile: str,
    runs: int,
    temp_dir: Path,
) -> Measurement:
    queue_path = temp_dir / f"{spec.version}-abc.yaml"
    if spec.version == "v1" and case in {"mixed_abc", "six_abcabc"}:
        _write_v1_mixed_queue(spec.root, queue_path, quantity=2 if case == "six_abcabc" else 1)
    planned = _command(spec, case, profile, queue_path)
    if planned is None:
        return Measurement(
            case=case,
            version=spec.version,
            supported=False,
            completed=False,
            exit_code=None,
            simulation_seconds=None,
            wall_seconds=None,
            units=6 if case == "six_abcabc" else 3,
            throughput_per_sim_hour=None,
            parallel_install_seconds=None,
            error="该历史版本不支持此订单模式",
            command=[],
        )

    command, units = planned
    wall_samples: list[float] = []
    snapshots: list[dict[str, Any]] = []
    exit_codes: list[int] = []
    stderr = ""
    for _ in range(runs):
        started = time.perf_counter()
        process = subprocess.run(
            command,
            cwd=spec.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        wall_samples.append(time.perf_counter() - started)
        exit_codes.append(process.returncode)
        stderr = process.stderr.strip()
        snapshots.append(_parse_snapshot(process.stdout))

    snapshot = snapshots[-1]
    simulation_seconds = _simulation_seconds(snapshot, case, spec.version)
    completed = all(
        code == 0 and _is_complete(candidate, case, spec.version, units)
        for code, candidate in zip(exit_codes, snapshots)
    )
    throughput = None
    if completed and simulation_seconds and simulation_seconds > 0:
        throughput = units * 3600.0 / simulation_seconds
    return Measurement(
        case=case,
        version=spec.version,
        supported=True,
        completed=completed,
        exit_code=exit_codes[-1],
        simulation_seconds=simulation_seconds,
        wall_seconds=statistics.median(wall_samples),
        units=units,
        throughput_per_sim_hour=throughput,
        parallel_install_seconds=snapshot.get("scheduled_parallel_install_seconds"),
        error=str(snapshot.get("last_error") or stderr),
        command=command,
    )


def _write_outputs(output_dir: Path, profile: str, rows: list[Measurement]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path.cwd()
    revision = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout.strip()
    )
    payload = {
        "schema_version": 1,
        "measured_at": date.today().isoformat(),
        "machine": platform.platform(),
        "code_revision": revision,
        "working_tree_dirty": dirty,
        "profile": profile,
        "measurement_basis": {
            "simulation_seconds": "实际仿真事件完成时间",
            "wall_seconds": "subprocess 端到端 perf_counter 中位数",
            "throughput": "完成件数 / 仿真 makespan",
        },
        "results": [asdict(row) for row in rows],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(asdict(rows[0]).keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    by_key = {(row.case, row.version): row for row in rows}
    lines = [
        "# V2 / V1 / 早期 V1 复现结果",
        "",
        f"运行配置：`{profile}`。仿真时间来自实际完工事件，墙钟为 {profile} 进程实测。",
        "",
        "| 场景 | 版本 | 完成 | 仿真 makespan | 墙钟 | 吞吐 |",
        "|---|---|:---:|---:|---:|---:|",
    ]
    labels = {
        "single_a": "单件 A",
        "three_a": "三件 A",
        "mixed_abc": "A/B/C 各一件",
        "six_abcabc": "A/B/C 各两件",
    }
    for row in rows:
        simulation = "—" if row.simulation_seconds is None else f"{row.simulation_seconds:.2f} s"
        wall = "—" if row.wall_seconds is None else f"{row.wall_seconds:.2f} s"
        throughput = "—" if row.throughput_per_sim_hour is None else f"{row.throughput_per_sim_hour:.2f} 件/h"
        complete = "✅" if row.completed else ("不支持" if not row.supported else "❌")
        lines.append(
            f"| {labels[row.case]} | {row.version} | {complete} | " f"{simulation} | {wall} | {throughput} |"
        )
    lines.extend(["", "## V2 相对正式 V1", ""])
    for case in labels:
        v2 = by_key.get((case, "v2"))
        v1 = by_key.get((case, "v1"))
        if not v2 or not v1 or not v2.completed or not v1.completed:
            continue
        assert v2.simulation_seconds is not None and v1.simulation_seconds is not None
        reduction = (v1.simulation_seconds - v2.simulation_seconds) / v1.simulation_seconds * 100
        lines.append(f"- {labels[case]}：makespan 缩短 **{reduction:.1f}%**。")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, default=Path.cwd())
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument("--v1-early-root", type=Path, required=True)
    parser.add_argument("--profile", choices=("full", "fast"), default="full")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cases",
        default="single_a,three_a,mixed_abc,six_abcabc",
        help="comma-separated subset of single_a,three_a,mixed_abc,six_abcabc",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    cases = [item.strip() for item in args.cases.split(",") if item.strip()]
    unknown = sorted(set(cases) - {"single_a", "three_a", "mixed_abc", "six_abcabc"})
    if unknown:
        parser.error(f"unknown cases: {', '.join(unknown)}")

    specs = [
        VersionSpec("v2", args.v2_root.resolve(), "brazing_line_v2.py"),
        VersionSpec("v1", args.v1_root.resolve(), "brazing_line.py"),
        VersionSpec("v1-early", args.v1_early_root.resolve(), "brazing_line.py"),
    ]
    rows: list[Measurement] = []
    with tempfile.TemporaryDirectory(prefix="fr3-benchmark-") as temp:
        temp_dir = Path(temp)
        for case in cases:
            for spec in specs:
                print(f"[benchmark] {case} / {spec.version}", flush=True)
                rows.append(_run_case(spec, case, args.profile, args.runs, temp_dir))
    _write_outputs(args.output_dir, args.profile, rows)
    print(f"[benchmark] wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
