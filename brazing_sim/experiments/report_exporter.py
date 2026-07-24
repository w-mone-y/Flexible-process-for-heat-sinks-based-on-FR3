"""Export complete, auditable experiment artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping

from ..manufacturing_runtime import ManufacturingRuntime


class ExperimentReporter:
    def __init__(self, output_directory: str | Path) -> None:
        self.output_directory = Path(output_directory).expanduser().resolve()

    @staticmethod
    def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        values = [dict(row) for row in rows]
        fields = sorted({key for row in values for key in row})
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in values:
                writer.writerow(
                    {
                        key: (
                            json.dumps(value, ensure_ascii=False)
                            if isinstance(value, (dict, list))
                            else value
                        )
                        for key, value in row.items()
                    }
                )

    @staticmethod
    def _git_commit(root: Path) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            return result.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    def export(
        self,
        runtime: ManufacturingRuntime,
        metrics: Mapping[str, Any],
        *,
        config_files: Iterable[str | Path] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        output = self.output_directory
        output.mkdir(parents=True, exist_ok=False)
        snapshot_dir = output / "config_snapshot"
        snapshot_dir.mkdir()
        for value in config_files:
            source = Path(value).expanduser().resolve()
            if source.is_file():
                shutil.copy2(source, snapshot_dir / source.name)
        with (output / "events.jsonl").open("w", encoding="utf-8") as stream:
            for event in runtime.events.history:
                stream.write(json.dumps(event.as_dict(), ensure_ascii=False) + "\n")
        self._write_csv(output / "tasks.csv", runtime.graph.snapshot())
        self._write_csv(output / "resources.csv", runtime.resources.snapshot().values())
        self._write_csv(output / "orders.csv", (entry.as_dict() for entry in runtime.orders.values()))
        self._write_csv(output / "faults.csv", (fault.as_dict() for fault in runtime.faults.values()))
        (output / "metrics.json").write_text(
            json.dumps(dict(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        info = {
            "scheduler": runtime.scheduler_mode,
            "tick_count": runtime.tick_count,
            "git_commit": self._git_commit(Path(__file__).resolve().parents[2]),
            **dict(metadata or {}),
        }
        (output / "run.log").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = [
            "# 制造调度实验摘要",
            "",
            f"- 调度器：{runtime.scheduler_mode}",
            f"- Makespan：{float(metrics.get('makespan', 0.0)):.3f} s",
            f"- 完成单元：{int(metrics.get('completed_units', 0))}",
            f"- 平均机器人利用率：{100.0 * float(metrics.get('average_robot_utilization', 0.0)):.2f}%",
            f"- 故障恢复率：{100.0 * float(metrics.get('recovery_rate', 0.0)):.2f}%",
        ]
        (output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
        return output


__all__ = ["ExperimentReporter"]
