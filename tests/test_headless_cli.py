from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_case(fault: str | None = None) -> dict:
    command = [
        sys.executable,
        str(ROOT / "brazing_line.py"),
        "--headless",
        "--order",
        "A",
        "--fast",
        "--port",
        "0",
        "--no-terminal-commands",
        "--max-sim-time",
        "40",
    ]
    if fault:
        command.extend(("--fault", fault))
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    start = completed.stdout.find("{\n")
    assert start >= 0, completed.stdout
    return json.loads(completed.stdout[start:])


def run_batch_case() -> dict:
    command = [
        sys.executable,
        str(ROOT / "brazing_line.py"),
        "--headless",
        "--batch",
        "A",
        "--fast",
        "--port",
        "0",
        "--no-terminal-commands",
        "--max-sim-time",
        "60",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    start = completed.stdout.find("{\n")
    assert start >= 0, completed.stdout
    return json.loads(completed.stdout[start:])


@pytest.mark.parametrize(
    ("fault", "stage", "rework"),
    [
        (None, "PASS", None),
        ("fin_pose:fin_02", "PASS", "fin"),
        ("brazing_gap:slot_02_left", "PASS", "material"),
        ("furnace_profile:recoverable", "REWORK_REQUIRED", None),
        ("furnace_profile:severe", "SCRAPPED", None),
    ],
)
def test_headless_fault_matrix(fault: str | None, stage: str, rework: str | None) -> None:
    snapshot = run_case(fault)
    assert snapshot["stage"] == stage
    if rework:
        assert snapshot["kpi"]["rework_counts"][rework] >= 1


def test_headless_three_layer_batch() -> None:
    snapshot = run_batch_case()
    assert snapshot["stage"] == "COMPLETE"
    assert snapshot["disposition"] == "PASS"
    assert [unit["product_stage"] for unit in snapshot["batch"]["units"]] == [
        "PASS",
        "PASS",
        "PASS",
    ]
    assert [shelf["state"] for shelf in snapshot["rack"]["shelves"]] == [
        "UNLOADED",
        "UNLOADED",
        "UNLOADED",
    ]
