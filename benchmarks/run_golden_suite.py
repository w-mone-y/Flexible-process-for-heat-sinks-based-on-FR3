#!/usr/bin/env python3
"""Run and export the four competition-facing golden experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from brazing_sim.experiments import GoldenExperimentSuite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results/2026-08-02-golden-flexibility",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()
    result = GoldenExperimentSuite(seed=args.seed, step_seconds=args.step).run(args.output_dir)
    print(Path(args.output_dir).resolve())
    print(f"groups={len(result['groups'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
