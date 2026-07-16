"""MuJoCo-truth inspection gates and final quality classification."""

from __future__ import annotations

from math import dist
from statistics import fmean
from typing import Iterable

from .domain import (
    BrazingPathState,
    FinState,
    InspectionConfig,
    InspectionKind,
    InspectionResult,
    OrderStage,
    ProductState,
    TerminalDisposition,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class QualityEvaluator:
    """Evaluate geometry, material and completed-joint quality.

    Images are presentation-only in the MVP; all metrics consumed here are
    populated from simulation truth or injected deterministic faults.
    """

    COVERAGE_WEIGHT = 0.35
    GEOMETRY_WEIGHT = 0.25
    TEMPERATURE_WEIGHT = 0.25
    FIXTURE_WEIGHT = 0.15

    def __init__(self, config: InspectionConfig | None = None) -> None:
        self.config = config

    def _config(self, product: ProductState) -> InspectionConfig:
        return self.config or product.spec.inspection

    @staticmethod
    def _position_error(fin: FinState) -> float:
        measured = dist(fin.actual_position, fin.target_position)
        return max(fin.position_error_m, measured)

    def _geometry(
        self, fins: Iterable[FinState], config: InspectionConfig
    ) -> tuple[dict[str, float], tuple[str, ...], tuple[str, ...], float]:
        active = tuple(fin for fin in fins if fin.active)
        if not active:
            return {}, ("fins.missing",), (), 0.0
        failures: list[str] = []
        targets: list[str] = []
        position_errors: list[float] = []
        verticality_errors: list[float] = []
        root_gaps: list[float] = []
        pitch_errors: list[float] = []
        per_fin_scores: list[float] = []

        for fin in active:
            position = self._position_error(fin)
            position_errors.append(position)
            verticality_errors.append(abs(fin.verticality_error_deg))
            root_gaps.append(max(0.0, fin.root_gap_m))
            pitch_errors.append(abs(fin.pitch_error_m))
            ratios = (
                position / config.fin_position_m,
                abs(fin.verticality_error_deg) / config.fin_verticality_deg,
                max(0.0, fin.root_gap_m) / config.root_gap_m,
                abs(fin.pitch_error_m) / config.pitch_error_m,
            )
            per_fin_scores.append(_clamp01(1.0 - 0.25 * max(ratios)))
            fin_failed = False
            if position > config.fin_position_m:
                failures.append(f"{fin.fin_id}.position")
                fin_failed = True
            if abs(fin.verticality_error_deg) > config.fin_verticality_deg:
                failures.append(f"{fin.fin_id}.verticality")
                fin_failed = True
            if fin.root_gap_m > config.root_gap_m:
                failures.append(f"{fin.fin_id}.root_gap")
                fin_failed = True
            if abs(fin.pitch_error_m) > config.pitch_error_m:
                failures.append(f"{fin.fin_id}.pitch")
                fin_failed = True
            if fin_failed:
                targets.append(fin.fin_id)

        metrics = {
            "max_fin_position_error_m": max(position_errors),
            "max_fin_verticality_error_deg": max(verticality_errors),
            "max_root_gap_m": max(root_gaps),
            "max_pitch_error_m": max(pitch_errors),
            "geometry_score": fmean(per_fin_scores),
        }
        return metrics, tuple(failures), tuple(dict.fromkeys(targets)), fmean(per_fin_scores)

    def pre_inspection(self, product: ProductState, now: float = 0.0) -> InspectionResult:
        config = self._config(product)
        metrics, failures, targets, score = self._geometry(product.active_fins, config)
        result = InspectionResult(
            kind=InspectionKind.PRE_BRAZE,
            passed=not failures,
            metrics=metrics,
            hard_failures=failures,
            rework_targets=targets,
            score=score,
            timestamp=now,
        )
        product.add_inspection(result)
        return result

    pre = pre_inspection

    def _material(
        self, paths: Iterable[BrazingPathState], config: InspectionConfig
    ) -> tuple[dict[str, float], tuple[str, ...], tuple[str, ...], float]:
        active = tuple(path for path in paths if path.active)
        if not active:
            return {}, ("paths.missing",), (), 0.0
        failures: list[str] = []
        targets: list[str] = []
        path_scores: list[float] = []
        for path in active:
            failed = False
            if not path.applied:
                failures.append(f"{path.path_id}.not_applied")
                failed = True
            if path.coverage_ratio < config.coverage_ratio:
                failures.append(f"{path.path_id}.coverage")
                failed = True
            if path.longest_gap_m > config.longest_material_gap_m:
                failures.append(f"{path.path_id}.gap")
                failed = True
            if abs(path.lateral_error_m) > config.lateral_error_m:
                failures.append(f"{path.path_id}.lateral")
                failed = True
            if path.trajectory_rmse_m > config.trajectory_rmse_m:
                failures.append(f"{path.path_id}.trajectory_rmse")
                failed = True
            if path.trajectory_max_error_m > config.trajectory_max_error_m:
                failures.append(f"{path.path_id}.trajectory_max")
                failed = True
            if failed:
                targets.append(path.path_id)
            error_factor = max(
                abs(path.lateral_error_m) / config.lateral_error_m,
                path.trajectory_rmse_m / config.trajectory_rmse_m,
                path.trajectory_max_error_m / config.trajectory_max_error_m,
                path.longest_gap_m / config.longest_material_gap_m,
            )
            score = min(_clamp01(path.coverage_ratio), _clamp01(1.0 - 0.20 * error_factor))
            path_scores.append(score if path.applied else 0.0)

        metrics = {
            "minimum_coverage_ratio": min(path.coverage_ratio for path in active),
            "mean_coverage_ratio": fmean(path.coverage_ratio for path in active),
            "longest_material_gap_m": max(path.longest_gap_m for path in active),
            "max_lateral_error_m": max(abs(path.lateral_error_m) for path in active),
            "trajectory_rmse_m": max(path.trajectory_rmse_m for path in active),
            "trajectory_max_error_m": max(path.trajectory_max_error_m for path in active),
            "coverage_score": fmean(path_scores),
        }
        return metrics, tuple(failures), tuple(dict.fromkeys(targets)), fmean(path_scores)

    def material_inspection(self, product: ProductState, now: float = 0.0) -> InspectionResult:
        config = self._config(product)
        metrics, failures, targets, score = self._material(product.active_paths, config)
        result = InspectionResult(
            kind=InspectionKind.MATERIAL,
            passed=not failures,
            metrics=metrics,
            hard_failures=failures,
            rework_targets=targets,
            score=score,
            timestamp=now,
        )
        product.add_inspection(result)
        product.kpi.path_rmse_m = float(metrics.get("trajectory_rmse_m", 0.0))
        product.kpi.path_max_error_m = float(metrics.get("trajectory_max_error_m", 0.0))
        return result

    material = material_inspection

    def post_inspection(self, product: ProductState, now: float = 0.0) -> InspectionResult:
        config = self._config(product)
        geometry_metrics, geometry_failures, _, geometry_score = self._geometry(product.active_fins, config)
        material_metrics, material_failures, _, coverage_score = self._material(product.active_paths, config)
        furnace = product.furnace
        temperature_score = _clamp01(furnace.profile_score)
        fixture_score = 1.0 if product.fixture.cycle_locked else 0.0
        total = _clamp01(
            self.COVERAGE_WEIGHT * coverage_score
            + self.GEOMETRY_WEIGHT * geometry_score
            + self.TEMPERATURE_WEIGHT * temperature_score
            + self.FIXTURE_WEIGHT * fixture_score
        )

        failures = list(geometry_failures) + list(material_failures)
        if not product.fixture.cycle_locked:
            failures.append("fixture.not_locked_during_cycle")
        if not furnace.complete:
            failures.append("furnace.incomplete")
        if furnace.profile_fault == "recoverable":
            failures.append("furnace.profile_recoverable")
        if furnace.severe_violation or furnace.profile_fault == "severe":
            failures.append("furnace.profile_severe")

        if furnace.severe_violation or furnace.profile_fault == "severe":
            disposition = TerminalDisposition.SCRAPPED
        elif total < config.rework_score:
            disposition = TerminalDisposition.SCRAPPED
        elif failures or total < config.pass_score:
            disposition = TerminalDisposition.REWORK_REQUIRED
        else:
            disposition = TerminalDisposition.PASS

        metrics: dict[str, float | int | bool | str] = {
            **geometry_metrics,
            **material_metrics,
            "temperature_score": temperature_score,
            "fixture_score": fixture_score,
            "quality_score": total,
            "peak_temperature_c": furnace.peak_temperature_c,
        }
        result = InspectionResult(
            kind=InspectionKind.POST_BRAZE,
            passed=disposition is TerminalDisposition.PASS,
            metrics=metrics,
            hard_failures=tuple(failures),
            score=total,
            disposition=disposition,
            timestamp=now,
        )
        product.add_inspection(result)
        product.kpi.final_quality_score = total
        product.disposition = disposition
        if product.stage is OrderStage.POST_INSPECTION:
            product.transition(OrderStage(disposition.value), now)
        return result

    post = post_inspection

    def register_automatic_rework(self, product: ProductState, result: InspectionResult) -> bool:
        """Reserve one rework attempt for every failed target.

        Returns ``False`` and moves the product to ``MANUAL_REVIEW`` as soon as
        any target has exhausted the two-attempt MVP limit.
        """

        for target in result.rework_targets:
            if not product.record_rework(target):
                return False
        return True
