"""Simulation-clock-driven demonstration brazing furnace."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

from .domain import BrazingRecipe, FurnacePhase, FurnaceState


class FurnaceInterlockError(RuntimeError):
    """Raised when a door/load/cycle command violates a safety interlock."""


class DemoFurnace:
    """Asynchronous furnace state machine.

    ``update(now)`` advances from a supplied simulation timestamp and never
    sleeps.  ``start(now)`` is a convenience used by headless demos: it opens
    the door, loads, closes, executes the profile and opens for unloading.  The
    explicit door/load methods are available for actor-driven simulations and
    exercise the same interlocks.
    """

    VALID_FAULTS = {None, "recoverable", "severe"}

    def __init__(self, recipe: BrazingRecipe | None = None) -> None:
        self.recipe = recipe or BrazingRecipe()
        self.state = FurnaceState(
            temperature_c=self.recipe.ambient_c,
            target_temperature_c=self.recipe.ambient_c,
        )
        self._automatic_loading = False
        self._opening_for_unload = False
        self._pending_fault: str | None = None
        self._last_time = 0.0
        self.history: list[tuple[float, float, float, FurnacePhase]] = []

    @property
    def status(self) -> FurnacePhase:
        return self.state.phase

    @property
    def temperature(self) -> float:
        return self.state.temperature_c

    @property
    def complete(self) -> bool:
        return self.state.complete

    @property
    def running(self) -> bool:
        return self.state.phase in {
            FurnacePhase.DOOR_OPENING,
            FurnacePhase.DOOR_CLOSING,
            FurnacePhase.PREHEAT,
            FurnacePhase.RAMP,
            FurnacePhase.SOAK,
            FurnacePhase.COOLING,
        }

    def _validate_now(self, now: float) -> float:
        now = float(now)
        if not isfinite(now):
            raise ValueError("simulation time must be finite")
        if now + 1e-12 < self._last_time:
            raise ValueError("simulation time must be monotonic")
        self._last_time = max(self._last_time, now)
        return now

    def _set_phase(self, phase: FurnacePhase, start: float) -> None:
        self.state.phase = phase
        self.state.phase_started_at = start

    def reset(self, now: float = 0.0) -> FurnaceState:
        now = float(now)
        if not isfinite(now):
            raise ValueError("simulation time must be finite")
        self.state = FurnaceState(
            phase=FurnacePhase.IDLE,
            temperature_c=self.recipe.ambient_c,
            target_temperature_c=self.recipe.ambient_c,
            phase_started_at=now,
        )
        self._automatic_loading = False
        self._opening_for_unload = False
        self._pending_fault = None
        self._last_time = now
        self.history.clear()
        return self.state

    def start(self, now: float = 0.0, fault: str | None = None) -> FurnaceState:
        """Start the complete non-blocking demo sequence."""

        now = self._validate_now(now)
        if fault not in self.VALID_FAULTS:
            raise ValueError(f"unsupported furnace fault: {fault!r}")
        if self.state.phase not in {FurnacePhase.IDLE, FurnacePhase.COMPLETE}:
            raise FurnaceInterlockError(f"cannot start furnace while {self.state.phase.value}")
        if self.state.temperature_c > self.recipe.unload_c + 1e-9:
            raise FurnaceInterlockError("furnace is too hot to open for loading")
        if self.state.phase is FurnacePhase.COMPLETE:
            self.reset(now)
        self._automatic_loading = True
        self._opening_for_unload = False
        self._pending_fault = fault
        self.state.profile_fault = fault
        self.state.profile_score = 1.0
        self.state.severe_violation = False
        self.state.completed_at = None
        self._set_phase(FurnacePhase.DOOR_OPENING, now)
        return self.state

    def request_open(self, now: float) -> FurnaceState:
        now = self._validate_now(now)
        if self.state.phase not in {FurnacePhase.IDLE, FurnacePhase.READY, FurnacePhase.COMPLETE}:
            raise FurnaceInterlockError(f"door cannot open while {self.state.phase.value}")
        if self.state.temperature_c > self.recipe.unload_c + 1e-9:
            raise FurnaceInterlockError("door is interlocked above the unload temperature")
        if self.state.door_open:
            self._set_phase(FurnacePhase.LOADING, now)
            return self.state
        self._automatic_loading = False
        self._opening_for_unload = False
        self._set_phase(FurnacePhase.DOOR_OPENING, now)
        return self.state

    open_door = request_open

    def load_workpiece(self, now: float) -> FurnaceState:
        now = self._validate_now(now)
        self.update(now)
        if self.state.phase is not FurnacePhase.LOADING or not self.state.door_open:
            raise FurnaceInterlockError("workpiece loading requires a fully open door")
        if self.state.workpiece_loaded:
            raise FurnaceInterlockError("a workpiece is already loaded")
        self.state.workpiece_loaded = True
        return self.state

    load = load_workpiece

    def request_close(self, now: float) -> FurnaceState:
        now = self._validate_now(now)
        self.update(now)
        if self.state.phase is not FurnacePhase.LOADING or not self.state.door_open:
            raise FurnaceInterlockError("door closing requires the loading phase")
        if not self.state.workpiece_loaded:
            raise FurnaceInterlockError("cannot close for a cycle without a workpiece")
        self._set_phase(FurnacePhase.DOOR_CLOSING, now)
        return self.state

    close_door = request_close

    def start_cycle(self, now: float, fault: str | None = None) -> FurnaceState:
        now = self._validate_now(now)
        self.update(now)
        if fault not in self.VALID_FAULTS:
            raise ValueError(f"unsupported furnace fault: {fault!r}")
        if self.state.phase is not FurnacePhase.READY:
            raise FurnaceInterlockError("cycle start requires READY state")
        if not self.state.door_closed or not self.state.workpiece_loaded:
            raise FurnaceInterlockError("cycle start requires a closed door and loaded workpiece")
        self._pending_fault = fault
        self._begin_profile(now)
        return self.state

    def _begin_profile(self, now: float) -> None:
        fault = self._pending_fault
        self.state.profile_fault = fault
        self.state.profile_score = {None: 1.0, "recoverable": 0.82, "severe": 0.20}[fault]
        self.state.severe_violation = fault == "severe"
        self.state.cycle_started_at = now
        self.state.elapsed_seconds = 0.0
        self.state.peak_temperature_c = self.state.temperature_c
        self._set_phase(FurnacePhase.PREHEAT, now)

    @staticmethod
    def _lerp(start: float, end: float, fraction: float) -> float:
        fraction = min(1.0, max(0.0, fraction))
        return start + (end - start) * fraction

    def _faulted_temperature(self, target: float) -> float:
        if target <= self.recipe.preheat_c:
            return target
        fraction = min(
            1.0,
            max(0.0, (target - self.recipe.preheat_c) / (self.recipe.peak_c - self.recipe.preheat_c)),
        )
        if self.state.profile_fault == "recoverable":
            return target - 40.0 * fraction
        if self.state.profile_fault == "severe":
            return target + 180.0 * fraction
        return target

    def _update_temperature(self, now: float) -> None:
        elapsed = max(0.0, now - self.state.phase_started_at)
        phase = self.state.phase
        if phase is FurnacePhase.PREHEAT:
            fraction = elapsed / self.recipe.preheat_seconds
            target = self._lerp(self.recipe.ambient_c, self.recipe.preheat_c, fraction)
        elif phase is FurnacePhase.RAMP:
            fraction = elapsed / self.recipe.ramp_seconds
            target = self._lerp(self.recipe.preheat_c, self.recipe.peak_c, fraction)
        elif phase is FurnacePhase.SOAK:
            target = self.recipe.peak_c
        elif phase is FurnacePhase.COOLING:
            fraction = elapsed / self.recipe.cooling_seconds
            target = self._lerp(self.recipe.peak_c, self.recipe.unload_c, fraction)
        else:
            return
        self.state.target_temperature_c = target
        self.state.temperature_c = self._faulted_temperature(target)
        self.state.peak_temperature_c = max(self.state.peak_temperature_c, self.state.temperature_c)

    def update(self, now: float) -> FurnaceState:
        """Advance to ``now`` and return the same mutable state object."""

        now = self._validate_now(now)
        # Advance across any fully elapsed phases.  Exact phase boundaries are
        # retained so a large fake-clock jump produces the same result as many
        # small simulation steps.
        for _ in range(16):
            phase = self.state.phase
            elapsed = max(0.0, now - self.state.phase_started_at)
            boundary: float | None = None

            if phase is FurnacePhase.DOOR_OPENING:
                self.state.door_fraction = min(1.0, elapsed / self.recipe.door_seconds)
                if elapsed >= self.recipe.door_seconds:
                    boundary = self.state.phase_started_at + self.recipe.door_seconds
                    self.state.door_fraction = 1.0
                    if self._opening_for_unload:
                        self.state.workpiece_loaded = False
                        self.state.completed_at = boundary
                        self._set_phase(FurnacePhase.COMPLETE, boundary)
                    elif self._automatic_loading:
                        self.state.workpiece_loaded = True
                        self._set_phase(FurnacePhase.DOOR_CLOSING, boundary)
                    else:
                        self._set_phase(FurnacePhase.LOADING, boundary)
            elif phase is FurnacePhase.DOOR_CLOSING:
                self.state.door_fraction = max(0.0, 1.0 - elapsed / self.recipe.door_seconds)
                if elapsed >= self.recipe.door_seconds:
                    boundary = self.state.phase_started_at + self.recipe.door_seconds
                    self.state.door_fraction = 0.0
                    self._set_phase(FurnacePhase.READY, boundary)
                    if self._automatic_loading:
                        self._begin_profile(boundary)
            elif phase is FurnacePhase.PREHEAT:
                self._update_temperature(now)
                if elapsed >= self.recipe.preheat_seconds:
                    boundary = self.state.phase_started_at + self.recipe.preheat_seconds
                    self.state.target_temperature_c = self.recipe.preheat_c
                    self.state.temperature_c = self._faulted_temperature(self.recipe.preheat_c)
                    self._set_phase(FurnacePhase.RAMP, boundary)
            elif phase is FurnacePhase.RAMP:
                self._update_temperature(now)
                if elapsed >= self.recipe.ramp_seconds:
                    boundary = self.state.phase_started_at + self.recipe.ramp_seconds
                    self.state.target_temperature_c = self.recipe.peak_c
                    self.state.temperature_c = self._faulted_temperature(self.recipe.peak_c)
                    self.state.peak_temperature_c = max(
                        self.state.peak_temperature_c, self.state.temperature_c
                    )
                    self._set_phase(FurnacePhase.SOAK, boundary)
            elif phase is FurnacePhase.SOAK:
                self._update_temperature(now)
                if elapsed >= self.recipe.soak_seconds:
                    boundary = self.state.phase_started_at + self.recipe.soak_seconds
                    self._set_phase(FurnacePhase.COOLING, boundary)
            elif phase is FurnacePhase.COOLING:
                self._update_temperature(now)
                if elapsed >= self.recipe.cooling_seconds:
                    boundary = self.state.phase_started_at + self.recipe.cooling_seconds
                    self.state.target_temperature_c = self.recipe.unload_c
                    self.state.temperature_c = self.recipe.unload_c
                    self._opening_for_unload = True
                    self._set_phase(FurnacePhase.DOOR_OPENING, boundary)
            else:
                break

            if boundary is None or boundary > now:
                break

        if self.state.cycle_started_at is not None:
            profile_end = self.state.cycle_started_at + self.recipe.process_seconds
            self.state.elapsed_seconds = max(0.0, min(now, profile_end) - self.state.cycle_started_at)
        self.history.append(
            (
                now,
                self.state.temperature_c,
                self.state.target_temperature_c,
                self.state.phase,
            )
        )
        return self.state

    def stop(self, now: float) -> FurnaceState:
        now = self._validate_now(now)
        self.state.error = "stopped"
        self._set_phase(FurnacePhase.STOPPED, now)
        return self.state

    def fail(self, now: float, message: str) -> FurnaceState:
        now = self._validate_now(now)
        self.state.error = message
        self._set_phase(FurnacePhase.ERROR, now)
        return self.state

    def snapshot(self) -> FurnaceState:
        return replace(self.state)


FurnaceController = DemoFurnace
