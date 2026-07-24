"""Physical stage router for the shallow-U asynchronous pallet line."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .profiles import quintic_time_scaling

STAGE_STATIONS = {
    "BASE_LOADING": "s1",
    "MATERIAL_APPLICATION": "s2a",
    "MATERIAL_INSPECTION": "s2b",
    "COMB_CONFIGURATION": "s3",
    "FIN_ASSEMBLY": "s3",
    "PRE_INSPECTION": "s3",
    "FIXTURE_PRESSING": "s3",
    "FIXTURE_LOCKING": "s3",
    "FURNACE_LOADING": "rack_infeed",
}

TRANSFER_EDGES = {
    ("s1", "s2a"): "s1_s2a",
    ("s2a", "s2b"): "s2a_s2b",
    ("s2b", "s3"): "s2b_s3",
    ("s3", "rack_infeed"): "s3_rack",
}

STATION_ORDER = ("s1", "s2a", "s2b", "s3", "rack_infeed")


@dataclass(slots=True)
class AsyncRouteSnapshot:
    enabled: bool
    station: str | None
    target_station: str | None
    active_transfer: str | None
    progress: float
    product_token: str | None
    blocker: str

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "layout": "SHALLOW_U",
            "station": self.station,
            "target_station": self.target_station,
            "active_transfer": self.active_transfer,
            "progress": self.progress,
            "product_token": self.product_token,
            "blocker": self.blocker,
        }


class AsyncLineProcessRouter:
    """Move the reusable physical product through independent station slides.

    A stage is released only after the measured slide position and velocity
    satisfy the arrival tolerance.  Robot trajectories are therefore created
    from the destination station's live product frame, never from a stale
    central Table2 constant.
    """

    def __init__(
        self,
        scene: Any,
        *,
        duration_s: float = 1.6,
        settle_s: float = 0.20,
        fast: bool = False,
    ) -> None:
        self.scene = scene
        self.duration_s = max(0.4, float(duration_s))
        self.settle_s = max(0.0, float(settle_s))
        self.fast = bool(fast)
        self.enabled = False
        self.station: str | None = None
        self.target_station: str | None = None
        self.active_transfer: str | None = None
        self.product_token: str | None = None
        self.started_at: float | None = None
        self.settled_at: float | None = None
        self.progress = 0.0
        self.blocker = ""
        self._returning: set[str] = set()

    @staticmethod
    def station_for_stage(stage: Any) -> str | None:
        name = str(getattr(stage, "value", stage)).upper()
        return STAGE_STATIONS.get(name)

    def activate(self) -> None:
        self.enabled = True

    def deactivate(self) -> None:
        self.enabled = False
        self.active_transfer = None
        self.started_at = None
        self.settled_at = None
        self.target_station = None
        self.progress = 0.0
        self.blocker = ""

    def reset(self) -> None:
        self.station = None
        self.target_station = None
        self.active_transfer = None
        self.product_token = None
        self.started_at = None
        self.settled_at = None
        self.progress = 0.0
        self.blocker = ""
        self._returning.clear()
        self.scene.registry.reset_async_transfers(teleport=True)

    def _dock_new_product(self, token: str) -> None:
        self.scene.registry.reset_async_transfers(teleport=True)
        self.scene.registry.dock_assembly_tray_to_station("s1", snap=True)
        self.station = "s1"
        self.target_station = None
        self.active_transfer = None
        self.product_token = str(token)
        self.started_at = None
        self.settled_at = None
        self.progress = 1.0
        self.blocker = ""

    def _next_station(self, desired: str) -> str:
        if self.station is None:
            return "s1"
        current_index = STATION_ORDER.index(self.station)
        desired_index = STATION_ORDER.index(desired)
        if desired_index < current_index:
            # Material inspection is intentionally one station downstream
            # from dispensing.  A detected gap must not send the pallet
            # backwards through the one-way line: Arm2 reaches into S2B and
            # performs the local repair there, after which Arm3 re-inspects
            # the same stationary pallet.
            if self.station == "s2b" and desired == "s2a":
                return self.station
            raise RuntimeError(f"异步单向线禁止托盘逆流：{self.station} -> {desired}")
        return STATION_ORDER[min(current_index + 1, desired_index)]

    def _start_transfer(self, target: str, now: float) -> None:
        assert self.station is not None
        transfer = TRANSFER_EDGES[(self.station, target)]
        registry = self.scene.registry
        if abs(registry.async_transfer_position(transfer)) > 0.0015:
            self.blocker = f"{transfer}空滑台正在回零"
            self._returning.add(transfer)
            return
        source_weld = f"station_{self.station}_assembly_tray_weld"
        carriage = f"transfer_{transfer}_carriage"
        transfer_weld = f"transfer_{transfer}_assembly_tray_weld"
        registry.set_weld(source_weld, False)
        registry.set_weld(
            transfer_weld,
            True,
            recompute=(carriage, "assembly_tray"),
            forward=True,
        )
        self.active_transfer = transfer
        self.target_station = target
        self.started_at = float(now)
        self.settled_at = None
        self.progress = 0.0
        self.blocker = f"托盘正在移载：{self.station.upper()}→{target.upper()}"

    def _update_transfer(self, now: float) -> bool:
        assert self.active_transfer is not None
        assert self.target_station is not None
        assert self.started_at is not None
        registry = self.scene.registry
        elapsed = max(0.0, float(now) - self.started_at)
        linear = min(1.0, elapsed / self.duration_s)
        self.progress = quintic_time_scaling(linear)
        limit = registry.async_transfer_limit(self.active_transfer)
        command = limit if self.fast else limit * self.progress
        registry.set_async_transfer_target(
            self.active_transfer,
            command,
            teleport=self.fast,
        )
        error = abs(registry.async_transfer_position(self.active_transfer) - limit)
        speed = abs(registry.async_transfer_velocity(self.active_transfer))
        if linear < 1.0 or error > 0.0015 or speed > 0.015:
            self.settled_at = None
            self.blocker = f"异步移载中：剩余{error * 1000.0:.1f} mm，" f"速度{speed * 1000.0:.1f} mm/s"
            if elapsed > self.duration_s + 5.0:
                raise RuntimeError(f"托盘移载到位超时：{self.blocker}")
            return False
        if self.settled_at is None:
            self.settled_at = float(now)
            self.blocker = "托盘到位，正在停稳确认"
            return False
        if float(now) - self.settled_at < self.settle_s:
            return False

        transfer = self.active_transfer
        target = self.target_station
        registry.set_weld(f"transfer_{transfer}_assembly_tray_weld", False)
        anchor = f"station_{target}_anchor"
        registry.set_weld(
            f"station_{target}_assembly_tray_weld",
            True,
            recompute=(anchor, "assembly_tray"),
            forward=True,
        )
        self.station = target
        self.target_station = None
        self.active_transfer = None
        self.started_at = None
        self.settled_at = None
        self.progress = 1.0
        self.blocker = ""
        self._returning.add(transfer)
        registry.refresh_assembly_target_pose()
        if target == "rack_infeed":
            # The legacy furnace conveyor now starts at this exact rack-infeed
            # pose.  Recompute the measured relative transform; do not snap.
            registry.handoff_rack_infeed_to_conveyor()
        return True

    def tick(self, now: float) -> None:
        """Return empty slides in the background without moving a product."""

        del now
        for transfer in tuple(self._returning):
            self.scene.registry.set_async_transfer_target(transfer, 0.0, teleport=self.fast)
            position = abs(self.scene.registry.async_transfer_position(transfer))
            speed = abs(self.scene.registry.async_transfer_velocity(transfer))
            if position <= 0.0015 and speed <= 0.015:
                self._returning.discard(transfer)

    def gate(
        self,
        stage: Any,
        now: float,
        *,
        product_token: str,
        safe_to_transfer: bool,
    ) -> bool:
        if not self.enabled:
            return True
        desired = self.station_for_stage(stage)
        if desired is None:
            return True
        if self.product_token != product_token:
            if not safe_to_transfer:
                self.blocker = "新托盘等待三臂退出S1"
                return False
            self._dock_new_product(product_token)
        if self.active_transfer is not None:
            return self._update_transfer(now) and self.station == desired
        if self.station == desired:
            self.scene.registry.refresh_assembly_target_pose()
            self.blocker = ""
            return True
        if not safe_to_transfer:
            self.blocker = "等待机械臂退出移载滑台扫掠区"
            return False
        next_station = self._next_station(desired)
        if next_station == self.station:
            self.scene.registry.refresh_assembly_target_pose()
            self.blocker = "S2B原位局部补涂，无需逆向移载"
            return True
        transfer = TRANSFER_EDGES[(self.station, next_station)]
        if transfer in self._returning:
            self.blocker = f"{transfer}空滑台正在回零"
            return False
        self._start_transfer(next_station, now)
        return False

    def snapshot(self) -> dict[str, object]:
        return AsyncRouteSnapshot(
            enabled=self.enabled,
            station=self.station,
            target_station=self.target_station,
            active_transfer=self.active_transfer,
            progress=self.progress,
            product_token=self.product_token,
            blocker=self.blocker,
        ).as_dict()


__all__ = ["AsyncLineProcessRouter", "AsyncRouteSnapshot"]
