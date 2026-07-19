"""Three-tray batch coordinator built on the proven single-layer process."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from .batch_transfer import BatchTransferActor
from .config import create_batch_state
from .domain import (
    Actor,
    BatchStage,
    BatchState,
    FurnacePhase,
    OrderStage,
    ProductState,
    RackShelfState,
    TaskSpec,
    TaskStatus,
    TaskType,
    TransferPhase,
    TrayUnitPhase,
)
from .furnace import DemoFurnace
from .process import ActorResult, ProcessCoordinator

if TYPE_CHECKING:
    from .flexible.models import ProcessPlan


class BatchCoordinator:
    """Coordinate three complete A units and one shared furnace cycle."""

    def __init__(
        self,
        scene: object,
        single: ProcessCoordinator,
        *,
        fast: bool = False,
    ) -> None:
        self.scene = scene
        self.single = single
        self.fast = bool(fast)
        self.batch: BatchState | None = None
        self.transfer = BatchTransferActor(scene, lambda: self.batch, fast=fast)
        self.furnace: DemoFurnace | None = None
        self.paused = False
        self.status = "idle"
        self._waiting_for_index = False
        self._unload_cursor = -1
        self._paused_at: float | None = None
        self.transfer_demo = False
        self.inspection_task: TaskSpec | None = None
        self._inspection_unit_index: int | None = None
        self._transfer_resources: tuple[str, ...] = ()
        self._pending_load_index: int | None = None
        self._pending_furnace_fault: str | None = None
        self._prefetch_resources: tuple[str, ...] = ()
        self.process_plan: ProcessPlan | None = None
        self._unload_order: list[int] = []
        self._unload_position = -1

    @property
    def product(self) -> ProductState | None:
        if self.single.product is not None:
            return self.single.product
        if self.batch is None:
            return None
        return self.batch.active_unit.product

    @property
    def active_task(self) -> Any:
        return self.inspection_task or self.single.active_task

    @property
    def faults(self) -> list[Any]:
        return self.single.faults

    @property
    def running(self) -> bool:
        return self.batch is not None and not self.batch.terminal and not self.paused

    @property
    def terminal(self) -> bool:
        return self.batch is not None and self.batch.terminal

    def _start_unit(self, index: int, now: float, *, full_scene_reset: bool) -> None:
        assert self.batch is not None
        unit = self.batch.units[index]
        if self.process_plan is None:
            product = self.single.start_order(
                self.batch.preset,
                now=now,
                order_id=unit.product.order_id,
            )
        else:
            product = self.single.start_process_plan(
                self.process_plan,
                now=now,
                unit_index=index,
            )
        unit.product = product
        unit.phase = TrayUnitPhase.BUILDING
        self.single.pause_after_stage = OrderStage.READY_FOR_TRANSFER
        if full_scene_reset:
            self.scene.reset(product, raw=True)
            self.transfer.reset(show_empty_cache=True)
            self.scene.registry.set_furnace_door(0.0, teleport=True)
        else:
            self.scene.reset_workcell(product)
        self.scene.registry.set_batch_tray_visible(index, carrier=False, payload=False)
        self.status = f"building tray {index + 1}/{len(self.batch.units)}"

    def start_batch(
        self,
        preset: str = "A",
        *,
        layers: int = 3,
        now: float | None = None,
    ) -> BatchState:
        from .flexible.planner import build_preset_plan

        plan = build_preset_plan(preset, quantity=layers)
        return self.start_process_plan(plan, now=now)

    def start_process_plan(
        self,
        plan: ProcessPlan,
        *,
        now: float | None = None,
    ) -> BatchState:
        """Execute all units of one validated flexible order in one furnace cycle."""

        timestamp = self.scene.time if now is None else float(now)
        if self.batch is not None and not self.batch.terminal:
            raise RuntimeError("a batch is already active")
        if self.single.product is not None:
            self.single.reset()
        layer_indices = tuple(assignment.layer_index for assignment in plan.rack_assignments)
        self.batch = create_batch_state(
            plan.execution_spec.preset,
            layers=plan.quantity,
            spec=plan.execution_spec,
            layer_indices=layer_indices,
            batch_id=plan.order.order_id,
            created_at=timestamp,
        )
        self.process_plan = plan
        self.furnace = DemoFurnace(self.batch.units[0].product.spec.recipe)
        self.paused = False
        self._waiting_for_index = False
        self._unload_cursor = -1
        self._paused_at = None
        self.transfer_demo = False
        self.inspection_task = None
        self._inspection_unit_index = None
        self._transfer_resources = ()
        self._pending_load_index = None
        self._pending_furnace_fault = None
        self._prefetch_resources = ()
        self._unload_order = []
        self._unload_position = -1
        self.batch.transition(BatchStage.BUILDING_LAYER, timestamp)
        self._start_unit(0, timestamp, full_scene_reset=True)
        # The furnace is outside every robot work envelope. Pre-open it while
        # the first unit is being assembled so the first rack transfer does
        # not sit idle behind door travel.
        self.furnace.request_open(timestamp)
        self.batch.furnace = self.furnace.state
        self._sync_furnace_door()
        return self.batch

    def start_transfer_demo(self, *, now: float | None = None) -> BatchState:
        """Run one ready tray through lift and bottom-shelf insertion only."""

        timestamp = self.scene.time if now is None else float(now)
        batch = self.start_batch("A", layers=3, now=timestamp)
        product = batch.units[0].product
        product.fixture.base_weld_active = True
        product.fixture.comb_configured = True
        product.fixture.comb_aligned = True
        product.fixture.material_passed = True
        product.fixture.fins_passed = True
        product.fixture.press_force_held = True
        product.fixture.locked = True
        product.fixture.cycle_locked = True
        product.fixture.ready_for_transfer = True
        product.stage = OrderStage.READY_FOR_TRANSFER
        for fin in product.active_fins:
            fin.inserted = True
            fin.temporary_welded = True
        for path in product.active_paths:
            path.applied = True
            path.coverage_ratio = 1.0

        # This standalone segment begins after all upstream work has already
        # completed. Synchronize the visible Table2 workcell with those domain
        # prerequisites immediately; otherwise the command looks idle while
        # the coordinator is correctly waiting for the furnace door to open.
        registry = self.scene.registry
        registry.place_base_on_tray(snap=True)
        self.scene.fixture_controller.configure_product(product.spec, product.fixture)
        for fin in product.active_fins:
            registry.place_fin_in_slot(fin.fin_id, snap=True)
        for path in product.active_paths:
            registry.set_path_visible(path.path_id, True, coverage=1.0)
        registry.set_press_installed(True)
        self.scene.fixture_controller.complete_immediately(product.fixture)
        self.scene.fixture_controller.lock(product.fixture)

        self.single.tasks.clear()
        self.single.active_task = None
        self.single.paused = True
        unit = batch.units[0]
        unit.phase = TrayUnitPhase.READY_FOR_TRANSFER
        batch.transition(BatchStage.TRANSFERRING_LAYER, timestamp)
        self.transfer_demo = True
        self._queue_load(0, timestamp)
        return batch

    def inject_fault(self, *args: Any, **kwargs: Any) -> Any:
        return self.single.inject_fault(*args, **kwargs)

    def _acquire_transfer(self, resources: tuple[str, ...], now: float) -> None:
        owner = "batch_transfer"
        if not self.single.resources.acquire_many(resources, owner, now=now):
            raise RuntimeError(f"batch transfer resources are busy: {', '.join(resources)}")
        self._transfer_resources = resources

    def _release_transfer(self) -> None:
        self.single.resources.release_all("batch_transfer")
        self._transfer_resources = ()

    def _release_prefetch(self) -> None:
        self.single.resources.release_all("batch_prefetch")
        self._prefetch_resources = ()

    def _maybe_start_prefetch(self, now: float) -> None:
        """Overlap the next empty-tray index with the current rack insertion."""

        assert self.batch is not None
        if self.transfer_demo:
            return
        index = self.batch.active_unit_index
        next_index = index + 1
        if next_index >= len(self.batch.units) or self.transfer.operation != "load":
            return
        if self.transfer.prefetch_active_for(next_index):
            return
        if self.transfer.phase not in {
            TransferPhase.LIFTING,
            TransferPhase.PUSHING,
            TransferPhase.RETRACTING,
            TransferPhase.LOWERING,
        }:
            return
        resources = ("tray_indexer", "table2_zone")
        if not self.single.resources.acquire_many(resources, "batch_prefetch", now=now):
            return
        try:
            self.transfer.start_prefetch_index(next_index, now)
        except Exception:
            self.single.resources.release_all("batch_prefetch")
            raise
        self._prefetch_resources = resources
        self.status = f"loading tray {index + 1}; indexing tray {next_index + 1} concurrently"

    def _start_load(self, index: int, now: float) -> None:
        self._acquire_transfer(
            ("lift_transfer", "rack_pusher", "furnace_rack", "furnace_mouth"),
            now,
        )
        try:
            self.transfer.start_load(index, now)
        except Exception:
            self._release_transfer()
            raise

    def _sync_furnace_door(self) -> None:
        """Drive the physical door from the batch furnace state."""

        if self.furnace is None:
            return
        self.scene.registry.set_furnace_door(
            self.furnace.state.door_fraction,
            teleport=self.fast,
        )

    def _physical_door_open(self) -> bool:
        return self.scene.registry.furnace_door_fraction >= 0.98

    def _physical_door_closed(self) -> bool:
        return self.scene.registry.furnace_door_fraction <= 0.02

    def _queue_load(self, index: int, now: float) -> None:
        """Open the furnace and defer tray motion until the door is really open."""

        assert self.batch is not None and self.furnace is not None
        if self._pending_load_index is not None or self.transfer.operation:
            raise RuntimeError("another rack transfer is already pending")
        if self.furnace.status is FurnacePhase.IDLE:
            self.furnace.request_open(now)
        elif self.furnace.status not in {FurnacePhase.DOOR_OPENING, FurnacePhase.LOADING}:
            raise RuntimeError(f"rack loading requires an open furnace, not {self.furnace.status.value}")
        self._pending_load_index = index
        self.batch.furnace = self.furnace.state
        self._sync_furnace_door()
        self.status = f"opening furnace door for tray {index + 1}"

    def _dispatch_pending_load(self, now: float) -> None:
        if self._pending_load_index is None:
            return
        index = self._pending_load_index
        self._pending_load_index = None
        self._start_load(index, now)
        assert self.batch is not None
        layer = self.batch.units[index].layer_index
        self.status = f"loading tray {index + 1} into shelf {layer + 1}"

    def _start_index(self, index: int, now: float) -> None:
        self._acquire_transfer(("tray_indexer", "table2_zone"), now)
        try:
            self.transfer.start_index(index, now)
        except Exception:
            self._release_transfer()
            raise

    def _start_unload(self, index: int, now: float) -> None:
        self._acquire_transfer(
            (
                "lift_transfer",
                "rack_pusher",
                "furnace_rack",
                "furnace_mouth",
                "output_buffer",
            ),
            now,
        )
        try:
            self.transfer.start_unload(index, now)
        except Exception:
            self._release_transfer()
            raise

    def _begin_transfer(self, now: float) -> None:
        assert self.batch is not None
        index = self.batch.active_unit_index
        unit = self.batch.units[index]
        detached = self.single.detach_ready_product()
        unit.product = detached
        unit.phase = TrayUnitPhase.READY_FOR_TRANSFER
        self.batch.transition(BatchStage.TRANSFERRING_LAYER, now)
        self._queue_load(index, now)

    def _after_load(self, now: float) -> None:
        assert self.batch is not None
        self._release_transfer()
        loaded = self.batch.units[self.batch.active_unit_index]
        loaded.product.transition(OrderStage.FURNACE_LOADING, now)
        if self.transfer_demo:
            self.paused = True
            self.status = "bottom-shelf transfer demo completed"
            return
        index = self.batch.active_unit_index
        if index < len(self.batch.units) - 1:
            self.batch.active_unit_index = index + 1
            self._waiting_for_index = True
            next_index = index + 1
            if self.transfer.prefetch_complete_for(next_index):
                self.transfer.consume_prefetch(next_index)
                self._after_index(now)
            elif self.transfer.prefetch_active_for(next_index):
                self.status = f"waiting for prefetched tray {next_index + 1} to settle at Table2"
            else:
                # Fast-mode transfer may complete in one poll before an
                # overlap opportunity is observable; retain the normal path.
                self._start_index(next_index, now)
                self.status = f"indexing empty tray {next_index + 1} to Table2"
            return
        assigned = [unit.layer_index for unit in self.batch.units]
        if not self.batch.rack.planned_locked(assigned):
            self.batch.fail("furnace interlock: all planned shelves must be locked", now)
            return
        self.batch.transition(BatchStage.READY_FOR_BRAZING, now)
        self._start_furnace(now)

    def _after_index(self, now: float) -> None:
        assert self.batch is not None
        self._release_transfer()
        self._release_prefetch()
        self._waiting_for_index = False
        index = self.batch.active_unit_index
        self.scene.registry.set_batch_tray_visible(index, carrier=False, payload=False)
        self.batch.transition(BatchStage.BUILDING_LAYER, now)
        self._start_unit(index, now, full_scene_reset=False)

    def _start_furnace(self, now: float) -> None:
        assert self.batch is not None and self.furnace is not None
        assigned = [unit.layer_index for unit in self.batch.units]
        if not self.batch.rack.planned_locked(assigned):
            raise RuntimeError("all planned rack shelves are required before brazing")
        if self.transfer.busy:
            raise RuntimeError("transfer carrier must be clear before furnace closure")
        if (
            abs(self.scene.registry.batch_joint_position("batch_pusher_joint")) > 0.002
            or abs(self.scene.registry.batch_joint_position("batch_lift_joint")) > 0.002
            or abs(self.scene.registry.batch_joint_position("batch_outfeed_joint")) > 0.002
        ):
            raise RuntimeError("lift and pusher must return home before furnace closure")
        if not self.single.resources.acquire(
            "furnace_rack",
            "batch_furnace",
            now=now,
        ):
            raise RuntimeError("furnace rack is busy")
        profile_fault = next(
            (
                fault
                for fault in self.single.faults
                if fault.fault_type == "furnace_profile" and fault.armed and not fault.applied
            ),
            None,
        )
        fault_severity = None
        if profile_fault is not None:
            fault_severity = profile_fault.severity
            profile_fault.applied = True
            profile_fault.armed = False
        if self.furnace.status is not FurnacePhase.LOADING:
            raise RuntimeError("furnace must remain open while all planned trays are loaded")
        if not self._physical_door_open():
            raise RuntimeError("furnace door is not physically open")
        self._pending_furnace_fault = fault_severity
        self.furnace.load_workpiece(now)
        self.furnace.request_close(now)
        self.batch.furnace = self.furnace.state
        self._sync_furnace_door()
        self.status = f"{len(self.batch.units)} trays locked; closing furnace door"

    def _begin_unloading(self, now: float) -> None:
        assert self.batch is not None and self.furnace is not None
        self.single.resources.release("furnace_rack", "batch_furnace")
        furnace_state = self.furnace.snapshot()
        for unit in self.batch.units:
            shelf = self.batch.rack.shelves[unit.layer_index]
            unit.phase = TrayUnitPhase.BRAZED
            shelf.state = RackShelfState.BRAZED
            unit.product.furnace = furnace_state
            unit.product.transition(OrderStage.UNLOADING, now)
            for fin in unit.product.active_fins:
                fin.temporary_welded = False
                fin.board_welded = True
        self.batch.transition(BatchStage.UNLOADING, now)
        self._unload_order = sorted(
            range(len(self.batch.units)),
            key=lambda index: self.batch.units[index].layer_index,
            reverse=True,
        )
        self._unload_position = 0
        self._unload_cursor = self._unload_order[0]
        self.batch.active_unit_index = self._unload_cursor
        self._start_unload(self._unload_cursor, now)
        self.status = "unloading top shelf"

    def _start_post_inspection(self, now: float) -> None:
        """Move Arm3 to the unloaded tray before applying truth-based grading."""

        assert self.batch is not None
        self._release_transfer()
        index = self._unload_cursor
        self.batch.active_unit_index = index
        if self._unload_position == len(self._unload_order) - 1:
            self.batch.transition(BatchStage.POST_INSPECTION, now)
        owner = Actor.ARM3.value
        if not self.single.resources.acquire("inspection_zone", owner, now=now):
            raise RuntimeError("post-inspection zone is busy")
        site = self.scene.registry.site_pose(f"batch_output_slot_{index + 1:02d}_site")
        task = TaskSpec(
            task_id=f"batch_post_inspect_{index + 1:02d}",
            actor=Actor.ARM3,
            task_type=TaskType.POST_INSPECT,
            resources=("inspection_zone",),
            payload={
                "unit_index": index,
                "world_position": site.position.tolist(),
                "top_clearance_m": 0.22,
                "side_clearance_m": 0.12,
                "top_yaw_rad": 0.0,
                "side_yaw_rad": 0.0,
                "park_after": True,
            },
            timeout=45.0,
        )
        try:
            self.single.actors[Actor.ARM3.value].start_task(task, now)
        except Exception:
            self.single.resources.release("inspection_zone", owner)
            raise
        task.mark_running(now)
        self.inspection_task = task
        self._inspection_unit_index = index
        self.status = f"Arm3 inspecting unloaded tray {index + 1}"

    def _finish_post_inspection(self, now: float) -> None:
        assert self.batch is not None and self.furnace is not None
        assert self._inspection_unit_index is not None
        index = self._inspection_unit_index
        unit = self.batch.units[index]
        unit.product.furnace = self.furnace.snapshot()
        unit.product.transition(OrderStage.POST_INSPECTION, now)
        self.single.quality.post_inspection(unit.product, now)
        unit.phase = TrayUnitPhase.INSPECTED
        if self.inspection_task is not None:
            self.inspection_task.mark_succeeded(now)
        self.single.resources.release("inspection_zone", Actor.ARM3.value)
        self.inspection_task = None
        self._inspection_unit_index = None
        if self._unload_position + 1 < len(self._unload_order):
            self._unload_position += 1
            self._unload_cursor = self._unload_order[self._unload_position]
            self.batch.active_unit_index = self._unload_cursor
            self._start_unload(self._unload_cursor, now)
            self.status = f"unloading shelf {self._unload_cursor + 1}"
            return
        self.batch.transition(BatchStage.COMPLETE, now)
        self.status = f"{len(self.batch.units)}-unit flexible batch completed"

    def _poll_post_inspection(self, now: float) -> None:
        if self.inspection_task is None:
            return
        result = self.single.actors[Actor.ARM3.value].poll_task(now)
        if result is ActorResult.FAILED:
            self.inspection_task.mark_failed(now, "Arm3 post-inspection motion failed")
            self.single.resources.release("inspection_zone", Actor.ARM3.value)
            assert self.batch is not None
            self.batch.fail("Arm3 post-inspection motion failed", now)
            return
        if result is ActorResult.SUCCEEDED:
            self._finish_post_inspection(now)

    def tick(self, now: float | None = None) -> BatchState | None:
        timestamp = self.scene.time if now is None else float(now)
        batch = self.batch
        if batch is None or batch.terminal or self.paused:
            return batch
        try:
            if (
                self.furnace is not None
                and batch.stage is BatchStage.BUILDING_LAYER
                and self.furnace.status is FurnacePhase.DOOR_OPENING
            ):
                self.furnace.update(timestamp)
                batch.furnace = self.furnace.state
                self._sync_furnace_door()
            if batch.stage is BatchStage.BUILDING_LAYER:
                self.single.tick(timestamp)
                product = self.single.product
                if product is not None and product.stage in {
                    OrderStage.MANUAL_REVIEW,
                    OrderStage.ERROR,
                }:
                    batch.active_unit.phase = (
                        TrayUnitPhase.MANUAL_REVIEW
                        if product.stage is OrderStage.MANUAL_REVIEW
                        else TrayUnitPhase.ERROR
                    )
                    target = (
                        BatchStage.MANUAL_REVIEW
                        if product.stage is OrderStage.MANUAL_REVIEW
                        else BatchStage.ERROR
                    )
                    batch.transition(target, timestamp)
                    self.status = f"batch paused by tray {batch.active_unit_index + 1}"
                elif (
                    product is not None
                    and product.stage is OrderStage.READY_FOR_TRANSFER
                    and self.single.paused
                ):
                    self._begin_transfer(timestamp)
            elif batch.stage is BatchStage.TRANSFERRING_LAYER:
                if self._pending_load_index is not None:
                    assert self.furnace is not None
                    self.furnace.update(timestamp)
                    batch.furnace = self.furnace.state
                    self._sync_furnace_door()
                    if self.furnace.state.door_open and self._physical_door_open():
                        self._dispatch_pending_load(timestamp)
                else:
                    result = self.transfer.poll(timestamp)
                    self._maybe_start_prefetch(timestamp)
                    if result is TaskStatus.FAILED:
                        batch.fail(self.transfer.error or "batch transfer failed", timestamp)
                    elif result is TaskStatus.SUCCEEDED:
                        if self._waiting_for_index:
                            index = batch.active_unit_index
                            if self.transfer.prefetch_complete_for(index):
                                self.transfer.consume_prefetch(index)
                            self._after_index(timestamp)
                        else:
                            self._after_load(timestamp)
            elif batch.stage is BatchStage.READY_FOR_BRAZING:
                assert self.furnace is not None
                self.furnace.update(timestamp)
                batch.furnace = self.furnace.state
                self._sync_furnace_door()
                if self.furnace.status is FurnacePhase.READY and self._physical_door_closed():
                    self.furnace.start_cycle(timestamp, fault=self._pending_furnace_fault)
                    self._pending_furnace_fault = None
                    for unit in batch.units:
                        unit.product.transition(OrderStage.BRAZING, timestamp)
                    batch.transition(BatchStage.BRAZING, timestamp)
                    batch.furnace = self.furnace.state
                    self.status = f"{len(batch.units)}-unit 10 s brazing cycle started"
            elif batch.stage is BatchStage.BRAZING:
                assert self.furnace is not None
                self.furnace.update(timestamp)
                batch.furnace = self.furnace.state
                self._sync_furnace_door()
                if self.furnace.complete and self._physical_door_open():
                    self._begin_unloading(timestamp)
            elif batch.stage in {BatchStage.UNLOADING, BatchStage.POST_INSPECTION}:
                if self.inspection_task is not None:
                    self._poll_post_inspection(timestamp)
                else:
                    result = self.transfer.poll(timestamp)
                    if result is TaskStatus.FAILED:
                        batch.fail(self.transfer.error or "batch unloading failed", timestamp)
                    elif result is TaskStatus.SUCCEEDED:
                        self._start_post_inspection(timestamp)
        except Exception as exc:
            batch.fail(str(exc), timestamp)
            self.status = str(exc)
        return batch

    def pause(self, now: float | None = None) -> None:
        if self.batch is None or self.batch.terminal or self.paused:
            return
        timestamp = self.scene.time if now is None else float(now)
        if self.batch.stage is BatchStage.BUILDING_LAYER:
            self.single.pause(timestamp)
        if self.transfer.busy:
            self.transfer.pause(timestamp)
        if self.inspection_task is not None:
            self.single.actors[Actor.ARM3.value].cancel()
            self.single.resources.release("inspection_zone", Actor.ARM3.value)
            self.inspection_task.status = TaskStatus.READY
            self.inspection_task.started_at = None
        self.paused = True
        self._paused_at = timestamp
        self.status = "batch paused"

    def resume(self, now: float | None = None) -> None:
        if self.batch is None or self.batch.terminal:
            raise RuntimeError("no resumable batch")
        timestamp = self.scene.time if now is None else float(now)
        if not self.paused:
            return
        if self._paused_at is not None and self.furnace is not None:
            offset = max(0.0, timestamp - self._paused_at)
            self.furnace.state.phase_started_at += offset
            if self.furnace.state.cycle_started_at is not None:
                self.furnace.state.cycle_started_at += offset
        if self.batch.stage is BatchStage.BUILDING_LAYER:
            self.single.resume(timestamp)
            self.single.pause_after_stage = OrderStage.READY_FOR_TRANSFER
        if self.transfer.busy:
            self.transfer.resume(timestamp)
        if self.inspection_task is not None:
            if not self.single.resources.acquire(
                "inspection_zone",
                Actor.ARM3.value,
                now=timestamp,
            ):
                raise RuntimeError("post-inspection zone is busy")
            self.single.actors[Actor.ARM3.value].start_task(
                self.inspection_task,
                timestamp,
            )
            self.inspection_task.mark_running(timestamp)
        self.paused = False
        self._paused_at = None
        self.status = "batch resumed"

    def reset(self) -> None:
        if self.inspection_task is not None:
            self.single.actors[Actor.ARM3.value].cancel()
        self.single.resources.release_all()
        self._transfer_resources = ()
        self._prefetch_resources = ()
        if self.single.product is not None:
            self.single.reset()
        self.transfer.reset(show_empty_cache=False)
        self.scene.registry.set_furnace_door(0.0, teleport=True)
        self.batch = None
        self.furnace = None
        self.paused = False
        self._waiting_for_index = False
        self._unload_cursor = -1
        self._paused_at = None
        self.transfer_demo = False
        self.inspection_task = None
        self._inspection_unit_index = None
        self._pending_load_index = None
        self._pending_furnace_fault = None
        self.process_plan = None
        self._unload_order = []
        self._unload_position = -1
        self.status = "idle"

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        timestamp = self.scene.time if now is None else float(now)
        if self.batch is None:
            return self.single.snapshot(timestamp)
        batch = self.batch
        product = self.product
        dispositions = [unit.product.disposition for unit in batch.units]
        aggregate_disposition = None
        if all(disposition is not None for disposition in dispositions):
            values = {disposition.value for disposition in dispositions if disposition is not None}
            if "SCRAPPED" in values:
                aggregate_disposition = "SCRAPPED"
            elif "REWORK_REQUIRED" in values:
                aggregate_disposition = "REWORK_REQUIRED"
            else:
                aggregate_disposition = "PASS"
        # Always start from the single coordinator's complete schema. During
        # transfer and furnace phases it has no attached product, but its
        # resource/arm/fault defaults are still needed to prevent stale HTTP
        # fields from leaking from the preceding build stage.
        base = self.single.snapshot(timestamp)
        base.update(
            {
                "status": self.status,
                "paused": self.paused,
                "order_id": batch.batch_id,
                "preset": batch.preset,
                "stage": batch.stage.value,
                "disposition": aggregate_disposition,
                "fins": ({} if product is None else {fin.fin_id: asdict(fin) for fin in product.fins}),
                "paths": ({} if product is None else {path.path_id: asdict(path) for path in product.paths}),
                "fixture": {} if product is None else asdict(product.fixture),
                "inspections": (
                    [] if product is None else [asdict(result) for result in product.inspections]
                ),
                "furnace": {
                    **asdict(batch.furnace),
                    "status": batch.furnace.phase.value,
                    "door_open": batch.furnace.door_open,
                },
                "batch": {
                    "batch_id": batch.batch_id,
                    "stage": batch.stage.value,
                    "result": aggregate_disposition,
                    "layers": len(batch.units),
                    "active_layer": batch.active_unit.layer_index + 1,
                    "elapsed_seconds": max(0.0, timestamp - batch.created_at),
                    "completed_units": sum(unit.phase is TrayUnitPhase.INSPECTED for unit in batch.units),
                    "furnace_cycle_count": int(
                        self.furnace is not None and self.furnace.state.cycle_started_at is not None
                    ),
                    "units": [
                        {
                            "unit_id": unit.unit_id,
                            "layer": unit.layer_index + 1,
                            "phase": unit.phase.value,
                            "order_id": unit.product.order_id,
                            "product_stage": unit.product.stage.value,
                            "disposition": (
                                None if unit.product.disposition is None else unit.product.disposition.value
                            ),
                            "output_slot": unit.output_slot,
                        }
                        for unit in batch.units
                    ],
                },
                "rack": {
                    "load_order": list(batch.rack.load_order),
                    "unload_order": list(batch.rack.unload_order),
                    "all_locked": batch.rack.planned_locked(unit.layer_index for unit in batch.units),
                    "shelves": [
                        {
                            "index": shelf.index,
                            "height_m": shelf.height_m,
                            "state": shelf.state.value,
                            "unit_id": shelf.unit_id,
                            "lock_engaged": shelf.lock_engaged,
                        }
                        for shelf in batch.rack.shelves
                    ],
                },
                "transfer": {**asdict(batch.transfer), **self.transfer.state},
                "last_error": batch.errors[-1] if batch.errors else "",
                "plan": None if self.process_plan is None else self.process_plan.summary(),
            }
        )
        if self.inspection_task is not None:
            base.setdefault("arms", {}).setdefault("arm3", {}).update(
                {
                    "task_id": self.inspection_task.task_id,
                    "task_type": TaskType.POST_INSPECT.value,
                    "status": "busy",
                    "error": self.inspection_task.error or "",
                }
            )
        return base


__all__ = ["BatchCoordinator"]
