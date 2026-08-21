"""Thread-safe control surface for the brazing simulation.

Only commands cross the HTTP/terminal boundary.  The MuJoCo thread remains the
sole owner of simulation state and consumes :attr:`SharedState.commands`.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from enum import Enum
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Mapping, TextIO
from urllib import error, request
from urllib.parse import urlsplit

ARM_NAMES = ("arm1", "arm2", "arm3")
FAULT_TYPES = {"fin_pose", "brazing_gap", "furnace_profile"}
FAULT_SEVERITIES = {"recoverable", "severe"}
MAX_HTTP_CONCURRENCY = 32


def _is_loopback_host(host: str) -> bool:
    normalized = str(host).strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field}必须是整数")
    return value


def jsonable(value: Any) -> Any:
    """Convert domain dataclasses/enums and containers to JSON-safe values."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


class SharedState:
    """Small synchronized state mirror shared with HTTP and optional Qt UI."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self._state: dict[str, Any] = {
            "schema_version": 2,
            "status": "starting",
            "viewer_running": False,
            "order_id": "",
            "preset": "",
            "stage": "IDLE",
            "paused": False,
            "simulation_speed": 1.0,
            "simulation_actual_rtf": 0.0,
            "simulation_speed_saturated": False,
            "disposition": None,
            "arms": {
                name: {"task_id": "", "task_type": "", "status": "idle", "error": ""} for name in ARM_NAMES
            },
            "resources": {},
            "fins": {},
            "paths": {},
            "fixture": {},
            "preflight": {"ok": False, "issues": [], "checked_presets": []},
            "tools": {"arm1": {}, "arm2": {}},
            "arm2_process": {
                "current_path": "",
                "completed_paths": 0,
                "total_paths": 0,
            },
            "conveyor": {
                "phase": "IDLE",
                "position_m": 0.0,
                "target_m": 0.0,
                "travel_m": 0.63,
                "progress": 0.0,
                "moving": False,
            },
            "batch": {},
            "rack": {"shelves": []},
            "transfer": {
                "phase": "IDLE",
                "step": "",
                "unit_id": None,
                "shelf_index": None,
                "lift_height_m": 0.0,
                "outfeed_position_m": 0.0,
                "pusher_position_m": 0.0,
                "pusher_extension_ratio": 0.0,
                "conveyor_position_m": 0.0,
                "conveyor_progress": 0.0,
                "lock_position_m": 0.0,
                "output_position_m": 0.0,
                "comb_removal_progress": 0.0,
                "press_removal_progress": 0.0,
                "moving": False,
            },
            "furnace": {"status": "idle", "temperature_c": 25.0, "door_open": False},
            "inspections": [],
            "faults": [],
            "kpi": {},
            "last_error": "",
            "camera_width": 0,
            "camera_height": 0,
            "camera_active": False,
            "camera_status": "camera starting",
            "camera_frame_time": 0.0,
            "available_orders": ["A", "B", "C"],
            "scheduler": {
                "mode": "FIXED_SEQUENCE",
                "ready_count": 0,
                "running_count": 0,
                "replan_count": 0,
            },
            "tasks": [],
            "resources_v2": {},
            "zone_locks": {},
            "orders": [],
            "faults_v2": [],
            "recoveries": [],
            "manual_fault_requests": [],
            "experiment_metrics": {},
            "workstations": {},
            "async_line": {},
            "transfers": {},
            "tray_routes": {},
            "motion_plans": [],
            "space_time_reservations": [],
            "motion_blockers": {},
            "gantt_events": [],
            "server_time": time.time(),
        }
        self._camera_frame_ppm = b""

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            # Every write path normalizes values before they enter _state.
            # Re-running the recursive enum/dataclass conversion after a deep
            # copy doubled the cost of each /state request.
            return copy.deepcopy(self._state)

    def camera_status_snapshot(self) -> dict[str, Any]:
        """Return camera metadata without copying the full manufacturing DAG."""

        with self.lock:
            return {
                key: self._state.get(key)
                for key in (
                    "camera_width",
                    "camera_height",
                    "camera_active",
                    "camera_status",
                    "camera_frame_time",
                )
            }

    def update(self, payload: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        """Atomically update top-level keys.

        ``update(stage="BRAZING")`` and ``update(coordinator.snapshot())`` are
        both supported to keep the entry point uncomplicated.
        """

        values: dict[str, Any] = {}
        if payload is not None:
            values.update(dict(payload))
        values.update(kwargs)
        with self.lock:
            self._state.update(jsonable(values))
            self._state["server_time"] = time.time()

    def update_prepared(self, payload: Mapping[str, Any], **kwargs: Any) -> None:
        """Update with an already JSON-safe snapshot from the simulation.

        Coordinator snapshots have already converted dataclasses, enums and
        arrays. This hot-path variant avoids recursively walking the same
        12-fin/24-path state a second time before publishing it.
        """

        values = dict(payload)
        values.update(kwargs)
        with self.lock:
            self._state.update(values)
            self._state["server_time"] = time.time()

    def replace(self, payload: Mapping[str, Any]) -> None:
        with self.lock:
            camera_meta = {
                key: self._state[key]
                for key in (
                    "camera_width",
                    "camera_height",
                    "camera_active",
                    "camera_status",
                    "camera_frame_time",
                    "available_orders",
                )
            }
            self._state = jsonable(dict(payload))
            for key, value in camera_meta.items():
                self._state.setdefault(key, value)
            self._state["server_time"] = time.time()

    def enqueue(self, command: Mapping[str, Any]) -> dict[str, Any]:
        item = jsonable(dict(command))
        self.commands.put(item)
        return item

    @property
    def camera_frame_ppm(self) -> bytes:
        with self.lock:
            return bytes(self._camera_frame_ppm)

    @camera_frame_ppm.setter
    def camera_frame_ppm(self, value: bytes) -> None:
        with self.lock:
            self._camera_frame_ppm = bytes(value)

    def update_camera(
        self,
        frame_ppm: bytes,
        *,
        width: int,
        height: int,
        active: bool = False,
        status: str = "camera ready",
        timestamp: float | None = None,
    ) -> None:
        with self.lock:
            self._camera_frame_ppm = bytes(frame_ppm)
            self._state.update(
                camera_width=int(width),
                camera_height=int(height),
                camera_active=bool(active),
                camera_status=str(status),
                camera_frame_time=time.time() if timestamp is None else float(timestamp),
                server_time=time.time(),
            )

    def clear_commands(self) -> None:
        while True:
            try:
                self.commands.get_nowait()
            except queue.Empty:
                return


class _BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Keep abusive or stalled clients from creating unbounded worker threads."""

    daemon_threads = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_slots = threading.BoundedSemaphore(MAX_HTTP_CONCURRENCY)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._request_slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._request_slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


class RequestHandler(BaseHTTPRequestHandler):
    """HTTP adapter; subclasses receive a class-level ``shared`` instance."""

    shared: SharedState
    auth_token: str | None = None
    max_body_bytes = 1_000_000

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, payload: Mapping[str, Any], status: int = 200) -> None:
        body = json.dumps(jsonable(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_binary(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0:
            return {}
        if length > self.max_body_bytes:
            raise ValueError("request body too large")
        decoded = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("JSON body must be an object")
        return decoded

    def _authorized_post(self) -> bool:
        expected = self.auth_token
        if not expected:
            return True
        scheme, separator, supplied = self.headers.get("Authorization", "").partition(" ")
        if separator != " " or scheme.lower() != "bearer" or not hmac.compare_digest(supplied, expected):
            self._send_json({"ok": False, "error": "Bearer authorization required"}, 401)
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/state":
            self._send_json(self.shared.snapshot())
        elif path == "/camera/status":
            self._send_json(self.shared.camera_status_snapshot())
        elif path == "/camera.ppm":
            frame = self.shared.camera_frame_ppm
            if frame:
                self._send_binary(frame, "image/x-portable-pixmap")
            else:
                self._send_binary(b"camera frame not ready", "text/plain; charset=utf-8", 503)
        else:
            state = self.shared.snapshot()
            from .fault_catalog import fault_catalog_snapshot

            views = {
                "/scheduler/status": state.get("scheduler", {}),
                "/tasks": {"tasks": state.get("tasks", [])},
                "/resources": {
                    "resources": state.get("resources_v2") or state.get("resources", {}),
                    "zone_locks": state.get("zone_locks", {}),
                },
                "/orders": {"orders": state.get("orders", [])},
                "/faults": {
                    "faults": state.get("faults_v2") or state.get("faults", []),
                    "policy_summary": state.get("fault_policy_summary", {}),
                },
                "/recoveries": {
                    "recoveries": state.get("recoveries", []),
                    "policy_summary": state.get("fault_policy_summary", {}),
                },
                "/metrics": {
                    "live": state.get("experiment_metrics") or state.get("kpi", {}),
                    "golden_experiments": state.get("golden_experiments", {}),
                },
                "/fault-catalog": {"faults": fault_catalog_snapshot()},
                "/workstations": {
                    "workstations": state.get("workstations", {}),
                    "async_line": state.get("async_line", {}),
                    "transfers": state.get("transfers", {}),
                    "tray_routes": state.get("tray_routes", {}),
                },
                "/motion/reservations": {
                    "motion_plans": state.get("motion_plans", []),
                    "reservations": state.get("space_time_reservations", []),
                },
                # Deliberately not added to ``views``: the report reloads config
                # and compiles a routing, which is far too much work to run on
                # every /state poll.  See the explicit branch below.
            }
            if path in views:
                self._send_json(views[path])
            elif path == "/flexibility":
                self._send_json(flexibility_view(state))
            else:
                self._send_json({"ok": False, "error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized_post():
            return
        path = urlsplit(self.path).path
        try:
            payload = self._read_json()
            command = validate_http_command(path, payload)
            if command.get("type") != "order_plan":
                self.shared.enqueue(command)
            response = {"ok": True, **command}
            self._send_json(response, 202)
        except KeyError:
            self._send_json({"ok": False, "error": "not found"}, 404)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)


def _capability_summary(graph: Any) -> dict[str, Any]:
    """Summarise capability coverage and flexibility of a compiled task graph.

    ``flexible_task_count`` is the number of nodes with more than one eligible
    resource, i.e. the size of the scheduler's actual decision space.  Before
    steps A/B this was zero for every plan.
    """

    capability_tasks = 0
    flexible_tasks = 0
    alternative_tasks = 0
    capabilities: dict[str, int] = {}
    for task in graph:
        capability = task.payload.get("capability")
        if not capability:
            continue
        capability_tasks += 1
        capabilities[str(capability)] = capabilities.get(str(capability), 0) + 1
        if len(task.eligible_resources) > 1:
            flexible_tasks += 1
        if task.payload.get("capability_alternatives"):
            alternative_tasks += 1
    return {
        "total_task_count": len(graph),
        "capability_task_count": capability_tasks,
        "flexible_task_count": flexible_tasks,
        "alternative_route_task_count": alternative_tasks,
        "capabilities": dict(sorted(capabilities.items())),
    }


def flexibility_view(state: Mapping[str, Any]) -> dict[str, Any]:
    """Six-dimension flexibility report, computed lazily for ``/flexibility``."""

    from .flexibility_report import flexibility_report

    return flexibility_report(state)


def validate_http_command(path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if path == "/flexibility/demo":
        demo = str(payload.get("demo", "")).strip().lower()
        allowed = {
            "product_mix",
            "resource_parallel",
            "batch_three",
            "urgent_insert",
            "fault_loop",
        }
        if demo not in allowed:
            raise ValueError(f"flexibility demo must be one of {sorted(allowed)}")
        return {"type": "flexibility_demo", "demo": demo}
    if path == "/faults/inject":
        from .fault_catalog import validate_manual_fault_payload

        return validate_manual_fault_payload(payload)
    if path in {"/orders/plan", "/orders/insert"}:
        from datetime import datetime

        from .flexible import build_custom_plan, build_inline_plan
        from .planning import build_task_graph
        from .dual_line.admission import validate_v2_order_id

        preset = str(payload.get("preset", "A")).strip().upper()
        line_profile = str(payload.get("line_profile", "")).strip().upper()
        raw_order_id = payload.get("order_id")
        if raw_order_id is None or (isinstance(raw_order_id, str) and not raw_order_id.strip()):
            order_id = f"UI_{datetime.now().strftime('%H%M%S%f')}"
        elif not isinstance(raw_order_id, str):
            raise ValueError("订单ID必须是字符串")
        else:
            order_id = raw_order_id.strip()
        validate_v2_order_id(order_id)
        mode = str(payload.get("mode", "preset")).strip().lower()
        quantity = _strict_int(payload.get("quantity", 1), "quantity")
        priority = _strict_int(payload.get("priority", 10), "priority")
        if priority < 0:
            raise ValueError("priority必须是非负整数")
        due_time = payload.get("due_time")
        preferred = payload.get("preferred_rack_layer")
        if isinstance(preferred, str) and preferred in {"", "null"}:
            preferred = None
        if preferred is not None:
            preferred = _strict_int(preferred, "preferred_rack_layer")
            if preferred not in {0, 1, 2}:
                raise ValueError("首选料架层必须是0、1、2或空值")
            if line_profile in {"V2", "V2_DUAL_INSTALL"}:
                raise ValueError("V2 不支持首选料架层；炉层由实际装炉顺序分配")
        route_strategy = str(payload.get("route_strategy", "STANDARD")).strip().upper()
        urgent = payload.get("urgent", False)
        if not isinstance(urgent, bool):
            raise ValueError("urgent必须是布尔值")
        custom_product = payload.get("custom_product")
        if mode == "custom":
            if not isinstance(custom_product, dict):
                raise ValueError("custom mode requires custom_product object")
            plan = build_custom_plan(
                order_id=order_id,
                quantity=quantity,
                priority=priority,
                due_time=due_time,
                preferred_rack_layer=preferred,
                product=custom_product,
                route_strategy=route_strategy,
            )
            preset = "CUSTOM"
        elif mode == "preset":
            plan = build_inline_plan(
                preset=preset,
                order_id=order_id,
                quantity=quantity,
                priority=priority,
                due_time=due_time,
                preferred_rack_layer=preferred,
                route_strategy=route_strategy,
            )
        else:
            raise ValueError("mode must be preset or custom")
        if line_profile in {"V2", "V2_DUAL_INSTALL"}:
            from .dual_line.admission import validate_v2_plan

            preview_graph = validate_v2_plan(plan)
        else:
            preview_graph = build_task_graph(plan, flexible_cell=True)
        summary = plan.summary()
        summary["estimated_task_count"] = len(preview_graph)
        # Step A/B visibility: how much of the plan is flexible rather than
        # single-resource bound, straight from the capability-derived graph.
        summary["capability_summary"] = _capability_summary(preview_graph)
        command = {
            "type": "order_plan" if path.endswith("/plan") else "order_insert",
            "order_id": order_id,
            "preset": preset,
            "quantity": quantity,
            "priority": priority,
            "due_time": due_time,
            "preferred_rack_layer": preferred,
            "urgent": urgent,
            "mode": mode,
            "custom_product": custom_product if mode == "custom" else None,
            "route_strategy": route_strategy,
            "line_profile": line_profile or None,
            "plan": summary,
            "task_preview": preview_graph.snapshot(),
        }
        return command
    if path == "/scheduler/replan":
        return {"type": "scheduler_replan", "reason": str(payload.get("reason", "operator"))}
    if path.startswith("/resources/") and path.endswith(("/fault", "/recover")):
        parts = [part for part in path.split("/") if part]
        if len(parts) != 3 or not parts[1]:
            raise ValueError("invalid resource route")
        action = parts[2]
        result = {"type": f"resource_{action}", "resource_id": parts[1].upper()}
        if action == "fault":
            result["fault_code"] = str(payload.get("fault_code", "OPERATOR_FAULT"))
            result["duration_s"] = payload.get("duration_s", payload.get("duration"))
        return result
    if path.startswith("/recoveries/") and path.endswith("/action"):
        parts = [part for part in path.split("/") if part]
        if len(parts) != 3:
            raise ValueError("invalid recovery route")
        action = str(payload.get("action", "")).lower()
        if action not in {"pause", "resume", "retry", "manual_review"}:
            raise ValueError("recovery action must be pause/resume/retry/manual_review")
        return {"type": "recovery_action", "recovery_id": parts[1], "action": action}
    if path == "/order":
        preset = str(payload.get("preset", "A")).strip().upper()
        if preset not in {"A", "B", "C"}:
            raise ValueError(f"unknown order preset: {preset}")
        return {"type": "order", "preset": preset}
    if path == "/batch":
        preset = str(payload.get("preset", "A")).strip().upper()
        layers = int(payload.get("layers", 3))
        if preset != "A":
            raise ValueError("the three-layer MVP batch currently supports preset A only")
        if layers != 3:
            raise ValueError("the three-layer MVP batch requires layers=3")
        return {"type": "batch", "preset": preset, "layers": layers}
    if path == "/segment":
        segment = str(payload.get("segment", "")).strip().lower()
        allowed = {
            "pick_place",
            "inspection_1",
            "arm2_motion",
            "fin_assembly",
            "inspection_2",
            "furnace_cycle",
            "rack_transfer",
            "v2_base_loading",
            "v2_dispensing",
            "v2_material_inspection",
            "v2_install_a",
            "v2_install_b",
            "v2_parallel_install",
            "v2_merge_inspection",
            "v2_furnace_batch",
            "v2_post_braze_delivery",
        }
        if segment not in allowed:
            raise ValueError(f"segment must be one of {sorted(allowed)}")
        return {"type": "segment", "segment": segment}
    if path == "/fault":
        fault_type = str(payload.get("type", "")).strip().lower()
        target = str(payload.get("target", "")).strip()
        severity = str(payload.get("severity", "recoverable")).strip().lower()
        if fault_type not in FAULT_TYPES:
            raise ValueError(f"fault type must be one of {sorted(FAULT_TYPES)}")
        if severity not in FAULT_SEVERITIES:
            raise ValueError(f"fault severity must be one of {sorted(FAULT_SEVERITIES)}")
        if fault_type != "furnace_profile" and not target:
            raise ValueError("fault target is required")
        if fault_type == "furnace_profile" and not target:
            target = "furnace"
        return {"type": "fault", "fault_type": fault_type, "target": target, "severity": severity}
    if path == "/stop":
        return {"type": "stop"}
    if path == "/continue":
        return {"type": "continue"}
    if path == "/speed":
        action = str(payload.get("action", "")).strip().lower()
        if action not in {"accelerate", "decelerate"}:
            raise ValueError("speed action must be accelerate or decelerate")
        return {"type": "speed", "action": action}
    if path == "/reset":
        return {"type": "reset"}
    raise KeyError(path)


def start_http_server(
    shared: SharedState,
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    auth_token: str | None = None,
) -> ThreadingHTTPServer:
    token = None if auth_token is None else str(auth_token).strip()
    if not _is_loopback_host(host) and not token:
        raise ValueError("remote HTTP control requires --auth-token or BRAZING_V2_HTTP_TOKEN")
    handler = type("BrazingRequestHandler", (RequestHandler,), {"shared": shared, "auth_token": token})
    server = _BoundedThreadingHTTPServer((host, int(port)), handler)
    threading.Thread(target=server.serve_forever, name="brazing-http", daemon=True).start()
    return server


def parse_terminal_command(line: str) -> dict[str, Any] | None:
    """Parse one documented terminal command without mutating shared state."""

    parts = line.strip().lower().replace("-", "_").split()
    if not parts:
        return None
    command = parts[0]
    if command in {"order_a", "order_b", "order_c"}:
        if len(parts) != 1:
            raise ValueError(f"usage: {command}")
        return {"type": "order", "preset": command[-1].upper()}
    if command == "batch_a":
        if len(parts) != 1:
            raise ValueError("usage: batch_a")
        return {"type": "batch", "preset": "A", "layers": 3}
    if command == "fault":
        if len(parts) < 3 or len(parts) > 4:
            raise ValueError(
                "usage: fault fin_pose <fin_id> | fault brazing_gap <path_id> | "
                "fault furnace_profile recoverable|severe"
            )
        fault_type = parts[1]
        if fault_type not in FAULT_TYPES:
            raise ValueError(f"unknown fault type: {fault_type}")
        if fault_type == "furnace_profile":
            if len(parts) != 3 or parts[2] not in FAULT_SEVERITIES:
                raise ValueError("usage: fault furnace_profile recoverable|severe")
            return {
                "type": "fault",
                "fault_type": fault_type,
                "target": "furnace",
                "severity": parts[2],
            }
        target = parts[2]
        severity = parts[3] if len(parts) == 4 else "recoverable"
        if severity not in FAULT_SEVERITIES:
            raise ValueError(f"fault severity must be one of {sorted(FAULT_SEVERITIES)}")
        return {"type": "fault", "fault_type": fault_type, "target": target, "severity": severity}
    segment_commands = {
        "pick_place": "pick_place",
        "inspection_1": "inspection_1",
        "arm2_motion": "arm2_motion",
        "fin_assembly": "fin_assembly",
        "inspection_2": "inspection_2",
        "furnace_cycle": "furnace_cycle",
        "rack_transfer": "rack_transfer",
    }
    if command in segment_commands:
        if len(parts) != 1:
            raise ValueError(f"usage: {command}")
        return {"type": "segment", "segment": segment_commands[command]}
    if command in {"stop", "continue", "reset", "status", "help"}:
        if len(parts) != 1:
            raise ValueError(f"usage: {command}")
        return {"type": command}
    raise ValueError(f"unknown command: {command}")


TERMINAL_HELP = """
[Terminal commands]
  order_a | order_b | order_c          start A/B/C flexible-fixture order
  fault fin_pose fin_02                inject recoverable fin pose fault
  fault brazing_gap slot_02_left       inject recoverable material-gap fault
  fault furnace_profile recoverable    degrade final quality to rework
  fault furnace_profile severe         force scrap disposition
  pick_place | inspection_1 | arm2_motion | fin_assembly | inspection_2 | furnace_cycle
  rack_transfer | batch_a
  stop | continue | reset | status | help
""".strip()


def start_terminal_thread(
    shared: SharedState,
    *,
    stream: TextIO = sys.stdin,
    output: TextIO = sys.stdout,
    on_command: Callable[[dict[str, Any]], None] | None = None,
) -> threading.Thread:
    """Start a daemon command reader; EOF cleanly terminates the thread."""

    def loop() -> None:
        print(TERMINAL_HELP, file=output, flush=True)
        for line in stream:
            try:
                command = parse_terminal_command(line)
                if command is None:
                    continue
                if command["type"] == "help":
                    print(TERMINAL_HELP, file=output, flush=True)
                elif command["type"] == "status":
                    print(
                        json.dumps(shared.snapshot(), ensure_ascii=False, indent=2), file=output, flush=True
                    )
                elif on_command is not None:
                    on_command(command)
                else:
                    shared.enqueue(command)
            except ValueError as exc:
                print(f"[terminal] {exc}", file=output, flush=True)

    thread = threading.Thread(target=loop, name="brazing-terminal", daemon=True)
    thread.start()
    return thread


def _http_error_detail(exc: error.HTTPError) -> str:
    """Return the validation reason carried by a failed JSON response."""

    try:
        body = exc.read().decode("utf-8").strip()
    except (OSError, UnicodeDecodeError):
        body = ""
    if body:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping):
            for key in ("error", "message", "detail"):
                detail = str(payload.get(key, "")).strip()
                if detail:
                    return detail
        return body
    return f"HTTP {exc.code}: {exc.reason}"


def post_json(url: str, payload: Mapping[str, Any], timeout: float = 1.0) -> dict[str, Any]:
    body = json.dumps(jsonable(payload)).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("BRAZING_V2_HTTP_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        raise ValueError(_http_error_detail(exc)) from exc


def get_json(url: str, timeout: float = 1.0) -> dict[str, Any]:
    with request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def get_bytes(url: str, timeout: float = 1.0) -> bytes:
    with request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return response.read()


def rgb_frame_to_ppm(frame: Any) -> bytes:
    """Encode an HxWx3 uint8-compatible frame without an image dependency."""

    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) != 3 or int(shape[2]) != 3:
        raise ValueError(f"camera frame must be HxWx3 RGB, got {shape}")
    try:
        import numpy as np

        rgb = np.ascontiguousarray(frame, dtype=np.uint8)
        height, width = rgb.shape[:2]
        pixels = rgb.tobytes()
    except ImportError:
        height, width = int(shape[0]), int(shape[1])
        pixels = bytes(frame)
    return f"P6\n{width} {height}\n255\n".encode("ascii") + pixels


__all__ = [
    "ARM_NAMES",
    "FAULT_SEVERITIES",
    "FAULT_TYPES",
    "RequestHandler",
    "SharedState",
    "TERMINAL_HELP",
    "get_bytes",
    "get_json",
    "jsonable",
    "parse_terminal_command",
    "post_json",
    "rgb_frame_to_ppm",
    "start_http_server",
    "start_terminal_thread",
    "validate_http_command",
]
