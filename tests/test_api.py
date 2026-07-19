from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from brazing_sim.api import SharedState, parse_terminal_command, start_http_server


def post(url: str, payload: dict) -> tuple[int, dict]:
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=2.0) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture
def api_server():
    shared = SharedState()
    server = start_http_server(shared, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield shared, url
    finally:
        server.shutdown()
        server.server_close()


def test_state_and_camera_contract(api_server) -> None:
    shared, url = api_server
    with urlopen(url + "/state", timeout=2.0) as response:  # noqa: S310
        state = json.loads(response.read())
    assert state["stage"] == "IDLE"
    assert state["conveyor"]["phase"] == "IDLE"
    assert state["conveyor"]["travel_m"] == pytest.approx(0.63)
    with pytest.raises(HTTPError) as error:
        urlopen(url + "/camera.ppm", timeout=2.0)  # noqa: S310
    assert error.value.code == 503
    shared.update_camera(b"P6\n1 1\n255\n\x00\x00\x00", width=1, height=1)
    with urlopen(url + "/camera.ppm", timeout=2.0) as response:  # noqa: S310
        assert response.read().startswith(b"P6")


def test_order_fault_stop_reset_commands(api_server) -> None:
    shared, url = api_server
    for preset in ("A", "B", "C"):
        assert post(url + "/order", {"preset": preset})[0] == 202
        assert shared.commands.get(timeout=1) == {"type": "order", "preset": preset}
    status, payload = post(
        url + "/fault",
        {"type": "brazing_gap", "target": "slot_02_left", "severity": "recoverable"},
    )
    assert status == 202 and payload["fault_type"] == "brazing_gap"
    assert shared.commands.get(timeout=1)["type"] == "fault"
    assert post(url + "/stop", {})[0] == 202
    assert post(url + "/reset", {})[0] == 202


def test_segment_and_continue_commands(api_server) -> None:
    shared, url = api_server
    status, payload = post(url + "/segment", {"segment": "arm2_motion"})
    assert status == 202 and payload["segment"] == "arm2_motion"
    assert shared.commands.get(timeout=1) == {"type": "segment", "segment": "arm2_motion"}
    assert post(url + "/continue", {})[0] == 202
    assert shared.commands.get(timeout=1) == {"type": "continue"}
    assert post(url + "/speed", {"action": "accelerate"})[0] == 202
    assert shared.commands.get(timeout=1) == {"type": "speed", "action": "accelerate"}
    assert post(url + "/speed", {"action": "decelerate"})[0] == 202
    assert shared.commands.get(timeout=1) == {"type": "speed", "action": "decelerate"}
    assert post(url + "/speed", {"action": "invalid"})[0] == 400
    status, payload = post(url + "/segment", {"segment": "fin_assembly"})
    assert status == 202 and payload["segment"] == "fin_assembly"
    assert shared.commands.get(timeout=1) == {"type": "segment", "segment": "fin_assembly"}
    status, payload = post(url + "/segment", {"segment": "furnace_cycle"})
    assert status == 202 and payload["segment"] == "furnace_cycle"
    assert shared.commands.get(timeout=1) == {"type": "segment", "segment": "furnace_cycle"}
    status, payload = post(url + "/segment", {"segment": "rack_transfer"})
    assert status == 202 and payload["segment"] == "rack_transfer"
    assert shared.commands.get(timeout=1) == {"type": "segment", "segment": "rack_transfer"}
    assert post(url + "/segment", {"segment": "unknown"})[0] == 400


def test_three_layer_batch_http_and_terminal_commands(api_server) -> None:
    shared, url = api_server
    status, payload = post(url + "/batch", {"preset": "A", "layers": 3})
    assert status == 202
    assert payload == {"ok": True, "type": "batch", "preset": "A", "layers": 3}
    assert shared.commands.get(timeout=1) == {
        "type": "batch",
        "preset": "A",
        "layers": 3,
    }
    assert post(url + "/batch", {"preset": "B", "layers": 3})[0] == 400
    assert post(url + "/batch", {"preset": "A", "layers": 2})[0] == 400
    assert parse_terminal_command("batch_a") == {
        "type": "batch",
        "preset": "A",
        "layers": 3,
    }
    assert parse_terminal_command("rack_transfer") == {
        "type": "segment",
        "segment": "rack_transfer",
    }


def test_invalid_order_and_fault_are_rejected(api_server) -> None:
    _, url = api_server
    assert post(url + "/order", {"preset": "D"})[0] == 400
    assert post(url + "/fault", {"type": "brazing_gap"})[0] == 400
    assert post(url + "/fault", {"type": "unknown", "target": "x"})[0] == 400
