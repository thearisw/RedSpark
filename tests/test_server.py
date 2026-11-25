import pytest

pytest.importorskip("starlette")
pytest.importorskip("httpx")

from starlette.testclient import TestClient  # noqa: E402

from redspark.server import app  # noqa: E402

client = TestClient(app)


def test_lists_example_circuits():
    names = client.get("/api/circuits").json()
    assert {"not_gate", "and_gate", "or_gate"} <= set(names)
    assert client.get("/api/circuits/../pyproject").status_code == 404


def test_websocket_toggle_and_step():
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "load", "name": "not_gate"})
        state = ws.receive_json()
        lamp = next(b for b in state["circuit"]["blocks"] if b.get("label") == "OUT")
        assert lamp["active"]
        ws.send_json({"type": "toggle", "pos": [0, 0, 1]})
        ws.receive_json()
        for _ in range(6):  # torch 2 gt + lamp 4 gt
            ws.send_json({"type": "step"})
            state = ws.receive_json()
        lamp = next(b for b in state["circuit"]["blocks"] if b.get("label") == "OUT")
        assert not lamp["active"]
        ws.send_json({"type": "toggle", "pos": [1, 0, 1]})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "load", "name": "../secrets"})
        assert ws.receive_json()["type"] == "error"


def test_serves_the_built_viewer():
    from redspark.server import VIEWER_DIR
    if not (VIEWER_DIR / "index.html").is_file():
        pytest.skip("viewer not built")
    page = client.get("/")
    assert page.status_code == 200 and "RedSpark" in page.text


NOT_TASK = {
    "size": [5, 1, 3],
    "inputs": [{"label": "A", "pos": [0, 0, 1]}],
    "outputs": [{"label": "OUT1", "pos": [4, 0, 1]}],
    "rows": [{"in": [False], "out": [True]}, {"in": [True], "out": [False]}],
}


def test_websocket_training_streams_progress_and_caps_circuits(tmp_path, monkeypatch):
    import time
    import redspark.server as server
    monkeypatch.setattr(server, "CIRCUITS_DIR", tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "train_start", "task": NOT_TASK, "gps": 0, "show": "all", "render_hz": 2})
        started = time.monotonic()
        gens, circuits = [], []
        while time.monotonic() - started < 1.6:
            msg = ws.receive_json()
            if msg["type"] == "train_progress":
                gens += [s["gen"] for s in msg["stats"]]
            elif msg["type"] == "train_circuit":
                circuits.append(msg)
        ws.send_json({"type": "train_pause"})
        ws.send_json({"type": "train_save", "name": "My NOT!"})
        # Pausing finishes the generation in progress and flushes it before the save reply.
        while (msg := ws.receive_json())["type"] != "saved":
            if msg["type"] == "train_progress":
                gens += [s["gen"] for s in msg["stats"]]
            elif msg["type"] == "train_circuit":
                circuits.append(msg)
        assert msg == {"type": "saved", "name": "my_not"}
        assert gens == list(range(len(gens))) and len(gens) > 10    # every generation reported once
        assert 1 <= len(circuits) <= 6                              # but only ~2 drawn per second
        assert len(circuits[0]["frames"]) == 2 and circuits[0]["rows_total"] == 2
        assert (tmp_path / "my_not.json").is_file()
        ws.send_json({"type": "train_start", "task": {**NOT_TASK, "inputs": []}})
        assert ws.receive_json()["type"] == "error"


def test_clicking_a_comparator_switches_its_mode():
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "load", "name": "comparator_and_not"})
        ws.receive_json()
        ws.send_json({"type": "toggle", "pos": [4, 0, 2]})
        state = ws.receive_json()
        comp = next(b for b in state["circuit"]["blocks"] if b["type"] == "comparator")
        assert comp["mode"] == "subtract"
