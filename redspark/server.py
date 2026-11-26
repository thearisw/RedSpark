"""Backend: serves circuits and streams simulation state over a WebSocket.

    python -m uvicorn redspark.server:app --port 8000

It also serves the prebuilt viewer from redspark/viewer/ at "/", so no Node is
needed to use the app (see redspark/app.py for the one-click launcher).
Built on Starlette (pure Python, no compiled dependencies).

WebSocket protocol on /ws (JSON messages):

client -> server
    {"type": "load", "name": "not_gate"}        open circuits/<name>.json
    {"type": "toggle", "pos": [x, y, z]}         flip a lever
    {"type": "step"}                             advance one game tick
    {"type": "play", "on": true, "tps": 20}      run at tps game ticks/second (20 = real time)
    {"type": "reset"}                            reload the current circuit

  truth-table training (see redspark/designer.py)
    {"type": "train_start", "task": {size, inputs: [{label, pos}], outputs: [{label, pos}],
       rows: [{in: [bool...], out: [bool|null...]}]}, "max_blocks": 12, "method": "layout" | "network",
       "gps": 10, "show": "success"|"best"|"all"|"samples", "render_hz": 4}
                                                 new population, start running
    {"type": "train_pause"} / {"type": "train_resume"}
    {"type": "train_config", "gps": 0, "show": ..., "render_hz": ...}   gps 0 = as fast as possible
    {"type": "train_show_best"}                  send the best circuit found so far
    {"type": "train_save", "name": "my_gate"}    save the best circuit to circuits/<name>.json
    {"type": "train_results"}                    send the list of best circuits so far
    {"type": "load_data", "circuit": {...}, "name": "Result 3"}
                                                 play a circuit dict (e.g. a training result) like a saved one

server -> client
    {"type": "state", "name": ..., "circuit": {size, ticks, blocks: [...]}}
    {"type": "train_progress", "running", "gen", "rows_total", "max_blocks", "samples", "simulated",
       "stats": [{gen, best, mean, solved, new_best, blocks, samples}],
       "best_ever": {gen, rows, blocks, dims, delay}}             stats for every generation since the last message
                                                 samples: designs scored so far; simulated: those not
                                                 taken from the cache (null for the network method)
                                                 max_blocks: the budget in use (raised if the asked one
                                                 is too small for a wire to reach every output)
    {"type": "train_circuit", "gen", "kind", "sample", "rows_passed", "rows_total", "passed", "got", "blocks", "dims", "delay",
       "frames": [circuit per truth-table row]}   at most render_hz per second; "sample" is the
                                                 design's number in the run for show "samples", else null
    {"type": "saved", "name": ...}
    {"type": "train_results", "results": [{id, gen, rows, rows_total, blocks, dims, delay, circuit}]}
                                                 every new best of this run, oldest first (the last 50)
    {"type": "error", "message": ...}
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from .session import CIRCUITS_DIR, SHOW_MODES, Session, Training, circuit_names, load_circuit  # noqa: F401

VIEWER_DIR = Path(__file__).resolve().parent / "viewer"  # built by `npm run build` in web/


async def list_circuits(request: Request) -> JSONResponse:
    return JSONResponse(circuit_names())


async def get_circuit(request: Request) -> JSONResponse:
    name = request.path_params["name"]
    try:
        return JSONResponse(load_circuit(name).to_dict())
    except KeyError:
        return JSONResponse({"detail": f"no circuit named {name}"}, status_code=404)


async def websocket(ws: WebSocket) -> None:
    await ws.accept()
    session = Session(ws.send_json)
    try:
        while True:
            timeout = 1.0 / session.tps if session.playing and session.circuit else None
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout)
            except asyncio.TimeoutError:
                session.circuit.step()
                await session.send_state()
                continue
            try:
                await session.handle(msg)
            except (KeyError, ValueError, TypeError) as exc:
                await session.send({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        pass
    finally:
        await session.training.shutdown()


routes = [
    Route("/api/circuits", list_circuits),
    Route("/api/circuits/{name}", get_circuit),
    WebSocketRoute("/ws", websocket),
]
if VIEWER_DIR.is_dir():
    routes.append(Mount("/", StaticFiles(directory=VIEWER_DIR, html=True), name="viewer"))

app = Starlette(
    routes=routes,
    middleware=[Middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                           allow_methods=["*"], allow_headers=["*"])],
)
