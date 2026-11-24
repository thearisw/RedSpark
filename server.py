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
       rows: [{in: [bool...], out: [bool|null...]}]}, "max_blocks": 12,
       "gps": 10, "show": "success"|"best"|"all", "render_hz": 4}
                                                 new population, start running
    {"type": "train_pause"} / {"type": "train_resume"}
    {"type": "train_config", "gps": 0, "show": ..., "render_hz": ...}   gps 0 = as fast as possible
    {"type": "train_show_best"}                  send the best circuit found so far
    {"type": "train_save", "name": "my_gate"}    save the best circuit to circuits/<name>.json

server -> client
    {"type": "state", "name": ..., "circuit": {size, ticks, blocks: [...]}}
    {"type": "train_progress", "running", "gen", "rows_total", "stats": [{gen, best, mean, solved, new_best}],
       "best_ever": {gen, rows}}             stats for every generation since the last message
    {"type": "train_circuit", "gen", "kind", "rows_passed", "rows_total", "passed", "got", "blocks",
       "frames": [circuit per truth-table row]}   at most render_hz per second
    {"type": "saved", "name": ...}
    {"type": "error", "message": ...}
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from .designer import Trainer, TruthTableTask, evaluate
from .sim import Block, Circuit

CIRCUITS_DIR = Path(__file__).resolve().parents[1] / "circuits"
VIEWER_DIR = Path(__file__).resolve().parent / "viewer"  # built by `npm run build` in web/


def circuit_names() -> list[str]:
    return sorted(p.stem for p in CIRCUITS_DIR.glob("*.json"))


def load_circuit(name: str) -> Circuit:
    if name not in circuit_names():  # also blocks path tricks like "../x"
        raise KeyError(name)
    return Circuit.from_dict(json.loads((CIRCUITS_DIR / f"{name}.json").read_text()))


async def list_circuits(request: Request) -> JSONResponse:
    return JSONResponse(circuit_names())


async def get_circuit(request: Request) -> JSONResponse:
    name = request.path_params["name"]
    try:
        return JSONResponse(load_circuit(name).to_dict())
    except KeyError:
        return JSONResponse({"detail": f"no circuit named {name}"}, status_code=404)


class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.name: str | None = None
        self.circuit: Circuit | None = None
        self.playing = False
        self.tps = 20.0
        self._send_lock = asyncio.Lock()
        self.training = Training(self)

    async def send(self, msg: dict) -> None:
        async with self._send_lock:  # the training task sends too
            await self.ws.send_json(msg)

    async def send_state(self) -> None:
        await self.send({"type": "state", "name": self.name, "playing": self.playing,
                         "circuit": self.circuit.to_dict() if self.circuit else None})

    async def handle(self, msg: dict) -> None:
        kind = msg.get("type")
        if isinstance(kind, str) and kind.startswith("train_"):
            self.playing = False
            await self.training.handle(kind, msg)
            return
        if kind in ("load", "reset"):
            name = msg.get("name", self.name)
            self.circuit = load_circuit(name)
            self.name = name
        elif self.circuit is None:
            raise ValueError("load a circuit first")
        elif kind == "toggle":
            pos = tuple(int(v) for v in msg["pos"])
            if self.circuit.get(pos) != Block.LEVER:
                raise ValueError("only levers can be toggled")
            self.circuit.toggle(pos)
        elif kind == "step":
            self.circuit.step()
        elif kind == "play":
            self.playing = bool(msg.get("on", True))
            self.tps = min(60.0, max(0.5, float(msg.get("tps", self.tps))))
        else:
            raise ValueError(f"unknown message type {kind!r}")
        await self.send_state()


SHOW_MODES = ("success", "best", "all")


class Training:
    """Runs a Trainer generation by generation and streams the results.

    Every generation's numbers are sent (batched about 10 times a second), but
    circuits are sent at most ``render_hz`` times a second, so "show every
    generation" at full speed never asks the viewer to draw thousands of
    circuits. Generations in between still appear in the progress chart.
    """

    def __init__(self, session: Session):
        self.session = session
        self.trainer: Trainer | None = None
        self.task: asyncio.Task | None = None
        self.running = False
        self.gps = 10.0        # generations per second, 0 = as fast as possible
        self.show = "best"
        self.render_hz = 4.0
        self._wake = asyncio.Event()  # cuts the between-generations wait short on pause

    def configure(self, msg: dict) -> None:
        if "gps" in msg:
            self.gps = min(1000.0, max(0.0, float(msg["gps"])))
        if "show" in msg:
            if msg["show"] not in SHOW_MODES:
                raise ValueError(f"show must be one of {SHOW_MODES}")
            self.show = msg["show"]
        if "render_hz" in msg:
            self.render_hz = min(10.0, max(0.2, float(msg["render_hz"])))

    async def handle(self, kind: str, msg: dict) -> None:
        if kind == "train_start":
            task = TruthTableTask.from_dict(msg["task"])
            self.configure(msg)
            await self.stop()
            max_blocks = min(40, max(1, int(msg.get("max_blocks", 12))))
            self.trainer = Trainer(task, max_blocks=max_blocks)
            self.start()
        elif kind == "train_config":
            self.configure(msg)
        elif self.trainer is None:
            raise ValueError("start training first")
        elif kind == "train_pause":
            await self.stop()
        elif kind == "train_resume":
            self.start()
        elif kind == "train_show_best":
            if self.trainer.best is not None:
                await self.send_circuit(self.trainer.best, self.trainer.best_gen, "best_ever")
        elif kind == "train_save":
            await self.save(str(msg.get("name", "")))
            return
        else:
            raise ValueError(f"unknown message type {kind!r}")
        await self.send_progress([])

    def start(self) -> None:
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self.loop())

    async def stop(self) -> None:
        self.running = False
        self._wake.set()
        if self.task is not None:
            await self.task  # finishes the generation in progress, then flushes
            self.task = None

    async def save(self, name: str) -> None:
        if self.trainer.best is None:
            raise ValueError("nothing trained yet")
        name = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")[:40] or "designed"
        data = self.trainer.best.circuit.to_dict(include_state=False)
        (CIRCUITS_DIR / f"{name}.json").write_text(json.dumps(data, indent=1) + "\n")
        await self.session.send({"type": "saved", "name": name})

    async def send_progress(self, stats: list[dict]) -> None:
        t = self.trainer
        await self.session.send({
            "type": "train_progress", "running": self.running, "gen": t.gen if t else 0,
            "rows_total": t.n_rows if t else 0, "stats": stats,
            "best_ever": {"gen": t.best_gen, "rows": t.best.result.rows_passed} if t and t.best else None,
        })

    async def send_circuit(self, ind, gen: int, kind: str) -> None:
        res = evaluate(ind.circuit, self.trainer.task, frames=True)
        await self.session.send({
            "type": "train_circuit", "gen": gen, "kind": kind, "rows_passed": res.rows_passed,
            "rows_total": len(res.passed), "passed": res.passed, "got": res.got,
            "blocks": ind.blocks, "frames": res.frames,
        })

    def wanted(self, stats) -> bool:
        if self.show == "all":
            return True
        if self.show == "best":
            return stats.new_best
        return stats.best_rows == self.trainer.n_rows

    async def loop(self) -> None:
        pending: list[dict] = []
        candidate = None
        last_stats = last_circuit = 0.0
        while self.running:
            self._wake.clear()
            started = time.monotonic()
            stats = await asyncio.to_thread(self.trainer.generation)
            pending.append(stats.to_dict())
            if self.wanted(stats):
                candidate = stats
            now = time.monotonic()
            if candidate is not None and now - last_circuit >= 1.0 / self.render_hz:
                await self.send_circuit(candidate.best, candidate.gen, self.show)
                candidate, last_circuit = None, now
            if now - last_stats >= 0.1:
                await self.send_progress(pending)
                pending, last_stats = [], now
            wait = (1.0 / self.gps - (time.monotonic() - started)) if self.gps > 0 else 0.0
            try:  # also lets messages from the viewer in
                await asyncio.wait_for(self._wake.wait(), max(0.0, wait))
            except asyncio.TimeoutError:
                pass
        if candidate is not None:
            await self.send_circuit(candidate.best, candidate.gen, self.show)
        await self.send_progress(pending)


async def websocket(ws: WebSocket) -> None:
    await ws.accept()
    session = Session(ws)
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
        session.training.running = False


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
