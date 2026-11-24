"""FastAPI backend: serves circuits and streams simulation state over a WebSocket.

    uvicorn redspark.server:app --reload

WebSocket protocol on /ws (JSON messages):

client -> server
    {"type": "load", "name": "not_gate"}        open circuits/<name>.json
    {"type": "toggle", "pos": [x, y, z]}         flip a lever
    {"type": "step"}                             advance one tick
    {"type": "play", "on": true, "tps": 10}      run continuously at tps ticks/second
    {"type": "reset"}                            reload the current circuit

server -> client
    {"type": "state", "name": ..., "circuit": {size, ticks, blocks: [...]}}
    {"type": "error", "message": ...}
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .sim import Block, Circuit

CIRCUITS_DIR = Path(__file__).resolve().parents[1] / "circuits"

app = FastAPI(title="RedSpark")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"],
                   allow_headers=["*"])


def circuit_names() -> list[str]:
    return sorted(p.stem for p in CIRCUITS_DIR.glob("*.json"))


def load_circuit(name: str) -> Circuit:
    if name not in circuit_names():  # also blocks path tricks like "../x"
        raise KeyError(name)
    return Circuit.from_dict(json.loads((CIRCUITS_DIR / f"{name}.json").read_text()))


@app.get("/api/circuits")
def list_circuits() -> list[str]:
    return circuit_names()


@app.get("/api/circuits/{name}")
def get_circuit(name: str) -> dict:
    try:
        return load_circuit(name).to_dict()
    except KeyError:
        raise HTTPException(404, f"no circuit named {name}")


class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.name: str | None = None
        self.circuit: Circuit | None = None
        self.playing = False
        self.tps = 10.0

    async def send_state(self) -> None:
        await self.ws.send_json({"type": "state", "name": self.name, "playing": self.playing,
                                 "circuit": self.circuit.to_dict() if self.circuit else None})

    async def handle(self, msg: dict) -> None:
        kind = msg.get("type")
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


@app.websocket("/ws")
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
                await ws.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        pass
