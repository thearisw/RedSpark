"""Runs a viewer session inside the browser (Pyodide), for the static web demo.

The web worker (``web/src/worker.ts``) loads this package into Pyodide, then
calls ``start(post)`` once, where ``post(text)`` hands one JSON message to the
viewer, and ``receive(text)`` for every message from it. The messages are the
same as over the app's WebSocket (see ``redspark/server.py``), plus
``{"type": "list"}`` -> ``{"type": "circuits", "names": [...]}``, which stands
in for ``GET /api/circuits``. Circuits saved from training live in Pyodide's
in-memory file system, so they last until the page is reloaded.
"""

from __future__ import annotations

import asyncio
import json

from .session import Session, circuit_names

_session: Session | None = None
_post = None
_wake: asyncio.Event | None = None


async def _send(msg: dict) -> None:
    _post(json.dumps(msg))


def start(post) -> None:
    global _session, _post, _wake
    _post = post
    _session = Session(_send, threaded=False)
    _wake = asyncio.Event()
    asyncio.ensure_future(_play_loop())


async def _play_loop() -> None:
    """Steps the circuit while "play" is on, like the server's WebSocket loop."""
    while True:
        s = _session
        if s.playing and s.circuit is not None:
            try:
                await asyncio.wait_for(_wake.wait(), 1.0 / s.tps)
            except asyncio.TimeoutError:
                s.circuit.step()
                await s.send_state()
        else:
            await _wake.wait()
        _wake.clear()


async def receive(text: str) -> None:
    msg = json.loads(text)
    try:
        if msg.get("type") == "list":
            await _send({"type": "circuits", "names": circuit_names()})
        else:
            await _session.handle(msg)
    except (KeyError, ValueError, TypeError) as exc:
        await _send({"type": "error", "message": str(exc)})
    _wake.set()
