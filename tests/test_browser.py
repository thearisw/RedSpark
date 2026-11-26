"""redspark.browser: the entry point the web demo calls inside Pyodide."""

import asyncio
import json

from redspark import browser


def test_browser_session_lists_loads_and_steps():
    async def run():
        got = []
        browser.start(lambda text: got.append(json.loads(text)))
        for msg in ({"type": "list"}, {"type": "load", "name": "not_gate"}, {"type": "step"}, {"type": "nope"}):
            await browser.receive(json.dumps(msg))
        return got

    circuits, loaded, stepped, error = asyncio.run(run())
    assert circuits["type"] == "circuits" and "not_gate" in circuits["names"]
    assert loaded["type"] == "state" and loaded["name"] == "not_gate"
    assert stepped["circuit"]["ticks"] == loaded["circuit"]["ticks"] + 1
    assert error["type"] == "error"
