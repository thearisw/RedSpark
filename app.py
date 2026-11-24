"""One-click launcher: start the server and open the viewer in its own window.

    python -m redspark.app            (or double-click RedSpark.bat on Windows)

Picks a free local port, starts the server in the background and opens the
viewer as an app window (Edge or Chrome in app mode, else the default browser).
Closing that window quits RedSpark.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from .server import VIEWER_DIR, app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def find_app_browser() -> str | None:
    """A Chromium browser that supports --app windows (Edge ships with Windows)."""
    candidates = []
    for env in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
        base = os.environ.get(env)
        if base:
            candidates += [Path(base) / "Microsoft/Edge/Application/msedge.exe",
                           Path(base) / "Google/Chrome/Application/chrome.exe"]
    candidates += [Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
                   Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")]
    for c in candidates:
        if c.is_file():
            return str(c)
    for name in ("msedge", "google-chrome", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


def profile_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache")
    path = base / "RedSpark" / "window"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the RedSpark viewer.")
    parser.add_argument("--port", type=int, default=0, help="port to use (default: any free one)")
    parser.add_argument("--browser", action="store_true", help="open a normal browser tab instead")
    args = parser.parse_args()

    if not (VIEWER_DIR / "index.html").is_file():
        sys.exit("The viewer is not built. Run `npm install` and `npm run build` in web/ first.")

    port = args.port or free_port()
    url = f"http://127.0.0.1:{port}/"
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        if not thread.is_alive():
            sys.exit("The server could not start.")
        time.sleep(0.05)

    print(f"RedSpark is running at {url}")
    browser = None if args.browser else find_app_browser()
    if browser:
        # A separate profile gives the window its own process, so we can tell when it closes.
        window = subprocess.Popen([browser, f"--app={url}", f"--user-data-dir={profile_dir()}",
                                   "--window-size=1280,800", "--no-first-run",
                                   "--no-default-browser-check"])
        print("Close the RedSpark window to quit.")
        try:
            window.wait()
        except KeyboardInterrupt:
            window.terminate()
    else:
        webbrowser.open(url)
        print("Press Ctrl+C (or close this console) to quit.")
        try:
            thread.join()
        except KeyboardInterrupt:
            pass
    server.should_exit = True
    thread.join(timeout=3)


if __name__ == "__main__":
    main()
