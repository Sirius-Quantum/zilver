"""Running a node unattended: a Cloudflare tunnel for reachability, and a
launchd agent so the node survives sleep, crashes and reboots (macOS)."""

from __future__ import annotations

import os
import plistlib
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ZILVER_DIR    = Path.home() / ".zilver"
SERVICE_LABEL = "com.siriusquantum.zilver-node"

_TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


# ---------------------------------------------------------------------------
# Tunnel
# ---------------------------------------------------------------------------

def start_tunnel(port: int, timeout: float = 30.0) -> tuple[subprocess.Popen, str]:
    """
    Publish ``https://localhost:<port>`` through a Cloudflare quick tunnel.

    A home router accepts no incoming connections, so a node behind one cannot
    be reached at its own address. ``cloudflared`` dials out to Cloudflare
    instead and returns a public HTTPS URL that forwards to the port.

    Returns the running ``cloudflared`` process and the URL. The URL is new on
    every start, so the node re-registers with it each time.

    Raises RuntimeError when ``cloudflared`` is missing or no URL appears
    within *timeout* seconds.
    """
    exe = shutil.which("cloudflared")
    if exe is None:
        raise RuntimeError("cloudflared is not installed: brew install cloudflared")

    ZILVER_DIR.mkdir(exist_ok=True)
    log_path = ZILVER_DIR / f"tunnel-{port}.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [exe, "tunnel", "--no-autoupdate",
             "--url", f"https://localhost:{port}", "--no-tls-verify"],
            stdout=log, stderr=subprocess.STDOUT,
        )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = _TUNNEL_URL.search(log_path.read_text(errors="replace"))
        if found:
            return proc, found.group(0)
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"the tunnel did not come up; see {log_path}")


def exit_with_tunnel(proc: subprocess.Popen) -> None:
    """
    End this process if the tunnel dies, so the service manager restarts the
    node and the node re-registers with a fresh tunnel. A node whose tunnel is
    gone would keep heartbeating while no client can reach it.
    """
    def _watch() -> None:
        code = proc.wait()
        print(f"Tunnel exited (code {code}); stopping the node so it restarts.",
              file=sys.stderr, flush=True)
        os._exit(1)

    threading.Thread(target=_watch, daemon=True).start()


# ---------------------------------------------------------------------------
# launchd service
# ---------------------------------------------------------------------------

def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_LABEL}.plist"


def _launchctl(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, check=check)


def service_plist(start_args: list[str]) -> dict:
    """The launchd agent that runs ``zilver-node start <start_args>``."""
    exe = shutil.which("zilver-node") or os.path.abspath(sys.argv[0])
    log = str(ZILVER_DIR / "node.log")
    # launchd starts with a bare PATH; cloudflared lives in Homebrew's prefix.
    path_dirs = [os.path.dirname(exe), "/opt/homebrew/bin", "/usr/local/bin",
                 "/usr/bin", "/bin"]
    return {
        "Label": SERVICE_LABEL,
        # caffeinate -i keeps the Mac from idling to sleep while the node runs.
        "ProgramArguments": ["/usr/bin/caffeinate", "-i", exe, "start", *start_args],
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 30,
        "StandardOutPath": log,
        "StandardErrorPath": log,
        "EnvironmentVariables": {
            "PATH": ":".join(dict.fromkeys(path_dirs)),
            "PYTHONUNBUFFERED": "1",
        },
    }


def install_service(start_args: list[str]) -> Path:
    """Write the agent and (re)load it. Returns the plist path."""
    ZILVER_DIR.mkdir(exist_ok=True)
    path = _plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        plistlib.dump(service_plist(start_args), f)
    domain = f"gui/{os.getuid()}"
    _launchctl("bootout", domain, str(path))          # not loaded yet is fine
    _launchctl("bootstrap", domain, str(path), check=True)
    return path


def uninstall_service() -> bool:
    """Unload and delete the agent. Returns False if it was not installed."""
    path = _plist_path()
    if not path.exists():
        return False
    _launchctl("bootout", f"gui/{os.getuid()}", str(path))
    path.unlink()
    return True
