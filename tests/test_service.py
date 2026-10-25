"""
Node tunnel and launchd service, without opening a real tunnel or touching
the real LaunchAgents folder.
"""

import os
import plistlib
import subprocess
import time

import pytest

import sys; sys.path.insert(0, "src")
from zilver import service


def _fake_cloudflared(tmp_path, monkeypatch, body: str):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "cloudflared"
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setattr(service, "ZILVER_DIR", tmp_path / "zilver")


def test_tunnel_returns_its_url(tmp_path, monkeypatch):
    _fake_cloudflared(tmp_path, monkeypatch,
                      'echo "INF |  https://brave-new-node-1.trycloudflare.com  |" >&2\nsleep 30\n')
    proc, url = service.start_tunnel(7700, timeout=5)
    try:
        assert url == "https://brave-new-node-1.trycloudflare.com"
        assert proc.poll() is None
    finally:
        proc.terminate()


def test_tunnel_that_dies_raises(tmp_path, monkeypatch):
    _fake_cloudflared(tmp_path, monkeypatch, "exit 1\n")
    with pytest.raises(RuntimeError, match="did not come up"):
        service.start_tunnel(7700, timeout=5)


def test_missing_cloudflared_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="brew install cloudflared"):
        service.start_tunnel(7700)


def test_node_exits_when_tunnel_dies(monkeypatch):
    exited = []
    monkeypatch.setattr(service.os, "_exit", lambda code: exited.append(code))
    proc = subprocess.Popen(["/bin/sh", "-c", "exit 3"])
    service.exit_with_tunnel(proc)
    for _ in range(50):
        if exited:
            break
        time.sleep(0.05)
    assert exited == [1]


def test_service_plist_runs_start_awake_and_restarts():
    p = service.service_plist(["--registry", "https://r.example", "--port", "7700"])
    assert p["Label"] == service.SERVICE_LABEL
    assert p["ProgramArguments"][:2] == ["/usr/bin/caffeinate", "-i"]
    assert p["ProgramArguments"][3:] == ["start", "--registry", "https://r.example",
                                         "--port", "7700"]
    assert p["RunAtLoad"] and p["KeepAlive"]
    assert "/opt/homebrew/bin" in p["EnvironmentVariables"]["PATH"]


def test_install_and_uninstall(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(service.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(service, "ZILVER_DIR", tmp_path / ".zilver")
    monkeypatch.setattr(service, "_launchctl",
                        lambda *a, check=False: calls.append(a) or
                        subprocess.CompletedProcess(a, 0, "", ""))

    path = service.install_service(["--registry", "https://r.example"])
    assert path == tmp_path / "Library/LaunchAgents" / f"{service.SERVICE_LABEL}.plist"
    assert plistlib.loads(path.read_bytes())["ProgramArguments"][-1] == "https://r.example"
    assert [c[0] for c in calls] == ["bootout", "bootstrap"]
    assert calls[1][1] == f"gui/{os.getuid()}"

    assert service.uninstall_service() is True
    assert not path.exists()
    assert service.uninstall_service() is False
