#!/usr/bin/env bash
# Set up native Windows ROCm from the WSL terminal.
#
#     bash scripts/rocm-windows.sh
#
# WSL can invoke Windows executables, so everything below runs on the Windows
# side while being typed in the shell that works. No file copying, no long
# paths, no PowerShell quoting.
#
# Why native Windows: ROCm needs /dev/kfd, which WSL2 does not expose, so the
# GPU is unreachable from Linux. DirectML gets there through DirectX instead,
# but has no complex dtype (so the state must be split into real pairs, roughly
# doubling memory traffic), no kernel fusion, and a two-second watchdog that
# capped us at 26 qubits. ROCm has none of those.
set -uo pipefail

command -v powershell.exe >/dev/null || {
  echo "  powershell.exe not on PATH -- WSL interop is off. Enable it or run"
  echo "  the steps on the Windows side manually."; exit 1; }

ps() { powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$1" 2>&1 | tr -d '\r'; }

echo "=== 1. python on the Windows side ==="
# A bare `python` on Windows may be the Microsoft Store alias stub rather than
# an interpreter, so check for a real version string rather than exit status.
ver=$(ps 'python --version' | grep -o 'Python 3\.[0-9]*' || true)
if [ -z "$ver" ]; then
  echo "  not installed -- fetching the official installer (per-user, no admin)"
  # Not winget: its package source is broken on this box (0x8a15000f, "data
  # required by the source is missing") and repairing it wants admin. The
  # python.org installer needs neither -- InstallAllUsers=0 keeps it per-user.
  ps '$u="https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe"
      $f="$env:TEMP\py312.exe"
      Write-Output "  downloading..."
      Invoke-WebRequest -Uri $u -OutFile $f -UseBasicParsing
      Write-Output "  installing (quiet, per-user)..."
      Start-Process -FilePath $f -ArgumentList "/quiet","InstallAllUsers=0","PrependPath=1","Include_pip=1" -Wait
      Write-Output "  done"' | tail -4
  ver=$(ps 'python --version' | grep -o 'Python 3\.[0-9]*' || true)
  [ -z "$ver" ] && ver=$(ps '& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" --version' | grep -o 'Python 3\.[0-9]*' || true)
fi
[ -z "$ver" ] && { echo "  still no python on the Windows side; install it there and re-run"; exit 1; }
echo "  $ver"

echo
echo "=== 2. copy the repo across ==="
# Not `git clone`: git is a WSL-side tool here and is not necessarily installed
# on Windows. The working tree is already right here, so copy it rather than
# add a dependency. Only src/ and scripts/ are needed to run.
win_home=$(ps '$env:USERPROFILE' | tr -d '[:space:]')
[ -z "$win_home" ] && { echo "  could not read USERPROFILE"; exit 1; }
dest="/mnt/c${win_home#C:}"
dest="${dest//\\//}/zilver-rocm"
echo "  windows home : $win_home"
echo "  copying to   : $dest"
mkdir -p "$dest"
cp -r src scripts pyproject.toml "$dest"/ 2>/dev/null
ls "$dest/scripts/gpu.py" >/dev/null 2>&1 \
  && echo "  ok, $(find "$dest/src" -name '*.py' | wc -l | tr -d ' ') python files" \
  || { echo "  copy failed"; exit 1; }

echo
echo "=== 3. torch built for gfx1151 (AMD index) ==="
ps '$r="$env:USERPROFILE\zilver-rocm"; cd $r
    python -m venv .venv-rocm
    .\.venv-rocm\Scripts\python.exe -m pip install -q --upgrade pip
    .\.venv-rocm\Scripts\python.exe -m pip install -q --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch numpy
    if ($LASTEXITCODE -ne 0) { Write-Output "  AMD index failed -- see github.com/scottt/rocm-TheRock/releases" }'

echo
echo "=== 4. does ROCm see the 8060S, and does complex work there? ==="
ps '$r="$env:USERPROFILE\zilver-rocm"; cd $r
    .\.venv-rocm\Scripts\python.exe -c "import torch; ok=torch.cuda.is_available(); print(\"  torch\",torch.__version__); print(\"  device visible:\",ok); print(\"  name:\",torch.cuda.get_device_name(0)) if ok else None; x=torch.ones(4,dtype=torch.complex64,device=\"cuda\") if ok else None; print(\"  complex64 on device:\",(x*(2+1j)).sum().cpu()) if ok else None"'

echo
echo "=== 5. the measurement ==="
ps '$r="$env:USERPROFILE\zilver-rocm"; cd $r
    $env:ZILVER_BACKEND="torch"
    .\.venv-rocm\Scripts\python.exe scripts\gpu.py'
