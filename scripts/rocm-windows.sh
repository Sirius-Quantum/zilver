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
ps 'python --version' || {
  echo "  no python. Install 3.12 from the Microsoft Store, then re-run."; exit 1; }

echo
echo "=== 2. clone ==="
ps '$r="$env:USERPROFILE\zilver-rocm"
    if (Test-Path $r) { cd $r; git fetch -q --all; git checkout -q torch-backend; git pull -q }
    else { git clone -q -b torch-backend https://github.com/Sirius-Quantum/zilver.git $r }
    Write-Output "  at $r"'

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
