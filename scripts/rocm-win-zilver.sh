#!/usr/bin/env bash
# Run ZILVER ITSELF on the Radeon 8060S through the native-Windows ROCm interpreter that
# scripts/rocm-win.sh built. Driven from WSL; nothing is typed on the Windows side.
#
#   bash scripts/rocm-win-zilver.sh                 # 20 -> 31 qubits
#   FROM=28 TO=32 bash scripts/rocm-win-zilver.sh
#
# WHY THERE IS NO CODE CHANGE AND NO INSTALL
#   ROCm's torch reports itself as `cuda`, so _array.py's existing torch backend selects it on
#   the first branch -- the DirectML seam we built happens to be exactly the seam ROCm needs.
#   And because the device string is not `privateuseone`, HAS_COMPLEX stays True: the real-pair
#   lifting is bypassed and the statevector is carried as complex64, as it should be.
#   zilver's only runtime dependency is numpy, and it is a src-layout package, so the source
#   tree plus a cwd is a complete installation. Nothing is built, nothing is pinned, and the
#   ROCm venv is left exactly as rocm-win.sh made it.
set -uo pipefail

FROM=${FROM:-20}
TO=${TO:-31}

# Resolve the repo BEFORE moving: $BASH_SOURCE is relative to the invoking cwd, so the
# `cd /mnt/c` below would strand it. (powershell.exe warns and can refuse from a \\wsl.localhost
# cwd, which is why the cd has to happen at all.)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || { echo "STOP: cannot resolve repo root"; exit 1; }
[ -d "$REPO/src/zilver" ] || { echo "STOP: no src/zilver under $REPO"; exit 1; }

cd /mnt/c 2>/dev/null || { echo "STOP: no /mnt/c -- is this WSL with drive interop on?"; exit 1; }
command -v powershell.exe >/dev/null || { echo "STOP: powershell.exe not on PATH"; exit 1; }

WINHOME_W=$(powershell.exe -NoProfile -NonInteractive -Command '[Console]::Out.Write($env:USERPROFILE)' 2>/dev/null | tr -d '\r')
WINHOME=$(wslpath -u "$WINHOME_W")
ROOT="$WINHOME/siriusq-rocm"
ROOT_W="$WINHOME_W\\siriusq-rocm"
[ -d "$ROOT" ] || { echo "STOP: $ROOT_W not found -- run scripts/rocm-win.sh first"; exit 1; }

# Ship the source across. Copy rather than run over \\wsl.localhost: a UNC cwd makes
# powershell warn and can make python's relative sys.path insert miss.
DEST="$ROOT/zilver"
echo "-- copying source -> $ROOT_W\\zilver"
rm -rf "$DEST"; mkdir -p "$DEST/scripts"
cp -r "$REPO/src" "$DEST/src"
cp "$REPO/scripts/gpu.py" "$DEST/scripts/gpu.py"
find "$DEST" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
echo "   $(find "$DEST" -name '*.py' | wc -l) files"

cat > "$ROOT/runzilver.ps1" <<'PS1'
# Widths arrive as ARGUMENTS, not environment variables. WSLENV can drop them silently, and
# `$env:TO = $null` DELETES the variable rather than setting it -- so gpu.py fell back to its
# own default of 28 while the bash side cheerfully announced 20->31. Arguments cannot do that.
param([int]$From = 20, [int]$To = 31)
$ErrorActionPreference = 'Continue'
$root = Join-Path $env:USERPROFILE 'siriusq-rocm'
$py   = Join-Path $root 'venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = Join-Path $root 'pyembed\python.exe' }
if (-not (Test-Path $py)) { Write-Host "FAIL: no ROCm interpreter under $root"; exit 2 }
Write-Host "-- interpreter: $py"

# The AMD index carries no numpy -- that is the "Failed to initialize NumPy" warning torch
# prints. Pull it from PyPI. numpy has no dependencies, so this cannot disturb the torch pin.
& $py -c "import numpy" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "-- installing numpy from PyPI (the ROCm index does not carry it)"
  & $py -m pip install --disable-pip-version-check --quiet numpy
}

$env:ZILVER_BACKEND = 'torch'
$env:FROM = "$From"
$env:TO   = "$To"
Set-Location (Join-Path $root 'zilver')      # gpu.py does sys.path.insert(0, "src")
& $py scripts\gpu.py
exit $LASTEXITCODE
PS1

echo "-- running zilver on the ROCm device, $FROM -> $TO qubits"
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass \
  -File "$ROOT_W\\runzilver.ps1" "$FROM" "$TO" 2>&1 | tr -d '\r'

cat <<'NOTES'

Read the header, not just the table:
  device    : should be `cuda` -- that IS the Radeon through ROCm, not an NVIDIA card
  complex64 : True means the real-pair lifting is off and the state is genuinely complex64

Rerun a single width without recopying:
  FROM=31 TO=31 bash scripts/rocm-win-zilver.sh
NOTES
