# Zilver on native Windows ROCm — for the AMD Radeon 8060S (gfx1151).
#
# Run in PowerShell (NOT WSL):
#     irm https://raw.githubusercontent.com/Sirius-Quantum/zilver/torch-backend/scripts/windows-rocm.ps1 | iex
#
# Why this exists: WSL2 exposes no /dev/kfd, so ROCm cannot see the GPU from
# Linux at all. DirectML reaches it through /dev/dxg instead, but goes via
# DirectX and is in maintenance mode. Native Windows has real ROCm, and AMD
# publishes wheels built for this exact chip.

$ErrorActionPreference = "Stop"
$root = "$env:USERPROFILE\zilver-rocm"

Write-Host "`n=== python ===" -ForegroundColor Cyan
$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) {
    Write-Host "  python not found. Install 3.12 from python.org or the Store, then re-run." -ForegroundColor Yellow
    exit 1
}
python --version

Write-Host "`n=== clone ===" -ForegroundColor Cyan
if (Test-Path $root) {
    Push-Location $root; git fetch --all -q; git checkout -q torch-backend; git pull -q; Pop-Location
    Write-Host "  updated $root"
} else {
    git clone -q -b torch-backend https://github.com/Sirius-Quantum/zilver.git $root
    Write-Host "  cloned to $root"
}
Set-Location $root

Write-Host "`n=== venv + torch for gfx1151 ===" -ForegroundColor Cyan
python -m venv .venv-rocm
& .\.venv-rocm\Scripts\python.exe -m pip install -q --upgrade pip
# AMD's own index, built for this chip. If it 404s, the fallback below is the
# community build of the same thing.
& .\.venv-rocm\Scripts\python.exe -m pip install -q `
    --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ `
    torch torchvision torchaudio numpy
if ($LASTEXITCODE -ne 0) {
    Write-Host "  AMD nightly index failed; see" -ForegroundColor Yellow
    Write-Host "  https://github.com/scottt/rocm-TheRock/releases/v6.5.0rc-pytorch" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n=== does ROCm see the 8060S? ===" -ForegroundColor Cyan
& .\.venv-rocm\Scripts\python.exe -c @"
import torch
ok = torch.cuda.is_available()
print('  torch', torch.__version__)
print('  device visible:', ok)
if ok:
    print('  name:', torch.cuda.get_device_name(0))
    x = torch.ones(4, dtype=torch.complex64, device='cuda')
    print('  complex64 on device:', (x*(2+1j)).sum().cpu())
else:
    print('  NO DEVICE -- ROCm is installed but cannot see the GPU')
"@

Write-Host "`n=== the measurement ===" -ForegroundColor Cyan
$env:ZILVER_BACKEND = "torch"
& .\.venv-rocm\Scripts\python.exe -m pip install -q pytest
& .\.venv-rocm\Scripts\python.exe scripts\cpu_vs_gpu.py
