#!/usr/bin/env bash
# Native Windows ROCm on the Strix Halo box (Radeon 8060S = gfx1151), driven entirely from WSL.
#
#   bash deploy/rocm_win.sh 2>&1 | tee ~/rocm_win.log
#
# One short command over the KVM. Nothing else gets typed on the Windows side: every file
# this needs is written from WSL onto the Windows profile, and PowerShell is only ever invoked
# as `-File <script>`, never with an inline command string, so nothing can be mangled by a
# dropped character in a paste.
#
# WHAT THIS INSTALLS, AND WHY IT NEEDS NO ADMINISTRATOR
#   AMD's ROCm 10.0.0 ships an official Windows path for gfx1151 that is pure pip:
#     https://rocm.docs.amd.com/en/latest/install/rocm.html  (Windows tab, gfx1151 Ryzen APU)
#   The wheels at https://stable.repo.amd.com/rocm/whl-next/ carry the whole user-mode stack --
#   HIP runtime, compiler, math libraries and the precompiled gfx1151 kernel pack. There is no
#   HIP SDK, no MSI, no service. Everything lands in a per-user venv.
#   The ONE thing that is not per-user is the kernel-mode driver: ROCm 10.0.0 requires
#   AMD Software: Adrenalin Edition 26.8.1 (or Windows CDE/CPR 26.10.32). Installing or
#   upgrading THAT needs administrator. Phase 0 below reads the installed version and stops
#   with a plain verdict if it is older, because that is the only admin gate in the whole path.
#
# TRAPS THIS SCRIPT DEFUSES (each one has already cost a day somewhere)
#   * The Microsoft Store `python.exe` alias stub in %LOCALAPPDATA%\Microsoft\WindowsApps
#     prints an advert and exits 0. We never invoke a bare `python`; every call is an absolute
#     path, and we assert the resolved path does not contain WindowsApps.
#   * python.org's installer defaults InstallLauncherAllUsers=1, which raises UAC and makes a
#     /quiet install fail silently for a non-admin. We pass Include_launcher=0 AND
#     InstallLauncherAllUsers=0, and an explicit TargetDir so discovery never depends on PATH.
#   * PrependPath=1 cannot help anyway: a PATH edit does not reach an already-running WSL
#     interop session. Explicit TargetDir removes the question.
#   * USERPROFILE casing (C:\Users\AMD-RDP vs amd-rdp) is never guessed -- we ask Windows for
#     $env:USERPROFILE and run it through wslpath.
#   * powershell.exe launched from a \\wsl.localhost cwd emits a UNC warning and can refuse;
#     we cd to /mnt/c first.
#   * git is NOT installed on Windows and is not needed: the Windows profile is a normal WSL
#     directory, so files are copied, not cloned.

set -uo pipefail   # deliberately NOT -e: a failing phase is an ANSWER, and we want the
                   # remaining diagnostics printed rather than a bare exit.

PY_VER=3.12.10     # last 3.12 with an official Windows binary installer; cp312 wheels exist in
                   # the stable ROCm index. 3.13.x also works if you would rather stay current.
TORCH_PIN="torch[device-gfx1151]==2.13.0+rocm10.0.0"
INDEX=https://stable.repo.amd.com/rocm/whl-next/
MIN_ADRENALIN=26.8.1

say() { printf '\n== %s ==\n' "$*"; }
die() { printf '\nSTOP: %s\n' "$*"; exit 1; }

# powershell.exe inherits the cwd; from a WSL path it warns about UNC and can bail.
cd /mnt/c 2>/dev/null || die "no /mnt/c -- is this WSL with drive interop on?"
command -v powershell.exe >/dev/null || die "powershell.exe not on PATH (WSL interop is off)"

# ---------------------------------------------------------------- paths
# Ask Windows where the profile is. [Console]::Out.Write suppresses the trailing newline;
# tr strips the CR. Never hardcode the casing.
WINHOME_W=$(powershell.exe -NoProfile -NonInteractive -Command '[Console]::Out.Write($env:USERPROFILE)' 2>/dev/null | tr -d '\r')
[ -n "$WINHOME_W" ] || die "could not read USERPROFILE from Windows"
WINHOME=$(wslpath -u "$WINHOME_W") || die "wslpath could not translate $WINHOME_W"
[ -d "$WINHOME" ] || die "$WINHOME_W maps to $WINHOME which does not exist from WSL"

ROOT="$WINHOME/siriusq-rocm"; ROOT_W="$WINHOME_W\\siriusq-rocm"
mkdir -p "$ROOT" || die "cannot write to $ROOT"
echo "windows profile : $WINHOME_W"
echo "                  (seen from WSL as $WINHOME)"
echo "work dir        : $ROOT_W"

# ---------------------------------------------------------------- phase 0: the admin gate
say "0. preflight -- the only question that can end this"
cat > "$ROOT/preflight.ps1" <<'PS0'
$ErrorActionPreference='SilentlyContinue'
$os = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
Write-Host ("windows        : {0} build {1}.{2}" -f $os.DisplayVersion,$os.CurrentBuild,$os.UBR)
$adr = (Get-ItemProperty 'HKLM:\SOFTWARE\AMD\CN').RadeonSoftwareVersion
if (-not $adr) { $adr = (Get-ItemProperty 'HKLM:\SOFTWARE\WOW6432Node\AMD\CN').RadeonSoftwareVersion }
if (-not $adr) { $adr = (Get-ItemProperty 'HKLM:\SOFTWARE\AMD\CN').DriverVersion }
Write-Host ("adrenalin      : {0}" -f $(if ($adr) { $adr } else { '<not found in HKLM\SOFTWARE\AMD\CN>' }))
Get-CimInstance Win32_VideoController | ForEach-Object {
  Write-Host ("gpu            : {0}  wddm driver {1}" -f $_.Name, $_.DriverVersion)
}
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole('Administrators')
Write-Host ("elevated       : {0}   (expected False -- nothing below needs it)" -f $admin)
$tdr = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers').TdrDelay
Write-Host ("tdr delay      : {0}" -f $(if ($tdr) { "$tdr s" } else { '2 s (WDDM default -- see note in the script)' }))
Write-Host ("adrenalin_raw={0}" -f $adr)
PS0
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$ROOT_W\\preflight.ps1" 2>&1 | tr -d '\r' | tee "$ROOT/preflight.txt"

ADR=$(sed -n 's/^adrenalin_raw=//p' "$ROOT/preflight.txt" | tr -d ' ')
if [ -z "$ADR" ]; then
  echo
  echo "!! could not read the Adrenalin version. Continuing anyway -- pip cannot break anything,"
  echo "!! and if the driver is too old torch will simply report no device at the end."
else
  # dotted numeric compare; sort -V is fine for AMD's YY.M.P scheme
  OLDEST=$(printf '%s\n%s\n' "$ADR" "$MIN_ADRENALIN" | sort -V | head -1)
  if [ "$OLDEST" != "$MIN_ADRENALIN" ] && [ "$ADR" != "$MIN_ADRENALIN" ]; then
    echo
    echo "!! Adrenalin $ADR is older than the $MIN_ADRENALIN that ROCm 10.0.0 requires for gfx1151."
    echo "!! Updating the driver is the ONE step that needs administrator on this box."
    echo "!! Everything below still runs -- pip installs fine against an old driver and the"
    echo "!! failure shows up as 'no GPU' in phase 3, which is a cheap, definitive answer."
  fi
fi

# ---------------------------------------------------------------- phase 1: per-user Python
say "1. per-user Python $PY_VER on the Windows side"
EXE="$ROOT/python-$PY_VER-amd64.exe"
if [ ! -s "$EXE" ] || [ "$(stat -c%s "$EXE" 2>/dev/null || echo 0)" -lt 20000000 ]; then
  # Download from WSL, not with Invoke-WebRequest: WSL's curl is the network path we already
  # trust, and this writes straight into the Windows profile through DrvFs.
  echo "downloading python-$PY_VER-amd64.exe ..."
  curl -fL --retry 5 --connect-timeout 30 -o "$EXE" \
    "https://www.python.org/ftp/python/$PY_VER/python-$PY_VER-amd64.exe" \
    || die "download failed"
fi
echo "installer: $(stat -c%s "$EXE") bytes at $ROOT_W\\python-$PY_VER-amd64.exe"
# Prove the WSL->Windows path mapping before anything depends on it. This is the exact step
# that failed silently last time; make it loud.
powershell.exe -NoProfile -NonInteractive -Command \
  "if (Test-Path '$ROOT_W\\python-$PY_VER-amd64.exe') { 'windows can see the installer: OK' } else { 'windows CANNOT see the installer -- path mapping is wrong' }" \
  2>/dev/null | tr -d '\r'

# ---------------------------------------------------------------- phase 2+3: install and probe
say "2. venv + ROCm PyTorch, 3. GPU probe"
cat > "$ROOT/gpucheck.py" <<'PY0'
# Answers, in order: is there a GPU at all, is it gfx1151, does complex64 work (the thing
# DirectML aborts the process over), how much memory can we actually get, and what does a
# real statevector gate application cost.
import sys, time, torch

print("torch          :", torch.__version__)
print("hip            :", torch.version.hip)
print("executable     :", sys.executable)
if "WindowsApps" in sys.executable:
    print("FAIL: running under the Microsoft Store alias stub"); sys.exit(2)
if not torch.cuda.is_available():
    print("FAIL: no ROCm device visible.")
    print("      pip is fine; this is the kernel-mode driver. Check the Adrenalin version")
    print("      printed in phase 0 against 26.8.1.")
    sys.exit(3)

d = torch.device("cuda")
p = torch.cuda.get_device_properties(0)
print("device         :", torch.cuda.get_device_name(0))
print("arch           :", getattr(p, "gcnArchName", "?"))
free, total = torch.cuda.mem_get_info()
print("mem free/total : %.2f / %.2f GiB" % (free / 2**30, total / 2**30))

# complex64 -- the whole reason for leaving DirectML. If this line survives, the real/imag
# split and its doubled memory traffic can be deleted from the simulator.
z = torch.randn(4096, 4096, dtype=torch.complex64, device=d)
torch.cuda.synchronize()
print("complex64 gemm :", complex((z @ z)[0, 0].item()))
del z; torch.cuda.empty_cache()

def apply_1q(psi, q, g):
    """One gate on qubit q of a statevector -- the bandwidth-bound inner loop."""
    v = psi.view(-1, 2, 1 << q)
    a, b = v[:, 0, :], v[:, 1, :]
    na = g[0, 0] * a + g[0, 1] * b
    nb = g[1, 0] * a + g[1, 1] * b
    v[:, 0, :], v[:, 1, :] = na, nb

h = (torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device=d) / (2 ** 0.5))
print("\n n   state GiB   ms/gate   GB/s>=  ||psi||")   # >= : counts one read + one write
n = 20
while n <= 34:
    try:
        psi = torch.zeros(1 << n, dtype=torch.complex64, device=d); psi[0] = 1
        for q in range(4):                       # warm up + let the allocator settle
            apply_1q(psi, q, h)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for q in range(8):
            apply_1q(psi, q % n, h)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000 / 8
        gib = (1 << n) * 8 / 2**30
        # read state + write state, at minimum
        gbs = 2 * gib * 1.073741824 / (ms / 1000.0)
        print(" %-3d %-11.3f %-9.2f %-7.0f %.7f" % (n, gib, ms, gbs, psi.norm().item()))
        del psi; torch.cuda.empty_cache()
    except RuntimeError as e:
        print(" %-3d ceiling: %s" % (n, str(e).splitlines()[0][:90]))
        break
    n += 1
PY0

cat > "$ROOT/setup.ps1" <<'PS1'
$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'
$root   = Join-Path $env:USERPROFILE 'siriusq-rocm'
$pyDir  = Join-Path $root 'py312'
$pyExe  = Join-Path $pyDir 'python.exe'
$venv   = Join-Path $root 'venv'
$venvPy = Join-Path $venv 'Scripts\python.exe'
$setup  = Get-ChildItem (Join-Path $root 'python-*-amd64.exe') | Select-Object -First 1

if (-not (Test-Path $pyExe)) {
  if (-not $setup) { Write-Host "FAIL: no python-*-amd64.exe in $root (phase 1 did not land)"; exit 2 }
  Write-Host "-- installing Python per-user into $pyDir (no UAC)"
  # Include_launcher=0 removes the py.exe question entirely; InstallLauncherAllUsers=0 is belt
  # and braces. TargetDir means we never have to find the interpreter on PATH afterwards.
  $p = Start-Process -FilePath $setup.FullName -Wait -PassThru -ArgumentList @(
        '/quiet','InstallAllUsers=0','InstallLauncherAllUsers=0','Include_launcher=0',
        'PrependPath=0','AssociateFiles=0','Shortcuts=0',
        'Include_test=0','Include_doc=0','Include_tcltk=0',"TargetDir=$pyDir")
  Write-Host "   installer exit code: $($p.ExitCode)"
}
if (-not (Test-Path $pyExe)) { Write-Host "FAIL: no python.exe at $pyExe"; exit 2 }
& $pyExe -c "import sys; print('python         :', sys.version.split()[0], sys.executable)"

if (-not (Test-Path $venvPy)) { Write-Host "-- creating venv"; & $pyExe -m venv $venv }
if (-not (Test-Path $venvPy)) { Write-Host "FAIL: venv not created at $venv"; exit 2 }

Write-Host "-- pip install ROCm PyTorch (~1.1 GB: rocm-sdk-core, libraries, gfx1151 kernels, torch)"
# index-url ONLY, never extra-index-url: this index carries its own numpy/sympy/filelock etc,
# and adding PyPI would let pip resolve a plain CPU `torch` at a higher version and win.
& $venvPy -m pip install --disable-pip-version-check --timeout 120 --retries 10 `
    --index-url INDEX_URL_PLACEHOLDER TORCH_PIN_PLACEHOLDER
if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: pip install returned $LASTEXITCODE"; exit 3 }

& $venvPy (Join-Path $root 'gpucheck.py')
exit $LASTEXITCODE
PS1
# Substituted rather than interpolated so the PowerShell heredoc stays literal.
sed -i "s|INDEX_URL_PLACEHOLDER|$INDEX|; s|TORCH_PIN_PLACEHOLDER|\"$TORCH_PIN\"|" "$ROOT/setup.ps1"

powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$ROOT_W\\setup.ps1" 2>&1 | tr -d '\r'
RC=${PIPESTATUS[0]}

say "done (exit $RC)"
cat <<'NOTES'
Where things live on the Windows side:
    %USERPROFILE%\siriusq-rocm\py312\python.exe      per-user interpreter, not the Store stub
    %USERPROFILE%\siriusq-rocm\venv\Scripts\python.exe   ROCm PyTorch
    %USERPROFILE%\siriusq-rocm\gpucheck.py           edit from WSL, rerun with:
        cd /mnt/c && powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \
          "& \"$env:USERPROFILE\siriusq-rocm\venv\Scripts\python.exe\" \"$env:USERPROFILE\siriusq-rocm\gpucheck.py\""

Two things ROCm does NOT fix, so do not plan around them:
  * The 2 s watchdog is WDDM's TDR, not a DirectML feature. Native Windows ROCm is a WDDM
    client and is subject to it too (AMD lists TDR as a known HIP-on-Windows issue). Raising
    TdrDelay is an HKLM registry write = administrator. The portable fix is to keep each
    KERNEL short -- a single gate on a 30-qubit state is milliseconds, so chunk the work
    instead of buying a longer timeout.
  * Memory is the BIOS/Adrenalin graphics carve-out, not the 117 GB. If phase 3 shows a total
    far below what you need, the knob is Variable Graphics Memory in Adrenalin, or the UMA
    frame buffer in BIOS -- both outside pip.

If phase 3 says "no ROCm device" and the driver is current, the fallback worth 20 minutes is
ROCm INSIDE WSL, which AMD now supports on Strix Halo through /dev/dxg -- no /dev/kfd needed:
    sudo apt install ./rocdxg-roct_1.2.2_amd64.deb   # github.com/ROCm/librocdxg/releases
    pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1151]"
    export HSA_ENABLE_DXG_DETECTION=1                # not needed from ROCm 7.13 on
It needs sudo in WSL only -- never Windows administrator -- and its memory pool is sized by
.wslconfig rather than the graphics carve-out, which on this box is the larger number.
NOTES
