#!/usr/bin/env bash
# Can the MINT run on the Radeon? Ask, do not assume.
#
#   bash bench/rocm_chem_probe.sh
#
# We already measured what a GPU port of the mint is worth, on NVIDIA:
# demo/gpu_ladder_probe.py in the qleap repo puts SCF+ao2mo at 16.5 s against
# 2.1 s of CASCI at 15 heavy -- f = 0.887, so Amdahl caps the port at ~8.8x,
# and f RISES with molecule size because CASCI is flat at fixed (2k,2k). So the
# prize is real and it is the SCF, not the CI.
#
# The chain to reproduce that on this box is four links, and every one of them
# is a question rather than a known:
#   1. pyscf has no Windows wheels, so this must run in WSL, not the Windows
#      ROCm venv -- which means ROCm has to be reachable from Linux here.
#      That is librocdxg over /dev/dxg, since WSL2 exposes no /dev/kfd.
#   2. cupy needs a ROCm build for this architecture (gfx1151).
#   3. gpu4pyscf is CuPy plus custom CUDA integral kernels; whether it builds
#      or runs under ROCm is the link I am least sure of.
#   4. and then: do 40 compute units actually beat 32 Zen 5 cores? The RTX PRO
#      6000 is ALSO fp64-limited and still wins, so fp64 rate alone settles
#      nothing. It is an empirical question.
#
# This prints the state of each link. It installs nothing.

say() { printf '\n== %s ==\n' "$*"; }

say "1. can Linux see the GPU at all?"
for d in /dev/kfd /dev/dxg /dev/dri; do
  [ -e "$d" ] && echo "  $d      present" || echo "  $d      ABSENT"
done
echo "  (kfd absent + dxg present is the WSL2 signature: ROCm needs librocdxg,"
echo "   github.com/ROCm/librocdxg -- sudo in WSL only, never Windows admin)"
command -v rocminfo >/dev/null && rocminfo 2>&1 | grep -m2 -E "Name:|gfx" || echo "  rocminfo: not installed"

say "2. what does the AMD index actually publish?"
PY=$(command -v python3)
echo "  python: $PY"
for pkg in cupy cupy-rocm gpu4pyscf pyscf; do
  v=$("$PY" -m pip index versions "$pkg" 2>&1 | head -1)
  echo "  $pkg -> $v"
done
echo "  and on AMD's own index:"
for pkg in cupy gpu4pyscf; do
  v=$("$PY" -m pip index versions --index-url https://stable.repo.amd.com/rocm/whl-next/ "$pkg" 2>&1 | head -1)
  echo "  $pkg -> $v"
done

say "3. what is already importable here?"
"$PY" - <<'PY'
for mod in ("pyscf", "cupy", "gpu4pyscf", "torch"):
    try:
        m = __import__(mod)
        print(f"  {mod:<10} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  {mod:<10} -- {type(e).__name__}")
try:
    import torch
    print(f"  torch.version.hip = {torch.version.hip}   cuda_available={torch.cuda.is_available()}")
except Exception:
    pass
PY

cat <<'NOTES'

READ:
  Link 1 decides whether anything else is worth trying. If /dev/dxg is present
  and /dev/kfd is not, the route is librocdxg and it needs sudo in WSL only.
  Link 3 tells us whether the CPU mint's environment is even the one to extend.

  If cupy-rocm exists for gfx1151 but gpu4pyscf does not build, that is still a
  useful answer: the ao2mo/DF step is dense linear algebra and could be moved
  with cupy alone, which is most of the 88.7%.
NOTES
