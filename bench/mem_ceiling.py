"""How much of the 117 GB can the GPU actually have? Measure, do not assume.

    python3 bench/mem_ceiling.py

torch reports ~49 GiB total on this box against 117 GB of system RAM. On a
discrete card that is a wall. On an APU it is a POLICY: the CPU and the GPU are
reading the same DRAM, and the split is set by the driver, by BIOS, and by a
handful of runtime caps -- some of which are environment variables that cost
nothing to try.

This tries each in a FRESH process (they are read at runtime init, so setting
them after import does nothing) and reports what the runtime then says it has,
plus the largest single block it can actually get. The allocation is the real
answer; mem_get_info is only what it claims.
"""
import os
import subprocess
import sys

CASES = [
    ("baseline", {}),
    ("GPU_MAX_ALLOC_PERCENT=100", {"GPU_MAX_ALLOC_PERCENT": "100"}),
    ("GPU_SINGLE_ALLOC_PERCENT=100", {"GPU_SINGLE_ALLOC_PERCENT": "100"}),
    ("both alloc percents", {"GPU_MAX_ALLOC_PERCENT": "100",
                             "GPU_SINGLE_ALLOC_PERCENT": "100"}),
    ("GPU_MAX_HEAP_SIZE=100", {"GPU_MAX_HEAP_SIZE": "100"}),
    ("HSA_XNACK=1 (managed/pageable)", {"HSA_XNACK": "1"}),
    ("expandable_segments", {"PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"}),
    ("everything at once", {"GPU_MAX_ALLOC_PERCENT": "100",
                            "GPU_SINGLE_ALLOC_PERCENT": "100",
                            "GPU_MAX_HEAP_SIZE": "100",
                            "HSA_XNACK": "1",
                            "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"}),
]

CHILD = r'''
import sys, torch
if not torch.cuda.is_available():
    print("no device"); sys.exit(1)
free, total = torch.cuda.mem_get_info()
# Largest single block we can really get, by bisection on GiB.
lo, hi, best = 0.0, 160.0, 0.0
for _ in range(11):
    mid = (lo + hi) / 2
    try:
        n = int(mid * (2 ** 30) / 8)          # complex64 = 8 bytes
        t = torch.empty(n, dtype=torch.complex64, device="cuda")
        del t; torch.cuda.empty_cache()
        best, lo = mid, mid
    except Exception:
        hi = mid
qubits = 0
while (1 << (qubits + 1)) * 8 / 2**30 <= best:
    qubits += 1
print(f"{total/2**30:8.2f} {free/2**30:8.2f} {best:9.2f} {qubits:5d}")
'''

print("Reported vs ACTUALLY ALLOCATABLE. The allocation is the answer.\n")
print(f"{'setting':<34}{'total':>8}{'free':>8}{'biggest':>10}{'qubits':>7}")
for label, env in CASES:
    e = dict(os.environ, **env)
    try:
        out = subprocess.run([sys.executable, "-c", CHILD], env=e,
                             capture_output=True, text=True, timeout=600)
        line = (out.stdout or out.stderr).strip().splitlines()[-1] if (out.stdout or out.stderr) else "?"
    except Exception as exc:
        line = f"-- {type(exc).__name__}"
    print(f"{label:<34}{line}")

print("""
GiB, and qubits is the widest statevector that single block holds.
31 qubits needs 16 GiB, 32 needs 32, 33 needs 64.

If none of these move the ceiling, the split is set below the runtime -- the
Variable Graphics Memory slider in Adrenalin, or the UMA frame buffer in BIOS --
and neither needs AMD, only a reboot.""")
