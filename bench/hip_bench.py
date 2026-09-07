"""Compile the HIP kernel, prove it against a closed form, then price it.

    python3 bench/hip_bench.py

Three questions in order, and the second gates the third:

  1. does it COMPILE on this box at all -- at runtime, via hiprtc, so no ninja,
     no MSVC, no pybind11 and no administrator are involved
  2. is it RIGHT -- checked against H-on-every-qubit, whose amplitudes are
     (-1)^popcount(x&y)/2^(n/2), so one wrong index shows up as a sign. Not
     against our own torch path, which would only prove they agree.
  3. what does it cost, in COPIES per gate, against the same reference copy
     bench/copies_per_gate.py uses. Measured 4.28 for the torch path on the
     Radeon; the floor is 1.00, and a fused pass of m gates targets 1/m.

Pre-registered by the seat before any of this existed: a transliterated
in-place kernel should land at 0.95-1.20 copies. If it does not beat the torch
path by at least 2.5x, the traffic model is wrong and we stop.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

try:
    import torch
except ImportError:
    sys.exit("no torch here")

from zilver import hip_ext
from zilver.fused import kron_logical, pivots_for

H2 = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)


def _find_rocm():
    """torch's extension builder wants hipcc. The pip SDK does not advertise it."""
    if os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME"):
        return os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
    try:
        import rocm_sdk_core as _c                       # ROCm 10 pip layout
        p = os.path.dirname(_c.__file__)
    except ImportError:
        for cand in ("/opt/rocm", os.path.join(sys.prefix, "Lib", "site-packages", "_rocm_sdk_core")):
            if os.path.isdir(cand):
                p = cand
                break
        else:
            return None
    os.environ.setdefault("ROCM_PATH", p)
    os.environ.setdefault("ROCM_HOME", p)
    return p


print("== 1. build ==")
print("  torch          :", torch.__version__)
print("  hip            :", torch.version.hip)
print("  rocm path      :", _find_rocm())
if not torch.cuda.is_available():
    sys.exit("  no device -- nothing to build for")
os.environ.setdefault("ZILVER_BUILD_VERBOSE", "1")
t0 = time.perf_counter()
launch = hip_ext.kernel(verbose=True)
print(f"  compiled       : {launch is not None}   ({time.perf_counter()-t0:.1f} s)")
if launch is None:
    sys.exit("  compile failed -- the reason is printed above")

dev = torch.device("cuda")


def run_fused(state, n, positions, U):
    """Hand the kernel one pass over `positions`, no frame."""
    piv, masks, order = pivots_for([1 << p for p in positions])
    m = len(masks)
    launch(
        state,
        torch.tensor(masks, dtype=torch.int64, device=dev),
        torch.tensor(piv, dtype=torch.int32, device=dev),
        torch.zeros(m, dtype=torch.int64, device=dev),      # no frame
        torch.as_tensor(np.ascontiguousarray(U), dtype=torch.complex64, device=dev).contiguous(),
        m,
    )
    torch.cuda.synchronize()
    return order


print("\n== 2. correct? planted Walsh-Hadamard, every amplitude known ==")
ok = True
for n, m in ((16, 1), (16, 2), (16, 4), (20, 4)):
    x = 0b1011001011010110 % (1 << n)
    s = torch.zeros(1 << n, dtype=torch.complex64, device=dev)
    s[x] = 1.0
    for blk in range(0, n, m):
        pos = list(range(blk, min(blk + m, n)))
        run_fused(s, n, pos, kron_logical([H2] * len(pos)))
    y = np.arange(1 << n, dtype=np.int64)
    exact = ((-1.0) ** np.array([bin(x & int(v)).count("1") for v in y])
             ).astype(np.complex64) / np.sqrt(1 << n)
    err = np.abs(s.cpu().numpy() - exact).max()
    ok &= err < 1e-5
    print(f"  n={n:<3} m={m}   max|err| {err:.3e}   {'ok' if err < 1e-5 else 'WRONG'}")
if not ok:
    sys.exit("\n  stop: a wrong kernel has no interesting speed")

print("\n== 3. copies per gate, same reference copy as bench/copies_per_gate.py ==")
n = 24
N = 1 << n
src = torch.randn(N, dtype=torch.complex64, device=dev)
dst = torch.empty_like(src)


def timeit(fn, reps=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


ref = timeit(lambda: dst.copy_(src))
gb = 2 * N * 8 / 1e9
print(f"  reference copy {ref*1e3:8.2f} ms = {gb/ref:6.1f} GB/s  == 1.00 copies")
print(f"\n  {'m':>3}{'positions':>26}{'ms/pass':>10}{'copies/pass':>13}{'per gate':>10}")
for m, pos in ((1, [0]), (2, [0, 1]), (4, [0, 1, 2, 3]), (4, [0, 9, 17, 23])):
    s = torch.randn(N, dtype=torch.complex64, device=dev)
    U = kron_logical([H2] * m)
    dt = timeit(lambda: run_fused(s, n, pos, U))
    cp = dt / ref
    print(f"  {m:>3}{str(pos):>26}{dt*1e3:>10.2f}{cp:>13.2f}{cp/m:>10.2f}")

print("""
  torch path on this box measured 4.28 copies/gate (bench/copies_per_gate.py).
  Pre-registered: 0.95-1.20 for m=1. Below 2.5x improvement over 4.28 kills the
  traffic model. The mixed-stride row is the one that says whether arbitrary
  positions really do fuse in one pass, or only adjacent ones do.""")
