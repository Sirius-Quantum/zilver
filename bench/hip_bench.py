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
# Sweep POSITION, because it is the variable everything turns on and the first
# run compared our worst position against torch's best. copies_per_gate.py
# indexes by zilver's q, where stride = 1 << (n-1-q), so p = n-1-q.
#
# THESE ARE TRANSCRIBED CONSTANTS, NOT A MEASUREMENT MADE HERE, and the column header says so.
# They are copies_per_gate.py's output from THE SAME DAY, carried here so the rows line up --
# 2026-09-07 measured 4.28 4.26 4.26 4.30 6.07 5.48 6.99 7.00 and those were the values here.
#
# The hazard is not that they are wrong; it is that they go stale silently. Re-running
# copies_per_gate.py on 2026-09-08 gave 4.38 4.36 4.38 4.41 6.29 5.64 7.16 7.17, and this dict
# went on printing the 09-07 figures beside 09-08 kernel timings, so the same harvest contained
# two different reference columns and a paper drawing on both was wrong on its face.
#
# Updated to 2026-09-08 to match the kernel numbers printed above it. RENAMED so nothing implies
# these were timed alongside that column, and DATED so the next mismatch is visible rather than
# silent. The real fix is to time the reference path in this bench.
REF_XFORM_DATE = "2026-09-08"
REF_XFORM = {0: 4.38, 1: 4.36, 4: 4.38, 12: 4.41, 18: 6.29, 20: 5.64, 22: 7.16, 23: 7.17}

print(f"\n  m=1, one gate, by position ({'p':>2} = bit of the flat index)")
print(f"  {'p':>3}{'q':>4}{'stride':>12}{'ms':>9}{'copies':>9}{'ref@q*':>9}{'gain':>7}")
print(f"  * ref@q transcribed from copies_per_gate.py, {REF_XFORM_DATE}, not timed here")
for q in (0, 1, 4, 12, 18, 20, 22, 23):
    p_ = n - 1 - q
    s_ = torch.randn(N, dtype=torch.complex64, device=dev)
    U = kron_logical([H2])
    dt = timeit(lambda: run_fused(s_, n, [p_], U))
    cp = dt / ref
    t = REF_XFORM[q]
    print(f"  {p_:>3}{q:>4}{1 << p_:>12}{dt*1e3:>9.2f}{cp:>9.2f}{t:>9.2f}{t/cp:>6.1f}x")

print(f"\n  fusion: which POSITIONS are fused decides coalescing, not how many")
print(f"  {'m':>3}{'positions':>26}{'ms/pass':>10}{'copies':>9}{'per gate':>10}")
# The m-sweep on HIGH bits. m=1 and m=2 came back single-pass (1.24-1.34 copies) and m=4 did
# not (2.73), so the knee sits between them and m=3 was the one row missing. Pre-registered
# before this ran: m=3 stays single-pass at 1.25-1.45 copies/pass, and m=5, m=6 are monotone
# increasing above m=4's 2.73.
#   Falsified if m=3 comes back near 2.7 -- the knee is then at 2, this fuses PAIRS rather than
#   cosets, and the fusion claim must be restated as m=2 only (still true, still 0.62/gate).
#   Falsified also if m=5 or m=6 lands BELOW m=4, which would make 2.73 a compiler artifact
#   rather than occupancy, and the traffic model wrong.
# Low-bit rows stay in for contrast: they are the coalescing control, not the sweep.
for m, pos in ((2, [0, 1]), (2, [22, 23]),
               (3, [21, 22, 23]),
               (4, [0, 1, 2, 3]), (4, [20, 21, 22, 23]), (4, [0, 9, 17, 23]),
               (5, [19, 20, 21, 22, 23]),
               (6, [18, 19, 20, 21, 22, 23])):
    s_ = torch.randn(N, dtype=torch.complex64, device=dev)
    U = kron_logical([H2] * m)
    dt = timeit(lambda: run_fused(s_, n, pos, U))
    cp = dt / ref
    print(f"  {m:>3}{str(pos):>26}{dt*1e3:>10.2f}{cp:>9.2f}{cp/m:>10.2f}")

print("""
  Pre-registered before any of this existed: 0.95-1.20 copies at m=1, and below
  2.5x over the torch path kills the traffic model.
  m=1 came back 1.29-1.34 -- OUTSIDE the predicted band, by 27%. Recorded, not buried.
  Pre-registered for the m-sweep: m=3 single-pass at 1.25-1.45 copies/pass; m=5 and m=6
  monotone increasing above m=4's 2.73.

  * ref@q is TRANSCRIBED from bench/copies_per_gate.py's 2026-09-08 run, not timed here.
    Its reference copy was 1.26 ms against this bench's 1.30 ms -- 3% apart, which bounds
    any comparison drawn across the two columns.

  Read the FUSION table by which positions are fused, not by m. Coalescing is
  set by the bits left FREE for the thread id, not the bits being fused: fuse
  the low bits and consecutive threads land 2^m elements apart, wasting most of
  every cache line. Fuse high bits and they land adjacent. So [20,21,22,23]
  should beat [0,1,2,3] -- if it does not, that model is wrong.""")
