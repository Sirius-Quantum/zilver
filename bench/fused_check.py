"""Verify the fused/framed algorithm against CLOSED FORMS, on any machine.

Runs the same checks the HIP kernel must satisfy, using the numpy mirror in
zilver.fused, so the algorithm is proven before it is compiled -- and can be
re-proven on the box after it is.

    python3 bench/fused_check.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from zilver.fused import (GF2Frame, fused_apply_reference, kron_logical,
                          pivots_for)

H2 = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
rng = np.random.default_rng(20260906)


def plain_1q(state, n, p, U):
    out = state.copy()
    stride = 1 << p
    idx = np.arange(1 << (n - 1))
    i = ((idx >> p) << (p + 1)) | (idx & (stride - 1))
    j = i + stride
    a, b = state[i], state[j]
    out[i] = U[0, 0] * a + U[0, 1] * b
    out[j] = U[1, 0] * a + U[1, 1] * b
    return out


def materialize(state, frame):
    x = np.arange(1 << frame.n, dtype=np.int64)
    m = np.zeros_like(x)
    for p in range(frame.n):
        m ^= np.where((x >> p) & 1, frame.cols[p], 0)
    return state[m]


def walsh(n, x):
    y = np.arange(1 << n, dtype=np.int64)
    return ((-1.0) ** np.array([bin(x & int(v)).count("1") for v in y])
            ).astype(np.complex64) / np.sqrt(1 << n)


n, x = 14, 0b10110010110101
exact = walsh(n, x)
print("planted Walsh-Hadamard, n=%d  (every amplitude known in advance)" % n)
print("  %-8s %-12s %s" % ("fusion m", "max|err|", "passes"))
for m in (1, 2, 4, 5):
    s = np.zeros(1 << n, dtype=np.complex64); s[x] = 1.0
    passes = 0
    for blk in range(0, n, m):
        pos = list(range(blk, min(blk + m, n)))
        piv, masks, _ = pivots_for([1 << p for p in pos])
        s = fused_apply_reference(s, n, masks, piv, [0] * len(pos),
                                  kron_logical([H2] * len(pos)))
        passes += 1
    print("  %-8d %-12.3e %d" % (m, np.abs(s - exact).max(), passes))

s = np.zeros(1 << n, dtype=np.complex64); s[x] = 1.0
groups = [[0, 7, 13, 3], [1, 8, 12, 4], [2, 9, 11, 5], [6, 10]]
for pos in groups:
    piv, masks, _ = pivots_for([1 << p for p in pos])
    s = fused_apply_reference(s, n, masks, piv, [0] * len(pos),
                              kron_logical([H2] * len(pos)))
print("  %-8s %-12.3e %d   <- high and low strides in ONE block"
      % ("mixed", np.abs(s - exact).max(), len(groups)))

# ---- the deferred frame, and the silent bug it would otherwise hide --------
n = 10
s0 = (rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)).astype(np.complex64)
s0 /= np.linalg.norm(s0)
ladder = [(p, p + 1) for p in range(n - 1)]
late = [(2, H2), (5, H2), (0, H2)]

ref = s0.copy()
for c, t in ladder:
    idx = np.arange(1 << n, dtype=np.int64)
    ref = ref[np.where((idx >> c) & 1, idx ^ (1 << t), idx)]
for p, g in late:
    ref = plain_1q(ref, n, p, g)

print("\ndeferred GF(2) frame, n=%d: %d CNOTs + %d single-qubit gates"
      % (n, len(ladder), len(late)))
print("  explicit                    %d passes" % (len(ladder) + len(late)))
for label, use_row in (("frame, parity(i & row)  ", True),
                       ("frame, pivot bit  WRONG ", False)):
    frame = GF2Frame(n); buf = s0.copy()
    for c, t in ladder:
        frame.cnot(c, t)
    for p, g in late:
        piv, masks, _ = pivots_for([frame.mask(p)])
        buf = fused_apply_reference(buf, n, masks, piv,
                                    [frame.row(p)] if use_row else [0], g)
    got = materialize(buf, frame)
    print("  %s    %d passes   max|err| %.3e   norm %.7f"
          % (label, len(late), np.abs(got - ref).max(), np.linalg.norm(got)))

print("""
READ: the WRONG row keeps the norm at 1.0 and raises nothing -- it applies
X U X on half the pairs. A norm check passes it. Only a planted answer
catches it, which is why this file exists and why the kernel carries
__popcll rather than a bit test.""")
