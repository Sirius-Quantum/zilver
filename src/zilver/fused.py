"""Deferred GF(2) frame + coset fusion: the quantum structure the kernel exploits.

Two facts about quantum circuits, neither of which a generic GPU kernel can see:

1. **A CNOT never has to touch memory.** CNOT(c,t) sends index i to
   i XOR ((i>>c & 1) << t), which is GF(2)-LINEAR in i. A whole Clifford
   segment of CNOT/X/SWAP composes into one invertible n x n matrix M over
   GF(2). Carry M instead of applying it: the buffer holds s_buf, and the
   logical state is s_log[x] = s_buf[M x]. A later 1q gate on logical qubit q
   pairs buffer indices i and i XOR mask, mask = column q of M. One XOR.

   A CNOT ladder therefore costs ZERO passes instead of n-1.

2. **m gates fit in one pass.** A thread owning one coset of the span of m
   masks gathers 2^m amplitudes, applies a 2^m x 2^m unitary in registers and
   scatters back, in place. The masks need not be adjacent, so high- and
   low-stride qubits fuse into the same pass -- which is why none of the
   bit-reversal or transposition of an FFT appears here.

The trap, and the reason this module carries a planted-answer test: which
member of a pair is the logical |0> branch is NOT "pivot bit clear". It is
parity(i & row_q) with row_q from M^-1. Getting it wrong applies X U X on half
the pairs -- max error ~7e-2, norm preserved, no exception. It ships silently.

`fused_apply_reference` below mirrors src/zilver/hip/fused_gate.hip line for
line, so the algorithm is checked here, on any machine, before it is compiled
anywhere.
"""

from __future__ import annotations

import numpy as np


class GF2Frame:
    """The index relabelling a Clifford segment performs, carried as metadata.

    `cols[q]` is column q of M, `rows[q]` is row q of M^-1, both as int bitmasks
    over BIT POSITIONS (position p = bit p of the flat index; zilver's qubit q
    is position n-1-q). Identity is cols[q] = rows[q] = 1<<q.
    """

    __slots__ = ("n", "cols", "rows")

    def __init__(self, n: int):
        self.n = n
        self.cols = [1 << p for p in range(n)]
        self.rows = [1 << p for p in range(n)]

    def cnot(self, c: int, t: int) -> None:
        """CNOT with control at position c, target at position t.

        On the index, e_c -> e_c + e_t, so column c picks up column t; the
        inverse is the same map, so row t picks up row c.
        """
        self.cols[c] ^= self.cols[t]
        self.rows[t] ^= self.rows[c]

    def swap(self, a: int, b: int) -> None:
        self.cols[a], self.cols[b] = self.cols[b], self.cols[a]
        self.rows[a], self.rows[b] = self.rows[b], self.rows[a]

    def is_identity(self) -> bool:
        return all(self.cols[p] == 1 << p for p in range(self.n))

    def mask(self, p: int) -> int:
        """XOR partner mask for a 1q gate on logical position p."""
        return self.cols[p]

    def row(self, p: int) -> int:
        """Parity row selecting the logical |0> branch for position p."""
        return self.rows[p]


def pivots_for(masks) -> list[int]:
    """A distinct pivot bit per mask, by Gaussian elimination over GF(2).

    The kernel scatters a thread id across the NON-pivot bits to build a coset
    representative, so the pivots must be distinct or two threads collide.
    Returns pivots in ascending order together with the reduced masks.
    """
    red = list(masks)
    piv: list[int] = []
    for b in range(len(red)):
        for prev, p in enumerate(piv):
            if (red[b] >> p) & 1:
                red[b] ^= red[prev]
        if red[b] == 0:
            raise ValueError("masks are linearly dependent over GF(2); "
                             "these qubits cannot fuse in one pass")
        piv.append(red[b].bit_length() - 1)
    order = sorted(range(len(piv)), key=lambda i: piv[i])
    return [piv[i] for i in order], [masks[i] for i in order], order


def _parity(v):
    v = np.asarray(v, dtype=np.int64).copy()
    out = np.zeros_like(v)
    while v.any():
        out ^= v & 1
        v >>= 1
    return out


def fused_apply_reference(state, n, masks, pivots, rows, U):
    """numpy mirror of fused_gate.hip. Same arithmetic, vectorised over threads.

    state : (2^n,) complex, modified in place and returned
    masks : m XOR partner masks
    pivots: m distinct pivot bits, ascending
    rows  : m parity rows (0 for "no frame")
    U     : (2^m, 2^m) in LOGICAL order
    """
    m = len(masks)
    D = 1 << m
    t = np.arange(1 << (n - m), dtype=np.int64)

    # insert a zero bit at each pivot, ascending -- the coset representative
    base = t
    for p in pivots:
        low = base & ((1 << p) - 1)
        base = ((base ^ low) << 1) | low

    idx = np.empty((1 << (n - m), D), dtype=np.int64)
    for k in range(D):
        i = base.copy()
        for b in range(m):
            if (k >> b) & 1:
                i ^= masks[b]
        idx[:, k] = i

    amp = state[idx]

    framed = any(r for r in rows)
    if framed:
        lg = np.zeros((1 << (n - m), D), dtype=np.int64)
        for b in range(m):
            lg |= (_parity(idx & rows[b]) & 1) << b
        ordered = np.empty_like(amp)
        np.put_along_axis(ordered, lg, amp, axis=1)
        amp = ordered

    out = amp @ np.asarray(U, dtype=amp.dtype).T

    if framed:
        out = np.take_along_axis(out, lg, axis=1)
    state[idx] = out
    return state


def kron_logical(mats) -> np.ndarray:
    """2^m x 2^m for independent 1q gates, bit b of the local index <-> mats[b]."""
    U = np.eye(1, dtype=np.complex64)
    for g in mats:                       # bit 0 is the fastest-varying
        U = np.kron(np.asarray(g, dtype=np.complex64), U)
    return U
