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


# ---------------------------------------------------------------------------
# Running a whole circuit under the frame
# ---------------------------------------------------------------------------

_INDEX_GATES = {"cnot", "cx", "swap"}


def run_framed(ops, gate_of, n, state, apply_fused, apply_cnot):
    """Execute a circuit carrying CNOT/SWAP as a GF(2) index frame.

    apply_fused(state, masks, pivots, rows, U, m) -- one fused pass
    apply_cnot(state, control_pos, target_pos)    -- one CNOT, in place

    Returns (state, passes). The frame absorbs every CNOT it meets and is paid
    back ONCE at the end, so a circuit with d entangling layers costs one
    materialisation instead of d. On a single ladder there is nothing to win --
    the saving is in depth.

    Three-qubit gates are not index-linear (Toffoli is quadratic), so the frame
    is materialised before one and restarted after.
    """
    frame = GF2Frame(n)
    deferred = []                     # (control_pos, target_pos), in order
    passes = 0

    def materialise(state, passes):
        """Pay the frame back in ops that scale with n, not with circuit depth.

        Replaying the deferred CNOTs one by one costs exactly what not deferring
        them would have -- the first version of this did that and saved nothing,
        which the equivalence test showed at once. The accumulated M is ONE
        invertible matrix however many CNOTs built it, so factor it: eliminate M
        to the identity, recording the row operations, and each is a CNOT.
        d entangling layers then cost O(n^2) instead of d*(n-1).
        """
        cols = list(frame.cols)
        # M as rows of bits: bit p of row r is (cols[r] >> p) & 1, i.e. M[p][r].
        M = [sum(((cols[c] >> r) & 1) << c for c in range(n)) for r in range(n)]
        recorded = []
        for col in range(n):
            piv = next((r for r in range(col, n) if (M[r] >> col) & 1), None)
            if piv is None:
                continue
            if piv != col:
                M[piv], M[col] = M[col], M[piv]
                recorded.append(("swap", piv, col))
            for r in range(n):
                if r != col and (M[r] >> col) & 1:
                    M[r] ^= M[col]
                    recorded.append(("add", col, r))
        # FORWARD order, and CNOT(col -> row), not the transpose. Determined by
        # testing all four variants against the explicit gather out[x] = s[Mx]:
        # only this one gives 0.000e+00. The other three preserve the norm
        # exactly while being wrong by ~4, so a norm check cannot see them.
        for kind, a, b in recorded:
            if kind == "add":
                state = apply_cnot(state, a, b)
                passes += 1
            else:
                for c, t in ((a, b), (b, a), (a, b)):
                    state = apply_cnot(state, c, t)
                    passes += 3
        deferred.clear()
        frame.__init__(n)
        return state, passes

    for op in ops:
        kind = (op.kind or "").lower()
        qs = list(op.qubits)
        pos = [n - 1 - q for q in qs]

        if kind in _INDEX_GATES and len(qs) == 2:
            if kind == "swap":
                frame.swap(pos[0], pos[1])
                deferred.extend([(pos[0], pos[1]), (pos[1], pos[0]), (pos[0], pos[1])])
            else:
                frame.cnot(pos[0], pos[1])
                deferred.append((pos[0], pos[1]))
            continue

        if len(qs) > 2:
            state, passes = materialise(state, passes)
            raise NotImplementedError("3-qubit gates: materialise then use the plain path")

        g = np.asarray(gate_of(op), dtype=np.complex64).reshape(1 << len(qs), 1 << len(qs))
        masks0 = [frame.mask(p) for p in pos]
        rows0 = [frame.row(p) for p in pos]
        try:
            piv, masks, order = pivots_for(masks0)
        except ValueError:                       # dependent under this frame
            state, passes = materialise(state, passes)
            piv, masks, order = pivots_for([1 << p for p in pos])
            rows0 = [0] * len(pos)

        k = len(qs)
        gate_bit = [k - 1 - i for i in range(k)]
        perm = np.empty(1 << k, dtype=np.int64)
        for loc in range(1 << k):
            gi = 0
            for b in range(k):
                if (loc >> b) & 1:
                    gi |= 1 << gate_bit[order[b]]
            perm[loc] = gi
        U = np.ascontiguousarray(g[np.ix_(perm, perm)])
        rows = [rows0[i] for i in order]

        state = apply_fused(state, masks, piv, rows, U, k)
        passes += 1

    state, passes = materialise(state, passes)
    return state, passes
