"""Coset-gather fusion + deferred GF(2) frame: the two claims, hand-checked.

Claim 1: a gate on m qubits at ARBITRARY bit positions (any strides, high or
low) can be executed as one pass: each thread owns one coset of the span of
{2^p : p in positions}, gathers 2^m amplitudes, applies a 2^m x 2^m matrix in
registers, scatters back. In place. No transpose, no bit reversal.

Claim 2: a CNOT/X segment is an affine map on the index. It never has to touch
memory. Carry it as a GF(2) matrix M; a later 1q gate on logical qubit q pairs
buffer indices i and i XOR mask, mask = column q of M. One extra XOR.

Planted answer: H on all n qubits applied to |x> has the closed form
    amp(y) = (-1)^popcount(x & y) / 2^(n/2)
so every one of the 2^n amplitudes is known in advance and a single wrong
index shows up as a sign.

Bit convention here: position p = bit p of the flat index, p=0 is LSB.
Zilver's qubit q maps to p = n-1-q.
"""
import itertools
import numpy as np

rng = np.random.default_rng(20260906)

H2 = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)


# ---------------------------------------------------------------- reference

def apply_1q_reference(state, n, p, U):
    """One gate, one full pass. The thing we are trying to do fewer of."""
    out = state.copy()
    stride = 1 << p
    idx = np.arange(1 << (n - 1))
    l, r = idx >> p, idx & (stride - 1)
    i = (l << (p + 1)) | r
    j = i + stride
    a, b = state[i], state[j]
    out[i] = U[0, 0] * a + U[0, 1] * b
    out[j] = U[1, 0] * a + U[1, 1] * b
    return out


# ------------------------------------------------- claim 1: coset gather

def coset_indices(n, positions):
    """(2^(n-m), 2^m) index table. Row = one thread's coset.

    Column k of a row is the amplitude whose local index is k, where bit b of
    k selects positions[b]. This is exactly the address arithmetic a HIP
    thread would do; numpy is only vectorising over threads.
    """
    m = len(positions)
    free = [p for p in range(n) if p not in positions]
    t = np.arange(1 << (n - m), dtype=np.int64)
    base = np.zeros_like(t)
    for b, p in enumerate(free):          # scatter thread id into free bits
        base |= ((t >> b) & 1) << p
    k = np.arange(1 << m, dtype=np.int64)
    off = np.zeros_like(k)
    for b, p in enumerate(positions):     # scatter local index into fused bits
        off |= ((k >> b) & 1) << p
    return base[:, None] | off[None, :]


def fused_apply(state, n, positions, U):
    """One pass, m stages. In place: each thread owns a disjoint coset."""
    idx = coset_indices(n, positions)
    state[idx] = state[idx] @ U.T
    return state


def kron_ordered(mats, positions):
    """2^m x 2^m matrix for independent 1q gates, in coset_indices' local order
    (bit b of the local index <-> positions[b])."""
    m = len(positions)
    U = np.eye(1 << m, dtype=np.complex64)
    for b, g in enumerate(mats):
        left = np.eye(1 << b, dtype=np.complex64)
        right = np.eye(1 << (m - b - 1), dtype=np.complex64)
        U = np.kron(right, np.kron(g, left)) @ U
    return U


# --------------------------------- claim 2: deferred GF(2) index frame

def cnot_columns(n, cols, c, t):
    """CNOT(control=bit c, target=bit t) composed into the frame.

    Frame: buffer holds s_buf, logical s_log[x] = s_buf[M x]. cols[q] is column
    q of M as an int bitmask. CNOT sends e_c -> e_c + e_t on the index, so
    column c picks up column t.
    """
    cols = list(cols)
    cols[c] ^= cols[t]
    return cols


def cnot_rows(n, rows, c, t):
    """Same CNOT on M^-1: rows[t] ^= rows[c]."""
    rows = list(rows)
    rows[t] ^= rows[c]
    return rows


def _parity(v):
    v = v.copy()
    out = np.zeros_like(v)
    while v.any():
        out ^= v & 1
        v >>= 1
    return out


def apply_1q_masked(state, n, mask, row, U):
    """1q gate on a logical qubit under a deferred GF(2) frame.

    Pair partner of buffer index i is i XOR mask, mask = column q of M.
    WHICH member of the pair is the logical |0> branch is NOT "pivot bit
    clear" -- it is parity(i & row), row = row q of M^-1. Getting this wrong
    is a silent 7e-2 error (it applies X U X on half the pairs).

    Kernel cost over the plain strided gate: one XOR, one AND, one popcount
    parity, one select. Nothing touching memory.
    """
    out = state.copy()
    p = mask.bit_length() - 1                 # any valid pairing pivot
    idx = np.arange(1 << (n - 1))
    l, r = idx >> p, idx & ((1 << p) - 1)
    i = (l << (p + 1)) | r
    j = i ^ mask
    g = _parity(i & row)                      # 0 -> i is the |0> branch
    a, b = state[i], state[j]
    u = np.stack([np.where(g == 0, U[k, 0], U[k, 1]) for k in range(2)])
    v = np.stack([np.where(g == 0, U[k, 1], U[k, 0]) for k in range(2)])
    ai = np.where(g == 0, 0, 1)
    out[i] = np.where(g == 0, u[0] * a + v[0] * b, u[1] * a + v[1] * b)
    out[j] = np.where(g == 0, u[1] * a + v[1] * b, u[0] * a + v[0] * b)
    return out


def apply_cnot_reference(state, n, c, t):
    idx = np.arange(1 << n)
    src = np.where((idx >> c) & 1, idx ^ (1 << t), idx)
    return state[src]


# ------------------------------------------------------------------ tests

def test_planted_walsh_hadamard(n=14):
    """Every amplitude known in closed form. m=1 vs m=4 vs one m=n/... pass."""
    x = int(rng.integers(0, 1 << n))
    y = np.arange(1 << n, dtype=np.int64)
    exact = ((-1.0) ** np.array([bin(x & int(v)).count("1") for v in y])).astype(
        np.complex64) / np.sqrt(1 << n)

    results = {}
    for m in (1, 2, 4, 5):
        s = np.zeros(1 << n, dtype=np.complex64)
        s[x] = 1.0
        passes = 0
        for blk in range(0, n, m):
            pos = list(range(blk, min(blk + m, n)))
            U = kron_ordered([H2] * len(pos), pos)
            s = fused_apply(s, n, pos, U)
            passes += 1
        results[m] = (np.abs(s - exact).max(), passes)

    # and a deliberately awkward fused set: mixed high and low strides
    s = np.zeros(1 << n, dtype=np.complex64)
    s[x] = 1.0
    groups = [[0, 7, 13, 3], [1, 8, 12, 4], [2, 9, 11, 5], [6, 10]]
    for pos in groups:
        s = fused_apply(s, n, pos, kron_ordered([H2] * len(pos), pos))
    results["mixed"] = (np.abs(s - exact).max(), len(groups))
    return results


def test_fused_equals_sequential(n=12, trials=20):
    """Arbitrary (entangling) 2^m x 2^m unitary vs gate-at-a-time, and fused
    sets chosen adversarially: adjacent strides, half-array strides, mixtures."""
    worst = 0.0
    for _ in range(trials):
        m = int(rng.integers(1, 5))
        pos = list(rng.choice(n, size=m, replace=False))
        gates = [np.linalg.qr(rng.normal(size=(2, 2)) +
                              1j * rng.normal(size=(2, 2)))[0].astype(np.complex64)
                 for _ in pos]
        s0 = (rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)).astype(np.complex64)
        s0 /= np.linalg.norm(s0)

        seq = s0.copy()
        for p, g in zip(pos, gates):
            seq = apply_1q_reference(seq, n, int(p), g)
        fus = fused_apply(s0.copy(), n, [int(p) for p in pos],
                          kron_ordered(gates, pos))
        worst = max(worst, np.abs(seq - fus).max())
    return worst


def test_deferred_cnot_ladder(n=12):
    """A ladder of CNOTs, then a layer of 1q gates. Reference applies every
    CNOT to memory. Frame version applies ZERO of them and XORs instead."""
    s0 = (rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)).astype(np.complex64)
    s0 /= np.linalg.norm(s0)
    ladder = [(p, p + 1) for p in range(n - 1)]
    gates = [np.linalg.qr(rng.normal(size=(2, 2)) +
                          1j * rng.normal(size=(2, 2)))[0].astype(np.complex64)
             for _ in range(n)]

    ref = s0.copy()
    ref_passes = 0
    for c, t in ladder:
        ref = apply_cnot_reference(ref, n, c, t)
        ref_passes += 1
    for p, g in enumerate(gates):
        ref = apply_1q_reference(ref, n, p, g)
        ref_passes += 1

    buf = s0.copy()
    cols = [1 << q for q in range(n)]
    rows = [1 << q for q in range(n)]
    frame_passes = 0
    for c, t in ladder:
        cols = cnot_columns(n, cols, c, t)      # metadata only, no memory
        rows = cnot_rows(n, rows, c, t)
    for p, g in enumerate(gates):
        buf = apply_1q_masked(buf, n, cols[p], np.int64(rows[p]), g)
        frame_passes += 1
    # readout: materialise once, s_log[x] = s_buf[M x]
    M = np.zeros(1 << n, dtype=np.int64)
    for q in range(n):
        M |= 0
    xs = np.arange(1 << n, dtype=np.int64)
    Mx = np.zeros_like(xs)
    for q in range(n):
        Mx ^= np.where((xs >> q) & 1, cols[q], 0)
    out = buf[Mx]
    frame_passes += 1                            # the one deferred pass
    return np.abs(out - ref).max(), ref_passes, frame_passes


if __name__ == "__main__":
    print("planted Walsh-Hadamard (exact closed form, n=14)")
    print("  fusion m   max|err|      passes   (ideal = ceil(14/m))")
    for m, (err, passes) in test_planted_walsh_hadamard().items():
        print(f"  {str(m):>8}   {err:.3e}   {passes}")

    print("\nfused block == gate-at-a-time, random unitaries, random positions")
    print(f"  worst max|err| over 20 trials: {test_fused_equals_sequential():.3e}")

    print("\ndeferred GF(2) frame vs explicit CNOT ladder (n=12)")
    err, rp, fp = test_deferred_cnot_ladder()
    print(f"  max|err| {err:.3e}   reference passes {rp}   frame passes {fp}")
