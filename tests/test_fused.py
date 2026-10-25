"""The two claims, against planted answers rather than against each other."""
import numpy as np
import pytest

from zilver.fused import (GF2Frame, fused_apply_reference, kron_logical,
                          pivots_for)

H2 = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)


def _apply_1q_plain(state, n, p, U):
    """One gate, one full pass -- the thing we want fewer of."""
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


def _materialize(state, frame):
    """s_log[x] = s_buf[M x]. Only needed once, at readout."""
    x = np.arange(1 << frame.n, dtype=np.int64)
    mx_ = np.zeros_like(x)
    for p in range(frame.n):
        mx_ ^= np.where((x >> p) & 1, frame.cols[p], 0)
    return state[mx_]


@pytest.mark.parametrize("m", [1, 2, 4, 5])
def test_planted_walsh_hadamard(m):
    """H on every qubit of |x> has a closed form, so a wrong index is a sign."""
    n = 12
    x = 0b101100101101
    y = np.arange(1 << n, dtype=np.int64)
    exact = ((-1.0) ** np.array([bin(x & int(v)).count("1") for v in y])
             ).astype(np.complex64) / np.sqrt(1 << n)

    s = np.zeros(1 << n, dtype=np.complex64)
    s[x] = 1.0
    for blk in range(0, n, m):
        pos = list(range(blk, min(blk + m, n)))
        masks = [1 << p for p in pos]
        piv, masks, _ = pivots_for(masks)
        s = fused_apply_reference(s, n, masks, piv, [0] * len(pos),
                                  kron_logical([H2] * len(pos)))
    assert np.abs(s - exact).max() < 1e-6


def test_mixed_strides_fuse_in_one_pass():
    """High and low strides in the SAME block -- no permutation, no transpose."""
    n = 12
    x = 37
    y = np.arange(1 << n, dtype=np.int64)
    exact = ((-1.0) ** np.array([bin(x & int(v)).count("1") for v in y])
             ).astype(np.complex64) / np.sqrt(1 << n)
    s = np.zeros(1 << n, dtype=np.complex64)
    s[x] = 1.0
    for pos in ([0, 7, 11, 3], [1, 8, 10, 4], [2, 9, 5, 6]):
        piv, masks, order = pivots_for([1 << p for p in pos])
        s = fused_apply_reference(s, n, masks, piv, [0] * len(pos),
                                  kron_logical([H2] * len(pos)))
    assert np.abs(s - exact).max() < 1e-6


def test_fused_block_equals_gate_at_a_time():
    rng = np.random.default_rng(20260906)
    n = 10
    for _ in range(20):
        pos = sorted(rng.choice(n, size=3, replace=False).tolist())
        gs = []
        for _ in pos:
            a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            q, _r = np.linalg.qr(a)
            gs.append(q.astype(np.complex64))
        s0 = (rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)).astype(np.complex64)
        s0 /= np.linalg.norm(s0)

        ref = s0.copy()
        for p, g in zip(pos, gs):
            ref = _apply_1q_plain(ref, n, p, g)

        piv, masks, order = pivots_for([1 << p for p in pos])
        got = fused_apply_reference(s0.copy(), n, masks, piv, [0] * len(pos),
                                    kron_logical([gs[i] for i in order]))
        assert np.abs(got - ref).max() < 1e-5


def test_deferred_frame_costs_no_passes():
    """A CNOT ladder applied as metadata, then 1q gates paired by XOR mask.

    The orientation trap lives here: using the pivot bit instead of
    parity(i & row) passes the norm check and fails this one.
    """
    rng = np.random.default_rng(7)
    n = 9
    s0 = (rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)).astype(np.complex64)
    s0 /= np.linalg.norm(s0)
    ladder = [(p, p + 1) for p in range(n - 1)]
    late = [(2, H2), (5, H2), (0, H2)]

    # explicit: every CNOT is a full pass
    ref = s0.copy()
    passes = 0
    for c, t in ladder:
        idx = np.arange(1 << n, dtype=np.int64)
        src = np.where((idx >> c) & 1, idx ^ (1 << t), idx)
        ref = ref[src]
        passes += 1
    for p, g in late:
        ref = _apply_1q_plain(ref, n, p, g)
        passes += 1

    # deferred: the ladder never touches memory
    frame = GF2Frame(n)
    buf = s0.copy()
    fpasses = 0
    for c, t in ladder:
        frame.cnot(c, t)
    for p, g in late:
        piv, masks, _ = pivots_for([frame.mask(p)])
        buf = fused_apply_reference(buf, n, masks, piv, [frame.row(p)], g)
        fpasses += 1
    got = _materialize(buf, frame)

    assert np.abs(got - ref).max() < 1e-5
    assert fpasses == len(late)          # 3, not 3 + 8
    assert passes == len(ladder) + len(late)
