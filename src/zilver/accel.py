"""Multithreaded strided gate kernels for Apple Silicon CPU path.

This module is the single-statevector fast path. For batched / vmap workloads,
prefer :mod:`zilver.simulator` (Metal GPU). For one-shot statevector simulation
in the 8 → 22 qubit regime this beats the GPU path by 5-15× on M-series.

Two execution modes
-------------------
* :func:`run_circuit_tape` — pre-compile the circuit to a flat tape of
  ``(op_codes, qubits, params)`` arrays, then dispatch every gate in a single
  numba JIT function with inlined kernel bodies. **No Python overhead between
  gates.** Wins at n=8..18 where dispatch overhead matters.

* :func:`run_circuit` — fuses consecutive gates into ≤k-qubit blocks (Aer-style)
  and applies each block once. Wins at n≥18 where fewer memory passes is
  decisive.

Pick one explicitly, or use ``Circuit.statevector(method="auto")`` which routes
based on qubit count.

Design
------
Each gate type has a dedicated kernel that updates the statevector in place via
strided iteration. ``numba`` ``@njit(parallel=True, fastmath=True)`` JIT-compiles
the kernels to multithreaded AArch64 with NEON SIMD, hitting all P-cores on
M-series. No allocation per gate; no transpose; no matrix multiply.

For a gate on qubit ``q`` (qubit 0 = MSB, matching :mod:`zilver.circuit`):
the statevector view is ``(2**q, 2, 2**(n-1-q))``. The kernel updates the
size-2 axis in place. ``prange`` parallelises across the outer left-dim;
the inner stride-loop auto-vectorises.

Cross-qubit fusion
------------------
:func:`fuse_block` greedily merges consecutive gate ops whose qubit-support
union is ≤ ``max_qubits`` (default 5) into a single fused unitary. Applied
once via :func:`apply_kq` instead of N times. This is the Aer-style
optimisation that turns "many small dispatches" into "few big matmuls."
"""

from __future__ import annotations

import multiprocessing as _mp
import os
from typing import Sequence

import numpy as np
from numba import njit, prange
import numba as _numba


def _configure_threads() -> int:
    """Cap numba threads to the count of Apple Silicon P-cores (8 on M1/M2 Pro,
    12 on M3 Max, etc.). Spreading parallel work to E-cores actually slows
    statevector simulation because the slower cores stretch the synchronisation
    barriers.

    Override via env var ``ZILVER_NUM_THREADS`` (e.g. for benchmarking).
    """
    env = os.environ.get("ZILVER_NUM_THREADS")
    if env:
        try:
            n = int(env)
        except ValueError:
            n = _numba.get_num_threads()
    else:
        # Heuristic: M-series chips have 8 P-cores (Pro), 10 (Max), 12 (Max+),
        # and 2-4 E-cores. cpu_count() reports logical cores (no SMT on Apple),
        # so subtract a small constant for E-cores. Cap at 12 for safety.
        total = _mp.cpu_count()
        n = max(1, min(12, total - 2))
    try:
        _numba.set_num_threads(n)
    except Exception:
        pass
    return n


_NUM_THREADS = _configure_threads()


# ----------------------------------------------------------------------------
# Single-qubit specialised kernels
# ----------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def apply_ry(state: np.ndarray, theta: float, q: int, n: int) -> None:
    """In-place RY(theta) on qubit q. complex64."""
    c = np.complex64(np.cos(theta * 0.5))
    s = np.complex64(np.sin(theta * 0.5))
    stride = 1 << (n - 1 - q)
    block = 2 * stride
    left = 1 << q
    for l in prange(left):
        base = l * block
        for r in range(stride):
            i = base + r
            j = i + stride
            a = state[i]
            b = state[j]
            state[i] = c * a - s * b
            state[j] = s * a + c * b


@njit(parallel=True, fastmath=True, cache=True)
def apply_rx(state: np.ndarray, theta: float, q: int, n: int) -> None:
    c = np.complex64(np.cos(theta * 0.5))
    s = np.complex64(0.0 + 1j * np.sin(theta * 0.5))
    stride = 1 << (n - 1 - q)
    block = 2 * stride
    left = 1 << q
    for l in prange(left):
        base = l * block
        for r in range(stride):
            i = base + r
            j = i + stride
            a = state[i]
            b = state[j]
            state[i] = c * a - s * b
            state[j] = -s * a + c * b


@njit(parallel=True, fastmath=True, cache=True)
def apply_rz(state: np.ndarray, theta: float, q: int, n: int) -> None:
    half = theta * 0.5
    p0 = np.complex64(np.cos(half) - 1j * np.sin(half))   # e^{-i half}
    p1 = np.complex64(np.cos(half) + 1j * np.sin(half))   # e^{+i half}
    stride = 1 << (n - 1 - q)
    block = 2 * stride
    left = 1 << q
    for l in prange(left):
        base = l * block
        for r in range(stride):
            state[base + r]          *= p0
            state[base + r + stride] *= p1


@njit(parallel=True, fastmath=True, cache=True)
def apply_h(state: np.ndarray, q: int, n: int) -> None:
    INV = np.complex64(1.0 / np.sqrt(2.0))
    stride = 1 << (n - 1 - q)
    block = 2 * stride
    left = 1 << q
    for l in prange(left):
        base = l * block
        for r in range(stride):
            i = base + r
            j = i + stride
            a = state[i]
            b = state[j]
            state[i] = INV * (a + b)
            state[j] = INV * (a - b)


@njit(parallel=True, fastmath=True, cache=True)
def apply_1q_general(state: np.ndarray, m00: complex, m01: complex,
                      m10: complex, m11: complex, q: int, n: int) -> None:
    """Generic 2x2 unitary (e.g., fused single-qubit composition)."""
    M00 = np.complex64(m00); M01 = np.complex64(m01)
    M10 = np.complex64(m10); M11 = np.complex64(m11)
    stride = 1 << (n - 1 - q)
    block = 2 * stride
    left = 1 << q
    for l in prange(left):
        base = l * block
        for r in range(stride):
            i = base + r
            j = i + stride
            a = state[i]
            b = state[j]
            state[i] = M00 * a + M01 * b
            state[j] = M10 * a + M11 * b


# ----------------------------------------------------------------------------
# Two-qubit specialised kernels
# ----------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def apply_cnot(state: np.ndarray, c_q: int, t_q: int, n: int) -> None:
    """In-place CNOT: when control bit is 1, flip target bit (swap pair)."""
    pc = n - 1 - c_q
    pt = n - 1 - t_q
    cmask = 1 << pc
    tmask = 1 << pt
    N = 1 << n
    for i in prange(N):
        # only swap pairs where control bit is 1 AND target bit is 0,
        # to avoid double-swapping each pair.
        if (i & cmask) and not (i & tmask):
            j = i | tmask
            a = state[i]
            b = state[j]
            state[i] = b
            state[j] = a


@njit(parallel=True, fastmath=True, cache=True)
def apply_cz(state: np.ndarray, c_q: int, t_q: int, n: int) -> None:
    """In-place CZ: phase -1 on the |11> subspace of (c_q, t_q)."""
    pc = n - 1 - c_q
    pt = n - 1 - t_q
    cmask = 1 << pc
    tmask = 1 << pt
    N = 1 << n
    for i in prange(N):
        if (i & cmask) and (i & tmask):
            state[i] = -state[i]


@njit(parallel=True, fastmath=True, cache=True)
def apply_rzz(state: np.ndarray, theta: float, qa: int, qb: int, n: int) -> None:
    half = theta * 0.5
    p_neg = np.complex64(np.cos(half) - 1j * np.sin(half))   # bits agree → ZZ=+1 → e^{-i half}
    p_pos = np.complex64(np.cos(half) + 1j * np.sin(half))   # bits differ → ZZ=-1 → e^{+i half}
    pa = n - 1 - qa
    pb = n - 1 - qb
    mask_a = 1 << pa
    mask_b = 1 << pb
    N = 1 << n
    for i in prange(N):
        agree = ((i & mask_a) != 0) == ((i & mask_b) != 0)
        if agree:
            state[i] *= p_neg
        else:
            state[i] *= p_pos


@njit(parallel=True, fastmath=True, cache=True)
def _precompute_2q_bases(pa: int, pb: int, n: int) -> np.ndarray:
    """Return int64[2^(n-2)] of base indices i00 (qa-bit and qb-bit both 0).
    JIT-compiled — at n=22 this loop walks 1M elements, must not stay in Python."""
    quarter = 1 << (n - 2)
    bases = np.empty(quarter, dtype=np.int64)
    lo = pa if pa < pb else pb
    hi = pb if pa < pb else pa
    low_mask = (1 << lo) - 1
    if hi > lo:
        mid_mask = ((1 << (hi - 1)) - 1) ^ low_mask
    else:
        mid_mask = 0
    high_drop = (1 << (hi - 1)) - 1
    for t in prange(quarter):
        base = t & low_mask
        base |= (t & mid_mask) << 1
        base |= (t & ~high_drop) << 2
        bases[t] = base
    return bases


@njit(parallel=True, fastmath=True, cache=True)
def _apply_2q_with_bases(state: np.ndarray, G: np.ndarray, bases: np.ndarray,
                         mask_a: int, mask_b: int) -> None:
    quarter = bases.shape[0]
    for t in prange(quarter):
        i00 = bases[t]
        i01 = i00 | mask_b
        i10 = i00 | mask_a
        i11 = i00 | mask_a | mask_b
        v00 = state[i00]; v01 = state[i01]
        v10 = state[i10]; v11 = state[i11]
        state[i00] = G[0, 0] * v00 + G[0, 1] * v01 + G[0, 2] * v10 + G[0, 3] * v11
        state[i01] = G[1, 0] * v00 + G[1, 1] * v01 + G[1, 2] * v10 + G[1, 3] * v11
        state[i10] = G[2, 0] * v00 + G[2, 1] * v01 + G[2, 2] * v10 + G[2, 3] * v11
        state[i11] = G[3, 0] * v00 + G[3, 1] * v01 + G[3, 2] * v10 + G[3, 3] * v11


def apply_2q_general(state: np.ndarray, G: np.ndarray,
                     qa: int, qb: int, n: int) -> None:
    """Generic 4x4 two-qubit unitary applied in-place."""
    pa = n - 1 - qa
    pb = n - 1 - qb
    mask_a = 1 << pa
    mask_b = 1 << pb
    G64 = G.astype(np.complex64)
    bases = _precompute_2q_bases(pa, pb, n)
    _apply_2q_with_bases(state, G64, bases, mask_a, mask_b)


# Note: a generic k-qubit applier for k ≥ 3 is not provided in v0.4.0.
# Fusion is capped at 2 qubits, which captures the bulk of the win for
# nearest-neighbour entangled ansätze (hardware_efficient, real_amplitudes,
# qaoa_style). A k=3..5 path will land in a follow-up.


# ----------------------------------------------------------------------------
# Cross-qubit gate fusion
# ----------------------------------------------------------------------------

def fuse_block(ops: list, max_qubits: int = 5) -> list:
    """Greedy fusion of consecutive gates whose combined qubit support ≤ max_qubits.

    Returns a list of fused ops, each described as a dict:
        {"qubits": sorted tuple, "matrix_fn": callable(params) -> 2^k x 2^k matrix}

    The original ``ops`` are :class:`zilver.circuit.GateOp` instances. The
    fused ``matrix_fn`` composes the individual gate matrices in circuit order.

    No params-rebinding is done here — the returned fns capture references to
    the original param indices via closure over their input GateOp.
    """
    from .circuit import GateOp  # local to avoid circular import at module load

    fused: list[dict] = []
    pending_ops: list[GateOp] = []
    pending_qubits: set[int] = set()

    def _flush() -> None:
        nonlocal pending_ops, pending_qubits
        if not pending_ops:
            return
        qs_sorted = sorted(pending_qubits)
        captured = list(pending_ops)

        def matrix_fn(params, _qs=qs_sorted, _ops=captured):
            return _compose_block(_qs, _ops, params)

        fused.append({"qubits": tuple(qs_sorted), "matrix_fn": matrix_fn,
                      "kind": "fused" if len(captured) > 1 else captured[0].kind})
        pending_ops = []
        pending_qubits = set()

    for op in ops:
        op_qs = set(op.qubits)
        union = pending_qubits | op_qs
        if len(union) <= max_qubits:
            pending_ops.append(op)
            pending_qubits = union
        else:
            _flush()
            pending_ops = [op]
            pending_qubits = op_qs
    _flush()
    return fused


_SQRT_HALF = 1.0 / np.sqrt(2.0)


def _gate_matrix_np(op, params: np.ndarray, dtype=np.complex64) -> np.ndarray:
    """Build the small (2^k, 2^k) gate matrix for a single GateOp directly in
    NumPy. Avoids the MLX dispatch + sync that would otherwise dominate the
    fusion path. Falls back to the MLX gate_fn for unknown kinds.

    ``dtype`` controls precision: ``np.complex64`` (default) or
    ``np.complex128`` for the high-precision path."""
    kind = op.kind
    pidx = op.param_indices

    if kind == "ry":
        theta = float(params[pidx[0]])
        c, s = np.cos(theta * 0.5), np.sin(theta * 0.5)
        return np.array([[c, -s], [s, c]], dtype=dtype)
    if kind == "rx":
        theta = float(params[pidx[0]])
        c, s = np.cos(theta * 0.5), np.sin(theta * 0.5)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=dtype)
    if kind == "rz":
        theta = float(params[pidx[0]])
        p0 = np.exp(-1j * theta * 0.5)
        p1 = np.exp( 1j * theta * 0.5)
        return np.array([[p0, 0], [0, p1]], dtype=dtype)
    if kind == "h":
        return np.array([[ _SQRT_HALF,  _SQRT_HALF],
                         [ _SQRT_HALF, -_SQRT_HALF]], dtype=dtype)
    if kind == "x":
        return np.array([[0, 1], [1, 0]], dtype=dtype)
    if kind == "cnot":
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                         [0, 0, 0, 1], [0, 0, 1, 0]], dtype=dtype)
    if kind == "cz":
        return np.diag([1, 1, 1, -1]).astype(dtype)
    if kind == "rzz":
        theta = float(params[pidx[0]])
        p_neg = np.exp(-1j * theta * 0.5)
        p_pos = np.exp( 1j * theta * 0.5)
        return np.diag([p_neg, p_pos, p_pos, p_neg]).astype(dtype)
    if kind == "u3":
        theta = float(params[pidx[0]])
        phi   = float(params[pidx[1]])
        lam   = float(params[pidx[2]])
        ct, st = np.cos(theta * 0.5), np.sin(theta * 0.5)
        return np.array([
            [ct,                       -np.exp(1j * lam) * st],
            [np.exp(1j * phi) * st,     np.exp(1j * (phi + lam)) * ct],
        ], dtype=dtype)

    # Unknown kind: fall back to MLX gate_fn (one dispatch — acceptable as fallback).
    from ._array import mx
    if pidx:
        p_mx = mx.array(np.asarray(params[pidx], dtype=np.float32))
        G_mx = op.gate_fn(p_mx)
    else:
        G_mx = op.gate_fn(None)
    mx.eval(G_mx)
    return np.array(G_mx.tolist(), dtype=dtype)


def _compose_block(qubits_sorted: list[int], ops: list, params: np.ndarray) -> np.ndarray:
    """Compose a list of GateOps acting on a shared support of qubits into one 2^k x 2^k unitary.
    All gate matrices are built directly in NumPy — no MLX dispatch in the hot path."""
    k = len(qubits_sorted)
    dim = 1 << k
    pos_of = {q: idx for idx, q in enumerate(qubits_sorted)}

    U = np.eye(dim, dtype=np.complex64)
    for op in ops:
        gate_np = _gate_matrix_np(op, params)

        # Embed gate_np (acting on op.qubits) into the k-qubit block by tensor reshape
        ks = len(op.qubits)
        block_dim = 1 << ks
        # Reshape gate to (2,)*2*ks
        gate_t = gate_np.reshape((2,) * (2 * ks))
        # Build full block as identity tensor, then contract with gate on op.qubits' local positions
        local_pos = [pos_of[q] for q in op.qubits]

        # Build the full k-qubit gate matrix by inserting gate_t into identity on remaining qubits
        # Strategy: convert U from (dim, dim) to tensor of shape (2,)*2k and apply gate via tensordot.
        U_t = U.reshape((2,) * (2 * k))
        # U_t indices: (out_0, out_1, ..., out_{k-1}, in_0, in_1, ..., in_{k-1})
        # We compose on the LEFT: U' = G @ U  — so contract out-axes that match local_pos with gate's in-axes.
        in_axes = list(range(ks, 2 * ks))                                   # gate's input axes
        u_out_axes = local_pos                                              # U's output axes touched by gate
        contracted = np.tensordot(gate_t, U_t, axes=(in_axes, u_out_axes))
        # Re-order back: contracted now has gate-output axes first (ks), then unaffected U-output axes (k-ks),
        # then U-input axes (k). Need to put gate-output axes back into their local positions.
        new_order_out = list(range(ks))
        unaffected_out = [i for i in range(k) if i not in local_pos]
        # Build a length-k axis list interleaving gate-output and unaffected
        out_axis_map = [0] * k
        for i, p in enumerate(local_pos):
            out_axis_map[p] = i
        ua_iter = iter(range(ks, ks + len(unaffected_out)))
        for i in range(k):
            if i not in local_pos:
                out_axis_map[i] = next(ua_iter)
        # input axes come last (positions ks + len(unaffected_out) .. ks+len(unaffected_out)+k-1)
        in_axis_start = ks + len(unaffected_out)
        in_axis_map = list(range(in_axis_start, in_axis_start + k))
        perm = out_axis_map + in_axis_map
        U_t = np.transpose(contracted, perm)
        U = U_t.reshape(dim, dim)

    return U


# ----------------------------------------------------------------------------
# Circuit execution
# ----------------------------------------------------------------------------

def run_circuit(circuit, params: np.ndarray, fuse_max: int = 2) -> np.ndarray:
    """Execute a :class:`zilver.circuit.Circuit` on the accelerated CPU path.

    Returns the final ``2**n`` complex64 statevector.

    Parameters
    ----------
    circuit:
        A :class:`zilver.circuit.Circuit` instance.
    params:
        ``(P,)`` parameter vector (float). Converted to float64 internally.
    fuse_max:
        Maximum qubit support per fused block. ``1`` disables fusion and
        dispatches each gate via its specialised kernel. ``2`` (default) is
        the sweet spot for nearest-neighbour ansätze on M-series:
        consecutive single-qubit gates collapse into a single 2-qubit unitary
        with the surrounding entangler.
    """
    if fuse_max not in (1, 2):
        raise ValueError(f"fuse_max must be 1 or 2 in this release; got {fuse_max}")
    params = np.asarray(params, dtype=np.float64)
    n = circuit.n_qubits

    state = np.zeros(1 << n, dtype=np.complex64)
    state[0] = np.complex64(1.0)

    if fuse_max == 1:
        for op in circuit._ops:
            _apply_single_op(state, op, params, n)
        return state

    blocks = fuse_block(circuit._ops, max_qubits=fuse_max)
    for b in blocks:
        qs = b["qubits"]
        k = len(qs)
        U = b["matrix_fn"](params)
        if k == 1:
            apply_1q_general(state, U[0, 0], U[0, 1], U[1, 0], U[1, 1], qs[0], n)
        elif k == 2:
            apply_2q_general(state, U, qs[0], qs[1], n)
        else:
            raise NotImplementedError(
                f"k-qubit fused block (k={k}) is reserved for v0.4.x. "
                "Pass fuse_max=1 to bypass fusion."
            )
    return state


def _apply_single_op(state: np.ndarray, op, params: np.ndarray, n: int) -> None:
    """Dispatch an unfused GateOp to its specialised kernel."""
    from ._array import mx
    kind = op.kind
    qs = op.qubits

    if kind in ("ry", "rx", "rz") and len(qs) == 1:
        theta = float(params[op.param_indices[0]])
        if kind == "ry":   apply_ry(state, theta, qs[0], n)
        elif kind == "rx": apply_rx(state, theta, qs[0], n)
        else:              apply_rz(state, theta, qs[0], n)
        return
    if kind == "h":
        apply_h(state, qs[0], n); return
    if kind == "cnot":
        apply_cnot(state, qs[0], qs[1], n); return
    if kind == "cz":
        apply_cz(state, qs[0], qs[1], n); return
    if kind == "rzz":
        theta = float(params[op.param_indices[0]])
        apply_rzz(state, theta, qs[0], qs[1], n); return

    # Fallback: build matrix via the gate_fn and use the generic applier
    if op.param_indices:
        p_mx = mx.array(np.asarray(params[op.param_indices], dtype=np.float32))
        G_mx = op.gate_fn(p_mx)
    else:
        G_mx = op.gate_fn(None)
    mx.eval(G_mx)
    G = np.array(G_mx.tolist(), dtype=np.complex64)
    if len(qs) == 1:
        apply_1q_general(state, G[0, 0], G[0, 1], G[1, 0], G[1, 1], qs[0], n)
    elif len(qs) == 2:
        apply_2q_general(state, G, qs[0], qs[1], n)
    else:
        apply_kq_general(state, G, np.array(qs, dtype=np.int64), n)


# ----------------------------------------------------------------------------
# Tape-lowered dispatch (eliminates Python per-call overhead)
# ----------------------------------------------------------------------------

# Op codes (kept stable; values are part of the tape ABI)
OP_RY   = np.int32(0)
OP_RZ   = np.int32(1)
OP_RX   = np.int32(2)
OP_H    = np.int32(3)
OP_X    = np.int32(4)
OP_CNOT = np.int32(10)
OP_CZ   = np.int32(11)
OP_RZZ  = np.int32(12)
OP_U3   = np.int32(20)
# Op codes for fused 1q / 2q (matrix carried separately in tape_mat)
OP_GENERIC_1Q = np.int32(30)
OP_GENERIC_2Q = np.int32(31)


_KIND_TO_OP = {
    "ry": OP_RY, "rz": OP_RZ, "rx": OP_RX, "h": OP_H, "x": OP_X,
    "cnot": OP_CNOT, "cz": OP_CZ, "rzz": OP_RZZ, "u3": OP_U3,
}


def compile_tape(circuit) -> dict:
    """Lower a Circuit to numpy arrays for the JIT'd tape dispatcher.

    Returns a dict with:
      codes  : int32[M] op kind
      q0,q1  : int32[M] target qubit indices (q1 = -1 for 1q gates)
      p0,p1,p2 : int32[M] parameter indices (-1 if unused)
      n      : qubit count (mirror of circuit.n_qubits, kept here for cache key)

    Generic / fused gate matrices are NOT supported in this tape format yet.
    Unknown gate kinds raise NotImplementedError.
    """
    M = len(circuit._ops)
    codes = np.empty(M, dtype=np.int32)
    q0 = np.full(M, -1, dtype=np.int32)
    q1 = np.full(M, -1, dtype=np.int32)
    p0 = np.full(M, -1, dtype=np.int32)
    p1 = np.full(M, -1, dtype=np.int32)
    p2 = np.full(M, -1, dtype=np.int32)

    for i, op in enumerate(circuit._ops):
        kind = op.kind
        if kind not in _KIND_TO_OP:
            raise NotImplementedError(
                f"Tape compilation does not yet support gate kind '{kind}'. "
                "Use run_circuit (fused path) for circuits containing this gate."
            )
        codes[i] = _KIND_TO_OP[kind]
        q0[i] = op.qubits[0]
        if len(op.qubits) >= 2:
            q1[i] = op.qubits[1]
        if op.param_indices:
            p0[i] = op.param_indices[0]
            if len(op.param_indices) >= 2:
                p1[i] = op.param_indices[1]
            if len(op.param_indices) >= 3:
                p2[i] = op.param_indices[2]

    return {"codes": codes, "q0": q0, "q1": q1,
            "p0": p0, "p1": p1, "p2": p2, "n": circuit.n_qubits}


@njit(parallel=True, fastmath=True, cache=True)
def _run_tape(state: np.ndarray,
              codes: np.ndarray, q0: np.ndarray, q1: np.ndarray,
              p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
              params: np.ndarray, n: int) -> None:
    """Execute a flat gate tape in place on ``state``.

    Numba inlines all branches and prange-parallelises each gate's strided
    inner loop. Gates run sequentially (data dependency) but their per-gate
    workload is fully multi-threaded.
    """
    M = codes.shape[0]
    INV = np.complex64(1.0 / np.sqrt(2.0))
    for op_i in range(M):
        c = codes[op_i]
        a = q0[op_i]

        # ----- Single-qubit rotations: RY, RZ, RX -----
        if c == 0:   # OP_RY
            theta = params[p0[op_i]]
            cs = np.complex64(np.cos(theta * 0.5))
            sn = np.complex64(np.sin(theta * 0.5))
            stride = 1 << (n - 1 - a)
            block = 2 * stride
            left = 1 << a
            for l in prange(left):
                base = l * block
                for r in range(stride):
                    i = base + r
                    j = i + stride
                    aa = state[i]
                    bb = state[j]
                    state[i] = cs * aa - sn * bb
                    state[j] = sn * aa + cs * bb
        elif c == 1:  # OP_RZ
            half = params[p0[op_i]] * 0.5
            p_neg = np.complex64(np.cos(half) - 1j * np.sin(half))
            p_pos = np.complex64(np.cos(half) + 1j * np.sin(half))
            stride = 1 << (n - 1 - a)
            block = 2 * stride
            left = 1 << a
            for l in prange(left):
                base = l * block
                for r in range(stride):
                    state[base + r]          *= p_neg
                    state[base + r + stride] *= p_pos
        elif c == 2:  # OP_RX
            theta = params[p0[op_i]]
            cs = np.complex64(np.cos(theta * 0.5))
            sn = np.complex64(0.0 + 1j * np.sin(theta * 0.5))
            stride = 1 << (n - 1 - a)
            block = 2 * stride
            left = 1 << a
            for l in prange(left):
                base = l * block
                for r in range(stride):
                    i = base + r
                    j = i + stride
                    aa = state[i]
                    bb = state[j]
                    state[i] = cs * aa - sn * bb
                    state[j] = -sn * aa + cs * bb
        elif c == 3:  # OP_H
            stride = 1 << (n - 1 - a)
            block = 2 * stride
            left = 1 << a
            for l in prange(left):
                base = l * block
                for r in range(stride):
                    i = base + r
                    j = i + stride
                    aa = state[i]
                    bb = state[j]
                    state[i] = INV * (aa + bb)
                    state[j] = INV * (aa - bb)
        elif c == 4:  # OP_X
            stride = 1 << (n - 1 - a)
            block = 2 * stride
            left = 1 << a
            for l in prange(left):
                base = l * block
                for r in range(stride):
                    i = base + r
                    j = i + stride
                    tmp = state[i]
                    state[i] = state[j]
                    state[j] = tmp

        # ----- Two-qubit: CNOT, CZ, RZZ -----
        elif c == 10:  # OP_CNOT
            cq = q0[op_i]
            tq = q1[op_i]
            pc = n - 1 - cq
            pt = n - 1 - tq
            cmask = 1 << pc
            tmask = 1 << pt
            N = 1 << n
            for i in prange(N):
                if (i & cmask) != 0 and (i & tmask) == 0:
                    j = i | tmask
                    aa = state[i]
                    bb = state[j]
                    state[i] = bb
                    state[j] = aa
        elif c == 11:  # OP_CZ
            cq = q0[op_i]
            tq = q1[op_i]
            cmask = 1 << (n - 1 - cq)
            tmask = 1 << (n - 1 - tq)
            N = 1 << n
            for i in prange(N):
                if (i & cmask) != 0 and (i & tmask) != 0:
                    state[i] = -state[i]
        elif c == 12:  # OP_RZZ
            theta = params[p0[op_i]]
            half = theta * 0.5
            p_neg = np.complex64(np.cos(half) - 1j * np.sin(half))
            p_pos = np.complex64(np.cos(half) + 1j * np.sin(half))
            mask_a = 1 << (n - 1 - q0[op_i])
            mask_b = 1 << (n - 1 - q1[op_i])
            N = 1 << n
            for i in prange(N):
                bit_a = (i & mask_a) != 0
                bit_b = (i & mask_b) != 0
                if bit_a == bit_b:
                    state[i] *= p_neg
                else:
                    state[i] *= p_pos

        # ----- U3 (3 params) -----
        elif c == 20:  # OP_U3
            theta = params[p0[op_i]]
            phi   = params[p1[op_i]]
            lam   = params[p2[op_i]]
            ct = np.cos(theta * 0.5)
            st = np.sin(theta * 0.5)
            # 4 matrix elements
            m00 = np.complex64(ct)
            m01 = np.complex64(-np.cos(lam) * st - 1j * np.sin(lam) * st)
            m10 = np.complex64( np.cos(phi) * st + 1j * np.sin(phi) * st)
            m11 = np.complex64( np.cos(phi + lam) * ct + 1j * np.sin(phi + lam) * ct)
            stride = 1 << (n - 1 - a)
            block = 2 * stride
            left = 1 << a
            for l in prange(left):
                base = l * block
                for r in range(stride):
                    i = base + r
                    j = i + stride
                    aa = state[i]
                    bb = state[j]
                    state[i] = m00 * aa + m01 * bb
                    state[j] = m10 * aa + m11 * bb


def run_circuit_tape(circuit, params: np.ndarray, tape: dict | None = None) -> np.ndarray:
    """Execute a Circuit via the tape-lowered JIT dispatcher.

    Parameters
    ----------
    circuit:
        A :class:`zilver.circuit.Circuit`.
    params:
        (P,) parameter vector.
    tape:
        Optional pre-compiled tape dict (from :func:`compile_tape`). Pass this
        if you call the same circuit shape many times (training loop) to skip
        the per-call tape build.

    Returns
    -------
    state : np.ndarray
        ``(2**n,)`` complex64 statevector.
    """
    params = np.asarray(params, dtype=np.float64)
    n = circuit.n_qubits
    if tape is None:
        tape = compile_tape(circuit)
    state = np.zeros(1 << n, dtype=np.complex64)
    state[0] = np.complex64(1.0)
    _run_tape(state, tape["codes"], tape["q0"], tape["q1"],
              tape["p0"], tape["p1"], tape["p2"], params, n)
    return state


# ----------------------------------------------------------------------------
# Auto-routing entry point
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Pure-NumPy strided path (no numba — single-threaded, Accelerate-friendly)
# Wins at n ≤ ~14 where prange / JIT spawn overhead exceeds the actual work.
# ----------------------------------------------------------------------------

def _apply_1q_numpy_strided(state: np.ndarray, M: np.ndarray, q: int, n: int) -> np.ndarray:
    """In-place 1q gate via NumPy reshape + vectorised arithmetic.

    MEMORY, WHICH IS THE POINT AT LARGE n. The obvious form allocates a whole
    new statevector per gate (``out = np.empty_like(v)``) and one temporary per
    product, so a circuit of ~3n gates churns several multiples of the state.
    Measured on a 30-qubit run: 56 GB peak for an 8.6 GB state, and the process
    was OOM-killed at 31 qubits on a 60 GB box -- by allocation overhead, not by
    the state itself.

    A 1q gate is block-diagonal in the (left, 2, stride) view: the halves mix
    only with each other. Two HALF-sized scratch buffers therefore suffice, so
    the peak is one extra state instead of several, and it is reused across the
    two output rows. The order matters -- ``b`` is updated only after both rows
    have read the original ``b``, and ``a`` only after ``tmp2`` has read the
    original ``a``.
    """
    stride = 1 << (n - 1 - q)
    left   = 1 << q
    v = state.reshape(left, 2, stride)
    a = v[:, 0, :]
    b = v[:, 1, :]

    tmp  = np.empty_like(a)                 # holds the new upper half
    tmp2 = np.empty_like(a)                 # scratch, reused

    np.multiply(a, M[0, 0], out=tmp)        # tmp  = M00 a
    np.multiply(b, M[0, 1], out=tmp2)       # tmp2 = M01 b
    tmp += tmp2                             # tmp  = M00 a + M01 b

    np.multiply(a, M[1, 0], out=tmp2)       # tmp2 = M10 a   (a still original)
    b *= M[1, 1]                            # b    = M11 b   (b already consumed)
    b += tmp2                               # b    = M10 a + M11 b
    a[:] = tmp                              # a    = M00 a + M01 b

    return state


def _apply_2q_numpy_strided(state: np.ndarray, M: np.ndarray, qa: int, qb: int, n: int) -> np.ndarray:
    """Out-of-place 2q gate via NumPy gather/multiply/scatter."""
    pa = n - 1 - qa
    pb = n - 1 - qb
    quarter_count = 1 << (n - 2)

    # Precompute the i00 indices (qa-bit and qb-bit both 0) once per call.
    lo = min(pa, pb); hi = max(pa, pb)
    t = np.arange(quarter_count, dtype=np.int64)
    low_mask = (1 << lo) - 1
    mid_mask = (((1 << (hi - 1)) - 1) ^ low_mask) if hi > lo else 0
    high_drop = (1 << (hi - 1)) - 1
    i00 = (t & low_mask) | ((t & mid_mask) << 1) | ((t & ~high_drop) << 2)
    mask_a = 1 << pa
    mask_b = 1 << pb
    i01 = i00 | mask_b
    i10 = i00 | mask_a
    i11 = i00 | mask_a | mask_b

    v00 = state[i00]; v01 = state[i01]; v10 = state[i10]; v11 = state[i11]
    out = state.copy()
    out[i00] = M[0, 0] * v00 + M[0, 1] * v01 + M[0, 2] * v10 + M[0, 3] * v11
    out[i01] = M[1, 0] * v00 + M[1, 1] * v01 + M[1, 2] * v10 + M[1, 3] * v11
    out[i10] = M[2, 0] * v00 + M[2, 1] * v01 + M[2, 2] * v10 + M[2, 3] * v11
    out[i11] = M[3, 0] * v00 + M[3, 1] * v01 + M[3, 2] * v10 + M[3, 3] * v11
    return out


def run_circuit_strided(circuit, params: np.ndarray, dtype=np.complex64) -> np.ndarray:
    """Execute a Circuit via the pure-NumPy strided path (no numba, no MLX).

    Parameters
    ----------
    circuit:
        A :class:`zilver.circuit.Circuit`.
    params:
        ``(P,)`` parameter vector.
    dtype:
        Statevector complex dtype. ``np.complex64`` (default, ~10⁻⁷ precision)
        or ``np.complex128`` (~10⁻¹⁵ precision; ~2× memory, ~2× slower).

    Wins at n ≤ 14 on M-series where the JIT overhead of the numba kernels
    exceeds the per-gate compute. Above ~14 qubits, prefer
    :func:`run_circuit_tape` (complex64 only) or :func:`run_circuit` for
    complex64; this strided path is the only complex128 option in v0.4.0.
    """
    params = np.asarray(params, dtype=np.float64)
    n = circuit.n_qubits
    state = np.zeros(1 << n, dtype=dtype)
    state[0] = dtype(1.0) if hasattr(dtype, "__call__") else np.array(1.0, dtype=dtype).item()
    state[0] = 1.0
    for op in circuit._ops:
        M = _gate_matrix_np(op, params, dtype=dtype)
        if len(op.qubits) == 1:
            state = _apply_1q_numpy_strided(state, M, op.qubits[0], n)
        elif len(op.qubits) == 2:
            state = _apply_2q_numpy_strided(state, M, op.qubits[0], op.qubits[1], n)
        else:
            raise NotImplementedError(
                f"strided path supports only 1q/2q gates; got {len(op.qubits)}-qubit op"
            )
    return state


# ----------------------------------------------------------------------------
# Auto-routing entry point
# ----------------------------------------------------------------------------

def run_circuit_auto(circuit, params: np.ndarray, dtype=np.complex64) -> np.ndarray:
    """Route to the best CPU backend based on qubit count and dtype.

    Empirical thresholds on M-series (Apple M1 Pro / M2 / M3):

    *  dtype = complex128  → :func:`run_circuit_strided` (only complex128 path)
    *  dtype = complex64, n ≤ 14  → :func:`run_circuit_strided`
    *  dtype = complex64, n ≥ 15  → :func:`run_circuit_tape`  (numba multithreaded)

    For batched / vmap workloads (parameter sweeps, gradient batches, fidelity
    kernels) prefer the MLX GPU path :mod:`zilver.simulator` via
    ``Circuit.statevector_batch``.
    """
    n = circuit.n_qubits
    if dtype == np.complex128 or np.dtype(dtype) == np.complex128:
        return run_circuit_strided(circuit, params, dtype=np.complex128)
    try:
        if n <= 14:
            return run_circuit_strided(circuit, params, dtype=np.complex64)
        return run_circuit_tape(circuit, params)
    except NotImplementedError:
        return run_circuit(circuit, params, fuse_max=2)


__all__ = [
    "apply_ry", "apply_rx", "apply_rz", "apply_h",
    "apply_cnot", "apply_cz", "apply_rzz",
    "apply_1q_general", "apply_2q_general",
    "fuse_block",
    "run_circuit",           # fused (numba) path
    "run_circuit_tape",      # tape-lowered (numba) path
    "run_circuit_strided",   # pure-NumPy / Accelerate path
    "run_circuit_auto",      # auto-route
    "compile_tape",
]
