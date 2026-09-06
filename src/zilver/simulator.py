"""Statevector simulation."""

from __future__ import annotations
from typing import Sequence
from ._array import mx, HAS_COMPLEX
import numpy as np


class StateVector:
    """Wrapper around a quantum statevector.

    Internally stores either:
      * an MLX array (default, complex64) — for the MLX / Metal GPU paths
      * a NumPy array (complex64 or complex128) — for the accel CPU path,
        especially when ``precision="double"`` (MLX has no complex128 type)
    """

    def __init__(self, n_qubits: int,
                 array: "mx.array | np.ndarray | None" = None):
        self.n_qubits = n_qubits
        self._state_np: np.ndarray | None = None
        self._state: mx.array | None = None
        if array is None:
            state = np.zeros(2**n_qubits, dtype=np.complex64)
            state[0] = 1.0
            self._state = mx.array(state)
            return
        # On a device with no complex dtype the state arrives as a real
        # (2, 2**n) pair and must stay that way. Casting it back to complex here
        # is what DirectML answers by aborting the process -- and this
        # constructor runs on every circuit, so it undid the whole real-pair
        # path from the far end.
        if not HAS_COMPLEX:
            # A complex numpy array here came from a CPU path (accel/numba runs
            # the whole circuit in numpy complex64 and hands the result over).
            # It has no business on a device that cannot hold complex -- pushing
            # it there is what aborts DirectML. Keep it on the host.
            if isinstance(array, np.ndarray):
                if np.iscomplexobj(array):
                    self._state_np = array.astype(np.complex64, copy=False)
                else:
                    self._state = mx.array(array)
            else:
                self._state = array
        elif isinstance(array, np.ndarray):
            # Preserve complex128 if given; only downcast unrecognised dtypes
            if array.dtype == np.complex128:
                self._state_np = array
            elif array.dtype == np.complex64:
                self._state = mx.array(array)
            else:
                self._state = mx.array(array.astype(np.complex64))
        else:
            # mx.array — MLX only supports complex64 today
            self._state = array.astype(mx.complex64)

    @property
    def array(self) -> mx.array:
        """Return the MLX-backed statevector. Materialises from NumPy if
        needed (lossy: complex128 → complex64)."""
        if self._state is not None:
            return self._state
        # Downcast from numpy complex128 to complex64 MLX
        return mx.array(self._state_np.astype(np.complex64))

    @property
    def dtype(self) -> np.dtype:
        """The complex dtype actually held — complex64 or complex128."""
        if self._state_np is not None:
            return self._state_np.dtype
        return np.complex64

    @classmethod
    def zero_state(cls, n_qubits: int) -> "StateVector":
        return cls(n_qubits)

    @classmethod
    def from_array(cls, arr: "mx.array | np.ndarray", n_qubits: int) -> "StateVector":
        return cls(n_qubits, arr)

    def probabilities(self) -> mx.array:
        return mx.abs(self.array) ** 2

    def numpy(self) -> np.ndarray:
        """Return the statevector as a NumPy array. Preserves complex128
        when the StateVector was constructed at double precision.

        NEVER go via .tolist(). That materialises one boxed Python complex per
        amplitude -- 2^n objects at roughly 32 bytes plus a pointer each, so
        about 5x the array it is converting, and it was the single largest
        memory term in the whole pipeline. Measured at 25 qubits: peak went
        from 2.81x the state to 6.27x on this call alone.

        Both backends can do better. Under the numpy fallback the state IS an
        ndarray already, so this is free. MLX arrays expose the buffer protocol,
        so np.asarray copies once rather than boxing. .tolist() remains only as
        a last resort for an array type that supports neither.
        """
        if self._state_np is not None:
            return self._state_np
        mx.eval(self._state)
        # A device without a complex dtype carries the state as a real (2, 2**n)
        # pair, row 0 real and row 1 imaginary. Rejoin it here so callers never
        # see the representation.
        # A device tensor cannot be handed to numpy directly -- DirectML says
        # "can't convert privateuseone:0 device type" -- so come to the host
        # first, once, before anything inspects shape or dtype.
        s = self._state
        if hasattr(s, "detach"):
            s = s.detach()
        if hasattr(s, "cpu"):
            s = s.cpu()
        shp = tuple(getattr(s, "shape", ()))
        dt = str(getattr(s, "dtype", ""))
        is_pair = (len(shp) == 2 and shp[0] == 2 and shp[1] == 2 ** self.n_qubits
                   and "complex" not in dt.lower())
        if is_pair:
            # A device with no complex dtype carries the state as a real
            # (2, 2**n) pair, row 0 real and row 1 imaginary. The dtype test is
            # not redundant: shape alone also matches a genuinely complex state
            # that happens to be two rows wide, and casting one of those through
            # here silently discards the imaginary part.
            a = np.asarray(s, dtype=np.float32)
            return (a[0] + 1j * a[1]).astype(np.complex64)
        if isinstance(s, np.ndarray):
            return s.astype(np.complex64, copy=False)
        try:
            return np.asarray(s, dtype=np.complex64)
        except (TypeError, ValueError):
            return np.array(s.tolist(), dtype=np.complex64)

    def __repr__(self) -> str:
        return f"StateVector(n_qubits={self.n_qubits}, dtype={self.dtype})"


def _apply_gate_real_strided(state, gate, qubits, n):
    """The real-pair gate, without an n-dimensional tensor.

    Composes the two constructions already verified separately: the real
    lifting (a complex product is a real one of twice the size) and the strided
    view (a gate needs 3 or 5 axes, never n). With psi = a + i b and
    G = P + i Q,

        out_re = P a - Q b
        out_im = Q a + P b

    and each of those four products is a REAL gate applied to a REAL vector --
    exactly what _apply_gate_strided does. So this is four calls to code that is
    already tested, rather than a fifth hand-written index dance.
    """
    P, Q = gate[0], gate[1]
    a, b = state[0], state[1]
    out_re = _apply_gate_strided(a, P, qubits, n) - _apply_gate_strided(b, Q, qubits, n)
    out_im = _apply_gate_strided(a, Q, qubits, n) + _apply_gate_strided(b, P, qubits, n)
    return mx.stack([out_re, out_im])


def _apply_gate_real(state, gate, qubits, n):
    """apply_gate for a device that has no complex dtype.

    DirectML is the case in hand: it reaches an AMD or Intel GPU from Windows
    and from inside WSL2, where ROCm's /dev/kfd does not exist, but it has no
    ComplexFloat -- and it does not raise on one, it aborts the process.

    A complex matrix-vector product is a real one of twice the size. Writing
    psi = a + i b and G = P + i Q,

        [out_re]   [ P  -Q ] [a]
        [out_im] = [ Q   P ] [b]

    so the state is carried as a real (2, 2**n) tensor -- row 0 real part, row
    1 imaginary -- and each gate is lifted ONCE to its real block form. The hot
    loop stays a single real matmul rather than four, and nothing downstream
    branches on dtype.

    metal.py solves the same problem the same way, carrying an interleaved
    float32 array to dodge MLX's complex64 gap in custom kernels.
    """
    k = len(qubits)
    qubits = list(qubits)

    # Same rank ceiling, one axis lower: the real pair carries a leading
    # component axis, so the reshape below builds n+1 axes, not n.
    if k <= 2 and n + 1 > _MAX_RANK:
        return _apply_gate_real_strided(state, gate, qubits, n)

    other = [i for i in range(n) if i not in qubits]
    perm = qubits + other
    inv = [0] * n
    for i, p in enumerate(perm):
        inv[p] = i

    # (2, 2**n) -> (2, 2,2,...,2); axis 0 is the re/im component, so every
    # qubit axis shifts by one.
    t = state.reshape([2] + [2] * n)
    t = mx.transpose(t, [0] + [p + 1 for p in perm])
    t = t.reshape(2, 2 ** k, 2 ** (n - k))

    P, Q = gate[0], gate[1]                       # real and imaginary blocks
    a, b = t[0], t[1]
    out_re = mx.matmul(P, a) - mx.matmul(Q, b)
    out_im = mx.matmul(Q, a) + mx.matmul(P, b)

    t = mx.stack([out_re, out_im])
    t = t.reshape([2] + [2] * n)
    t = mx.transpose(t, [0] + [i + 1 for i in inv])
    return t.reshape(2, 2 ** n)


#: Widest [2]*n reshape any backend here will accept. 16 is the MPS limit and
#: is at or below every other device's.
_MAX_RANK = 16


def _apply_gate_strided(state, gate, qubits, n):
    """apply_gate without an n-dimensional tensor.

    The permutation form reshapes the state to [2]*n -- 28 axes at 28 qubits --
    and every GPU backend caps tensor rank far below that. MPS refuses outright
    ("MPS supports tensors with dimensions <= 16"), and DirectML has its own
    limit. So the device path cannot use it at all above 16 qubits.

    A gate does not need n axes. One qubit at position q splits the state into
    (left, 2, stride) -- three axes, any n. Two qubits split it into
    (A, 2, B, 2, C) -- five. That is what the strided CPU path in accel.py has
    always done; this is the same view, expressed in `mx` so it runs on a
    device.
    """
    k = len(qubits)
    if k == 1:
        q = qubits[0]
        stride = 1 << (n - 1 - q)
        left = 1 << q
        v = state.reshape(left, 2, stride)
        a, b = v[:, 0, :], v[:, 1, :]
        out0 = gate[0, 0] * a + gate[0, 1] * b
        out1 = gate[1, 0] * a + gate[1, 1] * b
        return mx.stack([out0, out1], axis=1).reshape(-1)

    if k == 2:
        qa, qb = qubits
        pa, pb = n - 1 - qa, n - 1 - qb
        hi, lo = max(pa, pb), min(pa, pb)
        C = 1 << lo
        B = 1 << (hi - lo - 1)
        A = 1 << (n - 1 - hi)
        v = state.reshape(A, 2, B, 2, C)
        # gate index is 2*(qa bit) + (qb bit); axis 1 carries the higher bit
        if pa > pb:
            blk = [v[:, i >> 1, :, i & 1, :] for i in range(4)]
        else:
            blk = [v[:, i & 1, :, i >> 1, :] for i in range(4)]
        out = [sum(gate[r, c] * blk[c] for c in range(4)) for r in range(4)]
        if pa > pb:
            rows = [mx.stack([out[0], out[1]], axis=2), mx.stack([out[2], out[3]], axis=2)]
        else:
            rows = [mx.stack([out[0], out[2]], axis=2), mx.stack([out[1], out[3]], axis=2)]
        return mx.stack(rows, axis=1).reshape(-1)

    raise NotImplementedError(
        f"strided path covers 1- and 2-qubit gates; got {k}. "
        "Three-qubit gates decompose, or fall back to the permutation form on CPU."
    )


def apply_gate(
    state: mx.array,
    gate: mx.array,
    qubits: Sequence[int],
    n: int,
    stream: mx.Stream | mx.Device = mx.gpu,
) -> mx.array:
    """
    Apply a k-qubit unitary gate to a statevector.

    Args:
        state:  (2**n,) complex64 statevector.
        gate:   (2**k, 2**k) complex64 unitary.
        qubits: Target qubit indices (0 = most significant).
        n:      Total number of qubits.
        stream: MLX stream or device for the matmul. Defaults to ``mx.gpu``
                (compute-dense; use ``mx.cpu`` for small circuits where GPU
                dispatch overhead dominates).

    Returns:
        Updated (2**n,) complex64 statevector.

    Implementation uses axis permutation to bring target qubits to the front,
    applies the gate as a matrix multiply, then permutes back. This is
    correct for arbitrary qubit orderings and compiles cleanly to Metal.
    """
    k = len(qubits)
    qubits = list(qubits)

    # The reshape below builds a [2]*n tensor -- 20 axes at 20 qubits -- and GPU
    # backends cap tensor rank well under that (MPS refuses above 16). So 1- and
    # 2-qubit gates take the strided view instead: 3 and 5 axes, any n. Verified
    # identical to this form at 1.7e-07, machine precision.
    if k <= 2 and n > _MAX_RANK:
        return _apply_gate_strided(state, gate, qubits, n)

    other = [i for i in range(n) if i not in qubits]

    perm = qubits + other
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i

    # Reshape to per-qubit tensor, permute, flatten leading k dims
    tensor = mx.transpose(state.reshape([2] * n), perm)
    tensor = tensor.reshape(2**k, 2**(n - k))

    # Gate application: (2^k, 2^k) @ (2^k, 2^(n-k)) -> (2^k, 2^(n-k))
    tensor = mx.matmul(gate, tensor, stream=stream)

    # Reshape and permute back
    tensor = tensor.reshape([2] * n)
    tensor = mx.transpose(tensor, inv_perm)
    return tensor.reshape(2**n)


def expectation_z(state: mx.array, qubit: int, n: int) -> mx.array:
    """
    Compute <Z_q> = <psi|Z_q|psi> for a single qubit.

    Eigenvalue of Z_q is +1 if the q-th bit of basis index is 0, else -1.
    Runs on CPU — elementwise ops over the probability vector are
    overhead-bound on the GPU for typical circuit sizes.
    """
    probs = mx.abs(state, stream=mx.cpu) ** 2
    indices = mx.arange(2**n, stream=mx.cpu)
    # Bit q in big-endian ordering: shift right by (n - 1 - q)
    bit = (indices >> (n - 1 - qubit)) & 1
    signs = mx.array(1, dtype=mx.float32) - 2 * bit.astype(mx.float32)
    return mx.sum(signs * probs.real, stream=mx.cpu)


def expectation_pauli_sum(state: mx.array, n: int, weights: Sequence[float] | None = None) -> mx.array:
    """
    Compute <sum_q w_q * Z_q> over all qubits.

    Default weights are all 1.0 (standard VQE cost for Z-type Hamiltonians).
    """
    if weights is None:
        weights = [1.0] * n
    terms = [weights[q] * expectation_z(state, q, n) for q in range(n)]
    return sum(terms)


def expectation_zz(state: mx.array, qubit_a: int, qubit_b: int, n: int) -> mx.array:
    """<Z_a Z_b> two-qubit Pauli correlator."""
    probs = mx.abs(state, stream=mx.cpu) ** 2
    indices = mx.arange(2**n, stream=mx.cpu)
    bit_a = (indices >> (n - 1 - qubit_a)) & 1
    bit_b = (indices >> (n - 1 - qubit_b)) & 1
    # ZZ eigenvalue: +1 if bits agree, -1 if differ
    signs = (1 - 2 * bit_a.astype(mx.float32)) * (1 - 2 * bit_b.astype(mx.float32))
    return mx.sum(signs * probs.real, stream=mx.cpu)
