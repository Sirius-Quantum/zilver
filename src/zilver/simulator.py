"""Statevector simulation."""

from __future__ import annotations
from typing import Sequence
from ._array import mx
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
        if isinstance(array, np.ndarray):
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
        when the StateVector was constructed at double precision."""
        if self._state_np is not None:
            return self._state_np
        mx.eval(self._state)
        return np.array(self._state.tolist(), dtype=np.complex64)

    def __repr__(self) -> str:
        return f"StateVector(n_qubits={self.n_qubits}, dtype={self.dtype})"


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
