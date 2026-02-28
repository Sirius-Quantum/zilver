"""
Matrix Product State (MPS) tensor network simulator.

Scales with entanglement entropy, not qubit count.
Target: 50+ qubits for circuits with bond dimension chi <= 256.

Qubit convention: qubit 0 = most significant bit.
Tensor shape: tensors[k] has shape (chi_left, 2, chi_right).
  - chi_left  = 1 at left boundary (k=0)
  - chi_right = 1 at right boundary (k=n-1)
  - physical dimension = 2 (qubit)

Two-qubit gates must act on adjacent qubits in the MPS ordering.
Non-adjacent gates are handled automatically via SWAP decomposition.

Note: SVD truncation makes MPS circuits approximate for chi < chi_exact.
      For exact simulation (chi_max=None), results match statevector exactly.
      SVD is not vmappable — batched VQA evaluation uses sequential evaluation
      or distribution across nodes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Sequence
import numpy as np
import mlx.core as mx

from . import gates as G


# ---------------------------------------------------------------------------
# MPS core operations
# ---------------------------------------------------------------------------

def init_mps(n_qubits: int) -> list[np.ndarray]:
    """
    Initialize |0...0> as MPS with bond dimension chi=1.
    Returns list of numpy arrays (SVD uses numpy throughout).
    """
    tensors = []
    for _ in range(n_qubits):
        t = np.zeros((1, 2, 1), dtype=np.complex64)
        t[0, 0, 0] = 1.0   # |0> at each site
        tensors.append(t)
    return tensors


def apply_single_qubit_gate_mps(
    tensors: list[np.ndarray],
    gate: np.ndarray,
    qubit: int,
) -> list[np.ndarray]:
    """
    Apply single-qubit gate: A[q][alpha, i, beta] -> Σ_j gate[i,j] A[q][alpha, j, beta]

    O(chi^2) time, no SVD needed.
    """
    A = tensors[qubit]                              # (chi_l, 2, chi_r)
    chi_l, _, chi_r = A.shape
    # Reshape to (chi_l*chi_r, 2), apply gate from right, reshape back
    A_flat = A.transpose(0, 2, 1).reshape(chi_l * chi_r, 2)   # (chi_l*chi_r, 2)
    new_A  = (A_flat @ gate.T).reshape(chi_l, chi_r, 2).transpose(0, 2, 1)
    tensors = list(tensors)
    tensors[qubit] = new_A
    return tensors


def apply_two_qubit_gate_mps(
    tensors: list[np.ndarray],
    gate: np.ndarray,
    qubit_a: int,
    qubit_b: int,
    chi_max: int | None = 64,
) -> list[np.ndarray]:
    """
    Apply two-qubit gate on adjacent qubits qubit_a, qubit_b = qubit_a + 1.

    Steps:
      1. Contract A[qubit_a] and A[qubit_b] into two-site tensor Theta.
      2. Apply gate to physical indices.
      3. SVD-decompose Theta back into A[qubit_a], S, A[qubit_b].
      4. Truncate bond dimension to chi_max (approximation if chi_max < exact chi).

    O(chi^3) time due to SVD.
    """
    assert qubit_b == qubit_a + 1, (
        f"Two-qubit gate requires adjacent qubits; got {qubit_a}, {qubit_b}. "
        "Use MPSCircuit which handles non-adjacent gates via SWAP decomposition."
    )

    A = tensors[qubit_a]    # (chi_l, 2, chi_mid)
    B = tensors[qubit_b]    # (chi_mid, 2, chi_r)
    chi_l, _, chi_mid = A.shape
    _,     _, chi_r   = B.shape

    # Step 1: contract into Theta of shape (chi_l*2, 2*chi_r)
    Theta = (A.reshape(chi_l * 2, chi_mid) @ B.reshape(chi_mid, 2 * chi_r))
    # Shape: (chi_l*2, 2*chi_r)

    # Step 2: apply gate
    # gate: (4, 4) acting on (i, j) physical indices
    # Theta[alpha_i, j_beta] = Theta[(alpha, i), (j, beta)]
    # new_Theta[(alpha, i'), (j', beta)] = Σ_{i,j} gate[(i',j'), (i,j)] Theta[(alpha,i), (j,beta)]
    # Reshape Theta to (chi_l, 2, 2, chi_r) -> (chi_l, 4, chi_r)
    Theta_4 = Theta.reshape(chi_l, 2, 2, chi_r)
    # Permute to (chi_l, chi_r, 4) to apply gate in the middle
    Theta_4 = Theta_4.transpose(0, 3, 1, 2).reshape(chi_l * chi_r, 4)
    # gate: (4, 4), apply from right to physical index
    new_Theta = (Theta_4 @ gate.T).reshape(chi_l, chi_r, 2, 2)
    # Permute to (chi_l, 2, 2, chi_r) -> (chi_l*2, 2*chi_r)
    new_Theta = new_Theta.transpose(0, 2, 3, 1).reshape(chi_l * 2, 2 * chi_r)

    # Step 3: SVD
    U, S, Vt = np.linalg.svd(new_Theta, full_matrices=False)
    # U:  (chi_l*2, min_dim)
    # S:  (min_dim,)
    # Vt: (min_dim, 2*chi_r)

    # Step 4: truncate
    chi_new = len(S)
    if chi_max is not None:
        chi_new = min(chi_new, chi_max)

    U  = U[:, :chi_new]
    S  = S[:chi_new]
    Vt = Vt[:chi_new, :]

    # Absorb singular values into right tensor (left-canonical form for A[qubit_a])
    new_A = U.reshape(chi_l, 2, chi_new)
    new_B = (np.diag(S) @ Vt).reshape(chi_new, 2, chi_r)

    tensors = list(tensors)
    tensors[qubit_a] = new_A
    tensors[qubit_b] = new_B
    return tensors


def apply_swap_mps(
    tensors: list[np.ndarray],
    qubit_a: int,
    chi_max: int | None = 64,
) -> list[np.ndarray]:
    """SWAP adjacent qubits qubit_a and qubit_a+1."""
    swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.complex64)
    return apply_two_qubit_gate_mps(tensors, swap, qubit_a, qubit_a + 1, chi_max)


def mps_to_statevector(tensors: list[np.ndarray], n: int) -> np.ndarray:
    """
    Contract full MPS to statevector. Only feasible for small n (≤ 20).
    Used for testing and validation.

    O(chi * 2^n) time and memory.
    """
    sv = tensors[0][0, :, :]                          # (2, chi_1)
    for k in range(1, n):
        chi_k    = tensors[k].shape[0]
        chi_next = tensors[k].shape[2]
        sv = (sv @ tensors[k].reshape(chi_k, 2 * chi_next)).reshape(2 ** (k + 1), chi_next)
    return sv[:, 0]                                    # (2^n,)


# ---------------------------------------------------------------------------
# Expectation values
# ---------------------------------------------------------------------------

def _update_left_env(L: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    L: (chi_l, chi_l) left environment
    A: (chi_l, 2, chi_r) MPS tensor
    Returns: (chi_r, chi_r)

    new_L[beta, beta'] = Σ_{i,alpha,alpha'} L[alpha,alpha'] conj(A[alpha',i,beta']) A[alpha,i,beta]
                       = Σ_i A[:,i,:]^† @ L @ A[:,i,:]
    """
    new_L = np.zeros((A.shape[2], A.shape[2]), dtype=np.complex64)
    for i in range(2):
        Ai = A[:, i, :]                # (chi_l, chi_r)
        new_L += Ai.conj().T @ L @ Ai
    return new_L


def _update_right_env(R: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    R: (chi_r, chi_r) right environment, indexed as R[bra, ket]
    A: (chi_l, 2, chi_r) MPS tensor
    Returns: (chi_l, chi_l)

    new_R[alpha, alpha'] = Σ_{i,beta,beta'} conj(A[alpha,i,beta]) R[beta,beta'] A[alpha',i,beta']
                         = Σ_i conj(A[:,i,:]) @ R @ A[:,i,:]^T
    """
    new_R = np.zeros((A.shape[0], A.shape[0]), dtype=np.complex64)
    for i in range(2):
        Ai = A[:, i, :]                # (chi_l, chi_r)
        new_R += Ai.conj() @ R @ Ai.T
    return new_R


def expectation_z_mps(tensors: list[np.ndarray], qubit: int, n: int) -> float:
    """
    <Z_qubit> = Σ_{beta,beta'} T_Z[beta,beta'] R[beta,beta']

    where T_Z is the operator transfer matrix at site `qubit` with Z inserted,
    contracted with the left environment up to site qubit-1.
    """
    L = np.array([[1.0 + 0j]], dtype=np.complex64)
    for k in range(qubit):
        L = _update_left_env(L, tensors[k])

    A = tensors[qubit]                               # (chi_l, 2, chi_r)
    signs = [1.0, -1.0]                              # Z eigenvalues for |0>, |1>
    T_Z = np.zeros((A.shape[2], A.shape[2]), dtype=np.complex64)
    for i in range(2):
        Ai = A[:, i, :]
        T_Z += signs[i] * Ai.conj().T @ L @ Ai

    R = np.array([[1.0 + 0j]], dtype=np.complex64)
    for k in range(n - 1, qubit, -1):
        R = _update_right_env(R, tensors[k])

    return float(np.sum(T_Z * R).real)


def expectation_sum_z_mps(
    tensors: list[np.ndarray],
    n: int,
    weights: Sequence[float] | None = None,
) -> float:
    """Σ_q w_q <Z_q>"""
    if weights is None:
        weights = [1.0] * n
    return sum(weights[q] * expectation_z_mps(tensors, q, n) for q in range(n))


def bond_dimensions(tensors: list[np.ndarray]) -> list[int]:
    """Return list of bond dimensions [chi_01, chi_12, ..., chi_{n-2,n-1}]."""
    return [tensors[k].shape[2] for k in range(len(tensors) - 1)]


def max_bond_dimension(tensors: list[np.ndarray]) -> int:
    return max(bond_dimensions(tensors)) if len(tensors) > 1 else 1


# ---------------------------------------------------------------------------
# MPSCircuit
# ---------------------------------------------------------------------------

@dataclass
class _GateRecord:
    gate_fn: Callable       # () -> np.ndarray for fixed; (params) -> np.ndarray for param
    qubits: list[int]
    param_indices: list[int] = field(default_factory=list)
    is_two_qubit: bool = False


class MPSCircuit:
    """
    Parameterized quantum circuit executed via MPS simulation.

    Supports circuits with 50+ qubits for shallow/sparse entanglement structures.
    Non-adjacent two-qubit gates are decomposed via SWAP networks automatically.

    Limitations vs statevector Circuit:
      - Approximate for chi_max < exact bond dimension (entanglement truncation)
      - Not vmappable (SVD is sequential) — use sequential evaluation or node distribution
      - Two-qubit gate cost: O(chi^3) per gate vs O(2^n) for statevector

    For shallow circuits (depth << n) with nearest-neighbor gates, chi stays small
    and MPS is exponentially faster than statevector beyond ~20 qubits.
    """

    def __init__(self, n_qubits: int, chi_max: int = 64):
        self.n_qubits = n_qubits
        self.chi_max  = chi_max
        self.n_params = 0
        self._ops: list[_GateRecord] = []

    # --- Gate builders -------------------------------------------------------

    def h(self, qubit: int) -> "MPSCircuit":
        mat = np.array(G.H().tolist(), dtype=np.complex64)
        self._ops.append(_GateRecord(gate_fn=lambda _mat=mat: _mat, qubits=[qubit]))
        return self

    def x(self, qubit: int) -> "MPSCircuit":
        mat = np.array(G.X().tolist(), dtype=np.complex64)
        self._ops.append(_GateRecord(gate_fn=lambda _mat=mat: _mat, qubits=[qubit]))
        return self

    def z(self, qubit: int) -> "MPSCircuit":
        mat = np.array(G.Z().tolist(), dtype=np.complex64)
        self._ops.append(_GateRecord(gate_fn=lambda _mat=mat: _mat, qubits=[qubit]))
        return self

    def cnot(self, control: int, target: int) -> "MPSCircuit":
        mat = np.array(G.CNOT().tolist(), dtype=np.complex64)
        self._ops.append(_GateRecord(
            gate_fn=lambda _mat=mat: _mat, qubits=[control, target], is_two_qubit=True
        ))
        return self

    def cz(self, control: int, target: int) -> "MPSCircuit":
        mat = np.array(G.CZ().tolist(), dtype=np.complex64)
        self._ops.append(_GateRecord(
            gate_fn=lambda _mat=mat: _mat, qubits=[control, target], is_two_qubit=True
        ))
        return self

    def ry(self, qubit: int, param_idx: int) -> "MPSCircuit":
        self.n_params = max(self.n_params, param_idx + 1)
        def _ry(params, _idx=param_idx):
            theta = float(params[_idx])
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            return np.array([[c, -s], [s, c]], dtype=np.complex64)
        self._ops.append(_GateRecord(gate_fn=_ry, qubits=[qubit], param_indices=[param_idx]))
        return self

    def rx(self, qubit: int, param_idx: int) -> "MPSCircuit":
        self.n_params = max(self.n_params, param_idx + 1)
        def _rx(params, _idx=param_idx):
            theta = float(params[_idx])
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            return np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex64)
        self._ops.append(_GateRecord(gate_fn=_rx, qubits=[qubit], param_indices=[param_idx]))
        return self

    def rz(self, qubit: int, param_idx: int) -> "MPSCircuit":
        self.n_params = max(self.n_params, param_idx + 1)
        def _rz(params, _idx=param_idx):
            theta = float(params[_idx])
            return np.array([
                [np.exp(-1j*theta/2), 0],
                [0, np.exp(1j*theta/2)]
            ], dtype=np.complex64)
        self._ops.append(_GateRecord(gate_fn=_rz, qubits=[qubit], param_indices=[param_idx]))
        return self

    def rzz(self, qubit_a: int, qubit_b: int, param_idx: int) -> "MPSCircuit":
        self.n_params = max(self.n_params, param_idx + 1)
        def _rzz(params, _idx=param_idx):
            theta = float(params[_idx])
            return np.diag([
                np.exp(-1j*theta/2),
                np.exp( 1j*theta/2),
                np.exp( 1j*theta/2),
                np.exp(-1j*theta/2),
            ]).astype(np.complex64)
        self._ops.append(_GateRecord(
            gate_fn=_rzz, qubits=[qubit_a, qubit_b],
            param_indices=[param_idx], is_two_qubit=True
        ))
        return self

    # --- Execution -----------------------------------------------------------

    def _run(self, params) -> list[np.ndarray]:
        """Execute circuit, return MPS tensor list."""
        tensors = init_mps(self.n_qubits)
        params_np = np.array(params.tolist(), dtype=np.float32) if len(params) > 0 else np.array([], dtype=np.float32)

        for op in self._ops:
            gate = op.gate_fn(params_np) if op.param_indices else op.gate_fn()

            if op.is_two_qubit:
                qa, qb = op.qubits[0], op.qubits[1]
                tensors = _apply_two_qubit_nonlocal(
                    tensors, gate, qa, qb, self.chi_max
                )
            else:
                tensors = apply_single_qubit_gate_mps(tensors, gate, op.qubits[0])

        return tensors

    def compile(
        self,
        observable: str = "sum_z",
    ) -> Callable[[mx.array], float]:
        """
        Return a function: params (mx.array) -> scalar expectation value.

        Not vmappable — MPS gate application uses sequential SVD.
        For batched VQA, evaluate sequentially or distribute across nodes.
        """
        n = self.n_qubits

        def eval_fn(params: mx.array) -> float:
            tensors = self._run(params)
            if observable == "sum_z":
                return expectation_sum_z_mps(tensors, n)
            elif observable == "z0":
                return expectation_z_mps(tensors, 0, n)
            else:
                raise ValueError(f"Unknown observable: {observable}")

        return eval_fn

    def statevector(self, params: mx.array) -> np.ndarray:
        """Contract full MPS. Only for small n (≤ ~20) — for validation."""
        return mps_to_statevector(self._run(params), self.n_qubits)

    def max_bond_dim(self, params: mx.array) -> int:
        """Max bond dimension after executing circuit with given params."""
        return max_bond_dimension(self._run(params))

    def __repr__(self) -> str:
        return (
            f"MPSCircuit(n_qubits={self.n_qubits}, "
            f"chi_max={self.chi_max}, n_params={self.n_params})"
        )


# ---------------------------------------------------------------------------
# Non-adjacent two-qubit gate via SWAP decomposition
# ---------------------------------------------------------------------------

def _apply_two_qubit_nonlocal(
    tensors: list[np.ndarray],
    gate: np.ndarray,
    qa: int,
    qb: int,
    chi_max: int | None,
) -> list[np.ndarray]:
    """
    Apply two-qubit gate on possibly non-adjacent qubits using SWAP network.

    Strategy: bubble qubit qb leftward to qb = qa + 1 using SWAPs,
    apply gate, then reverse SWAPs.

    SWAP overhead: 2 * (qb - qa - 1) additional adjacent gates.
    For nearest-neighbor circuits, this is zero.
    """
    if qb == qa + 1:
        return apply_two_qubit_gate_mps(tensors, gate, qa, qb, chi_max)

    assert qa < qb, f"Expected qa < qb, got {qa}, {qb}"

    swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.complex64)

    # Bubble qb leftward to position qa+1
    for k in range(qb, qa + 1, -1):
        tensors = apply_two_qubit_gate_mps(tensors, swap, k - 1, k, chi_max)

    # Apply gate at adjacent positions (qa, qa+1)
    tensors = apply_two_qubit_gate_mps(tensors, gate, qa, qa + 1, chi_max)

    # Reverse SWAPs: bubble back to original position
    for k in range(qa + 1, qb):
        tensors = apply_two_qubit_gate_mps(tensors, swap, k, k + 1, chi_max)

    return tensors


# ---------------------------------------------------------------------------
# MPS circuit factories — mirrors circuit.py for the Sirius benchmark families
# ---------------------------------------------------------------------------

def hardware_efficient_mps(n_qubits: int, depth: int, chi_max: int = 64) -> MPSCircuit:
    """MPS version of hardware_efficient ansatz."""
    c = MPSCircuit(n_qubits, chi_max)
    p = 0
    for layer in range(depth + 1):
        for q in range(n_qubits):
            c.ry(q, p); p += 1
            c.rz(q, p); p += 1
        if layer < depth:
            for q in range(n_qubits - 1):
                c.cnot(q, q + 1)
    c.n_params = p
    return c


def qaoa_style_mps(n_qubits: int, depth: int, chi_max: int = 64) -> MPSCircuit:
    """MPS version of QAOA-style ansatz."""
    c = MPSCircuit(n_qubits, chi_max)
    for q in range(n_qubits):
        c.h(q)
    p = 0
    for _ in range(depth):
        gamma_idx = p; p += 1
        beta_idx  = p; p += 1
        for q in range(n_qubits - 1):
            c.rzz(q, q + 1, gamma_idx)
        for q in range(n_qubits):
            c.rx(q, beta_idx)
    c.n_params = p
    return c
