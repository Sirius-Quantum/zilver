"""
Density matrix simulator for noise-aware quantum circuit simulation.

State: (2^n, 2^n) complex64 — quadratic memory vs statevector's linear.
Memory ceiling: M4 Pro (48GB) → 16 qubits, M2 Ultra (192GB) → 17 qubits.

Qubit convention: qubit 0 = most significant bit (consistent with simulator.py).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Sequence
import mlx.core as mx
import numpy as np

from . import gates as G
from .simulator import apply_gate


# ---------------------------------------------------------------------------
# Core density matrix operations
# ---------------------------------------------------------------------------

def apply_gate_dm(rho: mx.array, gate: mx.array, qubits: Sequence[int], n: int) -> mx.array:
    """
    Apply unitary gate to density matrix: rho -> U rho U†

    Uses axis permutation on the (2,)*2n tensor representation.
    No explicit Kronecker products — runs in O(4^n) time.

    Row application:  U @ rho  (contract gate with row/ket axes)
    Column application: (U rho) @ U†  (contract gate† with col/bra axes)
    """
    dim = 2 ** n
    k = len(qubits)
    qubits = list(qubits)

    # Reshape to (2, 2, ..., 2) with 2n axes
    # Axes 0..n-1 = row (ket) indices; axes n..2n-1 = col (bra) indices
    rho_t = rho.reshape([2] * (2 * n))

    # --- Row application: U @ rho ---
    row_targets = qubits
    row_rest = [i for i in range(n) if i not in row_targets]
    col_all = list(range(n, 2 * n))

    perm_row = row_targets + row_rest + col_all
    inv_row = [0] * (2 * n)
    for new_i, old_i in enumerate(perm_row):
        inv_row[old_i] = new_i

    rho_t = rho_t.transpose(perm_row)
    rho_t = rho_t.reshape(2 ** k, 2 ** (n - k) * 2 ** n)
    rho_t = gate.reshape(2 ** k, 2 ** k) @ rho_t
    rho_t = rho_t.reshape([2] * (2 * n))
    rho_t = rho_t.transpose(inv_row)

    # --- Column application: (U rho) @ U† ---
    # Move target col axes to the END for right-multiply by U†
    col_targets = [n + q for q in qubits]
    col_rest = [n + i for i in range(n) if i not in qubits]
    row_all = list(range(n))

    perm_col = row_all + col_rest + col_targets
    inv_col = [0] * (2 * n)
    for new_i, old_i in enumerate(perm_col):
        inv_col[old_i] = new_i

    rho_t = rho_t.transpose(perm_col)
    rho_t = rho_t.reshape(2 ** n * 2 ** (n - k), 2 ** k)
    gate_dag = gate.reshape(2 ** k, 2 ** k).conj().T
    rho_t = rho_t @ gate_dag
    rho_t = rho_t.reshape([2] * (2 * n))
    rho_t = rho_t.transpose(inv_col)

    return rho_t.reshape(dim, dim)


def apply_kraus_channel(
    rho: mx.array,
    kraus_ops: Sequence[mx.array],
    qubits: Sequence[int],
    n: int,
) -> mx.array:
    """
    Apply Kraus channel: rho -> sum_k K_k rho K_k†

    Completeness: sum_k K_k† K_k = I (preserved by each noise model below).
    """
    dim = 2 ** n
    new_rho = mx.zeros((dim, dim), dtype=mx.complex64)
    for K in kraus_ops:
        new_rho = new_rho + apply_gate_dm(rho, K, qubits, n)
    return new_rho


# ---------------------------------------------------------------------------
# Standard noise channels (Kraus operator sets)
# ---------------------------------------------------------------------------

def depolarizing_kraus(p: float) -> list[mx.array]:
    """
    Single-qubit depolarizing channel: rho -> (1-p)*rho + p*(I/2)

    Kraus decomposition: K0 = sqrt(1 - 3p/4)*I,  K1,2,3 = sqrt(p/4)*{X,Y,Z}
    Completeness: K0†K0 + K1†K1 + K2†K2 + K3†K3 = I  ✓
    p=0: identity,  p=1: fully mixed state I/2.
    Valid for 0 ≤ p ≤ 1.
    """
    s = np.sqrt(p / 4)
    return [
        mx.array(np.sqrt(1 - 3 * p / 4) * np.eye(2, dtype=np.complex64)),
        mx.array(s * np.array([[0, 1], [1, 0]], dtype=np.complex64)),
        mx.array(s * np.array([[0, -1j], [1j, 0]], dtype=np.complex64)),
        mx.array(s * np.array([[1, 0], [0, -1]], dtype=np.complex64)),
    ]


def amplitude_damping_kraus(gamma: float) -> list[mx.array]:
    """
    Amplitude damping (T1 decay): |1> -> |0> with probability gamma.
    gamma=0: identity, gamma=1: all population decays to |0>.
    """
    return [
        mx.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=mx.complex64),
        mx.array([[0, np.sqrt(gamma)], [0, 0]], dtype=mx.complex64),
    ]


def phase_damping_kraus(gamma: float) -> list[mx.array]:
    """
    Phase damping (T2 dephasing): destroys off-diagonal coherence.
    gamma=0: identity, gamma=1: fully dephased (diagonal rho only).
    """
    return [
        mx.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=mx.complex64),
        mx.array([[0, 0], [0, np.sqrt(gamma)]], dtype=mx.complex64),
    ]


def bit_flip_kraus(p: float) -> list[mx.array]:
    """
    Bit flip channel: X error with probability p.
    p=0: identity, p=0.5: maximally mixed in computational basis.
    """
    return [
        mx.array(np.sqrt(1 - p) * np.eye(2, dtype=np.complex64)),
        mx.array(np.sqrt(p) * np.array([[0, 1], [1, 0]], dtype=np.complex64)),
    ]


def phase_flip_kraus(p: float) -> list[mx.array]:
    """
    Phase flip channel: Z error with probability p.
    p=0: identity, p=0.5: maximally dephased.
    """
    return [
        mx.array(np.sqrt(1 - p) * np.eye(2, dtype=np.complex64)),
        mx.array(np.sqrt(p) * np.array([[1, 0], [0, -1]], dtype=np.complex64)),
    ]


# ---------------------------------------------------------------------------
# Expectation values on density matrix
# ---------------------------------------------------------------------------

def expectation_z_dm(rho: mx.array, qubit: int, n: int) -> mx.array:
    """
    <Z_q> = Tr(rho Z_q) = sum_i rho[i,i] * sign(bit q of i)

    sign = +1 if bit q is 0, -1 if bit q is 1.
    """
    dim = 2 ** n
    diag = mx.array([rho[i, i] for i in range(dim)]).real
    indices = mx.arange(dim)
    bit_q = (indices >> (n - 1 - qubit)) & 1
    signs = mx.array(1, dtype=mx.float32) - 2 * bit_q.astype(mx.float32)
    return mx.sum(signs * diag)


def expectation_sum_z_dm(rho: mx.array, n: int, weights: Sequence[float] | None = None) -> mx.array:
    """Tr(rho * sum_q w_q Z_q)"""
    if weights is None:
        weights = [1.0] * n
    return sum(weights[q] * expectation_z_dm(rho, q, n) for q in range(n))


def trace(rho: mx.array) -> mx.array:
    """Tr(rho) — should be 1.0 for a valid density matrix."""
    n = rho.shape[0]
    return sum(rho[i, i] for i in range(n)).real


# ---------------------------------------------------------------------------
# NoisyCircuit
# ---------------------------------------------------------------------------

@dataclass
class NoiseOp:
    """A noise channel inserted after a gate."""
    kraus_fn: Callable[[], list[mx.array]]  # returns Kraus operators
    qubits: list[int]


@dataclass
class DMGateOp:
    """Gate operation for density matrix circuit."""
    gate_fn: Callable        # None-args -> (2^k, 2^k) matrix, OR params -> matrix
    qubits: list[int]
    param_indices: list[int] = field(default_factory=list)


class NoisyCircuit:
    """
    Parametrized quantum circuit with noise channels, simulated via density matrix.

    Usage:
        nc = NoisyCircuit(n_qubits=4)
        nc.h(0)
        nc.cnot(0, 1)
        nc.ry(1, param_idx=0)
        nc.noise(depolarizing_kraus(0.01), qubits=[0, 1])

        rho = nc.run(params)
        exp = expectation_sum_z_dm(rho, nc.n_qubits)
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.n_params = 0
        self._ops: list[DMGateOp | NoiseOp] = []

    def _add_gate(self, op: DMGateOp) -> "NoisyCircuit":
        self._ops.append(op)
        return self

    def noise(self, kraus_ops: list[mx.array], qubits: list[int]) -> "NoisyCircuit":
        """Insert a noise channel after the preceding gate."""
        kraus_list = list(kraus_ops)
        self._ops.append(NoiseOp(kraus_fn=lambda: kraus_list, qubits=qubits))
        return self

    # --- Gate builder methods ------------------------------------------------

    def h(self, qubit: int) -> "NoisyCircuit":
        mat = G.H()
        return self._add_gate(DMGateOp(gate_fn=lambda _=None: mat, qubits=[qubit]))

    def x(self, qubit: int) -> "NoisyCircuit":
        mat = G.X()
        return self._add_gate(DMGateOp(gate_fn=lambda _=None: mat, qubits=[qubit]))

    def z(self, qubit: int) -> "NoisyCircuit":
        mat = G.Z()
        return self._add_gate(DMGateOp(gate_fn=lambda _=None: mat, qubits=[qubit]))

    def cnot(self, control: int, target: int) -> "NoisyCircuit":
        mat = G.CNOT()
        return self._add_gate(DMGateOp(gate_fn=lambda _=None: mat, qubits=[control, target]))

    def cz(self, control: int, target: int) -> "NoisyCircuit":
        mat = G.CZ()
        return self._add_gate(DMGateOp(gate_fn=lambda _=None: mat, qubits=[control, target]))

    def ry(self, qubit: int, param_idx: int) -> "NoisyCircuit":
        self.n_params = max(self.n_params, param_idx + 1)
        def _ry(p):
            theta = p[0]
            c = mx.cos(theta / 2)
            s = mx.sin(theta / 2)
            return mx.stack([mx.stack([c, -s]), mx.stack([s, c])]).astype(mx.complex64)
        return self._add_gate(DMGateOp(gate_fn=_ry, qubits=[qubit], param_indices=[param_idx]))

    def rx(self, qubit: int, param_idx: int) -> "NoisyCircuit":
        self.n_params = max(self.n_params, param_idx + 1)
        def _rx(p):
            theta = p[0]
            c = mx.cos(theta / 2)
            s = mx.sin(theta / 2)
            z = mx.zeros_like(c)
            real = mx.stack([mx.stack([c, z]), mx.stack([z, c])])
            imag = mx.stack([mx.stack([z, -s]), mx.stack([-s, z])])
            return real.astype(mx.complex64) + 1j * imag.astype(mx.complex64)
        return self._add_gate(DMGateOp(gate_fn=_rx, qubits=[qubit], param_indices=[param_idx]))

    def rz(self, qubit: int, param_idx: int) -> "NoisyCircuit":
        self.n_params = max(self.n_params, param_idx + 1)
        def _rz(p):
            theta = p[0]
            c = mx.cos(theta / 2)
            s = mx.sin(theta / 2)
            z = mx.zeros_like(c)
            real = mx.stack([mx.stack([c, z]), mx.stack([z, c])])
            imag = mx.stack([mx.stack([-s, z]), mx.stack([z, s])])
            return real.astype(mx.complex64) + 1j * imag.astype(mx.complex64)
        return self._add_gate(DMGateOp(gate_fn=_rz, qubits=[qubit], param_indices=[param_idx]))

    def rzz(self, qubit_a: int, qubit_b: int, param_idx: int) -> "NoisyCircuit":
        self.n_params = max(self.n_params, param_idx + 1)
        def _rzz(p):
            theta = p[0]
            c = mx.cos(theta / 2)
            s = mx.sin(theta / 2)
            z = mx.zeros_like(c)
            real = mx.stack([
                mx.stack([c, z, z, z]),
                mx.stack([z, c, z, z]),
                mx.stack([z, z, c, z]),
                mx.stack([z, z, z, c]),
            ])
            imag = mx.stack([
                mx.stack([-s, z,  z,  z]),
                mx.stack([ z, s,  z,  z]),
                mx.stack([ z, z,  s,  z]),
                mx.stack([ z, z,  z, -s]),
            ])
            return real.astype(mx.complex64) + 1j * imag.astype(mx.complex64)
        return self._add_gate(DMGateOp(gate_fn=_rzz, qubits=[qubit_a, qubit_b], param_indices=[param_idx]))

    # --- Execution -----------------------------------------------------------

    def run(self, params: mx.array) -> mx.array:
        """
        Execute circuit, return final density matrix.

        Returns (2^n, 2^n) complex64 density matrix.
        """
        dim = 2 ** self.n_qubits
        rho_np = np.zeros((dim, dim), dtype=np.complex64)
        rho_np[0, 0] = 1.0
        rho = mx.array(rho_np)

        for op in self._ops:
            if isinstance(op, NoiseOp):
                rho = apply_kraus_channel(rho, op.kraus_fn(), op.qubits, self.n_qubits)
            else:
                if op.param_indices:
                    gate_params = params[mx.array(op.param_indices)]
                    gate = op.gate_fn(gate_params)
                else:
                    gate = op.gate_fn()
                rho = apply_gate_dm(rho, gate, op.qubits, self.n_qubits)

        return rho

    def compile(
        self,
        observable: str = "sum_z",
        noise_model: dict | None = None,
    ) -> Callable[[mx.array], mx.array]:
        """
        Return a function: params -> scalar expectation value.

        Unlike the statevector Circuit.compile(), this is NOT vmappable
        for large n (density matrix is 4^n — only one fits in memory at n=16).
        For small n (≤ 10), vmap over parameter batches is feasible.
        """
        n = self.n_qubits

        def eval_fn(params: mx.array) -> mx.array:
            rho = self.run(params)
            if observable == "sum_z":
                return expectation_sum_z_dm(rho, n)
            elif observable == "z0":
                return expectation_z_dm(rho, 0, n)
            else:
                raise ValueError(f"Unknown observable: {observable}")

        return eval_fn

    def __repr__(self) -> str:
        n_noise = sum(1 for op in self._ops if isinstance(op, NoiseOp))
        n_gates = len(self._ops) - n_noise
        return (
            f"NoisyCircuit(n_qubits={self.n_qubits}, "
            f"n_params={self.n_params}, gates={n_gates}, noise_ops={n_noise})"
        )
