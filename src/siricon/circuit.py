"""
Circuit IR and pure-function executor.

Design goal: a Circuit compiles to a pure function
    f(params: mx.array) -> mx.array (scalar expectation value)
with no Python overhead inside, enabling mx.vmap and mx.compile
to dispatch all 400 grid evaluations as a single Metal kernel.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Sequence
import mlx.core as mx
import numpy as np

from .simulator import apply_gate, expectation_pauli_sum, StateVector
from . import gates as G


@dataclass
class GateOp:
    """
    A single gate operation in the circuit.

    gate_fn: callable(params) -> (2^k, 2^k) gate matrix, or a fixed matrix.
    qubits:  target qubit indices.
    param_indices: indices into the parameter vector this gate consumes.
                   Empty list for fixed (non-parameterized) gates.
    """
    gate_fn: Callable
    qubits: list[int]
    param_indices: list[int] = field(default_factory=list)

    @classmethod
    def fixed(cls, matrix: mx.array, qubits: list[int]) -> "GateOp":
        return cls(gate_fn=lambda _: matrix, qubits=qubits, param_indices=[])

    @classmethod
    def rx(cls, qubit: int, param_idx: int) -> "GateOp":
        def _rx(p):
            # MLX-native: no float() calls, stays in compute graph for vmap/compile
            theta = p[0]
            c = mx.cos(theta / 2)
            s = mx.sin(theta / 2)
            z = mx.zeros_like(c)
            real = mx.stack([mx.stack([c, z]), mx.stack([z, c])])
            imag = mx.stack([mx.stack([z, -s]), mx.stack([-s, z])])
            return real.astype(mx.complex64) + 1j * imag.astype(mx.complex64)
        return cls(gate_fn=_rx, qubits=[qubit], param_indices=[param_idx])

    @classmethod
    def ry(cls, qubit: int, param_idx: int) -> "GateOp":
        def _ry(p):
            theta = p[0]
            c = mx.cos(theta / 2)
            s = mx.sin(theta / 2)
            return mx.stack([mx.stack([c, -s]), mx.stack([s, c])]).astype(mx.complex64)
        return cls(gate_fn=_ry, qubits=[qubit], param_indices=[param_idx])

    @classmethod
    def rz(cls, qubit: int, param_idx: int) -> "GateOp":
        def _rz(p):
            theta = p[0]
            c = mx.cos(theta / 2)
            s = mx.sin(theta / 2)
            z = mx.zeros_like(c)
            real = mx.stack([mx.stack([c, z]), mx.stack([z, c])])
            imag = mx.stack([mx.stack([-s, z]), mx.stack([z, s])])
            return real.astype(mx.complex64) + 1j * imag.astype(mx.complex64)
        return cls(gate_fn=_rz, qubits=[qubit], param_indices=[param_idx])

    @classmethod
    def cnot(cls, control: int, target: int) -> "GateOp":
        return cls.fixed(G.CNOT(), [control, target])

    @classmethod
    def cz(cls, control: int, target: int) -> "GateOp":
        return cls.fixed(G.CZ(), [control, target])

    @classmethod
    def h(cls, qubit: int) -> "GateOp":
        return cls.fixed(G.H(), [qubit])


class Circuit:
    """
    Parameterized quantum circuit with a fixed gate sequence.

    Usage:
        c = Circuit(n_qubits=4)
        c.h(0)
        c.cnot(0, 1)
        c.ry(qubit=0, param_idx=0)
        c.ry(qubit=1, param_idx=1)

        # Compile to a pure function for vmap/compile
        f = c.compile(observable="sum_z")

        params = mx.array([0.3, 1.2, ...])
        expectation = f(params)                  # single evaluation
        batch_f = mx.vmap(f)
        grid_expectations = batch_f(params_grid) # (400,) in one Metal dispatch
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.n_params = 0
        self._ops: list[GateOp] = []

    # --- Gate builder methods ------------------------------------------------

    def add(self, op: GateOp) -> "Circuit":
        self._ops.append(op)
        return self

    def h(self, qubit: int) -> "Circuit":
        return self.add(GateOp.h(qubit))

    def x(self, qubit: int) -> "Circuit":
        return self.add(GateOp.fixed(G.X(), [qubit]))

    def cnot(self, control: int, target: int) -> "Circuit":
        return self.add(GateOp.cnot(control, target))

    def cz(self, control: int, target: int) -> "Circuit":
        return self.add(GateOp.cz(control, target))

    def rx(self, qubit: int, param_idx: int) -> "Circuit":
        self.n_params = max(self.n_params, param_idx + 1)
        return self.add(GateOp.rx(qubit, param_idx))

    def ry(self, qubit: int, param_idx: int) -> "Circuit":
        self.n_params = max(self.n_params, param_idx + 1)
        return self.add(GateOp.ry(qubit, param_idx))

    def rz(self, qubit: int, param_idx: int) -> "Circuit":
        self.n_params = max(self.n_params, param_idx + 1)
        return self.add(GateOp.rz(qubit, param_idx))

    def rzz(self, qubit_a: int, qubit_b: int, param_idx: int) -> "Circuit":
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
                mx.stack([-s, z, z,  z]),
                mx.stack([ z, s, z,  z]),
                mx.stack([ z, z, s,  z]),
                mx.stack([ z, z, z, -s]),
            ])
            return real.astype(mx.complex64) + 1j * imag.astype(mx.complex64)
        op = GateOp(gate_fn=_rzz, qubits=[qubit_a, qubit_b], param_indices=[param_idx])
        return self.add(op)

    # --- Execution -----------------------------------------------------------

    def _run(self, params: mx.array) -> mx.array:
        """Execute circuit, return final statevector."""
        init = np.zeros(2**self.n_qubits, dtype=np.complex64)
        init[0] = 1.0
        state = mx.array(init)

        for op in self._ops:
            if op.param_indices:
                gate_params = params[mx.array(op.param_indices)]
                gate = op.gate_fn(gate_params)
            else:
                gate = op.gate_fn(None)
            state = apply_gate(state, gate, op.qubits, self.n_qubits)

        return state

    def compile(
        self,
        observable: str = "sum_z",
        z_weights: Sequence[float] | None = None,
    ) -> Callable[[mx.array], mx.array]:
        """
        Return a pure function  params -> scalar expectation value.

        Args:
            observable: "sum_z" for sum_i Z_i (standard VQE cost),
                        or "z0" for single-qubit Z on qubit 0.
            z_weights:  Per-qubit weights for "sum_z". Default: all 1.0.

        The returned function is vmappable and compilable.
        """
        n = self.n_qubits
        weights = list(z_weights) if z_weights else [1.0] * n

        def eval_fn(params: mx.array) -> mx.array:
            state = self._run(params)
            if observable == "sum_z":
                return expectation_pauli_sum(state, n, weights)
            elif observable == "z0":
                from .simulator import expectation_z
                return expectation_z(state, 0, n)
            else:
                raise ValueError(f"Unknown observable: {observable}")

        return eval_fn

    def statevector(self, params: mx.array) -> StateVector:
        """Execute and return the full statevector."""
        state = self._run(params)
        mx.eval(state)
        return StateVector.from_array(state, self.n_qubits)

    def __repr__(self) -> str:
        return f"Circuit(n_qubits={self.n_qubits}, n_params={self.n_params}, n_ops={len(self._ops)})"


# --- Circuit factory functions for Sirius circuit families -------------------

def hardware_efficient(n_qubits: int, depth: int, entanglement: str = "linear") -> Circuit:
    """
    Hardware-efficient ansatz: alternating RY/RZ layers with CNOT entanglement.
    Parameter layout: [ry_00, rz_00, ry_01, rz_01, ..., ry_d_n, rz_d_n]
    Total params: 2 * n_qubits * (depth + 1)
    """
    c = Circuit(n_qubits)
    p = 0
    for layer in range(depth + 1):
        for q in range(n_qubits):
            c.ry(q, p); p += 1
            c.rz(q, p); p += 1
        if layer < depth:
            _add_entanglement(c, n_qubits, entanglement)
    c.n_params = p
    return c


def real_amplitudes(n_qubits: int, depth: int, entanglement: str = "linear") -> Circuit:
    """
    RealAmplitudes ansatz: RY layers with CNOT entanglement (real-valued states only).
    Total params: n_qubits * (depth + 1)
    """
    c = Circuit(n_qubits)
    p = 0
    for layer in range(depth + 1):
        for q in range(n_qubits):
            c.ry(q, p); p += 1
        if layer < depth:
            _add_entanglement(c, n_qubits, entanglement)
    c.n_params = p
    return c


def qaoa_style(n_qubits: int, depth: int) -> Circuit:
    """
    QAOA-style ansatz: alternating problem (ZZ) and mixer (RX) layers.
    Total params: 2 * depth  (one gamma and one beta per layer)
    """
    c = Circuit(n_qubits)
    for q in range(n_qubits):
        c.h(q)
    p = 0
    for layer in range(depth):
        gamma_idx = p; p += 1
        beta_idx  = p; p += 1
        for q in range(n_qubits - 1):
            c.rzz(q, q + 1, gamma_idx)
        for q in range(n_qubits):
            c.rx(q, beta_idx)
    c.n_params = p
    return c


def efficient_su2(n_qubits: int, depth: int, entanglement: str = "linear") -> Circuit:
    """
    EfficientSU2 ansatz: RY + RZ layers with CNOT entanglement.
    Same structure as hardware_efficient in this implementation.
    """
    return hardware_efficient(n_qubits, depth, entanglement)


def _add_entanglement(c: Circuit, n: int, pattern: str) -> None:
    if pattern == "linear":
        for q in range(n - 1):
            c.cnot(q, q + 1)
    elif pattern == "circular":
        for q in range(n):
            c.cnot(q, (q + 1) % n)
    elif pattern == "full":
        for q in range(n):
            for r in range(q + 1, n):
                c.cnot(q, r)
    else:
        raise ValueError(f"Unknown entanglement pattern: {pattern}")
