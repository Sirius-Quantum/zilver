"""Circuit cutting (QPD)."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import mlx.core as mx

from .circuit import Circuit
from . import gates as G


# ---------------------------------------------------------------------------
# QPD terms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QPDTerm:
    """
    One term in a quasi-probability decomposition.

    weight:        real coefficient (may be negative)
    left_ops:      list of (gate_matrix_np, qubit) to apply on the left subcircuit
    right_ops:     list of (gate_matrix_np, qubit) to apply on the right subcircuit
    """
    weight: float
    left_ops:  list[tuple[np.ndarray, int]]
    right_ops: list[tuple[np.ndarray, int]]


# Pauli matrices as numpy arrays (used in QPD construction)
_I  = np.eye(2,    dtype=np.complex64)
_X  = np.array([[0, 1], [1, 0]],  dtype=np.complex64)
_Y  = np.array([[0,-1j],[1j, 0]], dtype=np.complex64)
_Z  = np.array([[1, 0], [0,-1]],  dtype=np.complex64)
_H  = np.array([[1, 1], [1,-1]], dtype=np.complex64) / np.sqrt(2)
_Sd = np.array([[1, 0], [0,-1j]], dtype=np.complex64)   # S†


def wire_cut_terms(left_qubit: int, right_qubit: int) -> list[QPDTerm]:
    """
    QPD decomposition of the identity channel across a wire cut.

    Decomposes I(rho) = Σ_i c_i P_i rho P_i† into 4 (prep, meas) pairs.

    Each term: left subcircuit prepares one of {|0>,|1>,|+>,|i+>};
               right subcircuit measures in the matching Pauli basis.

    Encoding:
      term 0: c=+1/2, prep |0>  (no gate),    meas Z basis (no gate before meas)
      term 1: c=+1/2, prep |+>  (H on left),  meas X basis (H before meas on right)
      term 2: c=-1/2, prep |i+> (H,S on left),meas Y basis (S†,H before meas on right)
      term 3: c=+1/2, prep |1>  (X on left),  meas Z basis (no gate before meas)

    The right subcircuit measures in the given basis; the classical result
    (+1 or -1 eigenvalue) is multiplied by the weight during recombination.
    """
    return [
        QPDTerm(
            weight=+0.5,
            left_ops=[],
            right_ops=[],
        ),
        QPDTerm(
            weight=+0.5,
            left_ops=[(_H, left_qubit)],
            right_ops=[(_H, right_qubit)],
        ),
        QPDTerm(
            weight=-0.5,
            left_ops=[(_H, left_qubit), (_Sd, left_qubit)],
            right_ops=[(_Sd, right_qubit), (_H, right_qubit)],
        ),
        QPDTerm(
            weight=+0.5,
            left_ops=[(_X, left_qubit)],
            right_ops=[],
        ),
    ]


def cnot_cut_terms(control: int, target: int) -> list[QPDTerm]:
    """
    QPD decomposition of a CNOT gate spanning a partition boundary.

    Based on Mitarai & Fujii (2021): CNOT decomposes into 8 local terms.

    CNOT = Σ_{i=0}^{7} c_i  (L_i on control) ⊗ (R_i on target)

    Decomposition:
      CNOT = (1/2)[I⊗I + I⊗X + Z⊗I - Z⊗X]

    Expanded into 8 terms with all positive weights (using Pauli channel trick):
      c_i ∈ {+1/2, +1/2, +1/2, +1/2, +1/2, -1/2, +1/2, -1/2}
    """
    Rx_half    = np.array([[np.cos(np.pi/4), -1j*np.sin(np.pi/4)],
                            [-1j*np.sin(np.pi/4), np.cos(np.pi/4)]], dtype=np.complex64)
    Rx_neg     = Rx_half.conj().T

    return [
        QPDTerm(weight=+0.5, left_ops=[(_I, control)],  right_ops=[(_I, target)]),
        QPDTerm(weight=+0.5, left_ops=[(_I, control)],  right_ops=[(_X, target)]),
        QPDTerm(weight=+0.5, left_ops=[(_Z, control)],  right_ops=[(_I, target)]),
        QPDTerm(weight=-0.5, left_ops=[(_Z, control)],  right_ops=[(_X, target)]),
        QPDTerm(weight=+0.5, left_ops=[(_I, control)],  right_ops=[(_I,  target)]),
        QPDTerm(weight=-0.5, left_ops=[(_X, control)],  right_ops=[(_I,  target)]),
        QPDTerm(weight=+0.5, left_ops=[(_I, control)],  right_ops=[(_Z,  target)]),
        QPDTerm(weight=+0.5, left_ops=[(_X, control)],  right_ops=[(_Z,  target)]),
    ]


# ---------------------------------------------------------------------------
# Subcircuit pair
# ---------------------------------------------------------------------------

@dataclass
class SubcircuitPair:
    """
    One QPD term instantiated as a pair of concrete circuits.

    left:   Circuit acting on qubits [0, partition_qubit - 1]
    right:  Circuit acting on qubits [0, n_qubits - partition_qubit - 1]
    weight: QPD coefficient for this term
    term_index: index into the parent QPDTerm list
    """
    left:        Circuit
    right:       Circuit
    weight:      float
    term_index:  int


# ---------------------------------------------------------------------------
# Wire cut
# ---------------------------------------------------------------------------

@dataclass
class WireCut:
    """
    Specification for a single wire cut.

    partition_qubit: the qubit index in the original circuit where the cut
                     is applied. Qubits 0..partition_qubit-1 go to the left
                     subcircuit; qubits partition_qubit..n-1 go to the right.
    """
    partition_qubit: int


@dataclass
class CutCircuit:
    """
    Original circuit annotated with cut points.

    Construct via CutCircuit(base_circuit).add_wire_cut(qubit).
    Call .decompose() to generate the list of SubcircuitPairs for execution.
    """
    base: Circuit
    wire_cuts: list[WireCut] = field(default_factory=list)
    gate_cuts:  list[tuple[int, int]] = field(default_factory=list)  # (control, target)

    def add_wire_cut(self, partition_qubit: int) -> "CutCircuit":
        self.wire_cuts.append(WireCut(partition_qubit))
        return self

    def add_gate_cut(self, control: int, target: int) -> "CutCircuit":
        self.gate_cuts.append((control, target))
        return self

    def decompose(self) -> "CutDecomposition":
        """
        Generate all subcircuit pairs from QPD decomposition.

        Returns a CutDecomposition containing:
          - All SubcircuitPairs (one per QPD term per cut)
          - Recombination weights
          - Overhead factor (number of subcircuit evaluations needed)
        """
        if len(self.wire_cuts) == 1 and len(self.gate_cuts) == 0:
            return _decompose_single_wire_cut(self)
        if len(self.wire_cuts) == 0 and len(self.gate_cuts) == 1:
            return _decompose_single_gate_cut(self)
        raise NotImplementedError(
            "Multi-cut decomposition requires coordinator.py. "
            "Use single wire or gate cut for now."
        )

    @property
    def sampling_overhead(self) -> int:
        """Number of subcircuit evaluations required."""
        return 4 ** len(self.wire_cuts) * 9 ** len(self.gate_cuts)


# ---------------------------------------------------------------------------
# Decomposition result
# ---------------------------------------------------------------------------

@dataclass
class CutDecomposition:
    """
    All subcircuit pairs generated from one CutCircuit.

    pairs:    list of SubcircuitPair — one per QPD term
    overhead: total number of subcircuit evaluations (4^k_wire * 9^k_gate)

    Usage:
        decomp = cut_circuit.decompose()
        results = []
        for pair in decomp.pairs:
            left_val  = pair.left.compile()(params_left)
            right_val = pair.right.compile()(params_right)
            results.append((pair.weight, left_val, right_val))
        expectation = coordinator.recombine(results)
    """
    pairs:    list[SubcircuitPair]
    overhead: int

    def __repr__(self) -> str:
        return (
            f"CutDecomposition("
            f"n_pairs={len(self.pairs)}, overhead={self.overhead})"
        )


# ---------------------------------------------------------------------------
# Internal decomposition helpers
# ---------------------------------------------------------------------------

def _decompose_single_wire_cut(cc: CutCircuit) -> CutDecomposition:
    """
    Decompose a circuit with one wire cut into 4 subcircuit pairs.

    The cut is at cc.wire_cuts[0].partition_qubit = p.
    Left subcircuit:  qubits 0..p-1  (p qubits)
    Right subcircuit: qubits p..n-1  (n-p qubits), re-indexed from 0.
    """
    base  = cc.base
    cut   = cc.wire_cuts[0]
    p     = cut.partition_qubit
    n     = base.n_qubits
    n_l   = p
    n_r   = n - p

    # Pass absolute qubit indices; _build_subcircuit will re-index via qubit_offset
    terms = wire_cut_terms(left_qubit=p - 1, right_qubit=p)
    pairs = []

    for idx, term in enumerate(terms):
        left_circ  = _build_subcircuit(base, qubit_range=range(n_l),
                                        extra_ops=term.left_ops,
                                        n_qubits=n_l)
        right_circ = _build_subcircuit(base, qubit_range=range(p, n),
                                        extra_ops=term.right_ops,
                                        n_qubits=n_r,
                                        qubit_offset=p)
        pairs.append(SubcircuitPair(
            left=left_circ,
            right=right_circ,
            weight=term.weight,
            term_index=idx,
        ))

    return CutDecomposition(pairs=pairs, overhead=4)


def _decompose_single_gate_cut(cc: CutCircuit) -> CutDecomposition:
    """
    Decompose a circuit with one CNOT gate cut into 8 subcircuit pairs.
    """
    base            = cc.base
    control, target = cc.gate_cuts[0]
    n               = base.n_qubits

    # Determine partition: control on left, target on right
    # Left: qubits that include control; Right: qubits that include target
    p     = max(control, target - 1)   # partition after control
    n_l   = p + 1
    n_r   = n - n_l

    terms = cnot_cut_terms(control=control, target=target)
    pairs = []

    for idx, term in enumerate(terms):
        left_circ  = _build_subcircuit(base, qubit_range=range(n_l),
                                        extra_ops=term.left_ops,
                                        n_qubits=n_l,
                                        skip_gate=(control, target))
        right_circ = _build_subcircuit(base, qubit_range=range(n_l, n),
                                        extra_ops=term.right_ops,
                                        n_qubits=n_r,
                                        qubit_offset=n_l,
                                        skip_gate=(control, target))
        pairs.append(SubcircuitPair(
            left=left_circ,
            right=right_circ,
            weight=term.weight,
            term_index=idx,
        ))

    return CutDecomposition(pairs=pairs, overhead=8)


def _build_subcircuit(
    base: Circuit,
    qubit_range: range,
    extra_ops: list[tuple[np.ndarray, int]],
    n_qubits: int,
    qubit_offset: int = 0,
    skip_gate: tuple[int, int] | None = None,
) -> Circuit:
    """
    Extract gates that act within qubit_range from the base circuit.
    Re-index qubits by subtracting qubit_offset.
    Append extra_ops (QPD channel gates) at the boundary qubit.
    """
    from .circuit import GateOp
    import mlx.core as mx

    sub = Circuit(n_qubits)
    sub.n_params = base.n_params

    qset = set(qubit_range)

    for op in base._ops:
        # Skip the cut gate itself
        if skip_gate and set(op.qubits) == set(skip_gate):
            continue
        # Include only ops whose qubits are entirely within this subcircuit
        if all(q in qset for q in op.qubits):
            new_qubits = [q - qubit_offset for q in op.qubits]
            new_op = GateOp(
                gate_fn=op.gate_fn,
                qubits=new_qubits,
                param_indices=op.param_indices,
            )
            sub._ops.append(new_op)

    # Append QPD boundary operations
    for mat_np, qubit in extra_ops:
        mat_mx = mx.array(mat_np)
        new_q  = qubit - qubit_offset
        sub._ops.append(GateOp.fixed(mat_mx, [new_q]))

    return sub


# ---------------------------------------------------------------------------
# Overhead utilities
# ---------------------------------------------------------------------------

def wire_cut_overhead(n_cuts: int) -> int:
    """Sampling overhead for n wire cuts: 4^n subcircuit evaluations."""
    return 4 ** n_cuts


def gate_cut_overhead(n_cuts: int) -> int:
    """Sampling overhead for n gate cuts: 9^n subcircuit evaluations."""
    return 9 ** n_cuts


def max_feasible_cuts(overhead_budget: int, cut_type: str = "wire") -> int:
    """
    Maximum number of cuts given an overhead budget.

    overhead_budget: max acceptable number of subcircuit evaluations
    cut_type: 'wire' (4^k) or 'gate' (9^k)
    """
    base = 4 if cut_type == "wire" else 9
    k = 0
    while base ** (k + 1) <= overhead_budget:
        k += 1
    return k
