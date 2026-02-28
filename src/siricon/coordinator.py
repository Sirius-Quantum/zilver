"""
Coordinator: result recombination for circuit cutting.

Takes subcircuit evaluation results from a CutDecomposition and applies
quasi-probability weights to recover the full-circuit expectation value.

For a single wire cut with k=1:
  E_full = Σ_{i=0}^{3} w_i * f(left_i, right_i)

where f combines left and right expectation values per term.
For sum_Z observable: E_full = Σ_i w_i * (E_left_i + E_right_i)
The additive combination follows from the locality of Z:
  <Z_q> factorizes across partitions, so the total sum_Z is the
  weighted sum of left and right contributions per QPD term.

Multi-node execution:
  The coordinator assigns SubcircuitPairs to available nodes, collects
  results, and calls recombine(). In distributed mode each node runs
  its pair(s) via mx.vmap and returns a scalar.

Reference:
  Peng et al. (2020) — "Simulating Large Quantum Circuits on a Small Quantum Computer"
  Mitarai & Fujii (2021) — "Overhead for Simulating a Non-local Channel"
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np
import mlx.core as mx

from .cutting import CutDecomposition, SubcircuitPair


# ---------------------------------------------------------------------------
# Recombination result
# ---------------------------------------------------------------------------

@dataclass
class RecombinationResult:
    """
    Output of coordinator.recombine().

    expectation:   reconstructed full-circuit expectation value
    n_pairs:       number of subcircuit pairs evaluated
    overhead:      QPD sampling overhead used
    term_values:   per-term (weight, left_val, right_val) for diagnostics
    """
    expectation:  float
    n_pairs:      int
    overhead:     int
    term_values:  list[tuple[float, float, float]]

    def __repr__(self) -> str:
        return (
            f"RecombinationResult("
            f"expectation={self.expectation:.6f}, "
            f"n_pairs={self.n_pairs}, "
            f"overhead={self.overhead})"
        )


# ---------------------------------------------------------------------------
# Core recombination
# ---------------------------------------------------------------------------

def recombine(
    decomp: CutDecomposition,
    left_results:  Sequence[float],
    right_results: Sequence[float],
) -> RecombinationResult:
    """
    Recombine subcircuit results into a full-circuit expectation value.

    Args:
        decomp:        CutDecomposition from CutCircuit.decompose()
        left_results:  expectation value from each left subcircuit (one per pair)
        right_results: expectation value from each right subcircuit (one per pair)

    Returns:
        RecombinationResult with the reconstructed expectation value.

    The recombination formula for sum_Z observable:
        E_full = Σ_i w_i * (E_left_i + E_right_i)

    The additive form holds because Z is a local observable — each qubit's
    contribution is entirely in either the left or right partition.
    """
    if len(left_results) != len(decomp.pairs):
        raise ValueError(
            f"Expected {len(decomp.pairs)} left results, got {len(left_results)}"
        )
    if len(right_results) != len(decomp.pairs):
        raise ValueError(
            f"Expected {len(decomp.pairs)} right results, got {len(right_results)}"
        )

    total = 0.0
    term_values = []
    for pair, lv, rv in zip(decomp.pairs, left_results, right_results):
        contribution = pair.weight * (float(lv) + float(rv))
        total += contribution
        term_values.append((pair.weight, float(lv), float(rv)))

    return RecombinationResult(
        expectation=total,
        n_pairs=len(decomp.pairs),
        overhead=decomp.overhead,
        term_values=term_values,
    )


def recombine_multiplicative(
    decomp: CutDecomposition,
    left_results:  Sequence[float],
    right_results: Sequence[float],
) -> RecombinationResult:
    """
    Recombine using multiplicative combination: w_i * E_left_i * E_right_i.

    Use this when the observable factorizes across the cut, e.g. a product
    of left-only and right-only observables, or when testing correlators.
    For standard sum_Z recombination use recombine() instead.
    """
    if len(left_results) != len(decomp.pairs):
        raise ValueError(
            f"Expected {len(decomp.pairs)} left results, got {len(left_results)}"
        )

    total = 0.0
    term_values = []
    for pair, lv, rv in zip(decomp.pairs, left_results, right_results):
        contribution = pair.weight * float(lv) * float(rv)
        total += contribution
        term_values.append((pair.weight, float(lv), float(rv)))

    return RecombinationResult(
        expectation=total,
        n_pairs=len(decomp.pairs),
        overhead=decomp.overhead,
        term_values=term_values,
    )


# ---------------------------------------------------------------------------
# Local coordinator: run all pairs on the current device
# ---------------------------------------------------------------------------

def run_local(
    decomp: CutDecomposition,
    params: mx.array,
    observable: str = "sum_z",
) -> RecombinationResult:
    """
    Execute all subcircuit pairs locally and recombine.

    This is the single-node path. For multi-node distribution, use
    NodeCoordinator which dispatches pairs across the P2P network.

    Args:
        decomp:     CutDecomposition from CutCircuit.decompose()
        params:     parameter vector shared across all subcircuits
        observable: passed to Circuit.compile()

    Returns:
        RecombinationResult with reconstructed expectation value.
    """
    left_results  = []
    right_results = []

    for pair in decomp.pairs:
        lv = float(pair.left.compile(observable)(params).item())
        rv = float(pair.right.compile(observable)(params).item())
        left_results.append(lv)
        right_results.append(rv)

    return recombine(decomp, left_results, right_results)


# ---------------------------------------------------------------------------
# Node coordinator: distribute pairs across multiple local simulators
# ---------------------------------------------------------------------------

@dataclass
class NodeSpec:
    """Specification for one simulator node."""
    node_id:      str
    n_qubits_max: int
    backend:      str = "sv"      # "sv" | "dm" | "tn"


class NodeCoordinator:
    """
    Distributes SubcircuitPairs across a pool of NodeSpecs and collects results.

    In the full distributed network, each NodeSpec corresponds to a remote
    Apple Silicon device running zilver-node. Here we simulate distribution
    locally: each pair is routed to the first node with sufficient qubit capacity.

    Usage:
        nodes = [NodeSpec("node0", 20), NodeSpec("node1", 20)]
        coord = NodeCoordinator(nodes)
        result = coord.run(decomp, params)
    """

    def __init__(self, nodes: list[NodeSpec]):
        self.nodes = nodes

    def _select_node(self, n_qubits: int) -> NodeSpec | None:
        for node in self.nodes:
            if node.n_qubits_max >= n_qubits:
                return node
        return None

    def run(
        self,
        decomp: CutDecomposition,
        params: mx.array,
        observable: str = "sum_z",
    ) -> RecombinationResult:
        """
        Route each SubcircuitPair to an eligible node and recombine results.

        In this local simulation, eligible nodes are checked for qubit capacity
        but all execution happens on the current device via MLX.
        """
        left_results  = []
        right_results = []

        for pair in decomp.pairs:
            n_l = pair.left.n_qubits
            n_r = pair.right.n_qubits

            node_l = self._select_node(n_l)
            node_r = self._select_node(n_r)

            if node_l is None:
                raise RuntimeError(
                    f"No node with capacity >= {n_l} qubits for left subcircuit"
                )
            if node_r is None:
                raise RuntimeError(
                    f"No node with capacity >= {n_r} qubits for right subcircuit"
                )

            lv = float(pair.left.compile(observable)(params).item())
            rv = float(pair.right.compile(observable)(params).item())
            left_results.append(lv)
            right_results.append(rv)

        return recombine(decomp, left_results, right_results)

    def capacity_check(self, decomp: CutDecomposition) -> bool:
        """Return True if all pairs can be routed to available nodes."""
        for pair in decomp.pairs:
            if self._select_node(pair.left.n_qubits) is None:
                return False
            if self._select_node(pair.right.n_qubits) is None:
                return False
        return True
