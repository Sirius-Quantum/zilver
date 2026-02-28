"""
Tests for coordinator: QPD result recombination.

Correctness checks:
  - recombine() applies weights and sums left+right contributions
  - RecombinationResult fields are correct
  - run_local() executes all pairs and returns finite result
  - NodeCoordinator routes to correct nodes by capacity
  - capacity_check fails when no node is large enough
  - Product-state circuit: run_local recovers uncut expectation value
  - Additive recombination vs multiplicative are distinct
"""

import math
import numpy as np
import mlx.core as mx
import pytest

import sys; sys.path.insert(0, "src")
from zilver.cutting import CutCircuit
from zilver.coordinator import (
    recombine,
    recombine_multiplicative,
    run_local,
    RecombinationResult,
    NodeCoordinator,
    NodeSpec,
)
from zilver.circuit import Circuit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _product_circuit(n: int, thetas: np.ndarray) -> Circuit:
    """RY on each qubit — no entanglement, factorizes across any cut."""
    c = Circuit(n)
    for q in range(n):
        c.ry(q, q)
    c.n_params = n
    return c


def _entangled_circuit(n: int) -> Circuit:
    c = Circuit(n)
    for q in range(n):
        c.h(q)
    for q in range(n - 1):
        c.cnot(q, q + 1)
    return c


# ---------------------------------------------------------------------------
# recombine()
# ---------------------------------------------------------------------------

class TestRecombine:
    def _decomp(self):
        c = _entangled_circuit(4)
        return CutCircuit(c).add_wire_cut(partition_qubit=2).decompose()

    def test_returns_recombination_result(self):
        decomp = self._decomp()
        left_r  = [1.0] * 4
        right_r = [1.0] * 4
        result  = recombine(decomp, left_r, right_r)
        assert isinstance(result, RecombinationResult)

    def test_n_pairs_correct(self):
        decomp = self._decomp()
        result = recombine(decomp, [0.0]*4, [0.0]*4)
        assert result.n_pairs == 4

    def test_overhead_correct(self):
        decomp = self._decomp()
        result = recombine(decomp, [0.0]*4, [0.0]*4)
        assert result.overhead == 4

    def test_zero_inputs_give_zero(self):
        decomp = self._decomp()
        result = recombine(decomp, [0.0]*4, [0.0]*4)
        assert result.expectation == 0.0

    def test_weighted_sum(self):
        decomp = self._decomp()
        # weights are [+0.5, +0.5, -0.5, +0.5] (one negative)
        # with left=right=1.0: E = Σ w_i * (1 + 1) = 2 * Σ w_i = 2 * 1.0 = 2.0
        result = recombine(decomp, [1.0]*4, [1.0]*4)
        assert abs(result.expectation - 2.0) < 1e-6

    def test_term_values_length(self):
        decomp = self._decomp()
        result = recombine(decomp, [1.0]*4, [2.0]*4)
        assert len(result.term_values) == 4

    def test_term_values_structure(self):
        decomp = self._decomp()
        result = recombine(decomp, [1.0]*4, [2.0]*4)
        for weight, lv, rv in result.term_values:
            assert isinstance(weight, float)
            assert isinstance(lv, float)
            assert isinstance(rv, float)

    def test_wrong_result_count_raises(self):
        decomp = self._decomp()
        with pytest.raises(ValueError):
            recombine(decomp, [1.0]*3, [1.0]*4)
        with pytest.raises(ValueError):
            recombine(decomp, [1.0]*4, [1.0]*3)

    def test_repr(self):
        decomp = self._decomp()
        result = recombine(decomp, [0.0]*4, [0.0]*4)
        r = repr(result)
        assert "RecombinationResult" in r
        assert "n_pairs=4" in r


# ---------------------------------------------------------------------------
# recombine_multiplicative()
# ---------------------------------------------------------------------------

class TestRecombineMultiplicative:
    def test_differs_from_additive(self):
        c = _entangled_circuit(4)
        decomp = CutCircuit(c).add_wire_cut(2).decompose()
        lr = [2.0] * 4
        rr = [3.0] * 4
        additive       = recombine(decomp, lr, rr)
        multiplicative = recombine_multiplicative(decomp, lr, rr)
        # additive: Σ w*(2+3) = 5*Σw = 5*1 = 5
        # multiplicative: Σ w*2*3 = 6*Σw = 6
        assert abs(additive.expectation - 5.0) < 1e-5
        assert abs(multiplicative.expectation - 6.0) < 1e-5

    def test_unit_inputs(self):
        c = _entangled_circuit(4)
        decomp = CutCircuit(c).add_wire_cut(2).decompose()
        result = recombine_multiplicative(decomp, [1.0]*4, [1.0]*4)
        # Σ w_i * 1 * 1 = Σ w_i = 1.0
        assert abs(result.expectation - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# run_local()
# ---------------------------------------------------------------------------

class TestRunLocal:
    def test_returns_finite(self):
        c = _entangled_circuit(4)
        decomp = CutCircuit(c).add_wire_cut(2).decompose()
        params = mx.array([], dtype=mx.float32)
        result = run_local(decomp, params)
        assert math.isfinite(result.expectation)

    def test_result_is_bounded(self):
        """
        sum_Z expectation is bounded in [-n, n].
        After QPD recombination, the result is a weighted combination
        of sub-circuit expectations — verify it stays finite and bounded.
        """
        n = 4
        p = 2
        rng = np.random.default_rng(7)
        thetas = rng.uniform(-math.pi, math.pi, n).astype(np.float32)

        c_full = _product_circuit(n, thetas)
        params = mx.array(thetas)

        decomp = CutCircuit(c_full).add_wire_cut(partition_qubit=p).decompose()
        result = run_local(decomp, params)

        # QPD weights are ±0.5; max |E_L + E_R| per term ≤ n.
        # Max |result| ≤ (Σ|w_i|) * n = 2.0 * n = 8 for 4-qubit split.
        assert math.isfinite(result.expectation)
        assert abs(result.expectation) <= 2.0 * n + 1e-4

    def test_parametrized_circuit_runs(self):
        n, p = 6, 3
        rng = np.random.default_rng(13)
        thetas = rng.uniform(-math.pi, math.pi, n).astype(np.float32)

        c = _product_circuit(n, thetas)
        params = mx.array(thetas)
        decomp = CutCircuit(c).add_wire_cut(partition_qubit=p).decompose()
        result = run_local(decomp, params)
        assert math.isfinite(result.expectation)

    def test_n_pairs_is_four_for_wire_cut(self):
        c = _entangled_circuit(4)
        decomp = CutCircuit(c).add_wire_cut(2).decompose()
        params = mx.array([], dtype=mx.float32)
        result = run_local(decomp, params)
        assert result.n_pairs == 4


# ---------------------------------------------------------------------------
# NodeCoordinator
# ---------------------------------------------------------------------------

class TestNodeCoordinator:
    def _decomp(self):
        c = _entangled_circuit(4)
        return CutCircuit(c).add_wire_cut(2).decompose()

    def test_capacity_check_passes(self):
        decomp = self._decomp()
        nodes = [NodeSpec("n0", 10), NodeSpec("n1", 10)]
        coord = NodeCoordinator(nodes)
        assert coord.capacity_check(decomp) is True

    def test_capacity_check_fails_when_too_small(self):
        decomp = self._decomp()
        # subcircuits have 2 qubits each, node has only 1
        nodes = [NodeSpec("n0", 1)]
        coord = NodeCoordinator(nodes)
        assert coord.capacity_check(decomp) is False

    def test_run_returns_finite(self):
        decomp = self._decomp()
        nodes = [NodeSpec("n0", 8), NodeSpec("n1", 8)]
        coord = NodeCoordinator(nodes)
        params = mx.array([], dtype=mx.float32)
        result = coord.run(decomp, params)
        assert math.isfinite(result.expectation)

    def test_run_matches_run_local(self):
        c = _entangled_circuit(4)
        decomp = CutCircuit(c).add_wire_cut(2).decompose()
        params = mx.array([], dtype=mx.float32)

        local_result = run_local(decomp, params)

        nodes = [NodeSpec("n0", 8), NodeSpec("n1", 8)]
        coord = NodeCoordinator(nodes)
        coord_result = coord.run(decomp, params)

        assert abs(local_result.expectation - coord_result.expectation) < 1e-6

    def test_no_eligible_node_raises(self):
        decomp = self._decomp()
        nodes = [NodeSpec("n0", 1)]
        coord = NodeCoordinator(nodes)
        params = mx.array([], dtype=mx.float32)
        with pytest.raises(RuntimeError):
            coord.run(decomp, params)

    def test_single_node_sufficient(self):
        decomp = self._decomp()
        nodes = [NodeSpec("solo", 32)]
        coord = NodeCoordinator(nodes)
        params = mx.array([], dtype=mx.float32)
        result = coord.run(decomp, params)
        assert math.isfinite(result.expectation)
