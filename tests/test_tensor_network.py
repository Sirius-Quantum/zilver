"""
Tests for MPS tensor network simulator.

Correctness checks:
  - MPS matches statevector for exact chi (no truncation)
  - Single and two-qubit gate application
  - Non-adjacent gate via SWAP decomposition
  - Expectation values match statevector
  - Bond dimension growth with entanglement
  - chi_max truncation reduces bond dimension
  - Large-qubit circuits (50q) run without memory error
  - Factory circuits match statevector Circuit at small n
"""

import math
import numpy as np
import mlx.core as mx
import pytest

import sys; sys.path.insert(0, "src")
from zilver.tensor_network import (
    init_mps,
    apply_single_qubit_gate_mps,
    apply_two_qubit_gate_mps,
    mps_to_statevector,
    expectation_z_mps,
    expectation_sum_z_mps,
    bond_dimensions,
    max_bond_dimension,
    MPSCircuit,
    hardware_efficient_mps,
    qaoa_style_mps,
)
from zilver.circuit import Circuit, hardware_efficient, qaoa_style
from zilver.simulator import apply_gate, expectation_pauli_sum
from zilver import gates as G


def sv_init(n):
    sv = np.zeros(2**n, dtype=np.complex64); sv[0] = 1.0
    return sv


# ---------------------------------------------------------------------------
# MPS initialization
# ---------------------------------------------------------------------------

class TestMPSInit:
    def test_zero_state_shape(self):
        tensors = init_mps(4)
        assert len(tensors) == 4
        for t in tensors:
            assert t.shape == (1, 2, 1)

    def test_zero_state_statevector(self):
        n = 4
        tensors = init_mps(n)
        sv = mps_to_statevector(tensors, n)
        expected = np.zeros(2**n, dtype=np.complex64); expected[0] = 1.0
        np.testing.assert_allclose(np.abs(sv), np.abs(expected), atol=1e-6)


# ---------------------------------------------------------------------------
# Single-qubit gates
# ---------------------------------------------------------------------------

class TestSingleQubitGates:
    def test_h_on_qubit0(self):
        n = 1
        tensors = init_mps(n)
        H = np.array(G.H().tolist(), dtype=np.complex64)
        tensors = apply_single_qubit_gate_mps(tensors, H, 0)
        sv = mps_to_statevector(tensors, n)
        expected = np.array([1/math.sqrt(2), 1/math.sqrt(2)], dtype=np.complex64)
        np.testing.assert_allclose(np.abs(sv), np.abs(expected), atol=1e-5)

    def test_x_gate_flips_qubit(self):
        n = 3
        tensors = init_mps(n)
        X = np.array(G.X().tolist(), dtype=np.complex64)
        tensors = apply_single_qubit_gate_mps(tensors, X, 1)
        sv = mps_to_statevector(tensors, n)
        # |010> = index 2
        assert abs(abs(sv[2]) - 1.0) < 1e-5

    def test_single_qubit_matches_sv(self):
        n = 4
        tensors = init_mps(n)
        sv = sv_init(n)

        gates_np = [
            (np.array(G.H().tolist(), dtype=np.complex64), 0),
            (np.array(G.X().tolist(), dtype=np.complex64), 2),
            (np.array(G.Z().tolist(), dtype=np.complex64), 3),
        ]
        gates_mx = [(G.H(), 0), (G.X(), 2), (G.Z(), 3)]

        for g_np, q in gates_np:
            tensors = apply_single_qubit_gate_mps(tensors, g_np, q)
        for g_mx, q in gates_mx:
            sv = apply_gate(mx.array(sv), g_mx, [q], n)
            sv = np.array(sv.tolist(), dtype=np.complex64)

        sv_mps = mps_to_statevector(tensors, n)
        np.testing.assert_allclose(np.abs(sv_mps), np.abs(sv), atol=1e-5)


# ---------------------------------------------------------------------------
# Two-qubit gates
# ---------------------------------------------------------------------------

class TestTwoQubitGates:
    def test_cnot_creates_bell_state(self):
        n = 2
        H  = np.array(G.H().tolist(), dtype=np.complex64)
        CX = np.array(G.CNOT().tolist(), dtype=np.complex64)

        tensors = init_mps(n)
        tensors = apply_single_qubit_gate_mps(tensors, H, 0)
        tensors = apply_two_qubit_gate_mps(tensors, CX, 0, 1, chi_max=None)

        sv = mps_to_statevector(tensors, n)
        # Bell state: (|00> + |11>) / sqrt(2)
        assert abs(abs(sv[0]) - 1/math.sqrt(2)) < 1e-5
        assert abs(abs(sv[3]) - 1/math.sqrt(2)) < 1e-5
        assert abs(sv[1]) < 1e-5
        assert abs(sv[2]) < 1e-5

    def test_cnot_bond_dim_grows(self):
        n = 4
        tensors = init_mps(n)
        H  = np.array(G.H().tolist(), dtype=np.complex64)
        CX = np.array(G.CNOT().tolist(), dtype=np.complex64)

        for q in range(n):
            tensors = apply_single_qubit_gate_mps(tensors, H, q)
        for q in range(n - 1):
            tensors = apply_two_qubit_gate_mps(tensors, CX, q, q+1, chi_max=None)

        assert max_bond_dimension(tensors) > 1

    def test_two_qubit_matches_sv_exact(self):
        n = 4
        H  = np.array(G.H().tolist(), dtype=np.complex64)
        CX = np.array(G.CNOT().tolist(), dtype=np.complex64)

        tensors = init_mps(n)
        sv = sv_init(n)

        for q in range(n):
            tensors = apply_single_qubit_gate_mps(tensors, H, q)
            sv = apply_gate(mx.array(sv), G.H(), [q], n)
            sv = np.array(sv.tolist(), dtype=np.complex64)

        for q in range(n - 1):
            tensors = apply_two_qubit_gate_mps(tensors, CX, q, q+1, chi_max=None)
            sv = apply_gate(mx.array(sv), G.CNOT(), [q, q+1], n)
            sv = np.array(sv.tolist(), dtype=np.complex64)

        sv_mps = mps_to_statevector(tensors, n)
        np.testing.assert_allclose(np.abs(sv_mps), np.abs(sv), atol=1e-4)


# ---------------------------------------------------------------------------
# Non-adjacent gates via SWAP decomposition
# ---------------------------------------------------------------------------

class TestNonAdjacentGates:
    def test_cnot_non_adjacent_matches_sv(self):
        n = 4
        c_mps = MPSCircuit(n, chi_max=None)
        c_mps.h(0)
        c_mps.cnot(0, 3)   # non-adjacent: distance 3

        c_sv = Circuit(n)
        c_sv.h(0)
        c_sv.cnot(0, 3)

        params = mx.array([], dtype=mx.float32)
        sv_mps = c_mps.statevector(params)
        sv_ref = np.array(c_sv.statevector(params).numpy(), dtype=np.complex64)

        np.testing.assert_allclose(np.abs(sv_mps), np.abs(sv_ref), atol=1e-4)

    def test_cnot_distance2_matches_sv(self):
        n = 3
        c_mps = MPSCircuit(n, chi_max=None)
        c_mps.h(0)
        c_mps.cnot(0, 2)

        c_sv = Circuit(n)
        c_sv.h(0)
        c_sv.cnot(0, 2)

        params = mx.array([], dtype=mx.float32)
        sv_mps = c_mps.statevector(params)
        sv_ref = np.array(c_sv.statevector(params).numpy(), dtype=np.complex64)

        np.testing.assert_allclose(np.abs(sv_mps), np.abs(sv_ref), atol=1e-4)


# ---------------------------------------------------------------------------
# Expectation values
# ---------------------------------------------------------------------------

class TestExpectationValues:
    def test_z_expectation_zero_state(self):
        n = 3
        tensors = init_mps(n)
        for q in range(n):
            assert abs(expectation_z_mps(tensors, q, n) - 1.0) < 1e-5

    def test_z_expectation_x_state(self):
        # After X on qubit 0: |100>, <Z_0> = -1, <Z_1> = +1
        n = 3
        tensors = init_mps(n)
        X = np.array(G.X().tolist(), dtype=np.complex64)
        tensors = apply_single_qubit_gate_mps(tensors, X, 0)
        assert abs(expectation_z_mps(tensors, 0, n) + 1.0) < 1e-5
        assert abs(expectation_z_mps(tensors, 1, n) - 1.0) < 1e-5

    def test_sum_z_matches_sv(self):
        n = 5
        H  = np.array(G.H().tolist(), dtype=np.complex64)
        CX = np.array(G.CNOT().tolist(), dtype=np.complex64)

        tensors = init_mps(n)
        sv = sv_init(n)

        for q in [0, 2, 4]:
            tensors = apply_single_qubit_gate_mps(tensors, H, q)
            sv = np.array(apply_gate(mx.array(sv), G.H(), [q], n).tolist())
        for q in range(n - 1):
            tensors = apply_two_qubit_gate_mps(tensors, CX, q, q+1, chi_max=None)
            sv = np.array(apply_gate(mx.array(sv), G.CNOT(), [q, q+1], n).tolist())

        mps_val = expectation_sum_z_mps(tensors, n)
        sv_val  = float(expectation_pauli_sum(mx.array(sv.astype(np.complex64)), n).item())
        assert abs(mps_val - sv_val) < 1e-4


# ---------------------------------------------------------------------------
# chi_max truncation
# ---------------------------------------------------------------------------

class TestTruncation:
    def test_chi_max_limits_bond_dim(self):
        n = 6
        chi_max = 4
        H  = np.array(G.H().tolist(), dtype=np.complex64)
        CX = np.array(G.CNOT().tolist(), dtype=np.complex64)

        tensors = init_mps(n)
        for q in range(n):
            tensors = apply_single_qubit_gate_mps(tensors, H, q)
        for q in range(n - 1):
            tensors = apply_two_qubit_gate_mps(tensors, CX, q, q+1, chi_max=chi_max)

        assert max_bond_dimension(tensors) <= chi_max

    def test_small_chi_still_finite(self):
        # Truncation should give finite (not NaN/inf) expectation values
        n = 8
        c = MPSCircuit(n, chi_max=2)
        for q in range(n):
            c.h(q)
        for q in range(n - 1):
            c.cnot(q, q + 1)

        f = c.compile()
        result = f(mx.array([], dtype=mx.float32))
        assert math.isfinite(result)

    def test_exact_chi_matches_sv(self):
        # chi_max=None gives exact result matching statevector
        n = 6
        c_mps = MPSCircuit(n, chi_max=None)
        c_sv  = Circuit(n)

        for q in range(n):
            c_mps.h(q); c_sv.h(q)
        for q in range(n - 1):
            c_mps.cnot(q, q+1); c_sv.cnot(q, q+1)

        params = mx.array([], dtype=mx.float32)
        mps_val = c_mps.compile()(params)
        sv_val  = float(c_sv.compile()(params).item())
        assert abs(mps_val - sv_val) < 1e-4


# ---------------------------------------------------------------------------
# MPSCircuit parametrized
# ---------------------------------------------------------------------------

class TestMPSCircuitParametrized:
    def test_ry_parametrized_matches_sv(self):
        n = 4
        c_mps = MPSCircuit(n, chi_max=None)
        c_sv  = Circuit(n)

        p = 0
        for q in range(n):
            c_mps.ry(q, p); c_sv.ry(q, p); p += 1
        for q in range(n - 1):
            c_mps.cnot(q, q+1); c_sv.cnot(q, q+1)
        for q in range(n):
            c_mps.ry(q, p); c_sv.ry(q, p); p += 1

        rng = np.random.default_rng(0)
        params_np = rng.uniform(-math.pi, math.pi, p).astype(np.float32)
        params = mx.array(params_np)

        mps_val = c_mps.compile()(params)
        sv_val  = float(c_sv.compile()(params).item())
        assert abs(mps_val - sv_val) < 1e-3

    def test_mps_circuit_runs(self):
        n, d = 6, 3
        c = MPSCircuit(n, chi_max=32)
        p = 0
        for layer in range(d):
            for q in range(n):
                c.ry(q, p); p += 1
            for q in range(n - 1):
                c.cnot(q, q + 1)
        c.n_params = p

        rng = np.random.default_rng(1)
        params = mx.array(rng.uniform(-math.pi, math.pi, p).astype(np.float32))
        result = c.compile()(params)
        assert -n - 0.1 <= result <= n + 0.1


# ---------------------------------------------------------------------------
# Large qubit count — TN1 target (50 qubits)
# ---------------------------------------------------------------------------

class TestLargeQubitCount:
    def test_50_qubit_hw_efficient_runs(self):
        n = 50
        c = hardware_efficient_mps(n, depth=2, chi_max=32)
        assert c.n_params > 0

        rng = np.random.default_rng(42)
        params = mx.array(rng.uniform(-math.pi, math.pi, c.n_params).astype(np.float32))
        result = c.compile()(params)
        assert -n - 0.1 <= result <= n + 0.1

    def test_50_qubit_qaoa_runs(self):
        n = 50
        c = qaoa_style_mps(n, depth=2, chi_max=16)
        assert c.n_params == 4   # 2 * depth

        rng = np.random.default_rng(7)
        params = mx.array(rng.uniform(-math.pi, math.pi, c.n_params).astype(np.float32))
        result = c.compile()(params)
        assert -n - 0.1 <= result <= n + 0.1

    def test_bond_dimensions_logged(self):
        n = 10
        c = hardware_efficient_mps(n, depth=1, chi_max=8)
        params = mx.array(np.zeros(c.n_params, dtype=np.float32))
        chi = c.max_bond_dim(params)
        assert 1 <= chi <= 8


# ---------------------------------------------------------------------------
# Factory circuits match statevector Circuit at small n
# ---------------------------------------------------------------------------

class TestFactoryEquivalence:
    @pytest.mark.parametrize("n,d", [(4, 2), (6, 1)])
    def test_hardware_efficient_mps_matches_sv(self, n, d):
        c_mps = hardware_efficient_mps(n, d, chi_max=None)
        c_sv  = hardware_efficient(n, d)
        assert c_mps.n_params == c_sv.n_params

        rng = np.random.default_rng(99)
        params = mx.array(rng.uniform(-math.pi, math.pi, c_sv.n_params).astype(np.float32))
        mps_val = c_mps.compile()(params)
        sv_val  = float(c_sv.compile()(params).item())
        assert abs(mps_val - sv_val) < 1e-3

    @pytest.mark.parametrize("n,d", [(4, 2), (6, 1)])
    def test_qaoa_mps_matches_sv(self, n, d):
        c_mps = qaoa_style_mps(n, d, chi_max=None)
        c_sv  = qaoa_style(n, d)
        assert c_mps.n_params == c_sv.n_params

        rng = np.random.default_rng(13)
        params = mx.array(rng.uniform(-math.pi, math.pi, c_sv.n_params).astype(np.float32))
        mps_val = c_mps.compile()(params)
        sv_val  = float(c_sv.compile()(params).item())
        assert abs(mps_val - sv_val) < 1e-3
