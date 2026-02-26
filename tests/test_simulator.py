"""Statevector simulation correctness tests."""

import math
import numpy as np
import mlx.core as mx
import pytest

import sys; sys.path.insert(0, "src")
from zilver.simulator import apply_gate, expectation_z, expectation_pauli_sum, StateVector
from zilver import gates as G


def ev(x):
    mx.eval(x)
    return float(x.item())


class TestApplyGate:
    def test_x_flips_zero_state(self):
        n = 1
        state = mx.array([1.0, 0.0], dtype=mx.complex64)
        result = apply_gate(state, G.X(), [0], n)
        mx.eval(result)
        arr = np.array(result.tolist())
        assert abs(arr[0]) < 1e-6
        assert abs(abs(arr[1]) - 1.0) < 1e-6

    def test_h_creates_superposition(self):
        n = 1
        state = mx.array([1.0, 0.0], dtype=mx.complex64)
        result = apply_gate(state, G.H(), [0], n)
        mx.eval(result)
        arr = np.array(result.tolist())
        s = 1 / math.sqrt(2)
        np.testing.assert_allclose(np.abs(arr), [s, s], atol=1e-6)

    def test_cnot_entangles(self):
        # |10> -> |11>
        n = 2
        state = mx.array([0.0, 0.0, 1.0, 0.0], dtype=mx.complex64)  # |10>
        result = apply_gate(state, G.CNOT(), [0, 1], n)
        mx.eval(result)
        arr = np.array(result.tolist())
        assert abs(abs(arr[3]) - 1.0) < 1e-6, f"Expected |11>, got {arr}"

    def test_cnot_does_not_flip_zero_control(self):
        # |01> -> |01> (control=0, target=1; control qubit is 0)
        n = 2
        state = mx.array([0.0, 1.0, 0.0, 0.0], dtype=mx.complex64)  # |01>
        result = apply_gate(state, G.CNOT(), [0, 1], n)
        mx.eval(result)
        arr = np.array(result.tolist())
        assert abs(abs(arr[1]) - 1.0) < 1e-6

    def test_ry_on_second_qubit(self):
        n = 2
        state = mx.array([1.0, 0.0, 0.0, 0.0], dtype=mx.complex64)  # |00>
        result = apply_gate(state, G.RY(math.pi), [1], n)
        mx.eval(result)
        arr = np.array(result.tolist())
        # RY(pi)|0> = |1>, so |00> -> |01>
        assert abs(abs(arr[1]) - 1.0) < 1e-5, f"Expected |01>, got {arr}"

    def test_gate_preserves_norm(self):
        n = 3
        rng = np.random.default_rng(0)
        raw = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        raw /= np.linalg.norm(raw)
        state = mx.array(raw.astype(np.complex64))

        for qubit in range(n):
            state = apply_gate(state, G.H(), [qubit], n)
        mx.eval(state)
        arr = np.array(state.tolist())
        assert abs(np.linalg.norm(arr) - 1.0) < 1e-5


class TestExpectation:
    def test_z_on_zero_state(self):
        # <0|Z|0> = +1
        n = 1
        state = mx.array([1.0, 0.0], dtype=mx.complex64)
        assert abs(ev(expectation_z(state, 0, n)) - 1.0) < 1e-6

    def test_z_on_one_state(self):
        # <1|Z|1> = -1
        n = 1
        state = mx.array([0.0, 1.0], dtype=mx.complex64)
        assert abs(ev(expectation_z(state, 0, n)) + 1.0) < 1e-6

    def test_z_on_superposition(self):
        # <+|Z|+> = 0
        s = 1 / math.sqrt(2)
        state = mx.array([s, s], dtype=mx.complex64)
        assert abs(ev(expectation_z(state, 0, 1))) < 1e-6

    def test_z1_on_two_qubit_state(self):
        # |01>: qubit 0 = 0, qubit 1 = 1
        # <Z_0> = +1, <Z_1> = -1
        n = 2
        state = mx.array([0.0, 1.0, 0.0, 0.0], dtype=mx.complex64)
        assert abs(ev(expectation_z(state, 0, n)) - 1.0) < 1e-6
        assert abs(ev(expectation_z(state, 1, n)) + 1.0) < 1e-6

    def test_pauli_sum_all_zeros(self):
        n = 3
        state = mx.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=mx.complex64)
        val = ev(expectation_pauli_sum(state, n))
        assert abs(val - 3.0) < 1e-5  # sum of +1 for each qubit in |000>


class TestStateVector:
    def test_zero_state_initialization(self):
        sv = StateVector.zero_state(4)
        np.testing.assert_allclose(sv.numpy()[0], 1.0)
        np.testing.assert_allclose(np.sum(np.abs(sv.numpy())**2), 1.0, atol=1e-6)

    def test_probabilities_sum_to_one(self):
        sv = StateVector.zero_state(3)
        probs = sv.probabilities()
        mx.eval(probs)
        assert abs(float(mx.sum(probs).item()) - 1.0) < 1e-6
