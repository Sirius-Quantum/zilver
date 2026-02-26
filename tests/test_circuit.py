"""Circuit execution and factory function tests."""

import math
import numpy as np
import mlx.core as mx
import pytest

import sys; sys.path.insert(0, "src")
from zilver.circuit import (
    Circuit, hardware_efficient, real_amplitudes, qaoa_style, efficient_su2
)


def ev(x):
    mx.eval(x)
    return float(x.item())


class TestCircuitBasic:
    def test_empty_circuit_is_zero_state(self):
        c = Circuit(2)
        f = c.compile()
        params = mx.array([], dtype=mx.float32)
        # <sum Z> for |00> = 2.0
        result = ev(f(params))
        assert abs(result - 2.0) < 1e-5

    def test_x_gate_flips_expectation(self):
        c = Circuit(1)
        c.x(0)
        f = c.compile(observable="z0")
        params = mx.array([], dtype=mx.float32)
        # |1> has <Z> = -1
        assert abs(ev(f(params)) + 1.0) < 1e-5

    def test_ry_pi_over_2(self):
        c = Circuit(1)
        c.ry(qubit=0, param_idx=0)
        f = c.compile(observable="z0")
        # RY(pi/2)|0>: <Z> = cos(pi/2) = 0
        params = mx.array([math.pi / 2], dtype=mx.float32)
        assert abs(ev(f(params))) < 0.02  # small tolerance for float32

    def test_bell_state_expectation(self):
        # H on qubit 0, CNOT(0,1): Bell state (|00> + |11>) / sqrt(2)
        # <Z_0> = <Z_1> = 0, but <Z_0 Z_1> = 1 (correlated)
        c = Circuit(2)
        c.h(0)
        c.cnot(0, 1)
        f = c.compile(observable="sum_z")
        params = mx.array([], dtype=mx.float32)
        # <Z_0 + Z_1> = 0 for Bell state
        assert abs(ev(f(params))) < 1e-5

    def test_circuit_repr(self):
        c = Circuit(4)
        c.h(0)
        c.cnot(0, 1)
        c.ry(2, 0)
        assert "n_qubits=4" in repr(c)
        assert "n_params=1" in repr(c)


class TestCircuitFactories:
    @pytest.mark.parametrize("n,d", [(2, 1), (4, 2), (6, 3)])
    def test_hardware_efficient_runs(self, n, d):
        c = hardware_efficient(n, d)
        params = mx.array(
            np.random.uniform(-math.pi, math.pi, c.n_params).astype(np.float32)
        )
        f = c.compile()
        result = ev(f(params))
        # Expectation value of sum Z is bounded by [-n, n]
        assert -n - 0.1 <= result <= n + 0.1

    @pytest.mark.parametrize("n,d", [(4, 2), (6, 3)])
    def test_real_amplitudes_runs(self, n, d):
        c = real_amplitudes(n, d)
        params = mx.array(
            np.random.uniform(0, math.pi, c.n_params).astype(np.float32)
        )
        f = c.compile()
        result = ev(f(params))
        assert -n - 0.1 <= result <= n + 0.1

    @pytest.mark.parametrize("n,d", [(4, 2), (6, 3)])
    def test_qaoa_style_runs(self, n, d):
        c = qaoa_style(n, d)
        assert c.n_params == 2 * d
        params = mx.array(
            np.random.uniform(-math.pi, math.pi, c.n_params).astype(np.float32)
        )
        f = c.compile()
        result = ev(f(params))
        assert -n - 0.1 <= result <= n + 0.1


class TestVmap:
    def test_vmap_over_param_batch(self):
        """Core Sirius use case: batch 400 param vectors in one call."""
        c = hardware_efficient(n_qubits=4, depth=2)
        f = c.compile()

        rng = np.random.default_rng(0)
        batch = rng.uniform(-math.pi, math.pi, (100, c.n_params)).astype(np.float32)
        params_batch = mx.array(batch)

        results = mx.vmap(f)(params_batch)
        mx.eval(results)
        arr = np.array(results.tolist())

        assert arr.shape == (100,)
        # All expectations in [-n, n]
        assert np.all(arr >= -4.1) and np.all(arr <= 4.1)

    def test_vmap_results_match_sequential(self):
        """vmap must give same results as sequential evaluation."""
        c = real_amplitudes(n_qubits=3, depth=1)
        f = c.compile()

        rng = np.random.default_rng(1)
        batch = rng.uniform(-math.pi, math.pi, (10, c.n_params)).astype(np.float32)
        params_batch = mx.array(batch)

        # Batched
        batched = np.array(mx.vmap(f)(params_batch).tolist())

        # Sequential
        sequential = np.array([
            float(f(mx.array(batch[i])).item())
            for i in range(10)
        ])

        np.testing.assert_allclose(batched, sequential, atol=1e-4)
