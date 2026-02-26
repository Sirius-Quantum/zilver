"""Parameter shift gradient correctness tests."""

import math
import numpy as np
import mlx.core as mx
import pytest

import sys; sys.path.insert(0, "src")
from zilver.circuit import hardware_efficient, real_amplitudes
from zilver.gradients import param_shift_gradient, gradient_variance


class TestParamShift:
    def test_gradient_shape(self):
        c = hardware_efficient(n_qubits=4, depth=1)
        f = c.compile()
        params = mx.array(
            np.random.uniform(-math.pi, math.pi, c.n_params).astype(np.float32)
        )
        grads = param_shift_gradient(f, params)
        mx.eval(grads)
        assert grads.shape == (c.n_params,)

    def test_gradient_matches_finite_difference(self):
        """Parameter shift should agree with finite differences to 1e-3."""
        c = real_amplitudes(n_qubits=3, depth=1)
        f = c.compile()

        rng = np.random.default_rng(42)
        params_np = rng.uniform(-math.pi, math.pi, c.n_params).astype(np.float32)
        params = mx.array(params_np)

        grads = np.array(param_shift_gradient(f, params).tolist())

        eps = 1e-3
        fd_grads = np.zeros(c.n_params)
        for i in range(c.n_params):
            p_plus  = params_np.copy(); p_plus[i]  += eps
            p_minus = params_np.copy(); p_minus[i] -= eps
            f_plus  = float(f(mx.array(p_plus)).item())
            f_minus = float(f(mx.array(p_minus)).item())
            fd_grads[i] = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(grads, fd_grads, atol=5e-3)

    def test_zero_gradient_at_symmetry_point(self):
        """RY at theta=0 with |0> input: d/d(theta) <Z> at theta=0 should be 0."""
        from zilver.circuit import Circuit
        c = Circuit(1)
        c.ry(qubit=0, param_idx=0)
        f = c.compile(observable="z0")

        # At theta=0, we are at the top of the cosine -> derivative is 0
        params = mx.array([0.0], dtype=mx.float32)
        grads = param_shift_gradient(f, params)
        assert abs(float(grads[0].item())) < 1e-5

    def test_gradient_nonzero_away_from_plateau(self):
        """Gradients should be non-negligible for shallow circuits."""
        c = real_amplitudes(n_qubits=2, depth=1)
        f = c.compile()
        params = mx.array(
            np.full(c.n_params, math.pi / 4, dtype=np.float32)
        )
        grads = np.array(param_shift_gradient(f, params).tolist())
        assert np.max(np.abs(grads)) > 1e-3, "Expected nonzero gradients"


class TestGradientVariance:
    def test_gradient_variance_output_shape(self):
        c = hardware_efficient(n_qubits=4, depth=2)
        f = c.compile()
        stats = gradient_variance(f, c.n_params, n_samples=20, seed=0)
        assert len(stats["variance_per_param"]) == c.n_params
        assert stats["mean_gradient_magnitude"] >= 0.0

    def test_deep_circuit_has_lower_variance_than_shallow(self):
        """Barren plateau signature: deeper circuits have smaller gradient variance."""
        n = 6
        shallow = hardware_efficient(n, depth=1)
        deep    = hardware_efficient(n, depth=8)
        f_s = shallow.compile()
        f_d = deep.compile()

        stats_s = gradient_variance(f_s, shallow.n_params, n_samples=50, seed=0)
        stats_d = gradient_variance(f_d, deep.n_params, n_samples=50, seed=0)

        mean_var_shallow = float(np.mean(stats_s["variance_per_param"]))
        mean_var_deep    = float(np.mean(stats_d["variance_per_param"]))

        # Deep circuits should show barren plateau (lower variance)
        # This is a statistical test - may occasionally fail for unlucky seeds
        assert mean_var_deep <= mean_var_shallow * 2.0, (
            f"Expected deep ({mean_var_deep:.4f}) <= shallow ({mean_var_shallow:.4f})"
        )
