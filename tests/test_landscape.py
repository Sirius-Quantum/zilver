"""Loss landscape generation tests."""

import math
import numpy as np
import mlx.core as mx
import pytest

import sys; sys.path.insert(0, "src")
from zilver.circuit import hardware_efficient, real_amplitudes, qaoa_style
from zilver.landscape import LossLandscape, LandscapeResult


class TestLossLandscape:
    def test_output_shape(self):
        c = hardware_efficient(n_qubits=4, depth=1)
        landscape = LossLandscape(c, sweep_params=(0, 1), resolution=5)
        result = landscape.compute()

        assert len(result.loss_landscape) == 5
        assert len(result.loss_landscape[0]) == 5
        assert len(result.gradient_landscape) == 5
        assert len(result.gradient_landscape[0]) == 5

    def test_loss_values_in_expected_range(self):
        n = 4
        c = hardware_efficient(n, depth=1)
        landscape = LossLandscape(c, sweep_params=(0, 1), resolution=10)
        result = landscape.compute()

        flat = [v for row in result.loss_landscape for v in row]
        # sum Z expectation is in [-n, n]
        assert all(-n - 0.1 <= v <= n + 0.1 for v in flat)

    def test_gradient_values_non_negative(self):
        c = real_amplitudes(n_qubits=4, depth=2)
        landscape = LossLandscape(c, sweep_params=(0, 1), resolution=8)
        result = landscape.compute()

        flat = [v for row in result.gradient_landscape for v in row]
        assert all(v >= -1e-5 for v in flat), "Gradient magnitudes must be non-negative"

    def test_plateau_coverage_between_0_and_1(self):
        c = hardware_efficient(n_qubits=6, depth=4)
        landscape = LossLandscape(c, sweep_params=(0, 1), resolution=10)
        result = landscape.compute()

        cov = result.plateau_coverage()
        assert 0.0 <= cov <= 1.0

    def test_trainability_score_complements_coverage(self):
        c = real_amplitudes(n_qubits=4, depth=2)
        landscape = LossLandscape(c, sweep_params=(0, 1), resolution=8)
        result = landscape.compute()

        assert abs(result.plateau_coverage() + result.trainability_score() - 1.0) < 1e-6

    def test_metadata_fields(self):
        c = hardware_efficient(n_qubits=4, depth=1)
        landscape = LossLandscape(c, sweep_params=(2, 3), resolution=5)
        result = landscape.compute()

        assert result.n_qubits == 4
        assert result.resolution == 5
        assert result.backend == "zilver-mlx"
        assert result.wall_time_seconds > 0
        assert result.metadata["sweep_params"] == [2, 3]

    def test_deterministic_with_fixed_seed(self):
        c = hardware_efficient(n_qubits=4, depth=2)
        l1 = LossLandscape(c, sweep_params=(0, 1), resolution=5, seed=123)
        l2 = LossLandscape(c, sweep_params=(0, 1), resolution=5, seed=123)
        r1 = l1.compute()
        r2 = l2.compute()

        np.testing.assert_allclose(
            np.array(r1.loss_landscape),
            np.array(r2.loss_landscape),
            atol=1e-5,
        )

    @pytest.mark.parametrize("family_fn,n,d", [
        (hardware_efficient, 6, 3),
        (real_amplitudes,    6, 3),
        (qaoa_style,         6, 3),
    ])
    def test_sirius_families_20x20(self, family_fn, n, d):
        """Full 20x20 grid for all Sirius circuit families."""
        c = family_fn(n, d)
        landscape = LossLandscape(c, sweep_params=(0, 1), resolution=20)
        result = landscape.compute()

        assert len(result.loss_landscape) == 20
        assert len(result.loss_landscape[0]) == 20
        assert result.n_qubits == n
