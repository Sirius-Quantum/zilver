"""
Tests for density matrix simulator.

Correctness checks:
  - Pure state equivalence: no-noise DM matches statevector
  - Trace preservation after gates and noise channels
  - Hermiticity preservation
  - Noise channel limiting cases (p=0, p=1 / gamma=0, gamma=1)
  - NoisyCircuit expectation values
"""

import math
import numpy as np
import mlx.core as mx
import pytest

import sys; sys.path.insert(0, "src")
from zilver.density_matrix import (
    apply_gate_dm,
    apply_kraus_channel,
    depolarizing_kraus,
    amplitude_damping_kraus,
    phase_damping_kraus,
    bit_flip_kraus,
    phase_flip_kraus,
    expectation_z_dm,
    expectation_sum_z_dm,
    trace,
    NoisyCircuit,
)
from zilver.circuit import Circuit
from zilver import gates as G


def ev(x):
    mx.eval(x)
    return float(x.item())


def to_np(m):
    mx.eval(m)
    return np.array(m.tolist(), dtype=np.complex64)


def zero_state_dm(n):
    """Density matrix for |0>^n."""
    dim = 2**n
    rho = np.zeros((dim, dim), dtype=np.complex64)
    rho[0, 0] = 1.0
    return mx.array(rho)


def pure_state_dm(sv: mx.array) -> mx.array:
    """Build density matrix rho = |psi><psi| from statevector."""
    mx.eval(sv)
    v = np.array(sv.tolist(), dtype=np.complex64).reshape(-1, 1)
    return mx.array(v @ v.conj().T)


def is_hermitian(rho, atol=1e-5):
    m = to_np(rho)
    return np.allclose(m, m.conj().T, atol=atol)


def is_valid_dm(rho, atol=1e-5):
    """Check trace=1 and Hermitian."""
    t = ev(trace(rho))
    return abs(t - 1.0) < atol and is_hermitian(rho, atol)


# ---------------------------------------------------------------------------
# Pure state equivalence
# ---------------------------------------------------------------------------

class TestPureStateEquivalence:
    def test_h_gate_matches_sv(self):
        n = 1
        rho = zero_state_dm(n)
        rho = apply_gate_dm(rho, G.H(), [0], n)

        # Statevector reference: H|0> = |+>
        sv = mx.array([1/math.sqrt(2), 1/math.sqrt(2)], dtype=mx.complex64)
        rho_ref = pure_state_dm(sv)
        np.testing.assert_allclose(to_np(rho), to_np(rho_ref), atol=1e-5)

    def test_cnot_matches_sv(self):
        n = 2
        rho = zero_state_dm(n)
        rho = apply_gate_dm(rho, G.H(), [0], n)
        rho = apply_gate_dm(rho, G.CNOT(), [0, 1], n)

        # Bell state |00> + |11> / sqrt(2)
        sv = mx.array([1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], dtype=mx.complex64)
        rho_ref = pure_state_dm(sv)
        np.testing.assert_allclose(to_np(rho), to_np(rho_ref), atol=1e-5)

    def test_expectation_z_matches_sv(self):
        from zilver.simulator import apply_gate, expectation_z
        n = 3
        gates_seq = [(G.H(), [0]), (G.X(), [1]), (G.CNOT(), [0, 2])]

        # Statevector path
        sv_np = np.zeros(2**n, dtype=np.complex64); sv_np[0] = 1.0
        sv = mx.array(sv_np)
        for g, q in gates_seq:
            sv = apply_gate(sv, g, q, n)

        # Density matrix path
        rho = zero_state_dm(n)
        for g, q in gates_seq:
            rho = apply_gate_dm(rho, g, q, n)

        for q in range(n):
            sv_val  = ev(expectation_z(sv, q, n))
            dm_val  = ev(expectation_z_dm(rho, q, n))
            assert abs(sv_val - dm_val) < 1e-5, f"Qubit {q}: sv={sv_val:.4f} dm={dm_val:.4f}"

    def test_sum_z_matches_sv(self):
        from zilver.simulator import apply_gate, expectation_pauli_sum
        n = 4

        # Random circuit of single-qubit gates
        sv_np = np.zeros(2**n, dtype=np.complex64); sv_np[0] = 1.0
        sv = mx.array(sv_np)
        rho = zero_state_dm(n)

        for q in range(n):
            sv  = apply_gate(sv,  G.H(),   [q], n)
            rho = apply_gate_dm(rho, G.H(), [q], n)

        sv_val = ev(expectation_pauli_sum(sv, n))
        dm_val = ev(expectation_sum_z_dm(rho, n))
        assert abs(sv_val - dm_val) < 1e-5


# ---------------------------------------------------------------------------
# Validity preservation
# ---------------------------------------------------------------------------

class TestValidityPreservation:
    def test_trace_preserved_after_gate(self):
        n = 3
        rho = zero_state_dm(n)
        rho = apply_gate_dm(rho, G.H(), [0], n)
        rho = apply_gate_dm(rho, G.CNOT(), [0, 1], n)
        assert abs(ev(trace(rho)) - 1.0) < 1e-5

    def test_hermitian_after_gate(self):
        n = 3
        rho = zero_state_dm(n)
        rho = apply_gate_dm(rho, G.H(), [0], n)
        rho = apply_gate_dm(rho, G.X(), [1], n)
        rho = apply_gate_dm(rho, G.CNOT(), [0, 2], n)
        assert is_hermitian(rho)

    def test_trace_preserved_after_depolarizing(self):
        n = 2
        rho = zero_state_dm(n)
        rho = apply_gate_dm(rho, G.H(), [0], n)
        rho = apply_kraus_channel(rho, depolarizing_kraus(0.1), [0], n)
        assert abs(ev(trace(rho)) - 1.0) < 1e-5

    def test_trace_preserved_after_amplitude_damping(self):
        n = 2
        rho = zero_state_dm(n)
        rho = apply_gate_dm(rho, G.X(), [0], n)   # prepare |1>
        rho = apply_kraus_channel(rho, amplitude_damping_kraus(0.3), [0], n)
        assert abs(ev(trace(rho)) - 1.0) < 1e-5

    def test_hermitian_after_noise(self):
        n = 2
        rho = zero_state_dm(n)
        rho = apply_gate_dm(rho, G.H(), [0], n)
        rho = apply_kraus_channel(rho, phase_damping_kraus(0.2), [0], n)
        assert is_hermitian(rho)


# ---------------------------------------------------------------------------
# Noise channel limiting cases
# ---------------------------------------------------------------------------

class TestNoiseLimits:
    def test_depolarizing_p0_is_identity(self):
        n = 1
        rho = zero_state_dm(n)
        rho = apply_gate_dm(rho, G.H(), [0], n)
        rho_before = to_np(rho)
        rho = apply_kraus_channel(rho, depolarizing_kraus(0.0), [0], n)
        np.testing.assert_allclose(to_np(rho), rho_before, atol=1e-5)

    def test_depolarizing_p1_gives_maximally_mixed(self):
        # At p=1: rho -> I/2 for any initial state
        n = 1
        rho = zero_state_dm(n)
        rho = apply_gate_dm(rho, G.H(), [0], n)   # start from |+>
        rho = apply_kraus_channel(rho, depolarizing_kraus(1.0), [0], n)
        expected = mx.array([[0.5, 0], [0, 0.5]], dtype=mx.complex64)
        np.testing.assert_allclose(to_np(rho), to_np(expected), atol=1e-5)

    def test_amplitude_damping_gamma0_is_identity(self):
        n = 1
        rho = zero_state_dm(n)
        rho = apply_gate_dm(rho, G.H(), [0], n)
        rho_before = to_np(rho)
        rho = apply_kraus_channel(rho, amplitude_damping_kraus(0.0), [0], n)
        np.testing.assert_allclose(to_np(rho), rho_before, atol=1e-5)

    def test_amplitude_damping_gamma1_collapses_to_zero(self):
        # gamma=1: |1> decays to |0> completely
        n = 1
        rho = zero_state_dm(n)
        rho = apply_gate_dm(rho, G.X(), [0], n)  # prepare |1>
        rho = apply_kraus_channel(rho, amplitude_damping_kraus(1.0), [0], n)
        expected = mx.array([[1, 0], [0, 0]], dtype=mx.complex64)
        np.testing.assert_allclose(to_np(rho), to_np(expected), atol=1e-5)

    def test_phase_damping_gamma1_destroys_coherence(self):
        # gamma=1: off-diagonal elements -> 0
        n = 1
        rho = zero_state_dm(n)
        rho = apply_gate_dm(rho, G.H(), [0], n)   # |+>: rho = [[0.5, 0.5],[0.5, 0.5]]
        rho = apply_kraus_channel(rho, phase_damping_kraus(1.0), [0], n)
        m = to_np(rho)
        # Off-diagonal should be zero
        assert abs(m[0, 1]) < 1e-5
        assert abs(m[1, 0]) < 1e-5
        # Diagonal preserved
        assert abs(m[0, 0] - 0.5) < 1e-5
        assert abs(m[1, 1] - 0.5) < 1e-5

    def test_bit_flip_p05_dephases_z_expectation(self):
        # At p=0.5, bit flip completely randomizes the Z basis
        n = 1
        rho = zero_state_dm(n)
        rho = apply_kraus_channel(rho, bit_flip_kraus(0.5), [0], n)
        # Should give I/2 (maximally mixed in Z basis)
        z_exp = ev(expectation_z_dm(rho, 0, n))
        assert abs(z_exp) < 1e-5

    def test_phase_flip_preserves_diagonal(self):
        # Phase flip doesn't change populations, only coherences
        n = 1
        rho = zero_state_dm(n)
        rho = apply_gate_dm(rho, G.H(), [0], n)
        diag_before = [ev(rho[0, 0].real), ev(rho[1, 1].real)]
        rho = apply_kraus_channel(rho, phase_flip_kraus(0.3), [0], n)
        diag_after  = [ev(rho[0, 0].real), ev(rho[1, 1].real)]
        assert abs(diag_before[0] - diag_after[0]) < 1e-5
        assert abs(diag_before[1] - diag_after[1]) < 1e-5


# ---------------------------------------------------------------------------
# NoisyCircuit
# ---------------------------------------------------------------------------

class TestNoisyCircuit:
    def test_noiseless_matches_statevector(self):
        nc = NoisyCircuit(n_qubits=3)
        nc.h(0)
        nc.cnot(0, 1)
        nc.x(2)

        c = Circuit(3)
        c.h(0)
        c.cnot(0, 1)
        c.x(2)

        params = mx.array([], dtype=mx.float32)
        f_sv = c.compile()
        f_dm = nc.compile()

        assert abs(ev(f_sv(params)) - ev(f_dm(params))) < 1e-4

    def test_noise_degrades_expectation(self):
        # Without noise: X gate on |0> gives <Z> = -1
        # With high depolarizing noise: <Z> moves toward 0
        nc_clean = NoisyCircuit(n_qubits=1)
        nc_clean.x(0)

        nc_noisy = NoisyCircuit(n_qubits=1)
        nc_noisy.x(0)
        nc_noisy.noise(depolarizing_kraus(0.5), qubits=[0])

        params = mx.array([], dtype=mx.float32)
        clean_val = ev(nc_clean.compile()(params))
        noisy_val = ev(nc_noisy.compile()(params))

        # Clean: <Z> = -1. Noisy: |<Z>| < 1
        assert abs(clean_val + 1.0) < 1e-4
        assert abs(noisy_val) < abs(clean_val)

    def test_parametrized_noisycircuit_runs(self):
        nc = NoisyCircuit(n_qubits=3)
        nc.h(0)
        nc.ry(0, param_idx=0)
        nc.noise(depolarizing_kraus(0.02), qubits=[0])
        nc.cnot(0, 1)
        nc.ry(1, param_idx=1)
        nc.noise(amplitude_damping_kraus(0.01), qubits=[1])
        nc.cnot(1, 2)

        assert nc.n_params == 2
        params = mx.array([math.pi / 4, math.pi / 3], dtype=mx.float32)
        f = nc.compile()
        result = ev(f(params))
        assert -3.1 <= result <= 3.1

    def test_trace_preserved_in_noisycircuit(self):
        nc = NoisyCircuit(n_qubits=2)
        nc.h(0)
        nc.cnot(0, 1)
        nc.noise(depolarizing_kraus(0.05), qubits=[0])
        nc.noise(phase_damping_kraus(0.03), qubits=[1])

        params = mx.array([], dtype=mx.float32)
        rho = nc.run(params)
        assert abs(ev(trace(rho)) - 1.0) < 1e-4

    def test_repr(self):
        nc = NoisyCircuit(n_qubits=4)
        nc.h(0)
        nc.noise(depolarizing_kraus(0.01), qubits=[0])
        r = repr(nc)
        assert "NoisyCircuit" in r
        assert "n_qubits=4" in r

    def test_small_noise_limit(self):
        # Tiny noise should give result close to noiseless
        n = 3
        nc_clean = NoisyCircuit(n_qubits=n)
        nc_noisy = NoisyCircuit(n_qubits=n)

        for q in range(n):
            nc_clean.ry(q, param_idx=q)
            nc_noisy.ry(q, param_idx=q)
            nc_noisy.noise(depolarizing_kraus(1e-6), qubits=[q])

        params = mx.array([math.pi/4, math.pi/3, math.pi/5], dtype=mx.float32)
        clean_val = ev(nc_clean.compile()(params))
        noisy_val = ev(nc_noisy.compile()(params))
        assert abs(clean_val - noisy_val) < 0.01
