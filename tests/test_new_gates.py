"""
Tests for: Toffoli, Fredkin, variational_simulator, gate fusion.
"""

import math
import numpy as np
import mlx.core as mx
import pytest

import sys; sys.path.insert(0, "src")
from zilver import gates as G
from zilver.circuit import Circuit, variational_simulator
from zilver.gradients import param_shift_gradient


def to_np(m):
    mx.eval(m)
    return np.array(m.tolist(), dtype=np.complex64)

def is_unitary(m, atol=1e-5):
    return np.allclose(m @ m.conj().T, np.eye(m.shape[0], dtype=np.complex64), atol=atol)

def ev(x):
    mx.eval(x)
    return float(x.item())


# ---------------------------------------------------------------------------
# Toffoli (CCX)
# ---------------------------------------------------------------------------

class TestToffoli:
    def test_toffoli_is_unitary(self):
        assert is_unitary(to_np(G.Toffoli()))

    def test_toffoli_flips_target_when_both_controls_one(self):
        # |110> (index 6) -> |111> (index 7)
        from zilver.simulator import apply_gate
        state = mx.array([0,0,0,0,0,0,1,0], dtype=mx.complex64)
        result = apply_gate(state, G.Toffoli(), [0, 1, 2], 3)
        mx.eval(result)
        arr = np.array(result.tolist())
        assert abs(abs(arr[7]) - 1.0) < 1e-5, f"|110> should map to |111>, got {arr}"

    def test_toffoli_no_flip_single_control(self):
        # |100> (index 4): only one control active, target unchanged
        from zilver.simulator import apply_gate
        state = mx.array([0,0,0,0,1,0,0,0], dtype=mx.complex64)
        result = apply_gate(state, G.Toffoli(), [0, 1, 2], 3)
        mx.eval(result)
        arr = np.array(result.tolist())
        assert abs(abs(arr[4]) - 1.0) < 1e-5

    def test_toffoli_self_inverse(self):
        # Toffoli is its own inverse
        mat = to_np(G.Toffoli())
        np.testing.assert_allclose(mat @ mat, np.eye(8, dtype=np.complex64), atol=1e-5)

    def test_toffoli_in_circuit(self):
        c = Circuit(3)
        c.x(0); c.x(1)       # set controls to |1>
        c.toffoli(0, 1, 2)   # should flip qubit 2
        f = c.compile(observable="z0")
        result = ev(f(mx.array([], dtype=mx.float32)))
        # After Toffoli on |110>: state is |111>, so <sum_z> = -3
        assert abs(result + 1.0) < 1e-4   # z0 component: qubit 0 = |1> -> <Z> = -1


# ---------------------------------------------------------------------------
# Fredkin (CSWAP)
# ---------------------------------------------------------------------------

class TestFredkin:
    def test_fredkin_is_unitary(self):
        assert is_unitary(to_np(G.Fredkin()))

    def test_fredkin_swaps_when_control_one(self):
        # |101> (index 5): control=1, q1=0, q2=1 -> swap -> |110> (index 6)
        from zilver.simulator import apply_gate
        state = mx.array([0,0,0,0,0,1,0,0], dtype=mx.complex64)
        result = apply_gate(state, G.Fredkin(), [0, 1, 2], 3)
        mx.eval(result)
        arr = np.array(result.tolist())
        assert abs(abs(arr[6]) - 1.0) < 1e-5, f"|101> should swap to |110>, got {arr}"

    def test_fredkin_no_swap_when_control_zero(self):
        # |001> (index 1): control=0, no swap
        from zilver.simulator import apply_gate
        state = mx.array([0,1,0,0,0,0,0,0], dtype=mx.complex64)
        result = apply_gate(state, G.Fredkin(), [0, 1, 2], 3)
        mx.eval(result)
        arr = np.array(result.tolist())
        assert abs(abs(arr[1]) - 1.0) < 1e-5

    def test_fredkin_self_inverse(self):
        mat = to_np(G.Fredkin())
        np.testing.assert_allclose(mat @ mat, np.eye(8, dtype=np.complex64), atol=1e-5)

    def test_fredkin_in_circuit(self):
        c = Circuit(3)
        c.x(0); c.x(2)          # prepare |101>
        c.fredkin(0, 1, 2)       # swap q1,q2 -> |110>
        sv = c.statevector(mx.array([], dtype=mx.float32))
        arr = sv.numpy()
        assert abs(abs(arr[6]) - 1.0) < 1e-5  # |110> = index 6


# ---------------------------------------------------------------------------
# Variational Simulator
# ---------------------------------------------------------------------------

class TestVariationalSimulator:
    def test_vs_runs(self):
        c = variational_simulator(n_qubits=4, depth=2)
        assert c.n_params > 0
        params = mx.array(
            np.random.uniform(-math.pi, math.pi, c.n_params).astype(np.float32)
        )
        f = c.compile()
        result = ev(f(params))
        assert -4.1 <= result <= 4.1

    def test_vs_param_count(self):
        # depth * (n + (n-1)) per layer: n RX + (n-1) RZZ
        n, d = 6, 3
        c = variational_simulator(n, d)
        expected = d * (n + (n - 1))
        assert c.n_params == expected, f"Expected {expected} params, got {c.n_params}"

    def test_vs_vmappable(self):
        c = variational_simulator(n_qubits=4, depth=2)
        f = c.compile()
        rng = np.random.default_rng(7)
        batch = rng.uniform(-math.pi, math.pi, (40, c.n_params)).astype(np.float32)
        results = mx.vmap(f)(mx.array(batch))
        mx.eval(results)
        arr = np.array(results.tolist())
        assert arr.shape == (40,)
        assert np.all(np.isfinite(arr))

    def test_vs_gradient_nonzero(self):
        c = variational_simulator(n_qubits=4, depth=1)
        f = c.compile()
        rng = np.random.default_rng(42)
        params = mx.array(
            rng.uniform(-math.pi, math.pi, c.n_params).astype(np.float32)
        )
        grads = param_shift_gradient(f, params)
        mx.eval(grads)
        arr = np.array(grads.tolist())
        assert np.max(np.abs(arr)) > 1e-4


# ---------------------------------------------------------------------------
# Gate Fusion — Apple Silicon execution optimization
# ---------------------------------------------------------------------------

class TestGateFusion:
    def test_fusion_reduces_op_count(self):
        # 3 consecutive fixed single-qubit gates on qubit 0 should fuse to 1
        c = Circuit(2)
        c.h(0); c.x(0); c.h(0)   # 3 fixed single-qubit ops on qubit 0
        c.cnot(0, 1)              # flush boundary
        assert c.n_ops() == 4

        fused = c.fuse()
        assert fused.n_ops() == 2  # 1 fused + 1 CNOT

    def test_fusion_preserves_output(self):
        # Fused circuit must give identical statevector
        c = Circuit(3)
        c.h(0); c.x(1); c.h(1); c.h(0)
        c.cnot(0, 1)
        c.h(2); c.x(2)

        params = mx.array([], dtype=mx.float32)
        sv_orig  = c.statevector(params).numpy()
        sv_fused = c.fuse().statevector(params).numpy()
        np.testing.assert_allclose(np.abs(sv_orig), np.abs(sv_fused), atol=1e-5)

    def test_fusion_preserves_expectation(self):
        c = Circuit(4)
        for q in range(4):
            c.h(q); c.x(q); c.h(q)  # HXH = Z for each qubit
        c.cnot(0, 1); c.cnot(2, 3)

        params = mx.array([], dtype=mx.float32)
        f_orig  = c.compile()
        f_fused = c.fuse().compile()
        orig  = float(f_orig(params).item())
        fused = float(f_fused(params).item())
        assert abs(orig - fused) < 1e-5

    def test_fusion_parameterized_gates_are_boundaries(self):
        # Parameterized gates must NOT be fused (they vary at runtime)
        c = Circuit(2)
        c.h(0)
        c.ry(0, param_idx=0)   # parameterized — flush boundary
        c.h(0)
        fused = c.fuse()
        # h before ry fuses alone, ry is kept, h after ry fuses alone
        assert fused.n_params == 1
        params = mx.array([math.pi / 3], dtype=mx.float32)
        sv_orig  = c.statevector(params).numpy()
        sv_fused = fused.statevector(params).numpy()
        np.testing.assert_allclose(np.abs(sv_orig), np.abs(sv_fused), atol=1e-5)

    def test_fusion_vmappable(self):
        c = Circuit(4)
        for q in range(4):
            c.h(q)
        c.cnot(0, 1); c.cnot(2, 3)
        for q in range(4):
            c.ry(q, param_idx=q)
        fused = c.fuse()
        f = fused.compile()
        rng = np.random.default_rng(0)
        batch = rng.uniform(-math.pi, math.pi, (30, fused.n_params)).astype(np.float32)
        results = mx.vmap(f)(mx.array(batch))
        mx.eval(results)
        assert np.array(results.tolist()).shape == (30,)
