"""Gate matrix correctness tests."""

import math
import numpy as np
import mlx.core as mx
import pytest

import sys; sys.path.insert(0, "src")
from zilver import gates as G


def to_np(m):
    mx.eval(m)
    return np.array(m.tolist(), dtype=np.complex64)


def is_unitary(m: np.ndarray, atol=1e-5) -> bool:
    n = m.shape[0]
    return np.allclose(m @ m.conj().T, np.eye(n, dtype=np.complex64), atol=atol)


@pytest.mark.parametrize("gate_fn", [G.I, G.X, G.Y, G.Z, G.H, G.S, G.T])
def test_fixed_gates_are_unitary(gate_fn):
    assert is_unitary(to_np(gate_fn()))


@pytest.mark.parametrize("gate_fn", [G.CNOT, G.CZ, G.SWAP, G.iSWAP])
def test_two_qubit_gates_are_unitary(gate_fn):
    assert is_unitary(to_np(gate_fn()))


@pytest.mark.parametrize("theta", [0.0, math.pi/4, math.pi/2, math.pi, 2*math.pi])
@pytest.mark.parametrize("gate_fn", [G.RX, G.RY, G.RZ])
def test_rotation_gates_are_unitary(gate_fn, theta):
    assert is_unitary(to_np(gate_fn(theta)))


def test_rx_zero_is_identity():
    np.testing.assert_allclose(to_np(G.RX(0.0)), to_np(G.I()), atol=1e-6)


def test_rx_pi_maps_zero_to_minus_i_one():
    # RX(pi)|0> = -i|1>
    state_0 = np.array([1.0, 0.0], dtype=np.complex64)
    result = to_np(G.RX(math.pi)) @ state_0
    assert abs(result[0]) < 1e-5           # amplitude on |0> is 0
    assert abs(abs(result[1]) - 1.0) < 1e-5  # amplitude on |1> has magnitude 1


def test_ry_pi_maps_zero_to_one():
    state_0 = np.array([1.0, 0.0], dtype=np.complex64)
    result = to_np(G.RY(math.pi)) @ state_0
    assert abs(abs(result[1]) - 1.0) < 1e-5


def test_h_twice_is_identity():
    h = to_np(G.H())
    np.testing.assert_allclose(h @ h, np.eye(2, dtype=np.complex64), atol=1e-6)


def test_rzz_is_unitary():
    for theta in [0.3, math.pi/2, math.pi]:
        assert is_unitary(to_np(G.RZZ(theta)))


@pytest.mark.parametrize("theta,phi,lam", [(0.3, 0.7, 1.1), (math.pi/2, 0, math.pi)])
def test_u_gate_is_unitary(theta, phi, lam):
    assert is_unitary(to_np(G.U(theta, phi, lam)))
