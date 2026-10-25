"""Metal kernels against an independent float64 NumPy reference."""

import numpy as np
import pytest

import sys; sys.path.insert(0, "src")
from zilver import metal
from zilver.circuit import Circuit, hardware_efficient

pytestmark = pytest.mark.skipif(not metal.AVAILABLE, reason="needs Metal")


def _rx(t): c, s = np.cos(t / 2), np.sin(t / 2); return np.array([[c, -1j * s], [-1j * s, c]])
def _ry(t): c, s = np.cos(t / 2), np.sin(t / 2); return np.array([[c, -s], [s, c]])
def _rz(t): return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
def _u3(t, p, l):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -np.exp(1j * l) * s], [np.exp(1j * p) * s, np.exp(1j * (p + l)) * c]])
_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_X = np.array([[0, 1], [1, 0]])
_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
_CZ = np.diag([1, 1, 1, -1])
def _rzz(t): return np.diag(np.exp(-0.5j * t * np.array([1, -1, -1, 1])))


def _apply(state, gate, qubits, n):
    """Qubit 0 is the most significant bit, as in zilver."""
    k = len(qubits)
    psi = state.reshape([2] * n)
    psi = np.tensordot(gate.reshape([2] * 2 * k), psi, axes=(list(range(k, 2 * k)), qubits))
    return np.moveaxis(psi, list(range(k)), qubits).reshape(-1)


def _random_circuit(n, n_layers, rng):
    """Every gate kind with a Metal kernel, on random qubits and angles."""
    c, ref_ops, p = Circuit(n), [], 0
    for _ in range(n_layers):
        for q in range(n):
            kind = rng.choice(["rx", "ry", "rz", "h", "x", "u3"])
            if kind == "u3":
                c.u3(q, p, p + 1, p + 2); ref_ops.append((kind, [q], [p, p + 1, p + 2])); p += 3
            elif kind in ("h", "x"):
                getattr(c, kind)(q); ref_ops.append((kind, [q], []))
            else:
                getattr(c, kind)(q, p); ref_ops.append((kind, [q], [p])); p += 1
        a, b = rng.choice(n, 2, replace=False).tolist()
        kind = rng.choice(["cnot", "cz", "rzz"])
        if kind == "rzz":
            c.rzz(a, b, p); ref_ops.append((kind, [a, b], [p])); p += 1
        else:
            getattr(c, kind)(a, b); ref_ops.append((kind, [a, b], []))
    return c, ref_ops, p


def _reference(ref_ops, params, n):
    mats = {"rx": _rx, "ry": _ry, "rz": _rz, "u3": _u3, "rzz": _rzz,
            "h": lambda: _H, "x": lambda: _X, "cnot": lambda: _CNOT, "cz": lambda: _CZ}
    psi = np.zeros(2 ** n, complex); psi[0] = 1
    for kind, qubits, idx in ref_ops:
        psi = _apply(psi, mats[kind](*[float(params[i]) for i in idx]), qubits, n)
    return psi


@pytest.mark.parametrize("seed", range(5))
def test_every_kernel_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = 6
    c, ref_ops, n_params = _random_circuit(n, 8, rng)
    params = rng.uniform(-np.pi, np.pi, n_params).astype(np.float32)
    assert metal.supports(c)
    got = np.asarray(c.statevector(params, method="metal").numpy())
    ref = _reference(ref_ops, params, n)
    assert np.max(np.abs(got - ref)) < 1e-6


def test_norm_does_not_drift_with_depth():
    """Fast-math sin/cos shrank the norm by ~1.5e-6 per layer, always downward."""
    c = hardware_efficient(10, 30)
    params = np.random.default_rng(0).uniform(-np.pi, np.pi, c.n_params)
    got = np.asarray(c.statevector(params, method="metal").numpy()).astype(np.complex128)
    assert abs(np.linalg.norm(got) - 1) < 1e-5
