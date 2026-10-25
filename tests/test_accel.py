"""
Every CPU path in zilver.accel against the MLX reference, including gates on
3+ qubits in arbitrary (non-adjacent, reversed) qubit order.
"""

import numpy as np
import pytest

import sys; sys.path.insert(0, "src")
from zilver import accel
from zilver.circuit import Circuit


def _random_circuit(n: int, seed: int) -> tuple[Circuit, np.ndarray]:
    rng = np.random.default_rng(seed)
    c = Circuit(n)
    n_params = 0
    for _ in range(4 * n):
        qs = [int(q) for q in rng.permutation(n)[:3]]
        kind = rng.integers(7)
        if kind == 0:
            c.ry(qs[0], n_params); n_params += 1
        elif kind == 1:
            c.rx(qs[0], n_params); n_params += 1
        elif kind == 2:
            c.h(qs[0])
        elif kind == 3:
            c.cnot(qs[0], qs[1])
        elif kind == 4:
            c.rzz(qs[0], qs[1], n_params); n_params += 1
        elif kind == 5:
            c.toffoli(*qs)
        else:
            c.fredkin(*qs)
    return c, rng.uniform(-np.pi, np.pi, n_params)


def _reference(c: Circuit, params: np.ndarray) -> np.ndarray:
    return c.statevector(params, method="mlx").numpy()


PATHS = {
    "fused":    lambda c, p: accel.run_circuit(c, p, fuse_max=2),
    "unfused":  lambda c, p: accel.run_circuit(c, p, fuse_max=1),
    "strided":  lambda c, p: accel.run_circuit_strided(c, p),
    "strided128": lambda c, p: accel.run_circuit_strided(c, p, dtype=np.complex128),
    "auto":     lambda c, p: accel.run_circuit_auto(c, p),
}


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("seed", range(4))
def test_matches_mlx_with_3q_gates(path, seed):
    c, params = _random_circuit(5, seed)
    got = PATHS[path](c, params)
    np.testing.assert_allclose(got, _reference(c, params), atol=1e-5)


def test_complex128_keeps_its_dtype_with_3q_gates():
    c, params = _random_circuit(4, 0)
    assert accel.run_circuit_auto(c, params, dtype=np.complex128).dtype == np.complex128


def test_statevector_accel_toffoli():
    c = Circuit(3); c.x(0); c.x(1); c.toffoli(0, 1, 2)
    probs = np.asarray(c.statevector([], method="accel").probabilities())
    np.testing.assert_allclose(probs, np.eye(8)[7], atol=1e-6)


@pytest.mark.parametrize("qubits", [(0, 1, 2), (2, 0, 1), (3, 1, 0), (1, 3, 2)])
def test_kq_kernel_matches_dense_embedding(qubits):
    """The kernel against an explicitly built full-register unitary."""
    n = 4
    rng = np.random.default_rng(1)
    M, _ = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
    state = rng.normal(size=16) + 1j * rng.normal(size=16)

    full = np.zeros((16, 16), dtype=complex)
    for i in range(16):
        bits = [(i >> (n - 1 - q)) & 1 for q in range(n)]
        col = sum(bits[q] << (2 - j) for j, q in enumerate(qubits))
        for row in range(8):
            out = list(bits)
            for j, q in enumerate(qubits):
                out[q] = (row >> (2 - j)) & 1
            full[sum(b << (n - 1 - q) for q, b in enumerate(out)), i] += M[row, col]

    got = accel._apply_kq_numpy_strided(state.copy(), M, qubits, n)
    np.testing.assert_allclose(got, full @ state, atol=1e-12)
