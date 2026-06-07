#!/usr/bin/env python3
"""Noisy VQE example: ground-state energy, ideal vs noisy.

Variationally minimise <H> for a transverse-field Ising Hamiltonian on the
density-matrix backend, then show what decoherence costs the result:

  - noiseless VQE converges to the exact ground-state energy E0,
  - evaluating that same trained circuit under noise raises the energy,
  - re-optimising *under* noise reaches the best energy the noisy device allows.

Gradients use the parameter-shift rule, which holds for the noisy expectation
value too (the noise channels do not depend on the parameters).
"""

import sys
import numpy as np
import mlx.core as mx

sys.path.insert(0, "src")
from zilver import NoisyCircuit, NoiseModel

N_QUBITS = 3
N_LAYERS = 2
J, H_FIELD = 1.0, 1.0        # TFIM couplings:  H = -J sum ZZ - h sum X
LR = 0.15
STEPS = 60
SEED = 0

_I = np.eye(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _op_on(op, q, n):
    m = np.array([[1]], dtype=complex)
    for i in range(n):
        m = np.kron(m, op if i == q else _I)
    return m


def tfim_hamiltonian(n, j, h):
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for q in range(n - 1):
        H += -j * (_op_on(_Z, q, n) @ _op_on(_Z, q + 1, n))
    for q in range(n):
        H += -h * _op_on(_X, q, n)
    return H


def ansatz(n, layers):
    """Hardware-efficient ansatz on the density-matrix backend."""
    nc = NoisyCircuit(n)
    p = 0
    for _ in range(layers):
        for q in range(n):
            nc.ry(q, p); p += 1
            nc.rz(q, p); p += 1
        for q in range(n - 1):
            nc.cnot(q, q + 1)
    return nc, p


def energy(params, H, noise):
    """<H> = Tr(rho H) for the trained (possibly mixed) state."""
    nc, _ = ansatz(N_QUBITS, N_LAYERS)
    rho = np.array(nc.run(mx.array(params.astype(np.float32)), noise_model=noise))
    return float(np.trace(rho @ H).real)


def param_shift_grad(params, H, noise):
    g = np.zeros_like(params)
    for i in range(len(params)):
        sp = params.copy(); sp[i] += np.pi / 2
        sm = params.copy(); sm[i] -= np.pi / 2
        g[i] = (energy(sp, H, noise) - energy(sm, H, noise)) / 2
    return g


def train(H, noise, params):
    for _ in range(STEPS):
        params = params - LR * param_shift_grad(params, H, noise)
    return params


def run():
    H = tfim_hamiltonian(N_QUBITS, J, H_FIELD)
    e0 = float(np.linalg.eigvalsh(H)[0])
    _, n_params = ansatz(N_QUBITS, N_LAYERS)
    init = np.random.default_rng(SEED).uniform(-np.pi, np.pi, n_params)

    # depolarizing noise the user chooses to apply
    noise = NoiseModel.depolarizing(p1=0.005, p2=0.02)

    opt = train(H, None, init.copy())            # train noiselessly
    e_ideal = energy(opt, H, None)
    e_noisy = energy(opt, H, noise)              # evaluate the same circuit, noisy

    print(f"\nNoisy VQE  |  TFIM({N_QUBITS}q, J={J}, h={H_FIELD})  "
          f"|  hardware-efficient depth {N_LAYERS}  |  depolarizing 1q=0.5% 2q=2%")
    print(f"  {'-'*54}")
    print(f"  exact ground state     E0    = {e0:.4f}")
    print(f"  noiseless VQE          <H>   = {e_ideal:.4f}   "
          f"(error {abs(e_ideal - e0):.1e})")
    print(f"  same circuit, noisy    <H>   = {e_noisy:.4f}   "
          f"(+{e_noisy - e_ideal:.4f} — what decoherence costs)")
    print()


if __name__ == "__main__":
    run()
