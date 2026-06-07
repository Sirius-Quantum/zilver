#!/usr/bin/env python3
"""Noisy simulation example: thermal relaxation and depolarizing noise."""

import sys
import numpy as np
import mlx.core as mx

sys.path.insert(0, "src")
from zilver import NoisyCircuit, NoiseModel
from zilver.density_matrix import expectation_z_dm, trace

N_QUBITS = 3


def ghz() -> NoisyCircuit:
    """A 3-qubit GHZ circuit: H on qubit 0, then a CNOT chain."""
    nc = NoisyCircuit(N_QUBITS)
    nc.h(0)
    nc.cnot(0, 1)
    nc.cnot(1, 2)
    return nc


def run():
    params = mx.array([])  # no trainable parameters in this circuit

    # 1) Ideal vs a device-style thermal-relaxation noise model -------------
    # Coherence times and gate durations share a unit here (nanoseconds).
    noise = NoiseModel.thermal_relaxation(
        t1=120_000, t2=80_000, gate_time_1q=35, gate_time_2q=300,
    )

    rho_ideal = ghz().run(params)
    rho_noisy = ghz().run(params, noise_model=noise)

    print(f"\nGHZ-{N_QUBITS}  |  ideal vs thermal_relaxation(T1=120us, T2=80us)")
    print(f"  {'qubit':>6}  {'<Z> ideal':>10}  {'<Z> noisy':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}")
    for q in range(N_QUBITS):
        zi = float(expectation_z_dm(rho_ideal, q, N_QUBITS))
        zn = float(expectation_z_dm(rho_noisy, q, N_QUBITS))
        print(f"  {q:>6}  {zi:>10.4f}  {zn:>10.4f}")
    print(f"  trace(rho_noisy) = {float(trace(rho_noisy)):.6f}  (a valid state stays 1.0)")

    # 2) Depolarizing strength sweep ----------------------------------------
    # As the per-gate error rises, the GHZ coherence (off-diagonal corner of
    # the density matrix) collapses toward the maximally mixed state.
    print(f"\nDepolarizing sweep  |  GHZ-{N_QUBITS} corner coherence |rho[0, -1]|")
    print(f"  {'p (per gate)':>12}  {'|coherence|':>12}")
    print(f"  {'-'*12}  {'-'*12}")
    for p in [0.0, 0.005, 0.02, 0.05, 0.1]:
        dep = NoiseModel.depolarizing(p1=p, p2=p)
        rho = np.array(ghz().run(params, noise_model=dep))
        coherence = abs(rho[0, -1])
        print(f"  {p:>12.3f}  {coherence:>12.4f}")
    print("  (0.5 = perfect GHZ coherence; -> 0 as noise increases)\n")


if __name__ == "__main__":
    run()
