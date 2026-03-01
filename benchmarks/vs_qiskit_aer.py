#!/usr/bin/env python3
"""Zilver vs Qiskit Aer benchmark."""

import argparse
import time
import sys
import numpy as np


def bench_zilver(n_qubits: int, depth: int, resolution: int, circuit_family: str) -> dict:
    import zilver
    from zilver.circuit import hardware_efficient, real_amplitudes, qaoa_style, efficient_su2
    from zilver.landscape import LossLandscape

    factory = {
        "hardware_efficient": lambda: hardware_efficient(n_qubits, depth),
        "real_amplitudes":    lambda: real_amplitudes(n_qubits, depth),
        "qaoa_style":         lambda: qaoa_style(n_qubits, depth),
        "efficient_su2":      lambda: efficient_su2(n_qubits, depth),
    }[circuit_family]

    circuit = factory()
    landscape = LossLandscape(circuit, sweep_params=(0, 1), resolution=resolution)

    t0 = time.perf_counter()
    result = landscape.compute()
    elapsed = time.perf_counter() - t0

    return {
        "backend": "zilver-mlx",
        "n_qubits": n_qubits,
        "depth": depth,
        "n_params": circuit.n_params,
        "resolution": resolution,
        "circuit_family": circuit_family,
        "wall_time_s": elapsed,
        "plateau_coverage": result.plateau_coverage(),
        "trainability_score": result.trainability_score(),
    }


def bench_qiskit_aer(n_qubits: int, depth: int, resolution: int, circuit_family: str) -> dict:
    try:
        from qiskit.circuit.library import RealAmplitudes, EfficientSU2
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        from qiskit_aer.primitives import Estimator as AerEstimator
        from qiskit.quantum_info import SparsePauliOp
    except ImportError:
        print("  [skip] qiskit-aer not installed", file=sys.stderr)
        return {}

    # Build circuit
    if circuit_family == "real_amplitudes":
        circuit = RealAmplitudes(n_qubits, reps=depth, entanglement="linear")
    elif circuit_family in ("hardware_efficient", "efficient_su2"):
        circuit = EfficientSU2(n_qubits, reps=depth, entanglement="linear")
    else:
        # QAOA-like
        circuit = QuantumCircuit(n_qubits)
        for layer in range(depth):
            gamma = Parameter(f"gamma_{layer}")
            beta  = Parameter(f"beta_{layer}")
            for q in range(n_qubits - 1):
                circuit.cx(q, q + 1)
                circuit.rz(gamma, q + 1)
                circuit.cx(q, q + 1)
            for q in range(n_qubits):
                circuit.rx(beta, q)

    param_list = list(circuit.parameters)
    hamiltonian = SparsePauliOp.from_list([
        ("I" * i + "Z" + "I" * (n_qubits - i - 1), 1.0) for i in range(n_qubits)
    ])
    estimator = AerEstimator()

    t0 = time.perf_counter()

    loss_landscape = []
    grad_landscape = []
    axis = np.linspace(-np.pi, np.pi, resolution)
    fixed = np.random.uniform(-np.pi, np.pi, len(param_list))

    for i, p0 in enumerate(axis):
        loss_row, grad_row = [], []
        for j, p1 in enumerate(axis):
            pv = fixed.copy()
            if len(param_list) > 0: pv[0] = p0
            if len(param_list) > 1: pv[1] = p1
            pdict = dict(zip(param_list, pv))

            bound = circuit.assign_parameters(pdict)
            loss = float(estimator.run([bound], [hamiltonian]).result().values[0])

            grads = []
            shift = np.pi / 2
            for k in range(min(2, len(param_list))):
                pv_plus  = pv.copy(); pv_plus[k]  += shift
                pv_minus = pv.copy(); pv_minus[k] -= shift
                pd_plus  = dict(zip(param_list, pv_plus))
                pd_minus = dict(zip(param_list, pv_minus))
                bp = circuit.assign_parameters(pd_plus)
                bm = circuit.assign_parameters(pd_minus)
                gp = float(estimator.run([bp], [hamiltonian]).result().values[0])
                gm = float(estimator.run([bm], [hamiltonian]).result().values[0])
                grads.append(0.5 * (gp - gm))

            loss_row.append(loss)
            grad_row.append(float(np.linalg.norm(grads)))

        loss_landscape.append(loss_row)
        grad_landscape.append(grad_row)

    elapsed = time.perf_counter() - t0
    flat = [g for row in grad_landscape for g in row]
    plateau_cov = sum(1 for g in flat if g < 0.1) / len(flat)

    return {
        "backend": "qiskit-aer",
        "n_qubits": n_qubits,
        "depth": depth,
        "n_params": len(param_list),
        "resolution": resolution,
        "circuit_family": circuit_family,
        "wall_time_s": elapsed,
        "plateau_coverage": plateau_cov,
        "trainability_score": 1.0 - plateau_cov,
    }


def run_benchmark(n_qubits: int, depth: int, resolution: int):
    families = ["hardware_efficient", "real_amplitudes", "qaoa_style"]

    print(f"\n{'='*72}")
    print(f"  Zilver vs Qiskit Aer  |  {n_qubits}q  depth={depth}  grid={resolution}x{resolution}")
    print(f"{'='*72}")
    print(f"  {'Circuit family':<22}  {'Backend':<14}  {'Time (s)':>9}  {'Plateau %':>10}  {'Speedup':>8}")
    print(f"  {'-'*22}  {'-'*14}  {'-'*9}  {'-'*10}  {'-'*8}")

    for family in families:
        aer = bench_qiskit_aer(n_qubits, depth, resolution, family)
        sir = bench_zilver(n_qubits, depth, resolution, family)

        speedup = f"{aer['wall_time_s']/sir['wall_time_s']:.1f}x" if aer else "N/A"

        if aer:
            print(f"  {family:<22}  {'qiskit-aer':<14}  {aer['wall_time_s']:>9.2f}  {aer['plateau_coverage']*100:>9.1f}%  {'':>8}")
        print(f"  {family:<22}  {'zilver-mlx':<14}  {sir['wall_time_s']:>9.2f}  {sir['plateau_coverage']*100:>9.1f}%  {speedup:>8}")
        print()

    print(f"{'='*72}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits",     type=int, default=6)
    parser.add_argument("--depth",      type=int, default=3)
    parser.add_argument("--resolution", type=int, default=20)
    args = parser.parse_args()

    run_benchmark(args.qubits, args.depth, args.resolution)
