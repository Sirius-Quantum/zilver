#!/usr/bin/env python3
"""Barren plateau detection example."""

import math
import numpy as np
import sys

sys.path.insert(0, "src")
from zilver.circuit import hardware_efficient
from zilver.landscape import LossLandscape
from zilver.gradients import gradient_variance


N_QUBITS = 6
DEPTHS   = [1, 2, 4, 8]
SEED     = 42


def run_depth_scan():
    print(f"\nBarren Plateau Depth Scan  |  {N_QUBITS} qubits  |  hardware_efficient")
    print(f"{'Depth':<8}  {'n_params':<10}  {'Mean Var':<14}  {'Mean |grad|':<14}  {'Landscape (s)':>14}")
    print(f"{'-'*8}  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*14}")

    for depth in DEPTHS:
        circuit = hardware_efficient(N_QUBITS, depth)
        f = circuit.compile()

        # Gradient variance across random parameter samples
        stats = gradient_variance(f, circuit.n_params, n_samples=100, seed=SEED)

        # Full 20x20 landscape
        landscape = LossLandscape(circuit, sweep_params=(0, 1), resolution=20, seed=SEED)
        result = landscape.compute()

        mean_var = float(np.mean(stats["variance_per_param"]))

        print(
            f"{depth:<8}  {circuit.n_params:<10}  {mean_var:<14.2e}  "
            f"{stats['mean_gradient_magnitude']:<14.4f}  {result.wall_time_seconds:>14.3f}s"
        )

    print()


def run_landscape_example():
    print("Generating 20x20 loss landscape for hardware_efficient(6q, depth=4)...")
    circuit = hardware_efficient(N_QUBITS, depth=4)
    landscape = LossLandscape(circuit, sweep_params=(0, 1), resolution=20, seed=SEED)
    result = landscape.compute()

    print(f"  Wall time:           {result.wall_time_seconds:.3f}s")
    print(f"  Plateau coverage:    {result.plateau_coverage():.1%}")
    print(f"  Trainability score:  {result.trainability_score():.3f}")

    loss_arr = np.array(result.loss_landscape)
    print(f"  Loss range:          [{loss_arr.min():.3f}, {loss_arr.max():.3f}]")
    print()


if __name__ == "__main__":
    run_landscape_example()
    run_depth_scan()
