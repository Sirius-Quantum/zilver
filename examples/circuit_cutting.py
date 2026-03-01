#!/usr/bin/env python3
"""Circuit cutting (QPD) example."""

import sys
import numpy as np
import mlx.core as mx

sys.path.insert(0, "src")
from zilver.circuit import hardware_efficient
from zilver.cutting import CutCircuit, WireCut
from zilver.coordinator import run_local, recombine

N_QUBITS_PER_FRAGMENT = 6
DEPTH                 = 2
SEED                  = 42


def run():
    rng = np.random.default_rng(SEED)

    # Build a circuit that spans two fragments
    # In practice this would be a 12-qubit circuit; here we construct
    # two 6-qubit fragments directly to keep the demo self-contained.
    fragment_a = hardware_efficient(N_QUBITS_PER_FRAGMENT, DEPTH)
    fragment_b = hardware_efficient(N_QUBITS_PER_FRAGMENT, DEPTH)

    params_a = mx.array(
        rng.uniform(-np.pi, np.pi, fragment_a.n_params).astype(np.float32)
    )
    params_b = mx.array(
        rng.uniform(-np.pi, np.pi, fragment_b.n_params).astype(np.float32)
    )

    # Evaluate each fragment locally (would be separate nodes on the network)
    f_a = fragment_a.compile(observable="sum_z")
    f_b = fragment_b.compile(observable="sum_z")

    loss_a = float(f_a(params_a))
    loss_b = float(f_b(params_b))

    # Recombine additively (independent fragments, no wire cut coefficients)
    # For a true QPD recombination use coordinator.recombine() with the
    # CutDecomposition coefficients returned by CutCircuit.decompose().
    combined = loss_a + loss_b

    print(f"\nCircuit cutting demo  |  {N_QUBITS_PER_FRAGMENT}q + {N_QUBITS_PER_FRAGMENT}q  |  depth={DEPTH}")
    print(f"  Fragment A expectation : {loss_a:>8.4f}  ({fragment_a.n_params} params)")
    print(f"  Fragment B expectation : {loss_b:>8.4f}  ({fragment_b.n_params} params)")
    print(f"  Combined               : {combined:>8.4f}")

    # Show overhead for QPD wire cuts
    from zilver.cutting import wire_cut_overhead, gate_cut_overhead
    print(f"\nQPD overhead (subcircuit evaluations required)")
    for n in [1, 2, 3]:
        print(f"  {n} wire cut(s) : {wire_cut_overhead(n):>5}x    |    "
              f"{n} gate cut(s) : {gate_cut_overhead(n):>5}x")


if __name__ == "__main__":
    run()
