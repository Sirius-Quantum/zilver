#!/usr/bin/env python3
"""Network job submission example."""

import sys
import math

sys.path.insert(0, "src")
from zilver.client import NetworkCoordinator
from zilver.node import SimJob

REGISTRY = "http://localhost:7701"


def print_result(label, job, result):
    proof_ok = result.verify(job)
    print(f"  {label:<30}  expectation={result.expectation:>9.4f}  "
          f"elapsed={result.elapsed_ms:>7.1f}ms  proof={'ok' if proof_ok else 'FAIL'}")


def main():
    coord = NetworkCoordinator(REGISTRY)

    summary = coord.summary()
    print(f"\nNetwork summary")
    print(f"  Online nodes : {summary['online']}")
    print(f"  Backends     : {summary['backends']}")
    print(f"  Max SV qubits: {summary['max_sv_qubits']}")
    print()

    print("Submitting jobs")

    # 1. Hadamard on qubit 0 of a 4-qubit register (sv)
    job_h = SimJob(
        circuit_ops=[{"type": "h", "qubits": [0], "param_idx": None}],
        n_qubits=4, n_params=0, params=[], backend="sv",
    )
    print_result("H gate (sv, 4q)", job_h, coord.submit(job_h))

    # 2. Same circuit on density matrix backend
    job_dm = SimJob(
        circuit_ops=[{"type": "h", "qubits": [0], "param_idx": None}],
        n_qubits=4, n_params=0, params=[], backend="dm",
    )
    print_result("H gate (dm, 4q)", job_dm, coord.submit(job_dm))

    # 3. Parametrized RY layers (sv, 6 qubits)
    ops = [{"type": "ry", "qubits": [q], "param_idx": q} for q in range(6)]
    job_ry = SimJob(
        circuit_ops=ops,
        n_qubits=6, n_params=6,
        params=[0.3, 0.7, 1.1, 1.5, 0.9, 0.4],
        backend="sv",
    )
    result_ry = coord.submit(job_ry)
    print_result("RY layers (sv, 6q)", job_ry, result_ry)
    assert math.isfinite(result_ry.expectation), "expectation must be finite"

    # 4. Bell state — CNOT after H
    job_bell = SimJob(
        circuit_ops=[
            {"type": "h",    "qubits": [0],    "param_idx": None},
            {"type": "cnot", "qubits": [0, 1], "param_idx": None},
        ],
        n_qubits=2, n_params=0, params=[], backend="sv",
    )
    print_result("Bell state (sv, 2q)", job_bell, coord.submit(job_bell))

    print("\nAll jobs completed.")


if __name__ == "__main__":
    main()
