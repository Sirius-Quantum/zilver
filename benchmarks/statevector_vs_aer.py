#!/usr/bin/env python3
"""The README performance table: one statevector, Zilver against Qiskit Aer.

    python benchmarks/statevector_vs_aer.py            # 12 16 20 22 24 26 qubits
    python benchmarks/statevector_vs_aer.py 20 24      # chosen widths

Circuit: hardware_efficient(n, depth=2) -- RY and RZ on every qubit, a CNOT chain,
three times over -- with fixed random angles. Time is wall clock until the final
state is a NumPy array on the host, best of REPEATS after one warm-up run.
Both sides run in single precision (complex64). Aer's transpile is done once,
outside the timing. Each row also reports the fidelity between the two states.

Needs `pip install "zilver[qiskit]"`.
"""
import platform
import subprocess
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import zilver
from zilver.circuit import hardware_efficient

REPEATS = 10


def best_ms(fn):
    fn()
    times = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times) * 1e3


def to_qiskit(circuit, params):
    # zilver qubit q is bit n-1-q of the flat index; Qiskit qubit j is bit j.
    n = circuit.n_qubits
    qc = QuantumCircuit(n)
    for op in circuit._ops:
        q = [n - 1 - i for i in op.qubits]
        if op.kind == "ry":
            qc.ry(float(params[op.param_indices[0]]), q[0])
        elif op.kind == "rz":
            qc.rz(float(params[op.param_indices[0]]), q[0])
        elif op.kind == "cnot":
            qc.cx(q[0], q[1])
        else:
            raise ValueError(op.kind)
    qc.save_statevector()
    return qc


def chip():
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor() or platform.machine()


def main(widths):
    sim = AerSimulator(method="statevector", precision="single")
    import qiskit_aer
    print(f"{chip()}, zilver {zilver.__version__}, qiskit-aer {qiskit_aer.__version__}")
    print(f"hardware_efficient(n, depth=2), best of {REPEATS}, ms\n")
    print(f"{'qubits':>6} {'zilver':>10} {'aer':>10} {'speedup':>8} {'fidelity':>10}")
    for n in widths:
        circuit = hardware_efficient(n, depth=2)
        params = np.random.default_rng(0).uniform(-np.pi, np.pi, circuit.n_params).astype(np.float32)
        tq = transpile(to_qiskit(circuit, params), sim)

        z = best_ms(lambda: np.asarray(circuit.statevector(params).numpy()))
        a = best_ms(lambda: np.asarray(sim.run(tq).result().get_statevector()))

        got = np.asarray(circuit.statevector(params).numpy()).astype(np.complex128)
        ref = np.asarray(sim.run(tq).result().get_statevector()).astype(np.complex128)
        fid = abs(np.vdot(ref, got)) ** 2 / (np.vdot(ref, ref).real * np.vdot(got, got).real)
        print(f"{n:>6} {z:>10.2f} {a:>10.2f} {a / z:>7.1f}x {fid:>10.7f}", flush=True)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [12, 16, 20, 22, 24, 26])
