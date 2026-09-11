"""M2, device half. A seeded random brickwork circuit, written out for an independent simulator.

    FROM=24 TO=32 bash scripts/rocm-win-zilver.sh --random
    python3 bench/report_aer.py <report_out>          # the CPU half, in WSL, Qiskit Aer

WHY. M1 checks Zilver against mathematics; this checks it against the reference outsiders
already trust. The circuit is generated HERE and written to disk with every gate matrix, so the
Aer side rebuilds the same circuit from the file -- no reliance on two RNGs agreeing.

Circuit: depth 12. Each layer is a Haar-random 1q gate on every qubit, then an entangler on
alternating pairs -- CZ on even layers, CX on odd, with the CX direction alternating along the
row so both 2-qubit orientations of the kernel's gate-index permutation are exercised (CZ is
symmetric and cannot see an orientation error; CX can).

Writes m2_n{n}.npz: the circuit, K seeded amplitudes, the norm, and the timing. At n <= 28 it
also writes the full state as m2_n{n}_state.npy so the Aer side can take a full fidelity.
"""
from __future__ import annotations

import os
import time

import numpy as np

import _report as R

DEPTH = 12
FULL_MAX = 28


def brickwork(n, depth, seed):
    rng = np.random.default_rng(seed)
    kinds, qubits, mats = [], [], []
    for L in range(depth):
        for q in range(n):
            kinds.append("u"); qubits.append((q, -1)); mats.append(R.haar(rng))
        for q in range(L % 2, n - 1, 2):
            if L % 2 == 0:
                kinds.append("cz"); qubits.append((q, q + 1))
            else:
                qubits.append((q + 1, q) if (q // 2) % 2 else (q, q + 1)); kinds.append("cx")
            mats.append(np.zeros((2, 2), dtype=np.complex128))
    return np.array(kinds), np.array(qubits, dtype=np.int64), np.array(mats)


def main():
    B = R.backend()
    print(f"== M2 random brickwork, device half   backend={B.name}   {B.info}")
    for n in R.widths([24, 28, 30, 32]):
        seed = 7000 + n
        kinds, qubits, mats = brickwork(n, DEPTH, seed)
        s = B.basis(n, 0)
        B.sync()
        t = time.perf_counter()
        for k, q, m in zip(kinds, qubits, mats):
            if k == "u":
                B.apply(s, m, (int(q[0]),), n)
            else:
                B.apply(s, R.CZ if k == "cz" else R.CNOT, (int(q[0]), int(q[1])), n)
        B.sync()
        dt = time.perf_counter() - t
        idx = R.sample_indices(n, 2000 + n)
        amp = B.sample(s, idx)
        nrm = B.norm(s)
        path = os.path.join(R.out_dir(), f"m2_n{n}.npz")
        np.savez(path, n=n, depth=DEPTH, seed=seed, kinds=kinds, qubits=qubits, mats=mats,
                 idx=idx, amps=amp, norm=nrm, seconds=dt, gates=len(kinds),
                 device=str(B.info.get("device")))
        if n <= FULL_MAX:
            np.save(os.path.join(R.out_dir(), f"m2_n{n}_state.npy"), B.to_host(s))
        print(f"  n={n:<3} {len(kinds)} gates  {dt:8.2f} s  norm {nrm:.7f}  "
              f"k={idx.size}  -> {os.path.basename(path)}", flush=True)
        del s
        B.free()


if __name__ == "__main__":
    main()
