"""M2, reference half. Rebuild each m2_n{n}.npz circuit in Qiskit Aer and compare.

    python3 bench/report_aer.py <dir holding m2_n*.npz>

Runs on the CPU (WSL on the box), single precision like the device. Needs only numpy, qiskit
and qiskit-aer -- no torch, no zilver -- so it shares no code with the thing it checks.

Index convention: zilver qubit q is bit (n-1-q) of the flat index; Qiskit qubit j is bit j.
So zilver q maps to Qiskit qubit n-1-q and the flat index is the same integer on both sides.

  PASS  sampled amplitudes agree to <= 1e-4 relative to the typical modulus 2^(-n/2),
        and, where the full state was written (n <= 28), fidelity >= 1 - 1e-5
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

AMP_TOL = 1e-4
FID_TOL = 1e-5


def build(n, kinds, qubits, mats, full, idx):
    qc = QuantumCircuit(n)
    for k, q, m in zip(kinds, qubits, mats):
        if k == "u":
            qc.unitary(m, [n - 1 - int(q[0])])
        elif k == "cz":
            qc.cz(n - 1 - int(q[0]), n - 1 - int(q[1]))
        elif k == "cx":
            qc.cx(n - 1 - int(q[0]), n - 1 - int(q[1]))
        else:
            raise ValueError(k)
    if full:
        qc.save_statevector()
    else:
        qc.save_amplitudes([int(i) for i in idx])
    return qc


def main(d):
    files = sorted(glob.glob(os.path.join(d, "m2_n*.npz")),
                   key=lambda p: int(re.search(r"m2_n(\d+)\.npz", p).group(1)))
    if not files:
        print(f"  VERDICT M2 Aer comparison              FAIL   no m2_n*.npz in {d}")
        return 1
    sim = AerSimulator(method="statevector", precision="single")
    rows, ok = [], True
    for p in files:
        z = np.load(p)
        n = int(z["n"])
        full_path = p.replace(".npz", "_state.npy")
        full = os.path.exists(full_path)
        qc = build(n, z["kinds"], z["qubits"], z["mats"], full, z["idx"])
        t = time.perf_counter()
        try:
            data = sim.run(qc, shots=1).result().data(0)
        except Exception as e:                       # one width failing must not hide the rest
            ok = False
            print(f"  VERDICT M2 n={n} vs Aer{'':<22} FAIL   Aer raised {type(e).__name__}: "
                  f"{str(e)[:120]}", flush=True)
            rows.append({"n": n, "error": f"{type(e).__name__}: {e}"})
            continue
        dt = time.perf_counter() - t
        idx = z["idx"].astype(np.int64)
        dev_amp = z["amps"].astype(np.complex128)
        row = {"n": n, "gates": int(z["gates"]), "aer_seconds": round(dt, 2),
               "device_seconds": float(z["seconds"]), "k": int(idx.size)}
        if full:
            ref = np.asarray(data["statevector"], dtype=np.complex128)
            dev = np.load(full_path).astype(np.complex128)
            row["fidelity"] = float(abs(np.vdot(ref, dev)) ** 2
                                    / (np.vdot(ref, ref).real * np.vdot(dev, dev).real))
            ref_amp = ref[idx]
        else:
            ref_amp = np.asarray(data["amplitudes"], dtype=np.complex128)
        row["max_rel_amp_err"] = float(np.max(np.abs(dev_amp - ref_amp)) * 2 ** (n / 2))
        row["sampled_fidelity"] = float(abs(np.vdot(ref_amp, dev_amp)) ** 2
                                        / (np.vdot(ref_amp, ref_amp).real
                                           * np.vdot(dev_amp, dev_amp).real))
        good = row["max_rel_amp_err"] <= AMP_TOL and row.get("fidelity", 1.0) >= 1 - FID_TOL
        ok &= good
        fid = f"  fidelity={row['fidelity']:.9f}" if "fidelity" in row else ""
        print(f"  VERDICT M2 n={n} vs Aer{'':<22} {'PASS' if good else 'FAIL'}   "
              f"amp err={row['max_rel_amp_err']:.2e} (<= {AMP_TOL:g}){fid}  "
              f"aer {dt:.1f}s / device {row['device_seconds']:.1f}s", flush=True)
        rows.append(row)
    import qiskit, qiskit_aer
    out = {"qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
           "amp_tol": AMP_TOL, "fid_tol": FID_TOL, "rows": rows, "all_pass": bool(ok)}
    with open(os.path.join(d, "m2_aer.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n  M2 overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
