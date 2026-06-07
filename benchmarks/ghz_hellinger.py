#!/usr/bin/env python3
"""Sim-to-real bench card: one GHZ circuit, four numbers.

A buyer evaluating Zilver-vs-hardware asks four things; each maps to a standard,
recognized measurement, all from ONE GHZ-N circuit and ONE hardware job:

  gate fidelity  how exactly the sim builds the circuit (machine precision)
                 vs the hardware's physical 2q gate fidelity (RB)
  accuracy       Hellinger distance of the measured output: noisy Zilver vs the
                 real QPU, compared against Qiskit Aer vs the QPU
  speed          wall-time to result: Zilver / Aer on a Mac vs the QPU
  cost           $0 on a Mac vs N credits on the QPU

`--dry-run` fills the simulator side with zero credits (QPU columns pending);
the real run adds the QPU as a single job. RB gate fidelity is free metadata.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zilver.density_matrix import (  # noqa: E402
    NoisyCircuit, NoiseModel, depolarizing_kraus,
)
from benchmarks._providers import (  # noqa: E402
    Calibration, GateErrors, make_provider, hellinger, counts_to_probs,
)


# ---------------------------------------------------------------------------
# GHZ circuits
# ---------------------------------------------------------------------------

def qiskit_ghz(n: int):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    qc.measure(range(n), range(n))
    return qc


def zilver_ghz(n: int) -> NoisyCircuit:
    nc = NoisyCircuit(n)
    nc.h(0)
    for q in range(n - 1):
        nc.cnot(q, q + 1)
    return nc


def build_noise_model(cal: Calibration, ge: GateErrors) -> NoiseModel:
    tr = NoiseModel.thermal_relaxation(cal.t1_ns, cal.t2_ns, cal.tg1_ns, cal.tg2_ns)
    p1, p2 = ge.depol_1q, ge.depol_2q
    return NoiseModel(
        one_qubit=tr.one_qubit + [lambda: depolarizing_kraus(p1)],
        two_qubit=tr.two_qubit + [lambda: depolarizing_kraus(p2)],
    )


# ---------------------------------------------------------------------------
# Distributions + gate fidelity
# ---------------------------------------------------------------------------

def _diag_probs(rho: np.ndarray, n: int) -> dict[str, float]:
    probs = {}
    for i in range(2 ** n):
        p = float(rho[i, i].real)
        if p > 1e-9:
            probs[format(i, f"0{n}b")[::-1]] = p   # -> little-endian (Qiskit)
    return probs


def zilver_distribution(n: int, noise) -> tuple[dict[str, float], float]:
    t0 = time.perf_counter()
    rho = np.array(zilver_ghz(n).run(mx.array([]), noise_model=noise))
    dt = time.perf_counter() - t0
    return _diag_probs(rho, n), dt


def ideal_distribution(n: int) -> dict[str, float]:
    return {"0" * n: 0.5, "1" * n: 0.5}


def zilver_state_infidelity(n: int) -> float:
    """1 - <GHZ| rho_noiseless |GHZ>, machine-precision check that Zilver builds
    the exact circuit. Driver-of-accuracy 'gate fidelity' on the sim side."""
    rho = np.array(zilver_ghz(n).run(mx.array([]), noise_model=None))
    L = 2 ** n - 1
    fid = 0.5 * float((rho[0, 0] + rho[0, L] + rho[L, 0] + rho[L, L]).real)
    return abs(1.0 - fid)


def aer_distribution(n, cal, ge, shots) -> tuple[dict[str, float], float]:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel as AerNoise, thermal_relaxation_error, depolarizing_error,
    )
    from qiskit import transpile
    t1, t2, tg1, tg2 = cal.t1_ns*1e-9, cal.t2_ns*1e-9, cal.tg1_ns*1e-9, cal.tg2_ns*1e-9
    nm = AerNoise()
    e1 = thermal_relaxation_error(t1, t2, tg1).compose(depolarizing_error(ge.depol_1q, 1))
    nm.add_all_qubit_quantum_error(e1, ["h", "x", "sx", "rz", "u", "u3", "r"])
    e2 = thermal_relaxation_error(t1, t2, tg2).tensor(thermal_relaxation_error(t1, t2, tg2))
    e2 = e2.compose(depolarizing_error(ge.depol_2q, 2))
    nm.add_all_qubit_quantum_error(e2, ["cx", "cz", "ecr"])
    sim = AerSimulator(noise_model=nm)
    qc = transpile(qiskit_ghz(n), sim)
    _ = sim.run(qc, shots=64).result()                 # warmup
    t0 = time.perf_counter()
    counts = sim.run(qc, shots=shots).result().get_counts()
    return counts_to_probs(counts), time.perf_counter() - t0


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------

def card(n, cal, ge, provider, shots, with_aer):
    noise = build_noise_model(cal, ge)
    _ = zilver_distribution(n, noise)                  # warmup (MLX graph compile)
    zil, t_zil = zilver_distribution(n, noise)
    ide = ideal_distribution(n)
    infid = zilver_state_infidelity(n)

    aer = t_aer = None
    if with_aer:
        aer, t_aer = aer_distribution(n, cal, ge, shots)

    qpu = t_qpu = None
    if provider is not None:
        t0 = time.perf_counter()
        counts = provider.run([qiskit_ghz(n)], shots, layout=list(range(n)))
        t_qpu = time.perf_counter() - t0
        qpu = counts_to_probs(counts[0])

    hw_2q_fid = 1.0 - ge.depol_2q
    print()
    print(f"  GHZ-{n}   noise [{cal.source}]")
    print(f"           T1={cal.t1_ns/1e3:.1f}us T2={cal.t2_ns/1e3:.1f}us  "
          f"depol_1q={ge.depol_1q:.1e} depol_2q={ge.depol_2q:.1e}")
    print()
    print(f"    {'metric':<16}{'Zilver':>16}{'Qiskit Aer':>16}{'QPU':>16}")
    print(f"    {'-'*15:<16}{'-'*15:>16}{'-'*15:>16}{'-'*15:>16}")

    # gate fidelity
    print(f"    {'gate fidelity':<16}{f'1-F={infid:.1e}':>16}"
          f"{'(exact)':>16}{f'2q RB {hw_2q_fid*100:.2f}%':>16}")

    # accuracy (Hellinger vs QPU; offline -> vs ideal)
    if qpu is not None:
        a_zil = f"H={hellinger(qpu, zil):.4f}"
        a_aer = f"H={hellinger(qpu, aer):.4f}" if aer is not None else "--"
        a_qpu = "(reference)"
    else:
        a_zil = f"H_ideal={hellinger(zil, ide):.3f}"
        a_aer = f"H_ideal={hellinger(aer, ide):.3f}" if aer is not None else "--"
        a_qpu = "pending"
    print(f"    {'accuracy':<16}{a_zil:>16}{a_aer:>16}{a_qpu:>16}")

    # speed
    s_zil = f"{t_zil*1e3:.1f} ms"
    s_aer = f"{t_aer*1e3:.1f} ms" if t_aer is not None else "--"
    s_qpu = f"{t_qpu:.1f} s" if t_qpu is not None else "pending"
    print(f"    {'speed':<16}{s_zil:>16}{s_aer:>16}{s_qpu:>16}")

    # cost — provider-aware (IQM bills credits/job; IBM Open plan is free time-quota)
    if provider is None:
        c_qpu = "pending"
    elif provider.name == "iqm":
        c_qpu = "~3 cr (1 job)"
    else:
        c_qpu = "free (Open)"
    print(f"    {'cost':<16}{'$0':>16}{'$0':>16}{c_qpu:>16}")
    print()
    if qpu is not None and aer is not None:
        hz, ha = hellinger(qpu, zil), hellinger(qpu, aer)
        verdict = "as faithful as Aer" if hz <= ha + 0.02 else "less faithful than Aer"
        print(f"    => noisy Zilver vs real QPU: H={hz:.4f}  (Aer H={ha:.4f}) — {verdict}")
        print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=["ibm", "iqm"])
    p.add_argument("--backend", default=None,
                   help="backend name; IQM defaults to garnet, IBM to least-busy")
    p.add_argument("-n", type=int, default=4, help="GHZ qubit count (one circuit)")
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-aer", action="store_true")
    # Generic illustrative device numbers for --dry-run (NOT a specific device).
    # For real numbers, pass --provider and the bench reads live calibration.
    p.add_argument("--t1-us", type=float, default=100.0)
    p.add_argument("--t2-us", type=float, default=80.0)
    p.add_argument("--depol-1q", type=float, default=1.0e-3)
    p.add_argument("--depol-2q", type=float, default=1.0e-2)
    args = p.parse_args()

    print("=" * 66)
    print(f"  Zilver sim-to-real bench card  —  GHZ-{args.n}")
    print("=" * 66)

    if args.dry_run or not args.provider:
        cal = Calibration(args.t1_us*1e3, args.t2_us*1e3, 42.0, 130.0, 0,
                          f"dry-run preset (T1={args.t1_us}us T2={args.t2_us}us)")
        ge = GateErrors(args.depol_1q, args.depol_2q, "dry-run preset")
        card(args.n, cal, ge, None, args.shots, not args.no_aer)
        return

    provider = make_provider(args.provider, args.backend, list(range(args.n)))
    cal = provider.mean_calibration()
    ge = provider.gate_errors()
    card(args.n, cal, ge, provider, args.shots, not args.no_aer)


if __name__ == "__main__":
    main()
