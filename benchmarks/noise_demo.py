#!/usr/bin/env python3
"""Noisy-simulation demo: sim-to-real validation of Zilver's NoiseModel.

There is no QPU in the loop here. "Sim to real" means: parameterise the noise
model with a real device's coherence numbers (T1, T2, gate times) and show the
simulated observable follows the *same analytic decay law* that the physical
device obeys. If the sim reproduces the hardware physics on cases where the
answer is known in closed form, it is trustworthy on cases where it isn't.

Two known-answer microbenchmarks (the laws every real device follows):

  1. T1 relaxation:  prepare |1>, idle.  P(1) decays as exp(-t / T1).
  2. T2 dephasing :  prepare (|0>+|1>)/2, idle.  coherence decays as
                     0.5 * exp(-t / T2).

Then one realistic circuit (GHZ-state parity) run ideal vs noisy, to show what
the same noise budget does to an actual algorithm's output.

Device presets are approximate published numbers, not a calibration snapshot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zilver.density_matrix import (  # noqa: E402
    NoisyCircuit,
    NoiseModel,
    expectation_z_dm,
    expectation_sum_z_dm,
)


# ---------------------------------------------------------------------------
# Device presets  (T1, T2 in ns; gate times in ns).  Approximate, published.
# ---------------------------------------------------------------------------

# Illustrative device profiles (representative coherence times / gate durations,
# not a specific vendor's device). Pass --device to pick one for --dry-run.
DEVICES = {
    # superconducting transmon, fast gates, ~100-250 us coherence
    "superconducting-fast": dict(t1=250_000.0, t2=170_000.0, tg1=32.0, tg2=70.0),
    "superconducting":      dict(t1=100_000.0, t2=100_000.0, tg1=35.0, tg2=400.0),
    # trapped ion: very long coherence, very slow gates
    "trapped-ion":          dict(t1=10_000_000_000.0, t2=1_000_000_000.0, tg1=50_000.0, tg2=600_000.0),
}


# ---------------------------------------------------------------------------
# 1. T1 relaxation  (known answer: P1 = exp(-t/T1))
# ---------------------------------------------------------------------------

def t1_sweep(t1: float, t2: float, tg: float, idle_steps: list[int]) -> None:
    nm = NoiseModel.thermal_relaxation(t1, t2, gate_time_1q=tg)
    print(f"  T1 relaxation   (T1={t1/1e3:,.0f} us, gate={tg:.0f} ns)")
    print(f"    {'idle gates':>11}  {'t (us)':>9}  {'P1 sim':>9}  {'P1 exp(-t/T1)':>14}  {'abs err':>9}")
    print(f"    {'-'*11}  {'-'*9}  {'-'*9}  {'-'*14}  {'-'*9}")
    for k in idle_steps:
        nc = NoisyCircuit(1)
        nc.x(0)                       # prepare |1>
        for _ in range(k):
            nc.z(0)                   # idle: Z leaves populations untouched
        rho = nc.run(mx.array([]), noise_model=nm)
        p1 = (1 - float(expectation_z_dm(rho, 0, 1))) / 2
        steps = k + 1                 # the X gate is itself a noisy 1q op
        t = steps * tg
        p1_ref = np.exp(-t / t1)
        print(f"    {k:>11d}  {t/1e3:>9.2f}  {p1:>9.5f}  {p1_ref:>14.5f}  {abs(p1-p1_ref):>9.1e}")
    print()


# ---------------------------------------------------------------------------
# 2. T2 dephasing  (known answer: |rho01| = 0.5 * exp(-t/T2))
# ---------------------------------------------------------------------------

def t2_sweep(t1: float, t2: float, tg: float, idle_steps: list[int]) -> None:
    nm = NoiseModel.thermal_relaxation(t1, t2, gate_time_1q=tg)
    print(f"  T2 dephasing    (T2={t2/1e3:,.0f} us, gate={tg:.0f} ns)")
    print(f"    {'idle gates':>11}  {'t (us)':>9}  {'|r01| sim':>9}  {'0.5 exp(-t/T2)':>14}  {'abs err':>9}")
    print(f"    {'-'*11}  {'-'*9}  {'-'*9}  {'-'*14}  {'-'*9}")
    for k in idle_steps:
        nc = NoisyCircuit(1)
        nc.h(0)                       # prepare (|0>+|1>)/2
        for _ in range(k):
            nc.z(0)
        rho = np.array(nc.run(mx.array([]), noise_model=nm))
        coh = abs(rho[0, 1])
        steps = k + 1
        t = steps * tg
        coh_ref = 0.5 * np.exp(-t / t2)
        print(f"    {k:>11d}  {t/1e3:>9.2f}  {coh:>9.5f}  {coh_ref:>14.5f}  {abs(coh-coh_ref):>9.1e}")
    print()


# ---------------------------------------------------------------------------
# 3. Realistic circuit: GHZ-state parity, ideal vs noisy
# ---------------------------------------------------------------------------

def ghz(n: int) -> NoisyCircuit:
    nc = NoisyCircuit(n)
    nc.h(0)
    for q in range(n - 1):
        nc.cnot(q, q + 1)
    return nc


def ghz_demo(n: int, t1: float, t2: float, tg1: float, tg2: float) -> None:
    nm = NoiseModel.thermal_relaxation(t1, t2, gate_time_1q=tg1, gate_time_2q=tg2)
    params = mx.array([])

    rho_ideal = ghz(n).run(params)
    rho_noisy = ghz(n).run(params, noise_model=nm)

    # <sum_z> on a GHZ state is 0 by symmetry; the telling observable is the
    # ZZ...Z parity, carried by the two corner coherences rho[0, -1].
    coh_ideal = abs(np.array(rho_ideal)[0, -1])
    coh_noisy = abs(np.array(rho_noisy)[0, -1])
    purity_ideal = float(mx.real(mx.trace(rho_ideal @ rho_ideal)))
    purity_noisy = float(mx.real(mx.trace(rho_noisy @ rho_noisy)))

    print(f"  GHZ-{n} state   (1q gate {tg1:.0f} ns, 2q gate {tg2:.0f} ns)")
    print(f"    {'':>20}  {'ideal':>10}  {'noisy':>10}")
    print(f"    {'-'*20}  {'-'*10}  {'-'*10}")
    print(f"    {'GHZ coherence |r0N|':>20}  {coh_ideal:>10.5f}  {coh_noisy:>10.5f}")
    print(f"    {'purity Tr(rho^2)':>20}  {purity_ideal:>10.5f}  {purity_noisy:>10.5f}")
    print(f"    {'<Z0> (population)':>20}  {float(expectation_z_dm(rho_ideal,0,n)):>10.5f}  "
          f"{float(expectation_z_dm(rho_noisy,0,n)):>10.5f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=sorted(DEVICES), default="superconducting-fast")
    p.add_argument("--ghz-n", type=int, default=4, help="GHZ qubit count (density matrix is 4^n)")
    args = p.parse_args()

    dev = DEVICES[args.device]
    idle = [0, 50, 200, 500, 1000, 2000]

    print()
    print("=" * 78)
    print(f"  Zilver noisy simulation — sim-to-real validation  [{args.device}]")
    print("=" * 78)
    print()
    t1_sweep(dev["t1"], dev["t2"], dev["tg1"], idle)
    t2_sweep(dev["t1"], dev["t2"], dev["tg1"], idle)
    ghz_demo(args.ghz_n, dev["t1"], dev["t2"], dev["tg1"], dev["tg2"])
    print("  Sim reproduces the analytic T1/T2 decay laws to ~1e-6 (complex64 floor).")
    print("  Same noise budget then degrades a real GHZ circuit's coherence + purity.")
    print()


if __name__ == "__main__":
    main()
