"""M1 + M4. QFT on a planted basis state, checked against its closed form -- and made to fail.

    FROM=20 TO=32 bash scripts/rocm-win-zilver.sh --qft

WHY THIS CIRCUIT. A preserved norm is necessary, not sufficient: a wrong unitary preserves it
too. The QFT of |x> has every amplitude in closed form,

    psi[z] = exp(2 pi i * x * rev_n(z) / 2^n) / 2^(n/2)          (no final swaps, so rev_n)

and at n qubits it is n Hadamards plus n(n-1)/2 controlled phases -- 528 gates at n=32 -- which
touch every qubit pair, both stride extremes, and the 2-qubit (m=2) kernel path at all of them.
A wrong mask, stride or orientation anywhere shows up as a wrong phase.

READOUT. K seeded amplitudes (65,536 by default) read back from the device, error measured
RELATIVE to the exact modulus 2^(-n/2): |psi[z] * 2^(n/2) - exp(i theta)|. An absolute 1e-9 at
n=32, where every amplitude is 1.5e-5, would hide a great deal.

  PASS         max relative error <= 1e-4 at every width and seed
  M4 control   one CP(pi/2) moved from qubits (n-2, n-1) to (0, n-1). The check MUST fail it.
               x is forced odd for this run: with that bit clear, both CPs are the identity
               and the control could not fail -- a negative control that cannot fail proves
               nothing.
  M4 repeat    seed 0 run again; the sampled amplitudes must be bit-identical.
"""
from __future__ import annotations

import time

import numpy as np

import _report as R

TOL = 1e-4
SEEDS = (0, 1, 2)


def qft_ops(n):
    ops = []
    for j in range(n):
        ops.append((R.H, (j,)))
        for k in range(j + 1, n):
            ops.append((R.CP(np.pi / 2 ** (k - j)), (k, j)))
    return ops


def faulted(ops, n):
    """The same circuit with CP(pi/2) on (n-1, n-2) retargeted to (n-1, 0)."""
    out = list(ops)
    for i, (g, q) in enumerate(out):
        if q == (n - 1, n - 2):
            out[i] = (g, (n - 1, 0))
            return out
    raise RuntimeError("fault site not found")


def bitrev(z, n):
    z = z.astype(np.uint64)
    r = np.zeros_like(z)
    for b in range(n):
        r |= ((z >> np.uint64(b)) & np.uint64(1)) << np.uint64(n - 1 - b)
    return r


def expected_phase(n, x, z):
    mask = np.uint64((1 << n) - 1)
    prod = (np.uint64(x) * bitrev(z, n)) & mask         # wraps mod 2^64; 2^n divides it
    return np.exp(2j * np.pi * prod.astype(np.float64) / float(1 << n))


def run(B, n, x, ops, seed):
    s = B.basis(n, x)
    B.sync()
    t = time.perf_counter()
    for g, q in ops:
        B.apply(s, g, q, n)
    B.sync()
    dt = time.perf_counter() - t
    idx = R.sample_indices(n, 1000 + seed)
    amp = B.sample(s, idx)
    err = float(np.max(np.abs(amp.astype(np.complex128) * 2 ** (n / 2) - expected_phase(n, x, idx))))
    nrm = B.norm(s)
    del s
    B.free()
    return {"n": n, "seed": seed, "x": int(x), "gates": len(ops), "seconds": round(dt, 3),
            "per_gate_s": dt / len(ops), "k": int(idx.size), "max_rel_err": err,
            "norm": nrm}, amp


def main():
    B = R.backend()
    print(f"== M1 QFT closed form   backend={B.name}   {B.info}")
    rows, ok = [], True
    ws = R.widths([20, 24, 28, 30, 31, 32])
    amp0 = {}
    for n in ws:
        ops = qft_ops(n)
        for seed in SEEDS:
            x = int(np.random.default_rng(seed).integers(0, 1 << n, dtype=np.uint64))
            r, amp = run(B, n, x, ops, seed)
            if seed == 0:
                amp0[n] = amp
            rows.append(r)
            ok &= R.verdict(f"M1 n={n} seed={seed}", r["max_rel_err"] <= TOL,
                            f"err={r['max_rel_err']:.2e} (<= {TOL:g})  norm={r['norm']:.7f}  "
                            f"{r['gates']} gates {r['seconds']:.1f}s")

    top = ws[-1]
    print(f"\n== M4 negative control and repeat, n={top}")
    x = int(np.random.default_rng(0).integers(0, 1 << top, dtype=np.uint64)) | 1
    bad, _ = run(B, top, x, faulted(qft_ops(top), top), seed=0)
    bad["planted_fault"] = f"CP(pi/2) ({top-1},{top-2}) -> ({top-1},0), x forced odd"
    caught = bad["max_rel_err"] > 1e-2
    R.verdict(f"M4 fault caught n={top}", caught,
              f"err={bad['max_rel_err']:.2e} on the faulted circuit (must exceed 1e-2)")
    x0 = int(np.random.default_rng(0).integers(0, 1 << top, dtype=np.uint64))
    rep, amp_rep = run(B, top, x0, qft_ops(top), seed=0)
    same = bool(np.array_equal(amp_rep.view(np.uint32), amp0[top].view(np.uint32)))
    R.verdict(f"M4 bit-identical repeat n={top}", same,
              f"{amp_rep.size} sampled amplitudes, seed 0 run twice")

    R.write_json("m1_qft.json", {"backend": B.info, "tol": TOL, "rows": rows,
                                 "fault": bad, "repeat_identical": same,
                                 "all_pass": bool(ok and caught and same)})
    print(f"\n  M1+M4 overall: {'PASS' if ok and caught and same else 'FAIL'}")


if __name__ == "__main__":
    main()
