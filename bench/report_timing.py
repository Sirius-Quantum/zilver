"""M3. Where the time goes at the headline width, against a reference copy in the same process.

    FROM=30 TO=32 bash scripts/rocm-win-zilver.sh --timing

Two questions.

1. IS THE KERNEL STILL ~1.3 COPIES PER GATE AT n=32? At n=24 it measured 1.29-1.34. One copy
   is a bare device copy (one read + one write of the state), timed in this process. At the
   top width two states do not fit, so its reference is the width below it scaled by 2 --
   both are co-timed here and printed, so the scaling can be seen to hold. A figure far above
   the n=24 range would mean the state is not resident (paging or host copies).

2. WHAT IS IN THE 52.77 s / 54.39 s? scripts/gpu.py times H + RY on every qubit, a CNOT
   ladder, AND the copy of the whole state back to the host, not one Hadamard layer. This
   re-runs exactly that circuit and splits it: allocation, gates (each launch timed on its
   own, so the largest is the margin under the 2 s WDDM watchdog), readback, host norm.
"""
from __future__ import annotations

import time

import numpy as np

import _report as R

REPS = 3


def timed(B, fn, reps=REPS):
    fn(); B.sync()                                     # warm
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); B.sync(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), float(max(ts))


def positions(n):
    return [0, 1, n // 2, n - 2, n - 1]


def gate_table(B, s, n, ref):
    rows = []
    for q in positions(n):
        med, mx = timed(B, lambda: B.apply(s, R.H, (q,), n))
        rows.append({"gate": "H", "qubits": [q], "stride": 1 << (n - 1 - q),
                     "s": med, "max_s": mx, "copies": med / ref})
    for qs in ((0, 1), (n - 2, n - 1)):
        med, mx = timed(B, lambda: B.apply(s, R.CP(0.3), qs, n))
        rows.append({"gate": "CP", "qubits": list(qs), "s": med, "max_s": mx,
                     "copies": med / ref})
    for r in rows:
        print(f"    {r['gate']:<3}{str(r['qubits']):>10}  {r['s']*1e3:9.1f} ms  "
              f"{r['copies']:5.2f} copies", flush=True)
    return rows


def main():
    B = R.backend()
    print(f"== M3 timing   backend={B.name}   {B.info}")
    ws = R.widths([30, 31, 32])
    top = ws[-1]
    out = {"backend": B.info, "widths": {}}
    ref = {}

    for n in ws[:-1]:
        src = B.randn(n)
        dst = B.empty_like(src)
        med, _ = timed(B, lambda: B.copy_(dst, src))
        del dst
        B.free()
        ref[n] = med
        gb = 2 * (1 << n) * 8 / 1e9
        print(f"\n  n={n}: reference copy {med*1e3:.1f} ms = {gb/med:.1f} GB/s == 1.00 copies")
        out["widths"][n] = {"ref_copy_s": med, "ref_gbps": gb / med,
                            "gates": gate_table(B, src, n, med)}
        del src
        B.free()

    below = top - 1
    ref_top = ref[below] * 2 if below in ref else None
    print(f"\n  n={top}: reference = 2 x n={below} copy = "
          f"{(ref_top or float('nan'))*1e3:.1f} ms  (two states do not fit at this width)")

    t = time.perf_counter(); s = B.basis(top, 0); B.sync(); t_alloc = time.perf_counter() - t

    circ = []
    for q in range(top):
        circ.append((R.H, (q,)))
        circ.append((R.RY(0.1 * (q + 1)), (q,)))
    for q in range(top - 1):
        circ.append((R.CNOT, (q, q + 1)))
    launch = []
    t0 = time.perf_counter()
    for g, q in circ:
        t = time.perf_counter(); B.apply(s, g, q, top); B.sync(); launch.append(time.perf_counter() - t)
    t_gates = time.perf_counter() - t0
    t = time.perf_counter(); h = B.to_host(s); t_read = time.perf_counter() - t
    t = time.perf_counter()
    acc = 0.0
    for i in range(0, h.size, 1 << 26):
        c = h[i:i + (1 << 26)].astype(np.complex128)
        acc += float(np.vdot(c, c).real)
    host_norm = acc ** 0.5
    t_norm = time.perf_counter() - t
    del h
    split = {"circuit": "scripts/gpu.py: H+RY on every qubit, CNOT ladder",
             "gates": len(circ), "alloc_s": t_alloc, "gates_s": t_gates,
             "per_gate_median_s": float(np.median(launch)), "max_launch_s": float(max(launch)),
             "readback_s": t_read, "host_norm_s": t_norm, "norm": host_norm,
             "total_s": t_alloc + t_gates + t_read}
    if ref_top:
        split["per_gate_copies"] = split["per_gate_median_s"] / ref_top
    print(f"\n  the gpu.py circuit at n={top}, split:")
    for k in ("alloc_s", "gates_s", "readback_s", "host_norm_s", "total_s"):
        print(f"    {k:<12} {split[k]:8.2f} s")
    print(f"    {len(circ)} gates, median launch {split['per_gate_median_s']:.3f} s, "
          f"largest {split['max_launch_s']:.3f} s (WDDM watchdog is 2 s), norm {host_norm:.7f}")

    print(f"\n  single gates at n={top}:")
    gates_top = gate_table(B, s, top, ref_top) if ref_top else []
    out["widths"][top] = {"ref_copy_s_derived": ref_top, "split": split, "gates": gates_top,
                          "mem": B.mem()}
    del s
    B.free()

    if ref_top:
        cp = [r["copies"] for r in gates_top if r["gate"] == "H"]
        resident = max(cp) <= 2.6
        R.verdict(f"M3 resident n={top}", resident,
                  f"H at {min(cp):.2f}-{max(cp):.2f} copies (n=24 measured 1.29-1.34; "
                  f"above 2.6 = 2x that = not resident)")
    R.verdict(f"M3 watchdog margin n={top}", split["max_launch_s"] < 2.0,
              f"largest single launch {split['max_launch_s']:.3f} s against a 2 s TDR")
    R.write_json("m3_timing.json", out)


if __name__ == "__main__":
    main()
