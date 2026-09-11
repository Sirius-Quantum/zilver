"""M3. Memory traffic per gate at every width up to the top, against an in-place stream.

    FROM=24 TO=32 bash scripts/rocm-win-zilver.sh --timing

THE UNIT. Any single-qubit gate must read and write every amplitude once, so its compulsory
traffic is 2 x state bytes. We print that as effective GB/s, and as a ratio to an IN-PLACE
STREAM over the same buffer (view_as_real(state).neg_(): one read and one write of every
amplitude, the gate's own footprint), timed in the same process at the same width.

WHY NOT A PLAIN COPY (the first version of this bench). dst.copy_(src) streams TWO buffers, so
it needs twice the memory, and on this APU it degrades from 207 GB/s at 24 qubits to 116-128 GB/s
at 30-31 qubits while the in-place gate does not. The first version also DERIVED the top-width
reference as 2x the width below, because two top-width states do not fit. Together those printed
0.75 "copies" at 32 qubits -- below the floor, an artifact of the yardstick, not the kernel
(harvest amd-report-20260911T110052Z). The two-buffer copy is still timed below the top width,
labelled DIAGNOSTIC, so the degradation stays visible; nothing is derived any more.

At the top width it also re-runs scripts/gpu.py's circuit (H + RY on every qubit, a CNOT
chain, and the copy back to the host) split into allocation, gates, readback and host norm, with
every launch timed on its own so the largest is the margin under the 2 s WDDM watchdog.
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


def gbps(n, s):
    return 2 * (1 << n) * 8 / 1e9 / s


def gate_table(B, s, n, stream_s):
    rows = []
    for q in [0, 1, n // 2, n - 2, n - 1]:
        med, mx = timed(B, lambda: B.apply(s, R.H, (q,), n))
        rows.append({"gate": "H", "qubits": [q], "stride": 1 << (n - 1 - q), "s": med, "max_s": mx})
    for qs in ((0, 1), (n - 2, n - 1)):
        med, mx = timed(B, lambda: B.apply(s, R.CP(0.3), qs, n))
        rows.append({"gate": "CP", "qubits": list(qs), "s": med, "max_s": mx})
    for r in rows:
        r["gbps"] = gbps(n, r["s"])
        r["x_stream"] = r["s"] / stream_s
        print(f"    {r['gate']:<3}{str(r['qubits']):>10}  {r['s']*1e3:9.1f} ms  "
              f"{r['gbps']:6.1f} GB/s  {r['x_stream']:5.2f} x in-place stream", flush=True)
    return rows


def main():
    B = R.backend()
    print(f"== M3 timing   backend={B.name}   {B.info}")
    ws = R.widths([24, 30, 31, 32])
    top = ws[-1]
    out = {"backend": B.info, "unit": "2*state_bytes/time; ratio to in-place stream, same buffer",
           "widths": {}}

    for n in ws:
        w = {}
        if n != top:                                   # DIAGNOSTIC only: two buffers
            src = B.randn(n); dst = B.empty_like(src)
            cp, _ = timed(B, lambda: B.copy_(dst, src))
            del dst, src
            B.free()
            w["copy_two_buffer_s"] = cp
            w["copy_two_buffer_gbps"] = gbps(n, cp)
        s = B.randn(n)
        st, _ = timed(B, lambda: B.stream_(s))
        w["stream_s"] = st
        w["stream_gbps"] = gbps(n, st)
        diag = (f"   [diagnostic two-buffer copy {w['copy_two_buffer_gbps']:.1f} GB/s]"
                if "copy_two_buffer_gbps" in w else "")
        print(f"\n  n={n}: in-place stream {st*1e3:.1f} ms = {w['stream_gbps']:.1f} GB/s == 1.00{diag}")
        w["gates"] = gate_table(B, s, n, st)
        out["widths"][n] = w
        if n != top:
            del s
            B.free()

    # the gpu.py circuit, split, on a fresh |0...0> at the top width
    del s
    B.free()
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
    t_norm = time.perf_counter() - t
    del h
    split = {"circuit": "scripts/gpu.py: H+RY on every qubit, CNOT chain", "gates": len(circ),
             "alloc_s": t_alloc, "gates_s": t_gates, "per_gate_median_s": float(np.median(launch)),
             "max_launch_s": float(max(launch)), "readback_s": t_read, "host_norm_s": t_norm,
             "norm": acc ** 0.5, "total_s": t_alloc + t_gates + t_read}
    out["widths"][top]["split"] = split
    out["widths"][top]["mem"] = B.mem()
    print(f"\n  the gpu.py circuit at n={top}, split:")
    for k in ("alloc_s", "gates_s", "readback_s", "host_norm_s", "total_s"):
        print(f"    {k:<12} {split[k]:8.2f} s")
    print(f"    {len(circ)} gates, median launch {split['per_gate_median_s']:.3f} s, "
          f"largest {split['max_launch_s']:.3f} s (WDDM watchdog is 2 s), norm {split['norm']:.7f}")
    del s
    B.free()

    h_top = [r for r in out["widths"][top]["gates"] if r["gate"] == "H"]
    print(f"\n  n={top}: H at {min(r['gbps'] for r in h_top):.1f}-{max(r['gbps'] for r in h_top):.1f} GB/s, "
          f"{min(r['x_stream'] for r in h_top):.2f}-{max(r['x_stream'] for r in h_top):.2f} x the in-place stream")
    R.verdict(f"M3 watchdog margin n={top}", split["max_launch_s"] < 2.0,
              f"largest single launch {split['max_launch_s']:.3f} s against a 2 s TDR")
    R.write_json("m3_timing.json", out)


if __name__ == "__main__":
    main()
