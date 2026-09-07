"""What does the fused kernel buy, per circuit, before we build it?

Passes come from circuit structure alone (zilver.passes) and need no hardware.
The traffic column multiplies them by the MEASURED copies per gate on the box:
4.28 today (bench/copies_per_gate.py, Radeon 8060S) against a floor of 1.00 for
a fused in-place pass. It is a PREDICTION, stated here so it can be wrong.

    python3 bench/pass_price.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from zilver.circuit import Circuit
from zilver.passes import count_passes

NOW = float(os.environ.get("COPIES_NOW", "4.28"))   # measured, Radeon, q=0
GOAL = 1.00                                          # in-place fused pass

n = 28
rng = np.random.default_rng(0)
cases = []

c = Circuit(n); p = 0
for q in range(n): c.h(q); c.ry(q, p); p += 1
for q in range(n - 1): c.cnot(q, q + 1)
cases.append(("our benchmark circuit", c))

he = Circuit(n); p = 0
for _ in range(20):
    for q in range(n): he.u3(q, p, p + 1, p + 2); p += 3
    for q in range(0, n - 1, 2): he.cnot(q, q + 1)
    for q in range(1, n - 1, 2): he.cnot(q, q + 1)
cases.append(("hardware-efficient VQE, depth 20", he))

ho = Circuit(n); p = 0
for _ in range(560):
    ho.u3(int(rng.integers(n)), p, p + 1, p + 2); p += 3
cases.append(("hostile: u3 on random qubits", ho))

tf = Circuit(n); p = 0
for _ in range(20):
    for q in range(n): tf.ry(q, p); p += 1
    for q in range(n - 2): tf.toffoli(q, q + 1, q + 2)
cases.append(("Toffoli ladder (frame cannot help)", tf))

print(f"{'circuit':<36}{'gates':>7}{'naive':>7}{'frame':>7}{'fused':>7}"
      f"{'passes':>9}{'traffic':>9}")
for label, circ in cases:
    r = count_passes(circ._ops, n, m=4)
    pr = r["naive_passes"] / max(r["fused_passes"], 1)
    tr = (r["naive_passes"] * NOW) / (max(r["fused_passes"], 1) * GOAL)
    print(f"{label:<36}{r['gates']:>7}{r['naive_passes']:>7}"
          f"{r['frame_passes']:>7}{r['fused_passes']:>7}{pr:>8.1f}x{tr:>8.1f}x")

print(f"""
passes  : structure only, no hardware. traffic: passes x copies, using the
          measured {NOW} copies/gate today vs {GOAL:.2f} for a fused in-place pass.
The SPREAD is the point. A single headline number would be quoting the planner
rather than the kernel: the Toffoli row cannot use the frame at all (Toffoli is
quadratic on the index, not linear), and the hostile row cannot use its
structure. If every row came out alike, the measurement would be broken.""")
