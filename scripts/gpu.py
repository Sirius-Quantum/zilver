"""Run the simulator on the GPU. Nothing else.

    python3 scripts/gpu.py            # 20 -> 28 qubits
    FROM=24 TO=30 python3 scripts/gpu.py

Prints the device it got, then one line per width: seconds and the state norm.
Norm is the correctness check that costs nothing -- a unitary circuit ends at
1.0, so anything else means the arithmetic drifted.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, "src")
os.environ.setdefault("ZILVER_BACKEND", "torch")

import zilver._array as _a
from zilver.circuit import Circuit

print(f"\n  device      : {getattr(_a, 'TORCH_DEVICE', 'cpu')}")
print(f"  complex64   : {_a.HAS_COMPLEX}")
print(f"\n{'qubits':>7}{'state GB':>10}{'seconds':>10}{'norm':>12}")

for n in range(int(os.environ.get("FROM", "20")), int(os.environ.get("TO", "28")) + 1):
    gb = (2 ** n) * 8 / 1e9
    c = Circuit(n); pi = 0
    for q in range(n):
        c.h(q); c.ry(q, pi); pi += 1
    for q in range(n - 1):
        c.cnot(q, q + 1)
    p = [0.1 * (i + 1) for i in range(pi)]
    try:
        t = time.perf_counter()
        v = np.asarray(c.statevector(p).numpy())
        dt = time.perf_counter() - t
    except Exception as e:
        print(f"{n:>7}{gb:>10.2f}   {type(e).__name__}: {str(e)[:50]}")
        break
    if v.ndim == 2 and v.shape[0] == 2:
        v = v[0] + 1j * v[1]
    print(f"{n:>7}{gb:>10.2f}{dt:>10.2f}{np.linalg.norm(v):>12.7f}", flush=True)
