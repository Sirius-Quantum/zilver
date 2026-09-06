"""CPU vs GPU, one statevector, both devices. Runs anywhere python does.

    python scripts/cpu_vs_gpu.py            # 24 -> 28 qubits
    FROM=26 TO=29 python scripts/cpu_vs_gpu.py

Reports seconds per device and, in the last two columns, whether the two
devices agree. A speedup means nothing if the physics moved, so a row where
they disagree is void.
"""
import importlib, os, sys, time
import numpy as np

sys.path.insert(0, "src")
FROM = int(os.environ.get("FROM", "24"))
TO   = int(os.environ.get("TO", "28"))


def run(backend, n):
    for m in [k for k in list(sys.modules) if k.startswith("zilver")]:
        del sys.modules[m]
    os.environ["ZILVER_BACKEND"] = backend
    a = importlib.import_module("zilver._array")
    from zilver.circuit import Circuit
    dev = getattr(a, "TORCH_DEVICE", "cpu")
    c = Circuit(n); pi = 0
    for q in range(n):
        c.h(q); c.ry(q, pi); pi += 1
    for q in range(n - 1):
        c.cnot(q, q + 1)
    p = np.random.default_rng(0).uniform(0, 6.28, size=pi).astype(np.float32).tolist()
    t = time.perf_counter()
    v = np.asarray(c.statevector(p, method="mlx").numpy())
    dt = time.perf_counter() - t
    if v.ndim == 2 and v.shape[0] == 2:
        v = v[0] + 1j * v[1]
    return dt, v.ravel(), f"{dev} complex={a.HAS_COMPLEX}"


print(f"{chr(10)}{'qubits':>7}{'state GB':>10}{'CPU s':>9}{'GPU s':>9}{'speedup':>9}"
      f"{'max |diff|':>12}{'fidelity':>11}")
for n in range(FROM, TO + 1):
    gb = (2 ** n) * 8 / 1e9
    try:
        tc, vc, dc = run("numpy", n)
        tg, vg, dg = run("torch", n)
    except Exception as e:
        print(f"{n:>7}{gb:>10.2f}   {type(e).__name__}: {str(e)[:44]}")
        break
    d = np.abs(vc - vg).max()
    f = abs(np.vdot(vc, vg)) ** 2 / (np.vdot(vc, vc).real * np.vdot(vg, vg).real)
    if n == FROM:
        print(f"    cpu arm: {dc}{chr(10)}    gpu arm: {dg}{chr(10)}")
    print(f"{n:>7}{gb:>10.2f}{tc:>9.2f}{tg:>9.2f}{tc/tg:>8.2f}x{d:>12.2e}{f:>11.7f}")
print(f"{chr(10)}  a row counts only if max |diff| is at float32 precision (~1e-7).")
