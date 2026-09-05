#!/usr/bin/env bash
# The measurement promised to AMD: the SAME statevector, in the SAME memory,
# executed on the CPU cores and then on the iGPU.
#
#     bash scripts/cpu-vs-igpu.sh            # 24 -> 30 qubits
#     FROM=30 TO=33 bash scripts/cpu-vs-igpu.sh
#
# Reports, per qubit count and per device: seconds, peak resident memory, and
# the element-wise agreement between the two devices' final states. The last
# column is the one that matters -- a speedup means nothing if the physics
# moved, so the states must agree to float32 precision or the row is void.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -d .venv-x86 ] && . .venv-x86/bin/activate

python3 - <<'PY'
import json, os, sys, time, resource, importlib
import numpy as np
sys.path.insert(0, "src")

FROM = int(os.environ.get("FROM", "24"))
TO   = int(os.environ.get("TO",   "30"))
U    = 1024 if os.uname().sysname == "Linux" else 1024**3

def run(backend, n):
    """Fresh interpreter state per backend: _array picks at import."""
    for m in [k for k in list(sys.modules) if k.startswith("zilver")]:
        del sys.modules[m]
    os.environ["ZILVER_BACKEND"] = backend
    from zilver.circuit import Circuit
    from zilver._array import mx
    dev = getattr(importlib.import_module("zilver._array"), "TORCH_DEVICE", "cpu")
    c = Circuit(n); pi = 0
    for q in range(n):
        c.h(q); c.ry(q, pi); pi += 1
    for q in range(n - 1):
        c.cnot(q, q + 1)
    p = np.random.default_rng(0).uniform(0, 6.28, size=pi).astype(np.float32).tolist()
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / U
    t = time.perf_counter()
    v = np.asarray(c.statevector(p, method="mlx").numpy()).ravel()
    dt = time.perf_counter() - t
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / U
    return dict(seconds=dt, peak_gb=rss, base_gb=base, device=str(dev),
                norm=float(np.linalg.norm(v)), state=v)

print(f"\n{'qubits':>7}{'state GB':>10}{'CPU s':>9}{'iGPU s':>9}{'speedup':>9}"
      f"{'max |diff|':>12}{'fidelity':>11}")
rows = []
for n in range(FROM, TO + 1):
    gb = (2**n) * 8 / 1e9
    try:
        cpu = run("numpy", n)
        gpu = run("torch", n)
    except Exception as e:
        print(f"{n:>7}{gb:>10.2f}   {type(e).__name__}: {str(e)[:44]}")
        break
    d = np.abs(cpu["state"] - gpu["state"]).max()
    num = abs(np.vdot(cpu["state"], gpu["state"]))**2
    fid = num / (np.vdot(cpu["state"], cpu["state"]).real
                 * np.vdot(gpu["state"], gpu["state"]).real)
    print(f"{n:>7}{gb:>10.2f}{cpu['seconds']:>9.2f}{gpu['seconds']:>9.2f}"
          f"{cpu['seconds']/gpu['seconds']:>8.2f}x{d:>12.2e}{fid:>11.7f}")
    rows.append(dict(n=n, state_gb=round(gb, 3), device=gpu["device"],
                     cpu_s=round(cpu["seconds"], 3), gpu_s=round(gpu["seconds"], 3),
                     speedup=round(cpu["seconds"] / gpu["seconds"], 3),
                     max_diff=float(d), fidelity=float(fid),
                     cpu_peak_gb=round(cpu["peak_gb"], 2),
                     gpu_peak_gb=round(gpu["peak_gb"], 2)))

json.dump(rows, open("scripts/_cpu_vs_igpu.json", "w"), indent=1)
print(f"\n  device: {rows[-1]['device'] if rows else 'none'}")
print("  a row counts only if max |diff| is at float32 precision (~1e-7).")
print("  wrote scripts/_cpu_vs_igpu.json")
PY
