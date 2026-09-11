#!/usr/bin/env bash
# Can this machine be a Zilver node? Run from a zilver checkout:
#
#     bash scripts/x86-node-check.sh          # ladder to 30 qubits
#     MAX_Q=20 bash scripts/x86-node-check.sh # stop lower on a small box
#
# Four questions, in order, each gating the next:
#   1. does zilver install without MLX?
#   2. does it import?
#   3. does the suite pass?
#   4. how many qubits does this box actually hold, and how fast?
#
# Writes scripts/_x86_node_report.json.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="scripts/_x86_node_report.json"

echo "=== box ==="
uname -srm
python3 -c "import platform;print('  python', platform.python_version(), platform.machine())"
grep -m1 "model name" /proc/cpuinfo 2>/dev/null | sed 's/^/  /' || true
echo "  cores: $(python3 -c 'import os;print(os.cpu_count())')"
free -g 2>/dev/null | awk '/^Mem:/{print "  RAM:  " $2 " GB"}' || true

echo; echo "=== 1. install (MLX must NOT be pulled in) ==="
python3 -m venv .venv-x86 2>/dev/null || true
. .venv-x86/bin/activate
pip install -q --upgrade pip
pip install -q -e . && echo "  installed OK"
python3 -c "
import importlib.util as u
print('  mlx present:', u.find_spec('mlx') is not None, '  <- expect False on x86')"

echo; echo "=== 2. import ==="
python3 - <<'PY'
import zilver
from zilver._array import HAS_MLX
for m in ("circuit","gates","simulator","tensor_network","node",
          "density_matrix","metal","cutting","gradients","landscape"):
    __import__(f"zilver.{m}")
print(f"  all modules import.  HAS_MLX={HAS_MLX}")
PY

echo; echo "=== 3. correctness ==="
# tests/ is gitignored, so a clone has no suite to run. These checks are
# self-contained and are the ones that would actually catch a wrong array
# backend: exact states with known amplitudes, and norm preservation.
python3 - <<'PY'
import numpy as np, sys
from zilver.circuit import Circuit
fails = []

def amps(c, p=()):
    return np.asarray(c.statevector(list(p)).numpy()).ravel()

# Bell: (|00> + |11>)/sqrt(2)
c = Circuit(2); c.h(0); c.cnot(0, 1)
v = amps(c); want = np.array([1, 0, 0, 1]) / np.sqrt(2)
if not np.allclose(np.abs(v), np.abs(want), atol=1e-5): fails.append(f"Bell {v}")
print(f"  Bell        {np.round(v.real, 4)}   expect [0.7071 0 0 0.7071]")

# GHZ(n): only |0...0> and |1...1> populated
for n in (3, 5, 8):
    g = Circuit(n); g.h(0)
    for q in range(n - 1): g.cnot(q, q + 1)
    v = amps(g); nz = np.nonzero(np.abs(v) > 1e-4)[0]
    ok = nz.tolist() == [0, 2**n - 1] and np.allclose(np.abs(v[nz]), 1/np.sqrt(2), atol=1e-5)
    if not ok: fails.append(f"GHZ({n}) nonzero at {nz.tolist()}")
    print(f"  GHZ({n})      nonzero {nz.tolist()}   expect [0, {2**n - 1}]")

# unitarity: a random parameterised circuit must preserve the norm
rng = np.random.default_rng(20260904)
for n in (4, 7, 10):
    c = Circuit(n); pi = 0
    for _ in range(5 * n):
        k = int(rng.integers(0, 6)); q = int(rng.integers(0, n))
        q2 = int(rng.integers(0, n))
        while q2 == q: q2 = int(rng.integers(0, n))
        if   k == 0: c.h(q)
        elif k == 1: c.x(q)
        elif k == 2: c.ry(q, pi); pi += 1
        elif k == 3: c.rz(q, pi); pi += 1
        elif k == 4: c.cnot(q, q2)
        else:        c.cz(q, q2)
    p = rng.uniform(0, 2*np.pi, size=max(pi, 1)).astype(np.float32)
    nrm = float(np.linalg.norm(amps(c, p)))
    if abs(nrm - 1.0) > 1e-4: fails.append(f"norm({n}q) = {nrm}")
    print(f"  random {n:2d}q   norm {nrm:.7f}   expect 1.0000000")

print("\n  FAILED: " + "; ".join(fails) if fails else "\n  all correctness checks passed")
sys.exit(1 if fails else 0)
PY

echo; echo "=== 4. qubit ceiling and throughput ==="
python3 - <<'PY'
import json, time, os, numpy as np
from zilver.circuit import Circuit
from zilver._array import HAS_MLX
rows=[]
MAXQ = int(os.environ.get("MAX_Q", "30"))
for n in range(4, MAXQ + 1):
    gb = (2**n) * 8 / 1e9                      # interleaved float32 == complex64
    if gb > 40: rows.append({"n":n,"skipped":True,"state_gb":round(gb,2)}); break
    c=Circuit(n); pi=0
    for q in range(n):
        c.h(q); c.ry(q, pi); pi+=1
    for q in range(n-1): c.cnot(q,q+1)
    p=np.random.default_rng(0).uniform(0,6.28,size=pi).astype(np.float32).tolist()
    t=time.perf_counter(); v=c.statevector(p).numpy(); dt=time.perf_counter()-t
    nrm=float(np.linalg.norm(np.asarray(v)))
    rows.append({"n":n,"seconds":round(dt,4),"state_gb":round(gb,4),"norm":round(nrm,7)})
    print(f"  {n:2d} qubits  {dt:9.4f} s   state {gb:8.4f} GB   norm {nrm:.7f}")
    if dt > 120: print("  (stopping: over 2 minutes)"); break
json.dump({"has_mlx":HAS_MLX,"cores":os.cpu_count(),"rows":rows},
          open("scripts/_x86_node_report.json","w"), indent=1)
print("\n  wrote scripts/_x86_node_report.json")
PY
