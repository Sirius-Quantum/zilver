#!/usr/bin/env bash
# Can this machine be a Zilver node? Run from a zilver checkout:
#
#     bash scripts/x86-node-check.sh
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

echo; echo "=== 3. test suite ==="
pip install -q pytest
python3 -m pytest tests -q --no-header --tb=line 2>&1 | tail -3

echo; echo "=== 4. qubit ceiling and throughput ==="
python3 - <<'PY'
import json, time, numpy as np, os
from zilver.circuit import Circuit
from zilver._array import HAS_MLX
rows=[]
for n in range(4, 31):
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
