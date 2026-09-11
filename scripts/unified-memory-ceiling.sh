#!/usr/bin/env bash
# How many qubits does this box's MEMORY actually hold?
#
#     bash scripts/unified-memory-ceiling.sh          # 30 -> 33
#     FROM=32 TO=33 bash scripts/unified-memory-ceiling.sh
#
# The node-check script capped each circuit at 120 s and therefore stopped at
# 29 qubits -- which measured its own patience, not the machine. This one has
# no time cap and walks until allocation fails, reporting PEAK RESIDENT MEMORY
# at each rung. That number is the evidence: a statevector is one contiguous
# array, so the qubit ceiling is set by how much memory the process can hold,
# and nothing else. 33 qubits is 68.7 GB; 34 is 137.4 GB.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -d .venv-x86 ] && . .venv-x86/bin/activate

free -g 2>/dev/null | awk '/^Mem:/{print "  RAM total " $2 " GB, available " $7 " GB"}'
python3 - <<'PY'
import json, os, time, resource, numpy as np
from zilver.circuit import Circuit
from zilver._array import HAS_MLX

FROM = int(os.environ.get("FROM", "30"))
TO   = int(os.environ.get("TO",   "33"))
KB   = 1024 if os.uname().sysname == "Linux" else 1024**2   # ru_maxrss units

print(f"\n  HAS_MLX={HAS_MLX}   walking {FROM} -> {TO} qubits, no time cap\n")
print(f"{'qubits':>7}{'state GB':>10}{'seconds':>10}{'peak RSS GB':>13}{'norm':>12}")
rows = []
for n in range(FROM, TO + 1):
    gb = (2**n) * 8 / 1e9
    try:
        c = Circuit(n); pi = 0
        for q in range(n):
            c.h(q); c.ry(q, pi); pi += 1
        for q in range(n - 1):
            c.cnot(q, q + 1)
        p = np.random.default_rng(0).uniform(0, 6.28, size=pi).astype(np.float32).tolist()
        t = time.perf_counter()
        v = c.statevector(p).numpy()
        dt = time.perf_counter() - t
        nrm = float(np.linalg.norm(np.asarray(v)))
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / KB
        print(f"{n:>7}{gb:>10.1f}{dt:>10.1f}{rss:>13.1f}{nrm:>12.7f}")
        rows.append({"n": n, "state_gb": round(gb, 2), "seconds": round(dt, 1),
                     "peak_rss_gb": round(rss, 1), "norm": round(nrm, 7)})
        del v, c
    except MemoryError:
        print(f"{n:>7}{gb:>10.1f}   MemoryError -- this is the ceiling")
        rows.append({"n": n, "state_gb": round(gb, 2), "memory_error": True})
        break
    except Exception as e:
        print(f"{n:>7}{gb:>10.1f}   {type(e).__name__}: {str(e)[:50]}")
        rows.append({"n": n, "state_gb": round(gb, 2), "error": str(e)[:200]})
        break

json.dump({"has_mlx": HAS_MLX, "rows": rows},
          open("scripts/_ceiling_report.json", "w"), indent=1)
top = max((r["n"] for r in rows if "norm" in r), default=None)
print(f"\n  highest qubit count completed: {top}")
print(f"  wrote scripts/_ceiling_report.json")
PY
