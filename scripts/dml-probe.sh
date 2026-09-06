#!/usr/bin/env bash
# Does the DirectML device do what a statevector needs?
#
#     bash scripts/dml-probe.sh
#
# Three questions in order. The second decides how the backend is written:
# a statevector is complex, and if DirectML cannot do complex64 we carry the
# state as interleaved float32 -- which is exactly what metal.py already does
# to dodge the same gap in MLX's custom kernels.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -d .venv-x86 ] && . .venv-x86/bin/activate
python3 - <<'PY'
import sys
sys.path.insert(0, "src")
import torch

print("=== 1. device ===")
try:
    import torch_directml as dml
except ImportError:
    print("  torch_directml not installed"); raise SystemExit(1)
n = dml.device_count()
print(f"  {n} device(s): " + ", ".join(dml.device_name(i) for i in range(n)))
dev = dml.device()

print("\n=== 2. complex64 ===")
ok_complex = False
try:
    x = torch.ones(4, dtype=torch.complex64).to(dev)
    y = (x * (2 + 1j)).sum().cpu()
    print(f"  complex64 works: sum = {y}   expect (8+4j)")
    ok_complex = abs(complex(y) - (8 + 4j)) < 1e-5
except Exception as e:
    print(f"  complex64 FAILS: {type(e).__name__}: {str(e)[:90]}")

print("\n=== 3. the ops a gate needs, in float32 ===")
try:
    a = torch.arange(16, dtype=torch.float32).to(dev).reshape(2, 2, 2, 2)
    b = a.permute(0, 2, 1, 3).reshape(4, 4)
    c = torch.matmul(b, b)
    t = torch.empty_like(b); torch.multiply(b, 2.0, out=t); t += b
    r = float((c.sum() + t.sum()).cpu())
    print(f"  reshape/permute/matmul/multiply(out=)/+= all ran, checksum {r:.1f}")
    print("  float32 path is usable")
except Exception as e:
    print(f"  FAILS: {type(e).__name__}: {str(e)[:90]}")

print("\n=== verdict ===")
print("  complex64 supported -> backend runs unchanged" if ok_complex else
      "  no complex64 -> carry the state as interleaved float32, as metal.py does")
PY
