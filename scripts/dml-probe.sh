#!/usr/bin/env bash
# What can the DirectML device actually do?
#
#     bash scripts/dml-probe.sh
#
# The complex64 test runs in its own process on purpose: DirectML does not
# raise on an unsupported dtype, it ABORTS ("Invalid or unsupported data type
# ComplexFloat", core dumped), so an in-process try/except cannot catch it and
# everything after would never run.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -d .venv-x86 ] && . .venv-x86/bin/activate

echo "=== 1. device ==="
python3 -c "
import torch_directml as d
print('  ' + ', '.join(d.device_name(i) for i in range(d.device_count())))"

echo
echo "=== 2. complex64 (isolated -- a crash here is the answer, not a failure) ==="
if python3 -c "
import torch, torch_directml as d
x = torch.ones(4, dtype=torch.complex64).to(d.device())
print('  works:', (x*(2+1j)).sum().cpu())" 2>/dev/null; then
  echo "  COMPLEX OK"
else
  echo "  no complex64 -- state must be carried as real pairs"
fi

echo
echo "=== 3. float32: the ops a gate needs ==="
python3 - <<'PY'
import torch, torch_directml as dml
dev = dml.device()
def t(name, fn):
    try:
        fn(); print(f"  ok    {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {str(e)[:60]}")
a = torch.arange(64, dtype=torch.float32).to(dev)
t("reshape",          lambda: a.reshape(4, 4, 4))
t("permute",          lambda: a.reshape(4,4,4).permute(2,0,1).contiguous())
t("matmul",           lambda: torch.matmul(a.reshape(8,8), a.reshape(8,8)))
t("multiply(out=)",   lambda: torch.multiply(a, 2.0, out=torch.empty_like(a)))
t("in-place *= +=",   lambda: (a.mul_(1.0), a.add_(0.0)))
t("slice assign",     lambda: a.reshape(8,8).__setitem__((slice(0,2),), torch.zeros(2,8).to(dev)))
t("sub",              lambda: a - a)
t("cat",              lambda: torch.cat([a, a]))
t("stack",            lambda: torch.stack([a, a]))
t("sum",              lambda: a.sum())
t("sqrt/abs",         lambda: (a.abs().sqrt()))
t("float64",          lambda: torch.ones(4, dtype=torch.float64).to(dev) * 2)
PY
echo
echo "=== verdict: if float32 ops pass, the backend carries the state as"
echo "    real/imag pairs and lifts each complex gate to its real block form."
