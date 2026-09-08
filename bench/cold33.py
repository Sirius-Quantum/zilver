"""Is 33 qubits actually out of reach, or was the sweep just not first in line?

    FROM=33 TO=33 bash scripts/rocm-win-zilver.sh --cold33

WHY THIS EXISTS. bench/mem_shape.py allocated a single 64.00 GiB block on this box, and 33
qubits needs exactly 64.00 GiB -- "short by 0.00 GiB". Yet the width sweep OOM'd at 33 with
`free: 0`. Both can only be true if something was already resident when the sweep asked. The
sweep climbs 20, 21, ... 32 first, and the HIP caching allocator holds the blocks it has
already handed out, so by the time it reaches 33 the box is not empty.

So this asks the question with nothing else in the process: allocate the state FIRST, before
any warm-up, any gate matrix, any cached block, and see whether the driver gives it to us.

PRE-REGISTERED, before it ran:
  PASS      the state allocates and one gate leaves norm 0.999999x. 33 qubits is real on an
            integrated GPU, and the wall was our own allocator rather than the hardware.
  FALSIFIED it OOMs from a cold process. Then mem_shape's 64.00 GiB was a lazy reservation
            that never faulted its pages, the honest wall is 32, and we say so.

Either answer is publishable. What is not publishable is quoting a 64 GiB single block and a
32-qubit ceiling on the same page without explaining how both can hold.
"""
from __future__ import annotations
import time
import torch

N_QUBITS = 33


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: no ROCm device visible"); return 3
    dev = torch.device("cuda")
    print("device         :", torch.cuda.get_device_name(0))
    print("arch           :", getattr(torch.cuda.get_device_properties(0), "gcnArchName", "?"))
    free, total = torch.cuda.mem_get_info()
    print("mem free/total : %.2f / %.2f GiB   (reported; known to under-report on this part)"
          % (free / 2**30, total / 2**30))
    need = (1 << N_QUBITS) * 8 / 2**30
    print("state needed   : %.2f GiB at %d qubits" % (need, N_QUBITS))
    print("\n-- allocating the state as the FIRST allocation in this process")

    t0 = time.perf_counter()
    try:
        psi = torch.zeros(1 << N_QUBITS, dtype=torch.complex64, device=dev)
    except RuntimeError as e:
        print("\nFALSIFIED: cold allocation refused.")
        print("  %s" % str(e).splitlines()[0][:160])
        print("\n  So mem_shape's 64.00 GiB block was a reservation that never faulted its")
        print("  pages. The honest ceiling on this configuration is 32 qubits.")
        return 1
    torch.cuda.synchronize()
    print("   allocated in %.1f s" % (time.perf_counter() - t0))

    # A state, not a zero vector: |0...0>. Norm must be 1 before we touch it.
    psi[0] = 1
    torch.cuda.synchronize()
    print("   norm before  : %.7f" % psi.norm().item())

    # One Hadamard on qubit 0 -- the same inner loop the sweep times. Written out rather than
    # imported so this probe depends on nothing but torch and cannot fail for a second reason.
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device=dev) / (2 ** 0.5)
    print("-- one gate")
    t1 = time.perf_counter()
    v = psi.view(-1, 2, 1)
    a, b = v[:, 0, :], v[:, 1, :]
    na = h[0, 0] * a + h[0, 1] * b
    nb = h[1, 0] * a + h[1, 1] * b
    v[:, 0, :], v[:, 1, :] = na, nb
    torch.cuda.synchronize()
    dt = time.perf_counter() - t1

    norm = psi.norm().item()
    print("   gate         : %.2f s" % dt)
    print("   norm after   : %.7f" % norm)
    ok = abs(norm - 1.0) < 1e-5
    print("\n%s: %d qubits on an integrated GPU%s"
          % ("PASS" if ok else "FAIL", N_QUBITS,
             "" if ok else " -- state allocated but the gate did not preserve the norm"))
    if ok:
        print("  The 32-qubit wall was our own allocator holding earlier widths, not the hardware.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
