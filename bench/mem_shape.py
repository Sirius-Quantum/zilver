"""Is the 64 GiB ceiling PER ALLOCATION or TOTAL? Two different worlds.

    python3 bench/mem_shape.py

The device reports 49.12 GiB and hands out 63.98 in one block -- suspiciously
exactly 64 GiB, a round power of two, which is not the shape of a fraction of
117 GB. Two possibilities, and they lead to different engineering:

  per-allocation cap : the device has more, no single block may exceed 64 GiB.
                       33 qubits is then reachable by carrying the state as two
                       32 GiB blocks. Work, but possible.
  total cap          : the GPU can address 64 GiB of the 117 and 32 qubits is
                       the end of the road here without a BIOS change.

So: find the exact single-block ceiling, then keep allocating smaller blocks and
see whether the TOTAL goes past it. The second number is the one that decides.
"""
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("no device")

GiB = 2 ** 30
free, total = torch.cuda.mem_get_info()
print(f"reported total {total/GiB:.2f} GiB   free {free/GiB:.2f} GiB\n")


def can_alloc(gib):
    try:
        t = torch.empty(int(gib * GiB / 8), dtype=torch.complex64, device="cuda")
        del t
        torch.cuda.empty_cache()
        return True
    except Exception:
        torch.cuda.empty_cache()
        return False


# --- 1. the exact single-block ceiling, to 0.01 GiB -------------------------
lo, hi = 0.0, 200.0
while hi - lo > 0.01:
    mid = (lo + hi) / 2
    if can_alloc(mid):
        lo = mid
    else:
        hi = mid
print(f"largest SINGLE block      : {lo:.2f} GiB")
print(f"  33 qubits needs           64.00 GiB  -> {'fits' if lo >= 64.0 else 'short by %.2f GiB' % (64.0 - lo)}")

# --- 2. how much in aggregate, in blocks that each fit ----------------------
for block in (32, 16, 8, 4):
    held, n = [], 0
    try:
        while n * block < 200:
            held.append(torch.empty(int(block * GiB / 8),
                                    dtype=torch.complex64, device="cuda"))
            n += 1
    except Exception:
        pass
    got = n * block
    del held
    torch.cuda.empty_cache()
    print(f"aggregate in {block:>2} GiB blocks : {got:6.0f} GiB  ({n} blocks)")

print("""
READ: if the aggregate clearly exceeds the single-block number, the cap is PER
ALLOCATION and a segmented statevector reaches 33. If aggregate stops at the
same place, the device simply has that much and the rest of the 117 GB is not
addressable from the GPU on this configuration.""")
