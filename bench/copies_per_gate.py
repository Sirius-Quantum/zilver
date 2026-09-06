"""The instrument: measure everything in COPIES, not GB/s.

One copy = read the state once + write it once = 2 * 8 * 2^n bytes.
That is the theoretical floor for any gate that must touch every amplitude.

We never need to know the device's true peak bandwidth (nobody does, for an
APU on shared LPDDR5X). We measure a bare tensor copy of the SAME array at the
SAME size, call that 1.00 copies, and report every gate as a ratio. Achieved
bandwidth cancels out. The number is portable Mac <-> AMD <-> NVIDIA.

Run: python3 copies_per_gate.py [n]
Device: picks mps / cuda / cpu.
"""
import sys
import time
import torch

n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
dev = ("mps" if torch.backends.mps.is_available()
       else "cuda" if torch.cuda.is_available() else "cpu")
N = 1 << n
state_bytes = N * 8


def sync():
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def timeit(fn, reps=10, warm=3):
    for _ in range(warm):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - t0) / reps


s = torch.randn(N, dtype=torch.complex64, device=dev)
dst = torch.empty_like(s)

# --- the unit: one read + one write of the state, nothing else -------------
t_copy = timeit(lambda: dst.copy_(s))
print(f"device {dev}  n={n}  state {state_bytes/2**30:.3f} GiB")
print(f"reference copy  {t_copy*1e3:8.2f} ms  "
      f"= {2*state_bytes/t_copy/1e9:6.1f} GB/s  == 1.00 copies\n")

# --- arm A: what zilver's torch backend does now ---------------------------
G = torch.tensor([[0.7071, 0.7071], [0.7071, -0.7071]],
                 dtype=torch.complex64, device=dev)


def strided_gate(q):
    stride = 1 << (n - 1 - q)
    left = 1 << q
    v = s.reshape(left, 2, stride)
    a, b = v[:, 0, :], v[:, 1, :]
    o0 = G[0, 0] * a + G[0, 1] * b
    o1 = G[1, 0] * a + G[1, 1] * b
    return torch.stack([o0, o1], dim=1).reshape(-1)


# --- arm B: the floor an in-place kernel should hit -------------------------
# not a gate, but the exact traffic pattern of a stride-2^p gather/scatter:
# read two half-arrays at stride, write two half-arrays at stride.
def stride_traffic(q):
    stride = 1 << (n - 1 - q)
    left = 1 << q
    v = s.reshape(left, 2, stride)
    d = dst.reshape(left, 2, stride)
    d[:, 0, :] = v[:, 1, :]
    d[:, 1, :] = v[:, 0, :]


print(" q   stride(amps)   strided-arith        pure stride traffic")
print("                   ms    copies/gate     ms    copies/gate")
for q in [0, 1, 4, n // 2, n - 6, n - 4, n - 2, n - 1]:
    if not (0 <= q < n):
        continue
    ta = timeit(lambda q=q: strided_gate(q))
    tb = timeit(lambda q=q: stride_traffic(q))
    print(f"{q:3d}  {1<<(n-1-q):12d}  {ta*1e3:7.2f}  {ta/t_copy:8.2f}    "
          f"{tb*1e3:7.2f}  {tb/t_copy:8.2f}")

print("\nREAD: copies/gate is the number to drive down. 1.00 is the floor for")
print("one unfused gate. Anything above 1.00 is temporaries. Fusing m gates")
print("into one pass targets 1/m. Speedup available = (current)/(1/m).")
