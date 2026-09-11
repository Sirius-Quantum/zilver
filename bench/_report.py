"""Shared plumbing for the technical-report benches (report_qft, report_random, report_timing).

Every gate goes through zilver.simulator._hip_apply -- the path the simulator itself takes on
ROCm -- and a bench STOPS if that path declines a gate, so no number here can come from a
fallback path while printing the kernel's name.

ZILVER_REPORT_DRY=mirror|dense runs the same bench on numpy, so the bench's own logic (qubit
conventions, closed forms, the Aer mapping) is checked on a laptop before any box time:
  mirror  the kernel's algorithm (fused.fused_apply_reference) with _hip_apply's argument prep
  dense   an independent tensordot apply that shares no code with the kernel
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

DRY = os.environ.get("ZILVER_REPORT_DRY", "")
K_SAMPLES = int(os.environ.get("ZILVER_REPORT_K", "65536"))


def widths(default):
    """ZILVER_REPORT_WIDTHS=8,10 overrides; otherwise the default list clipped to FROM..TO."""
    w = os.environ.get("ZILVER_REPORT_WIDTHS")
    if w:
        return [int(x) for x in w.split(",")]
    lo, hi = int(os.environ.get("FROM", "0")), int(os.environ.get("TO", "99"))
    return [n for n in default if lo <= n <= hi]


def out_dir():
    """Outside the zilver copy: the runner deletes that copy on every invocation."""
    d = os.environ.get("ZILVER_REPORT_OUT")
    if not d:
        home = os.environ.get("USERPROFILE")
        if home and not DRY:
            d = os.path.join(home, "siriusq-rocm", "report_out")
        else:
            d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_report_out")
    os.makedirs(d, exist_ok=True)
    return d


def write_json(name, obj):
    p = os.path.join(out_dir(), name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print(f"  -> {p}")


def verdict(tag, ok, detail):
    print(f"  VERDICT {tag:<34} {'PASS' if ok else 'FAIL'}   {detail}", flush=True)
    return bool(ok)


# ---------------------------------------------------------------------------
# Gates, complex128. Zilver convention: qubit 0 is the MOST significant bit of the flat
# index, and a 2q gate's row index is 2*(bit of qubits[0]) + (bit of qubits[1]).
# ---------------------------------------------------------------------------
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)
CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)


def RY(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def CP(phi):
    return np.diag([1, 1, 1, np.exp(1j * phi)]).astype(np.complex128)


def haar(rng):
    z = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))


def sample_indices(n, seed, k=None):
    """Seeded, sorted, unique; always includes the first and last amplitude."""
    k = K_SAMPLES if k is None else k
    rng = np.random.default_rng(seed)
    N = 1 << n
    idx = rng.integers(0, N, size=min(k, N), dtype=np.uint64)
    idx = np.unique(np.concatenate([idx, np.array([0, N - 1], dtype=np.uint64)]))
    return idx


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class Hip:
    """The Radeon, through the same _hip_apply the simulator routes every gate through."""

    name = "hip"

    def __init__(self):
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            sys.exit("FAIL: no ROCm device visible")
        from zilver import hip_ext, simulator
        t = time.perf_counter()
        if hip_ext.kernel(verbose=False) is None:
            sys.exit("FAIL: the fused kernel did not compile -- nothing here would be the kernel")
        self.compile_s = time.perf_counter() - t
        self._apply = simulator._hip_apply
        self.dev = torch.device("cuda")
        p = torch.cuda.get_device_properties(0)
        self.info = {
            "device": torch.cuda.get_device_name(0),
            "arch": getattr(p, "gcnArchName", "?"),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "python": platform.python_version(),
            "compile_s": round(self.compile_s, 3),
        }

    def zeros(self, n):
        return self.torch.zeros(1 << n, dtype=self.torch.complex64, device=self.dev)

    def randn(self, n):
        return self.torch.randn(1 << n, dtype=self.torch.complex64, device=self.dev)

    def basis(self, n, x):
        s = self.zeros(n)
        s[int(x)] = 1.0
        return s

    def apply(self, s, g, qubits, n):
        g = np.ascontiguousarray(np.asarray(g, dtype=np.complex64))
        if not self._apply(s, g, list(qubits), n):
            raise RuntimeError(f"_hip_apply declined a gate on {list(qubits)} at n={n}; "
                               "refusing to measure a fallback path")

    def sync(self):
        self.torch.cuda.synchronize()

    def sample(self, s, idx):
        t = self.torch.as_tensor(idx.astype(np.int64), dtype=self.torch.int64, device=self.dev)
        return s[t].cpu().numpy()

    def norm(self, s, chunk=1 << 26):
        """Chunked, fp64 accumulation: a whole-state reduction would allocate a temporary."""
        tot = 0.0
        for i in range(0, s.numel(), chunk):
            c = self.torch.view_as_real(s[i:i + chunk]).double()
            tot += float((c * c).sum())
        return tot ** 0.5

    def to_host(self, s):
        return s.cpu().numpy()

    def copy_(self, dst, src):
        dst.copy_(src)

    def stream_(self, s):
        """One read and one write of every amplitude, in place: the gate's own footprint."""
        self.torch.view_as_real(s).neg_()

    def empty_like(self, s):
        return self.torch.empty_like(s)

    def free(self):
        self.sync()
        self.torch.cuda.empty_cache()

    def mem(self):
        f, t = self.torch.cuda.mem_get_info()
        return {"free_gib": f / 2**30, "total_gib": t / 2**30,
                "max_allocated_gib": self.torch.cuda.max_memory_allocated() / 2**30}


class Numpy:
    """Laptop stand-in. Same bench, numpy state; `mode` picks the apply."""

    def __init__(self, mode):
        self.name = f"dry-{mode}"
        self.mode = mode
        self.compile_s = 0.0
        self.info = {"device": self.name, "python": platform.python_version()}

    def zeros(self, n):
        return np.zeros(1 << n, dtype=np.complex64)

    def randn(self, n):
        r = np.random.default_rng(0)
        return (r.standard_normal(1 << n) + 1j * r.standard_normal(1 << n)).astype(np.complex64)

    def basis(self, n, x):
        s = self.zeros(n)
        s[int(x)] = 1.0
        return s

    def apply(self, s, g, qubits, n):
        g = np.asarray(g, dtype=np.complex64)
        k = len(qubits)
        if self.mode == "dense":
            t = s.reshape([2] * n)
            gt = g.reshape([2] * (2 * k))
            out = np.tensordot(gt, t, axes=(list(range(k, 2 * k)), list(qubits)))
            out = np.moveaxis(out, list(range(k)), list(qubits))
            s[:] = out.reshape(-1)
            return
        # mirror: _hip_apply's argument preparation, then the kernel's algorithm in numpy
        from zilver.fused import fused_apply_reference, pivots_for
        piv, masks, order = pivots_for([1 << (n - 1 - q) for q in qubits])
        gate_bit = [k - 1 - i for i in range(k)]
        perm = np.empty(1 << k, dtype=np.int64)
        for loc in range(1 << k):
            gi = 0
            for b in range(k):
                if (loc >> b) & 1:
                    gi |= 1 << gate_bit[order[b]]
            perm[loc] = gi
        U = np.ascontiguousarray(g.reshape(1 << k, 1 << k)[np.ix_(perm, perm)])
        fused_apply_reference(s, n, masks, piv, [0] * k, U)

    def sync(self):
        pass

    def sample(self, s, idx):
        return s[idx.astype(np.int64)].copy()

    def norm(self, s, chunk=None):
        return float(np.sqrt(np.sum(np.abs(s.astype(np.complex128)) ** 2)))

    def to_host(self, s):
        return s.copy()

    def copy_(self, dst, src):
        dst[:] = src

    def stream_(self, s):
        np.negative(s, out=s)

    def empty_like(self, s):
        return np.empty_like(s)

    def free(self):
        pass

    def mem(self):
        return {}


def backend():
    return Numpy(DRY) if DRY else Hip()
