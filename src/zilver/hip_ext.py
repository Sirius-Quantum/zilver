"""Compile and bind src/zilver/hip/fused_gate.hip, if this box has ROCm.

ROCm's pip wheels carry hipcc, so torch's extension builder compiles for
gfx1151 with no HIP SDK and no administrator. Everything here is best-effort:
`kernel()` returns None on any machine that cannot build it, and the caller
falls back to the torch path.
"""

from __future__ import annotations

import os
import pathlib

_SRC = pathlib.Path(__file__).parent / "hip" / "fused_gate.hip"
_MOD = None
_TRIED = False

# A thin shim so the kernel is reachable from Python with tensors. The launch
# geometry is the only decision here: one thread per coset, 2^m amplitudes each.
_SHIM = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>

extern "C" __global__ void fused_apply(
    float2*, const unsigned long long*, const int*,
    const unsigned long long*, const float2*, int, unsigned long long);

void fused(torch::Tensor state, torch::Tensor masks, torch::Tensor pivots,
           torch::Tensor rows, torch::Tensor U, int64_t m) {
  TORCH_CHECK(state.is_cuda(), "state must live on the device");
  TORCH_CHECK(state.scalar_type() == torch::kComplexFloat, "state must be complex64");
  TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
  const unsigned long long n_cosets = state.numel() >> m;
  const int threads = 256;
  const unsigned long long blocks = (n_cosets + threads - 1) / threads;
  hipLaunchKernelGGL(fused_apply, dim3(blocks), dim3(threads), 0,
      c10::cuda::getCurrentCUDAStream(),
      reinterpret_cast<float2*>(state.data_ptr()),
      reinterpret_cast<const unsigned long long*>(masks.data_ptr()),
      pivots.data_ptr<int>(),
      reinterpret_cast<const unsigned long long*>(rows.data_ptr()),
      reinterpret_cast<const float2*>(U.data_ptr()),
      static_cast<int>(m), n_cosets);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mm) { mm.def("fused", &fused); }
"""


def available() -> bool:
    """Is this a ROCm torch with a visible device? MPS and CUDA both say no."""
    try:
        import torch
        return bool(torch.version.hip) and torch.cuda.is_available()
    except Exception:
        return False


def kernel():
    """The compiled module, or None. Built once; torch caches it on disk."""
    global _MOD, _TRIED
    if _TRIED:
        return _MOD
    _TRIED = True
    if not available():
        return None
    try:
        from torch.utils.cpp_extension import load_inline
        _MOD = load_inline(
            name="zilver_fused",
            cpp_sources=[_SHIM],
            cuda_sources=[_SRC.read_text()],
            functions=["fused"],
            with_cuda=True,
            extra_cuda_cflags=["-O3"],
            verbose=bool(os.environ.get("ZILVER_BUILD_VERBOSE")),
        )
    except Exception as exc:                     # no compiler is an answer, not a crash
        _MOD = None
        if os.environ.get("ZILVER_BUILD_VERBOSE"):
            print(f"[zilver] HIP extension unavailable: {type(exc).__name__}: {exc}")
    return _MOD
