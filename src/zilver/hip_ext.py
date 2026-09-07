"""Run src/zilver/hip/fused_gate.hip on the device, with no build toolchain.

The obvious route -- torch.utils.cpp_extension -- drags in ninja, pybind11, and
on Windows an MSVC host compiler for the C++ shim. Visual Studio Build Tools
needs administrator, which we do not have on a borrowed box, so that route has a
wall in it that no amount of pip can move.

But the shim only ever existed to hand the kernel a device pointer, and torch
already gives us one: `tensor.data_ptr()`. So compile the kernel at RUNTIME with
hiprtc and launch it through the HIP driver API, both reached by ctypes. No
ninja, no MSVC, no pybind11, no build directory -- the ROCm runtime that torch
already loaded is the whole dependency.

`kernel()` returns a callable, or None with a printed reason. Nothing here
raises into the caller: a missing library is an answer.
"""

from __future__ import annotations

import ctypes
import glob
import os
import pathlib
import sys

_SRC = pathlib.Path(__file__).parent / "hip" / "fused_gate.hip"
_K = None
_TRIED = False


def _load(names, symbol):
    """First library that loads AND exports `symbol`.

    Names are not trustworthy here: ROCm ships hiprtc-builtins0715.dll beside
    hiprtc0715.dll, the first is a bitcode blob with no API in it, and a glob
    sorts '-' before '0' so it wins. Probing for the symbol is the only check
    that means anything, and it costs nothing.
    """
    roots = []
    try:
        import torch
        roots.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except Exception:
        pass
    for p in sys.path:
        for sub in ("rocm_sdk_core", "_rocm_sdk_core"):
            d = os.path.join(p, sub)
            if os.path.isdir(d):
                roots += [d, os.path.join(d, "bin"), os.path.join(d, "lib")]
    roots += ["/opt/rocm/lib", "/opt/rocm/bin", ""]

    # A ROCm DLL pulls in siblings, and ctypes will not find them unless the
    # directory is registered. Silent on Linux, essential on Windows.
    if hasattr(os, "add_dll_directory"):
        for root in roots:
            if root and os.path.isdir(root):
                try:
                    os.add_dll_directory(root)
                except OSError:
                    pass

    tried = []
    for stem in names:
        for root in roots:
            pat = os.path.join(root, stem) if root else stem
            cands = sorted(glob.glob(pat)) if any(c in stem for c in "*?") else [pat]
            for cand in cands:
                try:
                    lib = ctypes.CDLL(cand)
                except OSError:
                    continue
                if hasattr(lib, symbol):
                    return lib, cand
                tried.append(os.path.basename(cand))
    if tried:
        print(f"[zilver] loaded but no {symbol}: {', '.join(sorted(set(tried)))}")
    return None, None


def _arch():
    try:
        import torch
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        return "gfx1151"


def _compile(verbose=False):
    """hiprtc: source string in, device code object out. No host compiler."""
    rtc, rtc_path = _load(["hiprtc*.dll", "libhiprtc.so*", "amdhip64*.dll",
                           "libamdhip64.so*"], "hiprtcCreateProgram")
    if rtc is None:
        return None, "hiprtc library not found"
    if verbose:
        print(f"[zilver] hiprtc: {rtc_path}")

    # hiprtc supplies the HIP headers implicitly; including hip_runtime.h
    # confuses it, so drop that line and nothing else.
    src = "\n".join(l for l in _SRC.read_text().splitlines()
                    if "#include <hip/hip_runtime.h>" not in l)

    rtc.hiprtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
                                        ctypes.c_char_p, ctypes.c_int,
                                        ctypes.c_void_p, ctypes.c_void_p]
    rtc.hiprtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                         ctypes.POINTER(ctypes.c_char_p)]
    rtc.hiprtcGetCodeSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    rtc.hiprtcGetCode.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    rtc.hiprtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    rtc.hiprtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    prog = ctypes.c_void_p()
    rc = rtc.hiprtcCreateProgram(ctypes.byref(prog), src.encode(),
                                 b"fused_gate.hip", 0, None, None)
    if rc != 0:
        return None, f"hiprtcCreateProgram -> {rc}"

    opts = [f"--offload-arch={_arch()}".encode(), b"-O3"]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = rtc.hiprtcCompileProgram(prog, len(opts), arr)
    if rc != 0 or verbose:
        n = ctypes.c_size_t()
        rtc.hiprtcGetProgramLogSize(prog, ctypes.byref(n))
        if n.value > 1:
            buf = ctypes.create_string_buffer(n.value)
            rtc.hiprtcGetProgramLog(prog, buf)
            log = buf.value.decode(errors="replace").strip()
            if log:
                print("[zilver] hiprtc log:\n" + log)
    if rc != 0:
        return None, f"hiprtcCompileProgram -> {rc} (see log above)"

    n = ctypes.c_size_t()
    rtc.hiprtcGetCodeSize(prog, ctypes.byref(n))
    code = ctypes.create_string_buffer(n.value)
    rtc.hiprtcGetCode(prog, code)
    return code, None


def available() -> bool:
    try:
        import torch
        return bool(torch.version.hip) and torch.cuda.is_available()
    except Exception:
        return False


def kernel(verbose=None):
    """A callable (state, masks, pivots, rows, U, m) -> None, or None."""
    global _K, _TRIED
    if _TRIED:
        return _K
    _TRIED = True
    verbose = bool(os.environ.get("ZILVER_BUILD_VERBOSE")) if verbose is None else verbose
    if not available():
        return None

    code, err = _compile(verbose)
    if code is None:
        print(f"[zilver] hiprtc unavailable: {err}")
        return None

    hip, hip_path = _load(["amdhip64*.dll", "libamdhip64.so*"],
                          "hipModuleLaunchKernel")
    if hip is None:
        print("[zilver] HIP runtime not found")
        return None
    if verbose:
        print(f"[zilver] hip runtime: {hip_path}")

    import torch
    torch.zeros(1, device="cuda")          # force context creation before we use it

    hip.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                         ctypes.c_void_p, ctypes.c_char_p]
    mod = ctypes.c_void_p()
    rc = hip.hipModuleLoadData(ctypes.byref(mod), code)
    if rc != 0:
        print(f"[zilver] hipModuleLoadData -> {rc}")
        return None
    fn = ctypes.c_void_p()
    rc = hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"fused_apply")
    if rc != 0:
        print(f"[zilver] hipModuleGetFunction -> {rc}")
        return None

    hip.hipModuleLaunchKernel.restype = ctypes.c_int
    hip.hipModuleLaunchKernel.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
    ]

    def launch(state, masks, pivots, rows, U, m):
        n_cosets = state.numel() >> m
        threads = 256
        blocks = (n_cosets + threads - 1) // threads
        vals = [ctypes.c_void_p(state.data_ptr()),
                ctypes.c_void_p(masks.data_ptr()),
                ctypes.c_void_p(pivots.data_ptr()),
                ctypes.c_void_p(rows.data_ptr()),
                ctypes.c_void_p(U.data_ptr()),
                ctypes.c_int(int(m)),
                ctypes.c_ulonglong(int(n_cosets))]
        params = (ctypes.c_void_p * len(vals))(
            *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        rc = hip.hipModuleLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0,
                                       stream, params, None)
        if rc != 0:
            raise RuntimeError(f"hipModuleLaunchKernel -> {rc}")

    _K = launch
    return _K
