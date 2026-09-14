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
import re
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


def _maxm(default=5):
    """The widest fusion the kernel budgets registers for.

    Read it OUT of the kernel rather than restating it here. The number exists in
    fused_gate.hip as `#define MAXM`, and a second copy in Python is a copy that drifts: raise
    MAXM there and this file would keep refusing the widths the kernel had just gained.
    """
    found = re.search(r"#define\s+MAXM\s+(\d+)", _SRC.read_text())
    return int(found.group(1)) if found else default


_MAXM = _maxm()


def _specialise(src, m):
    """Bake the fused width in at COMPILE time.

    The kernel takes m as a runtime argument, so `D = 1 << m` is a runtime trip count and the
    per-thread arrays -- idx, amp, tmp, out, about 1 KB together -- cannot be promoted to
    registers. They were spilled to scratch, which is global memory, and spill traffic scales as
    2^m. Measured on gfx1151: 800 bytes of scratch on SEVEN registers, at every m. The comment
    in fused_gate.hip claiming the arithmetic happens "in registers" was never true.

    hiprtc compiles from a string, so the fix is a string. Measured at n=32 after this change:

        m    copies/pass before -> after     scratch     regs    waves/SIMD
        1         1.24  ->  0.98                 0 B       19     16 (full)
        2         1.30  ->  1.02                 0 B       41     16 (full)
        3         1.89  ->  1.11                96 B       70     16 (full)
        4         2.82  ->  1.32               160 B      119     12
        5         4.76  ->  1.82               288 B      192      8 (half)

    End to end on the 95-gate circuit at 32 qubits: 1.32 -> 0.33 copies per gate, 4.0x. m=1 at
    0.98 is the floor -- one read and one write of every amplitude, which is all a gate may do.

    Occupancy falls out of the register column and caps the useful width: do not schedule around
    m=5, where residency halves. m=3 is the widest that costs nothing.

    Verified bit-identical to the unspecialised kernel at every m (0.000e+00), against a planted
    closed form to 1.2e-10, and with a planted fault that fails the same check at 100% error.
    """
    out = src
    for old, new in (("const int D = 1 << m;", "const int D = 1 << M;"),
                     ("b < m; ++b", "b < M; ++b"),
                     ("[1 << MAXM]", "[1 << M]")):
        out = out.replace(old, new)

    # Check the OUTPUT, not the input. `b < m; ++b` occurs five times, so a kernel in which one
    # loop had been rewritten differently would still match on the other four: an input check
    # passes, and the result is a HALF-specialised kernel that compiles, still spills, and looks
    # like it worked. A post-condition cannot be fooled that way -- if any runtime trip count or
    # MAXM-sized declaration survives, the specialisation did not do its job.
    for leftover in ("1 << m", "b < m;", "1 << MAXM"):
        if leftover in out:
            raise RuntimeError(
                f"fused_gate.hip has changed shape: {leftover!r} survived specialisation, "
                f"so the kernel would still spill to scratch")
    return f"#define M {m}\n" + out


def _compile(m, verbose=False):
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
    src = _specialise(src, m)

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

    # ONE MODULE PER FUSED WIDTH, because m is now a compile-time constant. Each build is
    # ~0.3 s through hiprtc and the simulator only ever asks for 1 and 2, so widths are
    # compiled on first use and cached rather than all five up front.
    _fns = {}

    def _fn_for(m):
        if m in _fns:
            return _fns[m]
        if not 1 <= m <= _MAXM:
            print(f"[zilver] fused width m={m} outside 1..{_MAXM}")
            _fns[m] = None
            return None
        _fns[m] = None                     # cache the failure too: do not retry a bad compile
        code, err = _compile(m, verbose)
        if code is None:
            print(f"[zilver] hiprtc unavailable (m={m}): {err}")
            return None
        mod = ctypes.c_void_p()
        rc = hip.hipModuleLoadData(ctypes.byref(mod), code)
        if rc != 0:
            print(f"[zilver] hipModuleLoadData (m={m}) -> {rc}")
            return None
        fn = ctypes.c_void_p()
        rc = hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"fused_apply")
        if rc != 0:
            print(f"[zilver] hipModuleGetFunction (m={m}) -> {rc}")
            return None
        _fns[m] = fn
        return fn

    # Build m=1 eagerly. It is the width every one-qubit gate uses, and it turns a broken
    # toolchain into None HERE, where _hip_apply's caller still has a correct fallback, instead
    # of into an exception on the first gate of a run.
    if _fn_for(1) is None:
        return None

    hip.hipModuleLaunchKernel.restype = ctypes.c_int
    hip.hipModuleLaunchKernel.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
    ]

    def launch(state, masks, pivots, rows, U, m):
        fn = _fn_for(m)
        if fn is None:
            raise RuntimeError(f"no fused kernel compiled for m={m}")
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
