"""The array layer, and the one place that knows whether MLX is present.

Zilver was written against ``mlx.core``, which ships wheels only for Apple
silicon. Every module that touched an array therefore imported ``mlx.core``
at top level, which made the whole package unimportable on x86 -- not slow,
unimportable -- and confined the node fleet to Macs.

This module is the seam. It exports one name, ``mx``:

* on Apple silicon it IS ``mlx.core``, so the Metal path is untouched;
* everywhere else it is a numpy-backed stand-in covering the surface Zilver
  actually uses (~40 symbols, enumerated below).

Callers write ``from ._array import mx`` and are otherwise unchanged.

WHAT THE FALLBACK DOES NOT DO. It is not a reimplementation of MLX. Three
behaviours differ, and each is safe in exactly one direction:

* **Laziness.** MLX builds a graph and evaluates on demand; numpy is eager.
  ``mx.eval`` therefore becomes a no-op, which is correct -- the values are
  already materialised -- and ``mx.compile`` becomes identity, which costs
  the kernel fusion MLX would have done but changes no result.
* **Device placement.** ``mx.cpu``/``mx.gpu``/``Stream``/``Device`` become
  inert tokens. There is no second device to place anything on.
* **Metal kernels.** ``mx.fast.metal_kernel`` raises. That is deliberate:
  callers must fall back to the numpy gate path rather than silently get a
  wrong answer, and :data:`HAS_MLX` lets them choose before they try.

``HAS_MLX`` is the public flag. Prefer it to catching ImportError.
"""

from __future__ import annotations

import os

import numpy as np

__all__ = ["mx", "HAS_MLX"]

def _torch_backend():
    """mlx.core, backed by torch, so the statevector lives on whatever device
    torch can see -- CUDA, ROCm, or Apple MPS.

    This is the GPU path and it needs no gate kernels. simulator.apply_gate is
    written entirely against `mx` (transpose, matmul, reshape), so swapping the
    array layer moves the whole simulation onto the device unchanged. On an
    integrated GPU with unified memory the state is never copied: it is
    allocated once in shared memory and the device addresses it in place.

    Why this is the right precision story on RDNA 3.5 and Apple alike: a
    statevector is complex64, i.e. fp32 arithmetic, which consumer GPUs do
    well. It is fp64 they cripple -- which is why quantum chemistry does not
    belong here and simulation does.

    Selected with ZILVER_BACKEND=torch; device with ZILVER_DEVICE
    (default: cuda if visible -- ROCm reports as cuda -- else mps, else cpu).
    """
    import torch

    dev = os.environ.get("ZILVER_DEVICE")
    DEVICE = None
    if dev is None:
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            DEVICE = torch.device("mps")
        else:
            # DirectML: the route to an AMD or Intel GPU on Windows and inside
            # WSL2, where ROCm's /dev/kfd does not exist but DirectX's /dev/dxg
            # does. Not a torch.device string -- torch_directml hands back its
            # own device object.
            try:
                import torch_directml
                if torch_directml.device_count() > 0:
                    DEVICE = torch_directml.device()
            except ImportError:
                pass
            if DEVICE is None:
                DEVICE = torch.device("cpu")
    elif dev == "directml":
        import torch_directml
        DEVICE = torch_directml.device()
    else:
        DEVICE = torch.device(dev)

    class _ArrMeta(type):
        def __instancecheck__(cls, obj):
            return isinstance(obj, torch.Tensor)

    class _arr(metaclass=_ArrMeta):
        def __new__(cls, x, dtype=None, **_k):
            if isinstance(x, torch.Tensor):
                return x.to(dtype=dtype, device=DEVICE) if dtype else x.to(DEVICE)
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=DEVICE)

    class _Dev:
        __slots__ = ("name",)
        def __init__(self, name): self.name = name

    class _Fast:
        @staticmethod
        def metal_kernel(*_a, **_k):
            def _u(*_x, **_y):
                raise RuntimeError("Metal kernels need MLX on Apple silicon.")
            return _u

    class _Metal:
        @staticmethod
        def get_peak_memory():
            return int(torch.cuda.max_memory_allocated()) if DEVICE.type == "cuda" else 0
        @staticmethod
        def reset_peak_memory():
            if DEVICE.type == "cuda": torch.cuda.reset_peak_memory_stats()
        @staticmethod
        def is_available(): return DEVICE.type != "cpu"

    class _TorchMX:
        device = DEVICE
        complex64, complex128 = torch.complex64, torch.complex128
        float32, float64 = torch.float32, torch.float64
        int32, int64, uint32 = torch.int32, torch.int64, torch.int32
        cpu, gpu = _Dev("cpu"), _Dev("gpu")
        Device = Stream = _Dev
        fast, metal = _Fast(), _Metal()
        array = _arr

        @staticmethod
        def eval(*_a, **_k):
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            elif DEVICE.type == "mps": torch.mps.synchronize()

        @staticmethod
        def compile(fn=None, **_k):
            return (lambda f: f) if fn is None else fn

        # transpose(x, perm) in MLX is a full permutation; torch.transpose
        # swaps exactly two axes, so it must map to permute instead.
        @staticmethod
        def transpose(x, axes=None, **_k):
            return x.permute(*axes) if axes is not None else x.t()

        @staticmethod
        def matmul(a, b, **_k): return torch.matmul(a, b)

        @staticmethod
        def zeros(shape, dtype=None, **_k):
            return torch.zeros(shape, dtype=dtype or torch.float32, device=DEVICE)

        @staticmethod
        def eye(n, m=None, dtype=None, **_k):
            return torch.eye(n, m or n, dtype=dtype or torch.float32, device=DEVICE)

        @staticmethod
        def arange(*a, dtype=None, **_k):
            return torch.arange(*a, dtype=dtype, device=DEVICE)

        @staticmethod
        def vmap(fn, in_axes=0, out_axes=0, **_k):
            def mapped(batch):
                return torch.stack([fn(r) for r in batch]) if len(batch) else torch.tensor([])
            return mapped

        @staticmethod
        def tolist(x): return x.detach().cpu().tolist()

        def __getattr__(self, name):
            attr = getattr(torch, name, None)
            if attr is None:
                raise AttributeError(f"torch has no {name!r} (zilver._array)")
            if not callable(attr):
                return attr
            def _f(*a, **k):
                k.pop("stream", None); k.pop("device", None)
                return attr(*a, **k)
            return _f

    return _TorchMX(), DEVICE


_WANT = os.environ.get("ZILVER_BACKEND", "").lower()

try:                                     # Apple silicon: the real thing.
    if _WANT == "torch":
        raise ImportError("ZILVER_BACKEND=torch")
    import mlx.core as _mlx              # type: ignore[import-not-found]
    mx = _mlx
    HAS_MLX = True
except ImportError:                      # everywhere else: numpy, or torch.
    HAS_MLX = False
    if _WANT == "torch":
        mx, TORCH_DEVICE = _torch_backend()

    class _Device:
        """Inert stand-in for mx.Device / mx.Stream. There is one device."""
        __slots__ = ("name",)

        def __init__(self, name: str) -> None:
            self.name = name

        def __repr__(self) -> str:       # pragma: no cover - debugging aid
            return f"Device({self.name})"

        def __eq__(self, other: object) -> bool:
            return isinstance(other, _Device) and other.name == self.name

        def __hash__(self) -> int:
            return hash(self.name)

    class _Fast:
        """mx.fast. metal_kernel is the only member Zilver uses.

        MLX's metal_kernel COMPILES at construction and runs when called, so
        zilver/metal.py builds its nine gate kernels at module import. If the
        stub raised on construction, importing zilver.metal would explode and
        the backend probe in circuit.py could not ask ``metal.supports()``
        without catching it. So construction succeeds and INVOCATION raises --
        the module stays importable and inspectable, and the failure lands
        only if something actually tries to run a Metal kernel on hardware
        that has none.
        """

        @staticmethod
        def metal_kernel(*_a, **_k):
            def _unavailable(*_args, **_kwargs):
                raise RuntimeError(
                    "this Metal gate kernel needs MLX on Apple silicon; "
                    "use method='accel' (numba) or method='mlx' instead. "
                    "zilver._array.HAS_MLX is False on this machine."
                )
            return _unavailable

    class _Metal:
        """mx.metal. Memory telemetry only; report zero rather than lie."""

        @staticmethod
        def get_peak_memory() -> int:
            return 0

        @staticmethod
        def reset_peak_memory() -> None:
            return None

        @staticmethod
        def is_available() -> bool:
            return False

    class _ArrayMeta(type):
        def __instancecheck__(cls, obj) -> bool:
            return isinstance(obj, np.ndarray)

    class _array_type(metaclass=_ArrayMeta):
        """mx.array: constructs an ndarray, isinstance-checks as ndarray."""
        def __new__(cls, x, dtype=None, **_dev):
            return np.asarray(x, dtype=dtype)

    class _NumpyMX:
        """The numpy-backed subset of mlx.core that Zilver calls.

        Attribute lookup falls through to numpy, so the many element-wise
        functions (sin, cos, sqrt, abs, sum, min, max, mean, var, real, imag,
        conj, matmul, transpose, concatenate, stack, zeros, zeros_like, eye,
        arange, expand_dims, ...) need no individual definition. Only the
        symbols whose MLX semantics differ from numpy's are written out.
        """

        # dtypes and devices -------------------------------------------------
        complex64 = np.complex64
        complex128 = np.complex128
        float32 = np.float32
        float64 = np.float64
        uint32 = np.uint32
        int32 = np.int32
        int64 = np.int64
        bool_ = np.bool_

        cpu = _Device("cpu")
        gpu = _Device("gpu")            # kept so `stream=mx.gpu` defaults parse
        Device = _Device
        Stream = _Device

        fast = _Fast()
        metal = _Metal()

        # eager/lazy ---------------------------------------------------------
        @staticmethod
        def eval(*_args, **_kwargs) -> None:
            """No-op: numpy is eager, so every value is already materialised."""
            return None

        @staticmethod
        def compile(fn=None, **_kwargs):
            """Identity. Forfeits MLX's kernel fusion; changes no result."""
            if fn is None:
                return lambda f: f
            return fn

        @staticmethod
        def async_eval(*_args, **_kwargs) -> None:
            return None

        # array construction -------------------------------------------------
        # mx.array is a CLASS in MLX, and Zilver uses it both ways: as a
        # constructor, and as a type in `isinstance(params, mx.array)` and in
        # `-> mx.array` annotations. A plain function satisfies the first and
        # breaks the second, so this is a class whose construction returns a
        # real ndarray and whose isinstance check defers to np.ndarray.
        array = _array_type

        # Each takes **_dev and discards it: MLX threads stream=/device= through
        # its constructors too, and these are defined explicitly (rather than
        # reaching numpy via __getattr__) so they never see that wrapper.
        @staticmethod
        def zeros(shape, dtype=None, **_dev):
            return np.zeros(shape, dtype=dtype if dtype is not None else np.float32)

        @staticmethod
        def ones(shape, dtype=None, **_dev):
            return np.ones(shape, dtype=dtype if dtype is not None else np.float32)

        @staticmethod
        def eye(n, m=None, dtype=None, **_dev):
            return np.eye(n, m, dtype=dtype if dtype is not None else np.float32)

        @staticmethod
        def arange(*args, dtype=None, **_dev):
            return np.arange(*args, dtype=dtype)

        # the one real function ----------------------------------------------
        @staticmethod
        def vmap(fn, in_axes=0, out_axes=0, **_dev):
            """Map fn over one axis and restack.

            MLX vectorises into a single fused kernel; numpy cannot, so this
            loops. Zilver uses vmap only to evaluate a circuit across a batch
            of parameter vectors, where the loop is the honest cost of not
            having a GPU -- the results are identical.
            """
            if in_axes != 0 or out_axes != 0:
                raise NotImplementedError(
                    "the numpy fallback of mx.vmap supports in_axes=out_axes=0 only"
                )

            def mapped(batch):
                arr = np.asarray(batch)
                if arr.shape[0] == 0:
                    return np.asarray([])
                return np.stack([np.asarray(fn(row)) for row in arr], axis=0)

            return mapped

        @staticmethod
        def tolist(x):
            return np.asarray(x).tolist()

        # everything else is numpy -------------------------------------------
        def __getattr__(self, name: str):
            """Fall through to numpy, dropping MLX's device kwargs.

            Nearly every mlx.core op accepts ``stream=`` (and some ``device=``)
            to place work on a queue. numpy has one device and no queues, so
            the kwargs are meaningless here -- but they are passed positionally
            through Zilver's call sites (``mx.matmul(a, b, stream=stream)``),
            so silently dropping them is what keeps those sites unchanged.
            Anything callable therefore comes back wrapped; dtypes and
            constants come back as they are.
            """
            try:
                attr = getattr(np, name)
            except AttributeError as exc:
                raise AttributeError(
                    f"zilver's numpy fallback for mlx.core has no {name!r}. "
                    f"Add it to zilver/_array.py if a code path needs it."
                ) from exc

            if not callable(attr):
                return attr

            def _no_stream(*args, **kwargs):
                kwargs.pop("stream", None)
                kwargs.pop("device", None)
                return attr(*args, **kwargs)

            _no_stream.__name__ = name
            _no_stream.__doc__ = getattr(attr, "__doc__", None)
            return _no_stream

    mx = _NumpyMX()
