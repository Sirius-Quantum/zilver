"""Apple Silicon native gate kernels — the Zilver Metal path.

Custom Metal compute kernels dispatched via :func:`mlx.core.fast.metal_kernel`.
Each gate is one kernel call on ``2^(n-1)`` (1q gates) or ``2^n`` (2q gates,
phase gates) GPU threads. When the whole circuit is wrapped in
:func:`mlx.core.compile`, MLX fuses kernel launches and reuses buffers,
removing the per-gate allocation overhead that bottlenecks naïve usage.

Performance on a 16 GB M1 Pro for hardware-efficient depth-2 circuits
(min of 4 trials, milliseconds):

   n    metal+compile   qiskit-aer   speedup
   8        0.61            0.66       1.1×
  12        0.80            1.38       1.7×
  16        1.74            5.87       3.4×
  20       20.66           39.01       1.9×
  22       96.41          147.88       1.5×
  24      421.53          573.19       1.4×

The state is carried as a flat ``(2 * 2**n,)`` ``float32`` array with real and
imaginary parts interleaved (``state[2*k] = Re(amp_k)``, ``state[2*k+1] =
Im(amp_k)``). This sidesteps MLX's complex64 address-space quirks in custom
kernels while preserving exact complex64 semantics.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import mlx.core as mx


# ---------------------------------------------------------------------------
# Metal compute kernels (compiled once at module import)
# ---------------------------------------------------------------------------

_RY = mx.fast.metal_kernel(
    name="zilver_apply_ry",
    input_names=["state_in", "theta", "q", "n"],
    output_names=["state_out"],
    source="""
        uint id = thread_position_in_grid.x;
        uint nn = n[0];
        uint qq = q[0];
        uint stride = 1u << (nn - 1u - qq);
        uint l = id / stride;
        uint r = id - l * stride;
        uint i = l * 2u * stride + r;
        uint j = i + stride;
        float th2 = theta[0] * 0.5f;
        float c = metal::cos(th2);
        float s = metal::sin(th2);
        float ar = state_in[2u * i];
        float ai = state_in[2u * i + 1u];
        float br = state_in[2u * j];
        float bi = state_in[2u * j + 1u];
        state_out[2u * i]      = c * ar - s * br;
        state_out[2u * i + 1u] = c * ai - s * bi;
        state_out[2u * j]      = s * ar + c * br;
        state_out[2u * j + 1u] = s * ai + c * bi;
    """,
)

_RX = mx.fast.metal_kernel(
    name="zilver_apply_rx",
    input_names=["state_in", "theta", "q", "n"],
    output_names=["state_out"],
    source="""
        uint id = thread_position_in_grid.x;
        uint nn = n[0];
        uint qq = q[0];
        uint stride = 1u << (nn - 1u - qq);
        uint l = id / stride;
        uint r = id - l * stride;
        uint i = l * 2u * stride + r;
        uint j = i + stride;
        float th2 = theta[0] * 0.5f;
        float c = metal::cos(th2);
        float s = metal::sin(th2);
        float ar = state_in[2u * i];
        float ai = state_in[2u * i + 1u];
        float br = state_in[2u * j];
        float bi = state_in[2u * j + 1u];
        // RX = [[c, -is], [-is, c]]; multiplying:
        //   out_i = c*a - i s*b = (c*ar + s*bi) + i (c*ai - s*br)
        //   out_j = -i s*a + c*b = (s*ai + c*br) + i (-s*ar + c*bi)
        state_out[2u * i]      = c * ar + s * bi;
        state_out[2u * i + 1u] = c * ai - s * br;
        state_out[2u * j]      = s * ai + c * br;
        state_out[2u * j + 1u] = -s * ar + c * bi;
    """,
)

_RZ = mx.fast.metal_kernel(
    name="zilver_apply_rz",
    input_names=["state_in", "theta", "q", "n"],
    output_names=["state_out"],
    source="""
        uint id = thread_position_in_grid.x;
        uint nn = n[0];
        uint qq = q[0];
        uint bit = (id >> (nn - 1u - qq)) & 1u;
        float th2 = theta[0] * 0.5f;
        float c = metal::cos(th2);
        float s = metal::sin(th2);
        float pr = c;
        float pi_v = bit == 0u ? -s : s;
        float vr = state_in[2u * id];
        float vi = state_in[2u * id + 1u];
        state_out[2u * id]      = vr * pr - vi * pi_v;
        state_out[2u * id + 1u] = vr * pi_v + vi * pr;
    """,
)

_H = mx.fast.metal_kernel(
    name="zilver_apply_h",
    input_names=["state_in", "q", "n"],
    output_names=["state_out"],
    source="""
        uint id = thread_position_in_grid.x;
        uint nn = n[0];
        uint qq = q[0];
        uint stride = 1u << (nn - 1u - qq);
        uint l = id / stride;
        uint r = id - l * stride;
        uint i = l * 2u * stride + r;
        uint j = i + stride;
        float inv2 = 0.70710678118f;
        float ar = state_in[2u * i];
        float ai = state_in[2u * i + 1u];
        float br = state_in[2u * j];
        float bi = state_in[2u * j + 1u];
        state_out[2u * i]      = inv2 * (ar + br);
        state_out[2u * i + 1u] = inv2 * (ai + bi);
        state_out[2u * j]      = inv2 * (ar - br);
        state_out[2u * j + 1u] = inv2 * (ai - bi);
    """,
)

_X = mx.fast.metal_kernel(
    name="zilver_apply_x",
    input_names=["state_in", "q", "n"],
    output_names=["state_out"],
    source="""
        uint id = thread_position_in_grid.x;
        uint nn = n[0];
        uint qq = q[0];
        uint tmask = 1u << (nn - 1u - qq);
        // For thread id, output[id] = input[id ^ tmask]
        uint src = id ^ tmask;
        state_out[2u * id]      = state_in[2u * src];
        state_out[2u * id + 1u] = state_in[2u * src + 1u];
    """,
)

_CNOT = mx.fast.metal_kernel(
    name="zilver_apply_cnot",
    input_names=["state_in", "c_q", "t_q", "n"],
    output_names=["state_out"],
    source="""
        uint id = thread_position_in_grid.x;
        uint nn = n[0];
        uint cmask = 1u << (nn - 1u - c_q[0]);
        uint tmask = 1u << (nn - 1u - t_q[0]);
        uint src = id;
        if ((id & cmask) != 0u) {
            src = id ^ tmask;
        }
        state_out[2u * id]      = state_in[2u * src];
        state_out[2u * id + 1u] = state_in[2u * src + 1u];
    """,
)

_CZ = mx.fast.metal_kernel(
    name="zilver_apply_cz",
    input_names=["state_in", "c_q", "t_q", "n"],
    output_names=["state_out"],
    source="""
        uint id = thread_position_in_grid.x;
        uint nn = n[0];
        uint cmask = 1u << (nn - 1u - c_q[0]);
        uint tmask = 1u << (nn - 1u - t_q[0]);
        float sign = ((id & cmask) != 0u && (id & tmask) != 0u) ? -1.0f : 1.0f;
        state_out[2u * id]      = sign * state_in[2u * id];
        state_out[2u * id + 1u] = sign * state_in[2u * id + 1u];
    """,
)

_RZZ = mx.fast.metal_kernel(
    name="zilver_apply_rzz",
    input_names=["state_in", "theta", "qa", "qb", "n"],
    output_names=["state_out"],
    source="""
        uint id = thread_position_in_grid.x;
        uint nn = n[0];
        uint mask_a = 1u << (nn - 1u - qa[0]);
        uint mask_b = 1u << (nn - 1u - qb[0]);
        bool ba = (id & mask_a) != 0u;
        bool bb = (id & mask_b) != 0u;
        bool agree = ba == bb;
        float th2 = theta[0] * 0.5f;
        float c = metal::cos(th2);
        float s = metal::sin(th2);
        // bits agree (ZZ=+1)  ->  e^{-i th2}  = c - i s
        // bits differ (ZZ=-1) ->  e^{+i th2}  = c + i s
        float pr = c;
        float pi_v = agree ? -s : s;
        float vr = state_in[2u * id];
        float vi = state_in[2u * id + 1u];
        state_out[2u * id]      = vr * pr - vi * pi_v;
        state_out[2u * id + 1u] = vr * pi_v + vi * pr;
    """,
)

_U3 = mx.fast.metal_kernel(
    name="zilver_apply_u3",
    input_names=["state_in", "theta", "phi", "lam", "q", "n"],
    output_names=["state_out"],
    source="""
        uint id = thread_position_in_grid.x;
        uint nn = n[0];
        uint qq = q[0];
        uint stride = 1u << (nn - 1u - qq);
        uint l = id / stride;
        uint r = id - l * stride;
        uint i = l * 2u * stride + r;
        uint j = i + stride;
        float ct = metal::cos(theta[0] * 0.5f);
        float st = metal::sin(theta[0] * 0.5f);
        float cp = metal::cos(phi[0]);
        float sp = metal::sin(phi[0]);
        float cl = metal::cos(lam[0]);
        float sl = metal::sin(lam[0]);
        // U3 matrix elements
        // m00 = ct          (real)
        // m01 = -e^{i lam} * st     = (-cl*st, -sl*st)
        // m10 =  e^{i phi} * st     = ( cp*st,  sp*st)
        // m11 =  e^{i (phi+lam)} ct = (cos(phi+lam)*ct, sin(phi+lam)*ct)
        float m00r = ct,             m00i = 0.0f;
        float m01r = -cl * st,       m01i = -sl * st;
        float m10r =  cp * st,       m10i =  sp * st;
        float m11r = (cp*cl - sp*sl) * ct, m11i = (cp*sl + sp*cl) * ct;
        float ar = state_in[2u * i],     ai = state_in[2u * i + 1u];
        float br = state_in[2u * j],     bi = state_in[2u * j + 1u];
        // out_i = m00 * a + m01 * b
        state_out[2u * i]      = m00r*ar - m00i*ai + m01r*br - m01i*bi;
        state_out[2u * i + 1u] = m00r*ai + m00i*ar + m01r*bi + m01i*br;
        // out_j = m10 * a + m11 * b
        state_out[2u * j]      = m10r*ar - m10i*ai + m11r*br - m11i*bi;
        state_out[2u * j + 1u] = m10r*ai + m10i*ar + m11r*bi + m11i*br;
    """,
)


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def _grid_pairs(n: int):
    total = 1 << (n - 1)
    tg = min(256, total)
    return (total, 1, 1), (tg, 1, 1)


def _grid_all(n: int):
    total = 1 << n
    tg = min(256, total)
    return (total, 1, 1), (tg, 1, 1)


def _u32(x: int) -> mx.array: return mx.array([int(x)], dtype=mx.uint32)
def _f32(x: float) -> mx.array: return mx.array([float(x)], dtype=mx.float32)
def _f32_scalar(x: mx.array) -> mx.array:
    """Promote a 0-d MLX array to a 1-element float32 (for kernel arg)."""
    return mx.expand_dims(x, 0).astype(mx.float32) if x.ndim == 0 else x.astype(mx.float32)


def apply_ry(state, theta, q: int, n: int) -> mx.array:
    g, t = _grid_pairs(n)
    return _RY(inputs=[state, _f32_scalar(theta) if isinstance(theta, mx.array) else _f32(theta),
                       _u32(q), _u32(n)],
               output_shapes=[state.shape], output_dtypes=[mx.float32],
               grid=g, threadgroup=t)[0]


def apply_rx(state, theta, q: int, n: int) -> mx.array:
    g, t = _grid_pairs(n)
    return _RX(inputs=[state, _f32_scalar(theta) if isinstance(theta, mx.array) else _f32(theta),
                       _u32(q), _u32(n)],
               output_shapes=[state.shape], output_dtypes=[mx.float32],
               grid=g, threadgroup=t)[0]


def apply_rz(state, theta, q: int, n: int) -> mx.array:
    g, t = _grid_all(n)
    return _RZ(inputs=[state, _f32_scalar(theta) if isinstance(theta, mx.array) else _f32(theta),
                       _u32(q), _u32(n)],
               output_shapes=[state.shape], output_dtypes=[mx.float32],
               grid=g, threadgroup=t)[0]


def apply_h(state, q: int, n: int) -> mx.array:
    g, t = _grid_pairs(n)
    return _H(inputs=[state, _u32(q), _u32(n)],
              output_shapes=[state.shape], output_dtypes=[mx.float32],
              grid=g, threadgroup=t)[0]


def apply_x(state, q: int, n: int) -> mx.array:
    g, t = _grid_all(n)
    return _X(inputs=[state, _u32(q), _u32(n)],
              output_shapes=[state.shape], output_dtypes=[mx.float32],
              grid=g, threadgroup=t)[0]


def apply_cnot(state, c_q: int, t_q: int, n: int) -> mx.array:
    g, t = _grid_all(n)
    return _CNOT(inputs=[state, _u32(c_q), _u32(t_q), _u32(n)],
                 output_shapes=[state.shape], output_dtypes=[mx.float32],
                 grid=g, threadgroup=t)[0]


def apply_cz(state, c_q: int, t_q: int, n: int) -> mx.array:
    g, t = _grid_all(n)
    return _CZ(inputs=[state, _u32(c_q), _u32(t_q), _u32(n)],
               output_shapes=[state.shape], output_dtypes=[mx.float32],
               grid=g, threadgroup=t)[0]


def apply_rzz(state, theta, qa: int, qb: int, n: int) -> mx.array:
    g, t = _grid_all(n)
    return _RZZ(inputs=[state, _f32_scalar(theta) if isinstance(theta, mx.array) else _f32(theta),
                        _u32(qa), _u32(qb), _u32(n)],
                output_shapes=[state.shape], output_dtypes=[mx.float32],
                grid=g, threadgroup=t)[0]


def apply_u3(state, theta, phi, lam, q: int, n: int) -> mx.array:
    g, t = _grid_pairs(n)
    return _U3(inputs=[state,
                       _f32_scalar(theta) if isinstance(theta, mx.array) else _f32(theta),
                       _f32_scalar(phi)   if isinstance(phi,   mx.array) else _f32(phi),
                       _f32_scalar(lam)   if isinstance(lam,   mx.array) else _f32(lam),
                       _u32(q), _u32(n)],
               output_shapes=[state.shape], output_dtypes=[mx.float32],
               grid=g, threadgroup=t)[0]


# ---------------------------------------------------------------------------
# Circuit dispatch (compiled via mx.compile)
# ---------------------------------------------------------------------------

def _initial_state_real(n: int) -> mx.array:
    """Create the |0…0⟩ statevector in (2 * 2^n,) float32 layout."""
    arr = np.zeros(2 * (1 << n), dtype=np.float32)
    arr[0] = 1.0
    return mx.array(arr)


def _real_to_complex(state_real: mx.array) -> np.ndarray:
    """Materialise a (2 * 2^n,) float32 MLX array as a complex64 numpy array.

    Uses ``np.array(state_real)`` for direct buffer transfer (avoids the
    ~50× slowdown of going through ``.tolist()``), then reinterprets the
    (2*N,) float32 buffer as (N,) complex64 by view (no copy)."""
    mx.eval(state_real)
    flat = np.array(state_real, copy=False)
    # Reinterpret: pairs of consecutive floats become one complex64.
    # The MLX buffer may not be guaranteed to satisfy complex64 alignment
    # in every codepath, so we contiguous-ify if needed.
    if not flat.flags.c_contiguous:
        flat = np.ascontiguousarray(flat)
    return flat.view(np.complex64)


# A registry of "supported gate kinds" for fast routing.
_SUPPORTED = {"ry", "rx", "rz", "h", "x", "cnot", "cz", "rzz", "u3"}


def supports(circuit) -> bool:
    """True if every gate in the circuit has a hand-written Metal kernel."""
    return all(op.kind in _SUPPORTED for op in circuit._ops)


def _run_uncompiled(circuit, params: mx.array, n: int) -> mx.array:
    """Apply circuit's gates one by one (no mx.compile wrap). Used as the
    inner body that :func:`mx.compile` traces."""
    state = _initial_state_real(n)
    for op in circuit._ops:
        kind = op.kind
        qs = op.qubits
        if kind == "ry":  state = apply_ry(state, params[op.param_indices[0]], qs[0], n)
        elif kind == "rx": state = apply_rx(state, params[op.param_indices[0]], qs[0], n)
        elif kind == "rz": state = apply_rz(state, params[op.param_indices[0]], qs[0], n)
        elif kind == "h":  state = apply_h(state, qs[0], n)
        elif kind == "x":  state = apply_x(state, qs[0], n)
        elif kind == "cnot": state = apply_cnot(state, qs[0], qs[1], n)
        elif kind == "cz":   state = apply_cz(state, qs[0], qs[1], n)
        elif kind == "rzz":
            state = apply_rzz(state, params[op.param_indices[0]], qs[0], qs[1], n)
        elif kind == "u3":
            state = apply_u3(state,
                              params[op.param_indices[0]],
                              params[op.param_indices[1]],
                              params[op.param_indices[2]],
                              qs[0], n)
        else:
            raise NotImplementedError(
                f"Metal path does not yet support gate kind '{kind}'. "
                "Use Circuit.statevector(method='mlx') or method='accel' as fallback."
            )
    return state


_compiled_cache: dict[tuple, callable] = {}


def run_circuit(circuit, params, *, compile: bool = True) -> np.ndarray:
    """Execute a Circuit via custom Metal kernels and return a complex64 numpy
    statevector.

    Parameters
    ----------
    circuit:
        A :class:`zilver.circuit.Circuit`.
    params:
        Parameter vector — either ``np.ndarray`` or ``mx.array``.
    compile:
        Wrap the gate sequence in :func:`mx.compile`. Strongly recommended:
        the per-gate buffer-allocation overhead in lazy mode is significant
        at high qubit counts. With compile=True we observe 4-5× wins.

    Returns
    -------
    np.ndarray
        ``(2**n,)`` complex64 statevector.
    """
    if not supports(circuit):
        unsupported = sorted({op.kind for op in circuit._ops if op.kind not in _SUPPORTED})
        raise NotImplementedError(
            f"Metal path does not yet support gate kinds: {unsupported}. "
            "Use Circuit.statevector(method='accel') or method='mlx'."
        )

    n = circuit.n_qubits
    params_mx = mx.array(np.asarray(params, dtype=np.float32)) \
        if not isinstance(params, mx.array) else params.astype(mx.float32)

    if not compile:
        state = _run_uncompiled(circuit, params_mx, n)
        return _real_to_complex(state)

    # Cache the compiled function per (circuit-shape, n) tuple. The lambda
    # captures the circuit object; we key by id() to keep it cheap.
    key = (id(circuit), n)
    fn = _compiled_cache.get(key)
    if fn is None:
        fn = mx.compile(lambda p: _run_uncompiled(circuit, p, n))
        _compiled_cache[key] = fn
    state = fn(params_mx)
    return _real_to_complex(state)


__all__ = [
    "apply_ry", "apply_rx", "apply_rz", "apply_h", "apply_x",
    "apply_cnot", "apply_cz", "apply_rzz", "apply_u3",
    "supports", "run_circuit",
]
