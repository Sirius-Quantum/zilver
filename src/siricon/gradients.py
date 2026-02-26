"""
Gradient computation via the parameter shift rule.

The parameter shift rule for a gate G(theta) = exp(-i theta/2 * P) where P is
a Pauli operator gives exact gradients:
    df/d(theta_k) = 0.5 * [f(theta_k + pi/2) - f(theta_k - pi/2)]

Key optimization for Sirius: all 2*P shifted evaluations for a full gradient
are batched into a single mx.vmap call, dispatching one Metal kernel instead
of 2*P sequential calls.
"""

from __future__ import annotations
from typing import Callable
import mlx.core as mx
import numpy as np


SHIFT = np.pi / 2


def param_shift_gradient(
    f: Callable[[mx.array], mx.array],
    params: mx.array,
    shift: float = SHIFT,
) -> mx.array:
    """
    Full parameter shift gradient for all parameters.

    Args:
        f:      Pure function params -> scalar (compiled circuit expectation).
        params: (P,) parameter vector.
        shift:  Shift value, default pi/2.

    Returns:
        (P,) gradient vector.

    Batches 2*P evaluations via mx.vmap in a single Metal dispatch.
    """
    P = params.shape[0]
    eye = mx.eye(P, dtype=mx.float32)

    params_plus  = params[None, :] + shift * eye   # (P, P)
    params_minus = params[None, :] - shift * eye   # (P, P)
    all_params = mx.concatenate([params_plus, params_minus], axis=0)  # (2P, P)

    all_evals = mx.vmap(f)(all_params)  # (2P,)
    grads = 0.5 * (all_evals[:P] - all_evals[P:])
    mx.eval(grads)
    return grads


def param_shift_gradient_batched(
    f: Callable[[mx.array], mx.array],
    params_batch: mx.array,
    shift: float = SHIFT,
) -> mx.array:
    """
    Compute gradients for a batch of parameter vectors simultaneously.

    Args:
        f:             Pure function params -> scalar.
        params_batch:  (B, P) batch of parameter vectors.
        shift:         Shift value.

    Returns:
        (B, P) gradient matrix.

    Useful for computing gradients at every point of a 20x20 landscape grid
    simultaneously. Total circuit evaluations: 2 * B * P, all dispatched
    as one Metal kernel via nested vmap.
    """
    grad_fn = lambda p: param_shift_gradient(f, p, shift)
    grads = mx.vmap(grad_fn)(params_batch)
    mx.eval(grads)
    return grads


def gradient_magnitude(grads: mx.array) -> mx.array:
    """L2 norm of a gradient vector."""
    return mx.sqrt(mx.sum(grads ** 2))


def gradient_variance(
    f: Callable[[mx.array], mx.array],
    n_params: int,
    n_samples: int = 200,
    seed: int = 0,
) -> dict:
    """
    Estimate gradient variance at random parameter points.
    Low variance is the signature of a barren plateau.

    Uses a single flat vmap over all N*2P shifted evaluations - no nested vmap.

    Returns dict with per-parameter variance and mean gradient magnitude.
    """
    rng = np.random.default_rng(seed)
    samples = rng.uniform(-np.pi, np.pi, (n_samples, n_params)).astype(np.float32)
    params_mx = mx.array(samples)  # (N, P)

    # Build all shifted param vectors in one shot.
    # params_plus[n, i, :] = samples[n] + (pi/2) * e_i
    eye = mx.eye(n_params, dtype=mx.float32) * SHIFT   # (P, P)
    params_plus  = params_mx[:, None, :] + eye[None, :, :]  # (N, P, P)
    params_minus = params_mx[:, None, :] - eye[None, :, :]  # (N, P, P)

    # Flatten to single batch: (N*2P, P)
    all_shifted = mx.concatenate([params_plus, params_minus], axis=1)
    all_flat = all_shifted.reshape(n_samples * 2 * n_params, n_params)

    # Single vmap - no nesting
    all_evals = mx.vmap(f)(all_flat)  # (N*2P,)
    mx.eval(all_evals)

    evals = all_evals.reshape(n_samples, 2 * n_params)
    grads = 0.5 * (evals[:, :n_params] - evals[:, n_params:])  # (N, P)
    mx.eval(grads)

    var_per_param = mx.var(grads, axis=0)
    magnitudes = mx.sqrt(mx.sum(grads ** 2, axis=1))
    mx.eval(var_per_param, magnitudes)

    return {
        "variance_per_param": np.array(var_per_param.tolist()),
        "mean_gradient_magnitude": float(mx.mean(magnitudes).item()),
        "max_gradient_magnitude": float(mx.max(magnitudes).item()),
        "min_gradient_magnitude": float(mx.min(magnitudes).item()),
    }
