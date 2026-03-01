#!/usr/bin/env python3
"""VQA gradient descent example."""

import sys
import math
import numpy as np
import mlx.core as mx

sys.path.insert(0, "src")
from zilver.circuit import hardware_efficient
from zilver.gradients import param_shift_gradient

N_QUBITS   = 6
DEPTH      = 2
LR         = 0.1
MAX_STEPS  = 80
SEED       = 0


def run():
    rng = np.random.default_rng(SEED)
    circuit = hardware_efficient(N_QUBITS, DEPTH)
    f = circuit.compile(observable="sum_z")

    params = mx.array(
        rng.uniform(-np.pi, np.pi, circuit.n_params).astype(np.float32)
    )

    print(f"\nVQA optimization  |  hardware_efficient({N_QUBITS}q, depth={DEPTH})"
          f"  |  {circuit.n_params} params  |  lr={LR}")
    print(f"{'Step':<6}  {'Loss':>10}  {'|grad|':>10}")
    print(f"{'-'*6}  {'-'*10}  {'-'*10}")

    for step in range(MAX_STEPS):
        loss = float(f(params))
        grads = param_shift_gradient(f, params)
        grad_norm = float(mx.sqrt(mx.sum(grads ** 2)))

        if step % 10 == 0 or step == MAX_STEPS - 1:
            print(f"{step:<6}  {loss:>10.4f}  {grad_norm:>10.4f}")

        params = params - LR * grads
        mx.eval(params)

        if grad_norm < 1e-4:
            print(f"\nConverged at step {step}  (|grad| = {grad_norm:.2e})")
            break

    final_loss = float(f(params))
    print(f"\nFinal loss: {final_loss:.4f}  (min possible: {-N_QUBITS})")
    print(f"Optimized params (first 6): {[round(float(p), 3) for p in params[:6]]}")


if __name__ == "__main__":
    run()
