# Zilver Quickstart

Run quantum simulations locally on your Apple Silicon Mac — no accounts, no API
keys, no cloud. This walks from install to a trained circuit in a few minutes.

**Requirements:** Apple Silicon Mac · macOS 13+ · Python 3.10+

## Install

```bash
pip install zilver
```

Optional extra for the multithreaded CPU path and double precision:

```bash
pip install "zilver[accel]"
```

## 1. Run your first circuit

```python
import numpy as np
from zilver.circuit import hardware_efficient

circuit = hardware_efficient(n_qubits=8, depth=2)
params  = np.random.default_rng(0).uniform(-np.pi, np.pi, circuit.n_params).astype(np.float32)

sv = circuit.statevector(params)
print(sv.numpy().shape)     # (256,) complex64 statevector
```

That's the whole setup. The statevector is computed on the GPU via Metal kernels.

## 2. Build your own circuit

Gates are chained; parameter-free circuits take an empty parameter array.

```python
from zilver.circuit import Circuit

c = Circuit(2)
c.h(0)            # Hadamard on qubit 0
c.cnot(0, 1)      # entangle -> Bell state

sv = c.statevector(np.array([], dtype=np.float32))
print(np.round(sv.numpy(), 3))    # [0.707  0  0  0.707]
```

Parametrized gates (`ry`, `rz`, `rx`, `rzz`) take a `param_idx` into the
parameter vector you pass at run time:

```python
c = Circuit(2)
c.ry(0, param_idx=0)
c.rzz(0, 1, param_idx=1)
# c.n_params == 2
```

## 3. Expectation values

`compile()` returns a fast `params -> <observable>` function.

```python
import mlx.core as mx

f = circuit.compile(observable="sum_z")     # <Z0 + Z1 + ... >
energy = float(f(mx.array(params)))
print(energy)
```

## 4. Train a circuit (parameter-shift gradients)

```python
import mlx.core as mx
from zilver.gradients import param_shift_gradient

f = circuit.compile(observable="sum_z")
p = mx.array(params)

for step in range(50):
    grad = param_shift_gradient(f, p)       # exact gradient, on-device
    p = p - 0.1 * grad
    mx.eval(p)

print("final <sum_z>:", float(f(p)))
```

See `examples/vqa_optimization.py` for a complete VQA loop.

## 5. Quantum kernels

The fidelity kernel `K[i,j] = |<psi(x_i)|psi(x_j)>|^2` for a whole batch is one
on-device call — no Python loop.

```python
import mlx.core as mx

batch = mx.array(
    np.random.default_rng(1).uniform(-np.pi, np.pi, (32, circuit.n_params)).astype(np.float32)
)
K = circuit.fidelity_kernel(batch)          # (32, 32) numpy float32
```

## 6. Noisy simulation

`NoisyCircuit` runs on the density-matrix backend; a `NoiseModel` applies noise
channels automatically after every gate.

```python
import mlx.core as mx
from zilver import NoisyCircuit, NoiseModel

nc = NoisyCircuit(3)
nc.h(0); nc.cnot(0, 1); nc.cnot(1, 2)       # GHZ-3

# depolarizing per gate, or thermal_relaxation(t1, t2, gate_time_1q, gate_time_2q)
noise = NoiseModel.depolarizing(p1=0.01, p2=0.02)

rho = nc.run(mx.array([]), noise_model=noise)   # (8, 8) density matrix
```

See `examples/noisy_simulation.py` for thermal relaxation and a noise sweep.

## Where to next

- **Backends** — `circuit.statevector(params, method=..., precision=...)`
  selects `metal` (GPU), `accel` (CPU, supports float64), or `mlx` (batched).
  See the [README](README.md#statevector-backends).
- **Examples** — `examples/` has runnable scripts for VQA, barren plateaus,
  circuit cutting, and noisy simulation.
- **Joining the network** — to contribute compute or submit distributed jobs,
  see [NODES.md](NODES.md).
