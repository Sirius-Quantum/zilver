# Zilver

MLX-native statevector quantum circuit simulator for Apple Silicon.

Built to replace sequential Qiskit Aer calls with a single batched Metal dispatch — every point on a 20×20 loss landscape evaluated in one GPU kernel instead of 2000 sequential circuit executions.

---

## Requirements

- Apple Silicon Mac (M1 / M2 / M3 / M4 — MacBook, Mac Mini, Mac Studio, Mac Pro)
- macOS 13 Ventura or later
- Python 3.10+

---

## Install

```bash
git clone https://github.com/deryakarl/zilver
cd zilver
bash scripts/setup.sh
```

The setup script verifies Apple Silicon, installs dependencies, confirms Metal is active, installs the pre-commit guard hook, and runs the test suite.

---

## Quick start

```python
from zilver.circuit import hardware_efficient
from zilver.landscape import LossLandscape

circuit = hardware_efficient(n_qubits=6, depth=3)
result = LossLandscape(circuit, sweep_params=(0, 1), resolution=20).compute()

print(f"Plateau coverage:   {result.plateau_coverage():.1%}")
print(f"Trainability score: {result.trainability_score():.3f}")
print(f"Wall time:          {result.wall_time_seconds:.3f}s")
```

---

## Core concepts

**Circuit families**
```python
from zilver.circuit import hardware_efficient, real_amplitudes, qaoa_style, efficient_su2

c = hardware_efficient(n_qubits=8, depth=4)
c = real_amplitudes(n_qubits=6, depth=3)
c = qaoa_style(n_qubits=6, depth=4)        # QAOA: 2*depth params
```

**Universal single-qubit gate**
```python
from zilver.circuit import Circuit

c = Circuit(n_qubits=2)
c.u3(qubit=0, theta_idx=0, phi_idx=1, lam_idx=2)  # any SU(2)
c.cnot(0, 1)
```

**Batched evaluation via vmap**
```python
import mlx.core as mx
import numpy as np

f = circuit.compile(observable="sum_z")

# 400 parameter vectors evaluated in one Metal dispatch
params_grid = mx.array(np.random.uniform(-np.pi, np.pi, (400, circuit.n_params)).astype(np.float32))
losses = mx.vmap(f)(params_grid)
mx.eval(losses)
```

**Gradient computation**
```python
from zilver.gradients import param_shift_gradient, gradient_variance

# Full gradient vector (2P evaluations, batched)
grads = param_shift_gradient(f, params)

# Barren plateau detection: gradient variance across random samples
stats = gradient_variance(f, circuit.n_params, n_samples=200)
print(stats["variance_per_param"])   # low variance = barren plateau
```

---

## Gate library

| Gate | Type | Params |
|---|---|---|
| H, X, Y, Z, S, T | Fixed single-qubit | — |
| RX, RY, RZ | Rotation | 1 |
| U3 | Universal single-qubit | 3 (theta, phi, lambda) |
| CNOT, CZ, SWAP, iSWAP | Fixed two-qubit | — |
| RZZ, RXX | Ising coupling | 1 |
| CRZ | Controlled rotation | 1 |

All parameterized gates are MLX-native — fully compatible with `mx.vmap` and `mx.compile`.

---

## Development

```bash
make test          # run test suite (81 tests)
make bench         # benchmark vs Qiskit Aer
make guard         # scan repo for secrets / internal references
make install-hooks # (re)install pre-commit hook
```

---

## Security

A pre-commit guard runs on every commit and blocks: API keys, tokens, passwords, private key material, database connection strings, files over 500 KB, and internal path references.

Run a full scan anytime:
```bash
python scripts/guard.py --all
```

---

## License

Apache 2.0
