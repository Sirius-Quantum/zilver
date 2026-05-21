# Zilver

[![PyPI version](https://img.shields.io/pypi/v/zilver.svg)](https://pypi.org/project/zilver/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![MLX](https://img.shields.io/badge/MLX-0.18%2B-orange.svg)](https://github.com/ml-explore/mlx)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1--M4-black.svg)](https://www.apple.com/mac/)

**Distributed quantum simulation network for Apple Silicon.**

Zilver is a quantum simulation framework built on [MLX](https://github.com/ml-explore/mlx), Apple's unified memory compute framework. It runs natively on Apple Silicon — M1 through M4 — using Metal for GPU dispatch and the Neural Engine where applicable. There is no cloud dependency, no virtualisation layer, and no emulation.

The network is a distributed fabric of Apple Silicon nodes. Operators contribute their hardware. Researchers submit circuits. The registry coordinates matching, job dispatch, and result verification without acting as a compute intermediary.

> **v0.4 is under active development.** APIs and wire formats may change between minor releases.

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13 Ventura or later
- Python 3.10+

MLX does not run on Linux, Windows, or Intel Macs. This is a deliberate constraint — the architecture is built around Apple's unified memory model and Metal compute stack.

## Local simulation

Install and run circuits immediately, no account or network required.

```bash
pip install zilver
```

```python
from zilver.circuit import hardware_efficient
from zilver.gradients import param_shift_gradient
from zilver.landscape import LossLandscape
import mlx.core as mx

circuit = hardware_efficient(n_qubits=6, depth=3)

# Loss landscape analysis
result = LossLandscape(circuit, sweep_params=(0, 1), resolution=40).compute()
print(f"Trainability score: {result.trainability_score():.3f}")
print(f"Plateau coverage:   {result.plateau_coverage():.1%}")
print(f"Wall time:          {result.wall_time_seconds:.3f}s")

# Parameter-shift gradients, fully batched on Metal
f = circuit.compile(observable="sum_z")
params = mx.zeros([circuit.n_params])
grads = param_shift_gradient(f, params)
```

Three simulation backends ship in the base package:

| Backend | Flag | Max qubits (M1 Pro, 16 GB) |
|---|---|---|
| Statevector | `sv` | 30 |
| Density matrix | `dm` | 15 |
| Tensor network | `tn` | 50+ (circuit-dependent) |

### Statevector dispatch methods

`Circuit.statevector(params, method=..., precision=...)` selects how the statevector backend executes a circuit on Apple Silicon:

| Method | Implementation | Best for |
|---|---|---|
| `"metal"` | Custom Metal compute kernels for RY/RZ/RX/H/X/CNOT/CZ/RZZ/U3, fused via `mx.compile` | Single statevector simulation (complex64) |
| `"accel"` | Multithreaded CPU path (numba + Accelerate), strided + tape-lowered + k=2 fused kernels | Double-precision (`precision="double"`); CPU-only environments |
| `"mlx"` | Generic MLX gate application (transpose + matmul per gate) | Batched / `vmap` workloads (parameter sweeps, gradient batches, fidelity kernels) |
| `"auto"` | Picks `metal` for single-precision supported circuits, `accel` otherwise | Default |

The `accel` path requires the optional extra: `pip install "zilver[accel]"`.

```python
import numpy as np
from zilver.circuit import hardware_efficient

circuit = hardware_efficient(n_qubits=16, depth=2)
params  = np.random.default_rng(0).uniform(-np.pi, np.pi, circuit.n_params)

sv = circuit.statevector(params, method="auto", precision="single")
print(sv.dtype, sv.numpy().shape)
```

### Single-statevector performance on Apple M1 Pro (16 GB)

Hardware-efficient ansatz, depth 2. Lower is better.

| Qubits | zilver `metal` | qiskit-aer (statevector) | Speedup |
|---|---|---|---|
| 12 | 1.45 ms | 1.50 ms | 1.03× |
| 16 | 1.76 ms | 4.84 ms | **2.76×** |
| 20 | 19.93 ms | 40.84 ms | **2.05×** |
| 22 | 70.31 ms | 148.44 ms | **2.11×** |
| 24 | 334.63 ms | 588.61 ms | **1.76×** |

2-qubit gate process fidelity vs ideal unitary: CNOT and CZ are bit-exact on all paths; RZZ on `metal` (complex64) is within 3.4 × 10⁻⁸ of ideal; `accel` `precision="double"` and Aer match to numerical zero. Run `benchmarks/bench_2q_fidelity.py` and `benchmarks/bench_apple_silicon_ceiling.py` to reproduce.

## Network

The Zilver network connects Apple Silicon nodes into a shared simulation fabric. Researchers submit jobs to the registry; the registry matches to a capable node; the node executes and returns a cryptographically signed result.

### Join as a node operator

Node registration is invite-only during the current phase. If you have Apple Silicon hardware and want to contribute compute, open an issue with your chip model and intended uptime.

Once approved, you will receive a registry API key. Start your node:

```bash
pip install "zilver[network]"
zilver-node start \
  --registry https://registry.siriusquantum.com \
  --public-url https://your-public-url.example.com
```

See [NODES.md](NODES.md) for setup requirements, public URL options, and identity management.

### Submit jobs as a researcher

API access is by invitation. To request access, open an issue describing your use case and institution. Once approved you will receive a client key.

```python
from zilver.client import NetworkCoordinator
from zilver.node import SimJob

coord = NetworkCoordinator(
    "https://registry.siriusquantum.com",
    client_api_key="your-key",
)

job = SimJob(
    circuit_ops=[{"type": "ry", "qubits": [0], "param_idx": 0}],
    n_qubits=4, n_params=1, params=[1.57], backend="sv",
)

result = coord.submit(job)
print(result.expectation)
print(result.verify(job))  # cryptographic result verification
```

## Feedback & contributions

Open an issue on [GitHub](https://github.com/Sirius-Quantum/zilver/issues) or reach us at [dev@siriusquantum.com](mailto:dev@siriusquantum.com).

## License

Apache 2.0. See [LICENSE](LICENSE).

[Read the Sirius Quantum Manifesto](MANIFESTO.md)
