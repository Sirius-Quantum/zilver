# Zilver

[![PyPI version](https://img.shields.io/pypi/v/zilver.svg)](https://pypi.org/project/zilver/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![MLX](https://img.shields.io/badge/MLX-0.18%2B-orange.svg)](https://github.com/ml-explore/mlx)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1--M4-black.svg)](https://www.apple.com/mac/)

Zilver is a quantum circuit simulator built natively for Apple Silicon. It runs entirely on the Apple GPU through [MLX](https://github.com/ml-explore/mlx), with hand-written Metal compute kernels for the common gates and a multithreaded CPU path through Numba and Accelerate.

It is designed for QML researchers and engineers who want to develop, train, and benchmark variational quantum algorithms locally on their own Mac. No cloud dependency. No virtualisation. No Intel-Mac path.

## Install and run

```bash
pip install zilver
```

Apple Silicon Mac, macOS 13 or later, Python 3.10 or later.

```python
import numpy as np
from zilver.circuit import hardware_efficient

circuit = hardware_efficient(n_qubits=10, depth=3)
params  = np.random.default_rng(0).uniform(-np.pi, np.pi, circuit.n_params)

sv = circuit.statevector(params)
print(sv.numpy().shape, sv.dtype)
```

That is the entire setup. No accounts, no API keys, no registration.

For the multithreaded CPU path and double-precision simulation, install the optional extra:

```bash
pip install "zilver[accel]"
```

## What you can build

Zilver ships the primitives QML practitioners use daily.

**Variational algorithms.** Parameter-shift gradients, batched over a parameter vector or a parameter grid, fused into a single MLX dispatch.

```python
import mlx.core as mx
from zilver.gradients import param_shift_gradient, param_shift_gradient_batched

f = circuit.compile(observable="sum_z")
g = param_shift_gradient(f, mx.array(params.astype(np.float32)))
```

**Quantum kernel methods.** A fidelity kernel `|<psi_i|psi_j>|^2` for an N-sample batch is one call, computed on-device.

```python
K = circuit.fidelity_kernel(batch_params)  # (N, N) numpy float32
```

**Loss-landscape and barren-plateau analysis.** A 2D parameter sweep at fixed depth is one vmap dispatch.

```python
from zilver.landscape import LossLandscape

land = LossLandscape(circuit, sweep_params=(0, 1), resolution=32).compute()
print(land.trainability_score(), land.plateau_coverage())
```

## Statevector backends

`Circuit.statevector(params, method=..., precision=...)` selects how the circuit executes.

`method="metal"` runs hand-written Metal compute kernels for RY, RZ, RX, H, X, CNOT, CZ, RZZ, and U3, fused into a single graph by `mx.compile`. Single precision (complex64). The default for single-statevector evaluation.

`method="accel"` runs a multithreaded CPU path on Numba and Accelerate. It auto-routes between strided NumPy (small circuits), tape-lowered JIT dispatch, and k=2 fused blocks based on qubit count. Supports complex64 and complex128. Requires the `accel` extra.

`method="mlx"` is the generic MLX path. Use it for batched workloads such as parameter sweeps, gradient batches, and fidelity kernels where `mx.vmap` over the parameter axis is the dominant compute pattern.

`method="auto"` (the default) picks `metal` when the circuit uses only supported gate kinds and `precision="single"`, otherwise `accel`.

The wider simulation surface (density-matrix and tensor-network backends) is selected at job-submission time via the `backend` flag.

| Backend | Flag | Approximate ceiling on a 16 GB M-series |
|---------|------|----------------------------------------|
| Statevector | sv | 30 qubits |
| Density matrix | dm | 15 qubits |
| Tensor network | tn | 50+ qubits, circuit-dependent |

## Performance

Single statevector, hardware-efficient ansatz at depth 2, on Apple M1 Pro with 16 GB unified memory. Wall time in milliseconds, minimum of four trials; lower is better.

| Qubits | Zilver (Metal) | Qiskit Aer |
|-------:|---------------:|-----------:|
|     12 |           1.45 |       1.50 |
|     16 |           1.76 |       4.84 |
|     20 |          19.93 |      40.84 |
|     22 |          70.31 |     148.44 |
|     24 |         334.63 |     588.61 |

Two-qubit process fidelity against the ideal unitary: CNOT and CZ are bit-exact on every backend. RZZ on the Metal path is within 3.4e-08 of ideal, the float32 floor. The Accel path with `precision="double"` reproduces ideal to numerical zero.

## Optional: join the Zilver network

Everything above runs standalone on your Mac. If you want to contribute compute or run jobs across a distributed pool of Apple Silicon nodes, the Zilver network adds that as a separate, opt-in layer.

The network connects Apple Silicon nodes into a shared simulation fabric. Researchers submit jobs to the registry; the registry matches to a capable node; the node executes and returns a cryptographically signed result.

### Join as a node operator

Node registration is invite-only during the current phase. Open an issue with your chip model and intended uptime; on approval you will receive a registry API key.

```bash
pip install "zilver[network]"
zilver-node start \
  --registry https://registry.siriusquantum.com \
  --public-url https://your-public-url.example.com
```

See [NODES.md](NODES.md) for setup requirements, public URL options, and identity management.

### Submit jobs as a researcher

Client API access is also by invitation. Open an issue describing your use case and institution; on approval you will receive a client key.

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
print(result.verify(job))
```

## Status

v0.4 is under active development. Public APIs and wire formats may change between minor releases.

## Feedback and contributions

Open an issue on [GitHub](https://github.com/Sirius-Quantum/zilver/issues) or write to [info@siriusquantum.com](mailto:info@siriusquantum.com).

## License

Apache 2.0. See [LICENSE](LICENSE).

[Read the Sirius Quantum Manifesto](MANIFESTO.md)
