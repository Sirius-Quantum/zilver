# Zilver

[![Version](https://img.shields.io/badge/version-0.6.2-blue.svg)](https://pypi.org/project/zilver/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MLX%20%2B%20Metal-black.svg)](https://github.com/ml-explore/mlx)
[![AMD ROCm](https://img.shields.io/badge/AMD%20ROCm-fused%20HIP%20kernel-red.svg)](https://rocm.docs.amd.com/)

Zilver is a quantum circuit simulator and a distributed simulation network built on it.

The simulator runs on your own machine. On Apple silicon it uses the GPU through [MLX](https://github.com/ml-explore/mlx) and hand-written Metal kernels. On AMD GPUs it uses PyTorch for ROCm and a fused HIP kernel that updates the state in place, reaching 32 qubits on a Radeon 8060S. On any other machine it runs on the CPU with NumPy.

The network links Apple-silicon Macs into a shared pool. A registry sends each job to a Mac that has enough memory for it, and the job runs there.

It is built for people who develop, train and benchmark variational quantum algorithms. The simulator needs no account, no cloud service and no API key.

## Platforms

| Platform | Array backend | Fast path | Role |
|---|---|---|---|
| Apple silicon Mac, macOS 13+ | MLX | Metal compute kernels | Standalone, or a network node |
| AMD GPU with PyTorch for ROCm | PyTorch | Fused in-place HIP kernel | Standalone |
| Any x86 or ARM CPU | NumPy | Numba CPU kernels (optional) | Standalone |

The AMD path is verified on Windows 11 with native ROCm on a Radeon 8060S (gfx1151). The same PyTorch layer also targets CUDA, MPS and DirectML devices, without the fused kernel. Those devices are less tested.

## Install

```bash
pip install zilver
```

Python 3.10 or later. On Apple silicon, MLX is installed automatically. On other machines Zilver installs without MLX and falls back to NumPy.

Optional extras:

```bash
pip install "zilver[accel]"     # multithreaded CPU kernels and double precision
pip install "zilver[network]"   # node, registry and network client
pip install "zilver[qiskit]"    # Qiskit Aer, for the comparison benchmarks
```

### Run it on an AMD GPU

Install PyTorch for ROCm first, then:

```bash
pip install zilver
python -m zilver.gpu
```

This runs one circuit at each width from 20 to 32 qubits and stops early when memory runs out. For each width it prints the time and the norm of the final state. A norm of 1.0000000 means the arithmetic is correct. Use `FROM=24 TO=30 python -m zilver.gpu` to pick the widths.

New to Zilver? The [Quickstart](QUICKSTART.md) goes from install to a trained circuit in a few minutes.

## Quick start

```python
import numpy as np
from zilver.circuit import hardware_efficient

circuit = hardware_efficient(n_qubits=10, depth=3)
params  = np.random.default_rng(0).uniform(-np.pi, np.pi, circuit.n_params)

sv = circuit.statevector(params)
print(sv.numpy().shape, sv.dtype)   # (1024,) complex64
```

## What you can build

**Variational algorithms.** Parameter-shift gradients for one parameter vector or a whole batch, computed in a single MLX dispatch.

```python
import mlx.core as mx
from zilver.gradients import param_shift_gradient

f = circuit.compile(observable="sum_z")
g = param_shift_gradient(f, mx.array(params.astype(np.float32)))
```

**Quantum kernel methods.** The fidelity kernel `|<psi_i|psi_j>|^2` for N samples is one call, computed on the GPU.

```python
K = circuit.fidelity_kernel(batch_params)   # (N, N) float32
```

**Loss landscapes and barren plateaus.** A 2D parameter sweep is one vectorised (`vmap`) dispatch.

```python
from zilver.landscape import LossLandscape

land = LossLandscape(circuit, sweep_params=(0, 1), resolution=32).compute()
print(land.trainability_score(), land.plateau_coverage())
```

**Noisy simulation.** `NoisyCircuit` runs on the density-matrix backend. A `NoiseModel` applies Kraus channels after every gate. You can use depolarizing noise, or thermal relaxation built from a device's `T1`/`T2` and gate times.

```python
import mlx.core as mx
from zilver import NoisyCircuit, NoiseModel

nc = NoisyCircuit(4)
nc.h(0); nc.cnot(0, 1); nc.ry(1, param_idx=0)

# coherence times and gate durations share a unit (e.g. ns)
noise = NoiseModel.thermal_relaxation(t1=250_000, t2=170_000,
                                      gate_time_1q=32, gate_time_2q=70)
f = nc.compile(observable="sum_z", noise_model=noise)
exp = f(mx.array([0.7]))
```

**Wide, shallow circuits.** `MPSCircuit` simulates with matrix product states, so memory grows with entanglement instead of doubling with every qubit.

```python
from zilver.tensor_network import MPSCircuit

mps = MPSCircuit(40, chi_max=32)
for q in range(40):
    mps.ry(q, q)
for q in range(39):
    mps.cnot(q, q + 1)
print(mps.compile(observable="sum_z")(mx.array([0.3] * 40)))
```

More in [`examples/`](examples): VQA optimisation, barren plateaus, circuit cutting, noisy VQE.

## Execution paths

`Circuit.statevector(params, method=..., precision=...)` chooses how a circuit runs.

| `method` | Runs on | Precision | Use it for |
|---|---|---|---|
| `"auto"` (default) | Picks `metal` if every gate is supported and precision is single; otherwise `accel` | as requested | Most work |
| `"metal"` | Hand-written Metal kernels for RY, RZ, RX, H, X, CNOT, CZ, RZZ and U3, combined into one graph by `mx.compile` | complex64 | One statevector at a time on Apple silicon |
| `"accel"` | Multithreaded Numba CPU kernels; chooses between NumPy, compiled per-gate code and fused two-qubit blocks based on circuit size | complex64 or complex128 | Double precision; machines without a GPU. Needs `[accel]` |
| `"mlx"` | The array layer: MLX on Apple silicon, PyTorch or NumPy elsewhere | complex64 | Batched sweeps with `vmap`; the AMD GPU path |

Environment variables for the non-Apple paths:

| Variable | Effect |
|---|---|
| `ZILVER_BACKEND=torch` | Use PyTorch instead of MLX or NumPy. Set it before importing `zilver`. |
| `ZILVER_DEVICE` | PyTorch device: `cuda` (which also covers ROCm), `mps`, `cpu` or `directml`. By default Zilver tries `cuda`, then `mps`, then DirectML, then `cpu`. |
| `ZILVER_HIP=0` | Turn off the fused HIP kernel and use the generic PyTorch path. |

On a device with no complex dtype, such as DirectML, the state is stored as a pair of real arrays. Zilver refuses to create complex tensors there, because DirectML crashes the process instead of raising an error.

### Memory

A single-precision statevector takes 8 × 2ⁿ bytes: 1 GiB at 27 qubits, 32 GiB at 32. A density matrix takes 8 × 4ⁿ bytes. Most gate paths briefly need two to three times the state's memory. The fused HIP kernel updates the state in place, which is why a 64 GiB device can hold 32 qubits.

| Backend | Approximate ceiling, 16 GB Apple silicon |
|---|---|
| Statevector | 30 qubits |
| Density matrix | 15 qubits |
| Matrix product state | 50+ qubits, depending on entanglement |

## Performance

Single statevector, hardware-efficient ansatz at depth 2, on an Apple M1 Pro with 16 GB of unified memory. Wall time in milliseconds, best of four runs; lower is better.

| Qubits | Zilver (Metal) | Qiskit Aer |
|-------:|---------------:|-----------:|
|     12 |           1.45 |       1.50 |
|     16 |           1.76 |       4.84 |
|     20 |          19.93 |      40.84 |
|     22 |          70.31 |     148.44 |
|     24 |         334.63 |     588.61 |

CNOT and CZ reproduce the ideal two-qubit process exactly on every backend. RZZ on the Metal path is within 3.4e-08 of ideal, which is the limit of float32. The `accel` path with `precision="double"` matches the ideal unitary to numerical zero.

Reproduce the comparison with `python benchmarks/vs_qiskit_aer.py` (needs `[qiskit]`). The noise model is validated against real IBM and IQM hardware in [`benchmarks/`](benchmarks).

## The Zilver network

Everything above runs standalone. The network is a separate, opt-in layer for running jobs on other people's Macs, or lending yours.

**Status: invite-only preview.** Nodes are Apple-silicon Macs. Wire formats and endpoints may change between minor releases.

A job goes through three steps:

1. The client asks the registry for a node that supports the job's backend and qubit count.
2. The registry picks an online node that can hold the job and returns its address.
3. The client sends the circuit straight to the node. The node runs it and returns the result.

With `NetworkCoordinator.submit`, the circuit goes only to the node. The registry matches jobs to nodes and keeps the node list, but never receives the circuit.

### Submit jobs

Client access is by invitation. Open an issue describing your use case; once approved, you receive a client key.

```python
from zilver.circuit import Circuit
from zilver.client import NetworkCoordinator
from zilver.node import job_from_circuit

c = Circuit(4)
c.h(0); c.cnot(0, 1); c.ry(2, 0)

coord = NetworkCoordinator("https://registry.siriusquantum.com",
                           client_api_key="your-key")

job    = job_from_circuit(c, params=[1.57], observable="sum_z", backend="sv")
result = coord.submit(job)
print(result.expectation, result.elapsed_ms)
```

Network jobs run on the statevector backend (`sv`). Noisy and tensor-network simulation are available locally through `NoisyCircuit` and `MPSCircuit`.

`result.verify(job)` checks that the result's checksum matches the job id, the parameters and the returned value. It only checks that these fields agree. It does not re-run the circuit.

### Run a node

Node registration is invite-only. Open an issue with your chip model, unified memory and intended uptime.

```bash
pip install "zilver[network]"
zilver-node start \
  --registry https://registry.siriusquantum.com \
  --public-url https://your-node.example.com \
  --backends sv
```

The node must be reachable from the internet; a Cloudflare Tunnel is the simplest way. The node detects its chip and memory at startup and reports its qubit limits to the registry. See [NODES.md](NODES.md) for public-URL options, identity and troubleshooting.

Other commands:

```bash
zilver-node status      --registry URL   # network summary
zilver-node nodes       --registry URL   # online nodes
zilver-node dashboard   --registry URL   # live terminal view
zilver-node leaderboard --registry URL   # contribution ledger
```

## Status

Zilver is alpha software under active development. Public APIs and wire formats may change between minor releases. See the [changelog](CHANGELOG.md).

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). For anything else, write to [dev@siriusquantum.com](mailto:dev@siriusquantum.com).

## License

Apache 2.0. See [LICENSE](LICENSE).

[Read the Sirius Quantum Manifesto](MANIFESTO.md)
