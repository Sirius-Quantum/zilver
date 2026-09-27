# Zilver

[![Version](https://img.shields.io/badge/version-0.6.2-blue.svg)](https://pypi.org/project/zilver/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/sirius-quantum/zilver/blob/master/LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MLX%20%2B%20Metal-black.svg)](https://github.com/ml-explore/mlx)
[![AMD ROCm](https://img.shields.io/badge/AMD%20ROCm-fused%20HIP%20kernel-red.svg)](https://rocm.docs.amd.com/)

Zilver is a quantum circuit simulator and a distributed simulation network built on it.

**Fast statevector simulation on the GPU you already have.**

Zilver runs quantum circuits on Apple silicon through hand-written Metal kernels, and on AMD GPUs through a fused HIP kernel that updates the state in place. No GPU? It runs on the CPU with NumPy. No account, no cloud, no API key.

- **About 2× faster than Qiskit Aer** on a single statevector, 16 to 26 qubits on an M1 Pro.
- **32 qubits in 6.06 s** end to end on a Radeon 8060S integrated GPU.
- **Checked, not assumed.** Every example below matches Qiskit, or an exact formula, to float32 rounding.
- **Built for variational work:** parameter-shift gradients, fidelity kernels, loss landscapes, noisy simulation and matrix product states.

Need more memory than one machine has? The opt-in Zilver network pools Apple-silicon Macs and sends each job to one that can hold it. It is in invite-only preview.

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

New to Zilver? The [Quickstart](https://github.com/sirius-quantum/zilver/blob/master/QUICKSTART.md) goes from install to a trained circuit in a few minutes.

## Quick start

```python
import numpy as np
from zilver.circuit import hardware_efficient

circuit = hardware_efficient(n_qubits=10, depth=3)
params  = np.random.default_rng(0).uniform(-np.pi, np.pi, circuit.n_params)

sv = circuit.statevector(params)
print(sv.numpy().shape, sv.numpy().dtype)   # (1024,) complex64
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
batch_params = np.random.default_rng(1).uniform(-np.pi, np.pi, (8, circuit.n_params))
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

More in [`examples/`](https://github.com/sirius-quantum/zilver/tree/master/examples): VQA optimisation, barren plateaus, circuit cutting, noisy VQE.

## Execution paths

`Circuit.statevector(params, method=..., precision=...)` chooses how a circuit runs.

| `method` | Runs on | Precision | Use it for |
|---|---|---|---|
| `"auto"` (default) | Picks `metal` if every gate is supported and precision is single; otherwise `accel` if `[accel]` is installed, else `mlx` | as requested | Most work |
| `"metal"` | Hand-written Metal kernels for RY, RZ, RX, H, X, CNOT, CZ, RZZ and U3, combined into one graph by `mx.compile` | complex64 | One statevector at a time on Apple silicon |
| `"accel"` | Multithreaded Numba CPU kernels; chooses between NumPy, compiled per-gate code and fused two-qubit blocks based on circuit size | complex64 or complex128 | Double precision; machines without a GPU. Needs `[accel]` |
| `"mlx"` | The array layer: MLX on Apple silicon, PyTorch or NumPy elsewhere | complex64 | Batched sweeps with `vmap`; the AMD GPU path |

Environment variables for the non-Apple paths:

| Variable | Effect |
|---|---|
| `ZILVER_BACKEND=torch` | Use PyTorch instead of MLX or NumPy. Set it before importing `zilver`. |
| `ZILVER_DEVICE` | PyTorch device: `cuda` (which also covers ROCm), `mps`, `cpu` or `directml`. By default Zilver tries `cuda`, then `mps`, then DirectML, then `cpu`. |
| `ZILVER_HIP=0` | Turn off the fused HIP kernel and use the generic PyTorch path. |
| `ZILVER_SHRINK=1` | Shrink each pass to the touched support: a run starts at all-zeros, so until the schedule has touched K qubits a pass need only cover 2^K amplitudes instead of 2^n. Read at import. Off by default; `python -m zilver.gpu` turns it on. It reorders gates and renames qubits (the state is unchanged), and the saving is a fixed warm-up — large on shallow circuits, small on deep ones. |

On a device with no complex dtype, such as DirectML, the state is stored as a pair of real arrays. Zilver refuses to create complex tensors there, because DirectML crashes the process instead of raising an error.

### Memory

A single-precision statevector takes 8 × 2ⁿ bytes: 1 GiB at 27 qubits, 32 GiB at 32. A density matrix takes 8 × 4ⁿ bytes. Most gate paths briefly need two to three times the state's memory. The fused HIP kernel updates the state in place, which is why a 64 GiB device can hold 32 qubits.

| Backend | Approximate ceiling, 16 GB Apple silicon |
|---|---|
| Statevector | 30 qubits |
| Density matrix | 15 qubits |
| Matrix product state | 50+ qubits, depending on entanglement |

## Performance

### Apple M1 Pro

Single statevector, hardware-efficient ansatz at depth 2, on an Apple M1 Pro with 16 GB of unified memory. Wall time in milliseconds until the state is a NumPy array, best of ten runs; lower is better. Both simulators run in single precision (complex64); Zilver 0.6.2 with MLX 0.32.2, Qiskit Aer 0.17.2. The two final states agree to fidelity 1.0000000 at every width.

| Qubits | Zilver (Metal) | Qiskit Aer |
|-------:|---------------:|-----------:|
|     12 |           1.27 |       1.42 |
|     16 |           2.73 |       5.86 |
|     20 |          18.09 |      47.15 |
|     22 |          81.32 |     208.43 |
|     24 |         353.00 |     785.39 |
|     26 |       1,586.14 |   3,085.96 |

CNOT and CZ reproduce the ideal two-qubit process exactly on every backend. RZZ on the Metal path is within 3.4e-08 of ideal, which is the limit of float32. The `accel` path with `precision="double"` matches the ideal unitary to numerical zero.

Reproduce the table with `python benchmarks/statevector_vs_aer.py` (needs `[qiskit]`). The noise model is validated against real IBM and IQM hardware in [`benchmarks/`](https://github.com/sirius-quantum/zilver/tree/master/benchmarks).

### AMD Radeon 8060S

The circuit `python -m zilver.gpu` runs: a Hadamard and a Y-rotation on every qubit, then a CNOT chain, 95 gates at 32 qubits. Fused HIP kernel, complex64, on a Ryzen AI Max+ 395 with 128 GB of unified memory under native Windows ROCm. Measured 2026-09-17.

| Qubits | State | End to end (s) | Gates alone (s) | Norm |
|-------:|------:|---------------:|----------------:|:-----|
|     31 | 17.18 GB | 2.74 | 0.28 | 0.9999993 |
|     32 | 34.36 GB | 6.06 | 0.59 | 0.9999993 |

End to end includes allocating the state on the device and copying it back to the host, which is most of the time at this width. The kernel updates the state in place, so 32 qubits fits in the 64 GiB the GPU can address. These times use support shrinking (`ZILVER_SHRINK=1`, on in `python -m zilver.gpu`). Its saving is a fixed warm-up, so it is largest on shallow circuits like this one; with it off, 32 qubits takes 12.69 s end to end. Correctness is checked against closed forms: the quantum Fourier transform of a basis state at every width from 20 to 32 qubits agrees to 2.7e-6 of the exact amplitude.

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

```bash
pip install "zilver[network]"
```

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

The node must be reachable from the internet; a Cloudflare Tunnel is the simplest way. The node detects its chip and memory at startup and reports its qubit limits to the registry. See [NODES.md](https://github.com/sirius-quantum/zilver/blob/master/NODES.md) for public-URL options, identity and troubleshooting.

Other commands:

```bash
zilver-node status      # network summary
zilver-node nodes       # online nodes
zilver-node dashboard   # live terminal view
```

They use the public registry unless you pass `--registry URL`.

## Status

Zilver is alpha software under active development. Public APIs and wire formats may change between minor releases. See the [changelog](https://github.com/sirius-quantum/zilver/blob/master/CHANGELOG.md).

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](https://github.com/sirius-quantum/zilver/blob/master/CONTRIBUTING.md) and the [Code of Conduct](https://github.com/sirius-quantum/zilver/blob/master/CODE_OF_CONDUCT.md). Report security problems privately, as described in [SECURITY.md](https://github.com/sirius-quantum/zilver/blob/master/SECURITY.md). For anything else, write to [dev@siriusquantum.com](mailto:dev@siriusquantum.com).

## License

Apache 2.0. See [LICENSE](https://github.com/sirius-quantum/zilver/blob/master/LICENSE).

[Read the Sirius Quantum Manifesto](https://github.com/sirius-quantum/zilver/blob/master/MANIFESTO.md)
