# Zilver

[![PyPI](https://img.shields.io/pypi/v/zilver.svg)](https://pypi.org/project/zilver/)
[![Python](https://img.shields.io/pypi/pyversions/zilver.svg)](https://pypi.org/project/zilver/)
[![Tests](https://github.com/sirius-quantum/zilver/actions/workflows/tests.yml/badge.svg)](https://github.com/sirius-quantum/zilver/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/sirius-quantum/zilver/blob/master/LICENSE)

Zilver is a statevector simulator for quantum circuits that runs on the GPU in your own machine: Metal kernels on Apple silicon, a fused HIP kernel on AMD GPUs, and NumPy on any CPU. It is built for variational algorithms, and also does noisy simulation, matrix product states and circuit cutting. It needs no account and no cloud service.

Zilver is developed by [Sirius Quantum](https://siriusquantum.com) and released under the Apache 2.0 license.

## Install

```bash
pip install zilver
```

Zilver needs Python 3.10 or later. On Apple silicon it installs MLX automatically; elsewhere it runs on NumPy.

Optional extras:

```bash
pip install "zilver[accel]"     # multithreaded CPU kernels and double precision
pip install "zilver[qiskit]"    # Qiskit Aer, for the comparison benchmarks
pip install "zilver[network]"   # node, registry and network client
```

## Example

```python
import numpy as np
from zilver.circuit import hardware_efficient

circuit = hardware_efficient(n_qubits=10, depth=3)
params  = np.random.default_rng(0).uniform(-np.pi, np.pi, circuit.n_params)

sv = circuit.statevector(params)
print(sv.numpy().shape)          # (1024,) complex64

f = circuit.compile(observable="sum_z")   # params -> <sum of Z>
```

The [Quickstart](https://github.com/sirius-quantum/zilver/blob/master/QUICKSTART.md) goes from install to a trained circuit, and [`examples/`](https://github.com/sirius-quantum/zilver/tree/master/examples) has VQA optimisation, barren plateaus, circuit cutting and noisy VQE.

## Features

- Statevector simulation in single precision, or double precision on the CPU (`precision="double"`).
- Parameter-shift gradients, for one parameter vector or a batch at once (`zilver.gradients`).
- Fidelity kernels for quantum kernel methods (`Circuit.fidelity_kernel`).
- Loss landscapes and barren-plateau diagnostics (`zilver.landscape`).
- Density-matrix simulation with depolarizing or thermal-relaxation noise (`NoisyCircuit`, `NoiseModel`).
- Matrix product states for wide, shallow circuits (`zilver.tensor_network.MPSCircuit`).
- Circuit cutting (`zilver.cutting`).

## Platforms

| Platform | Backend | Notes |
|---|---|---|
| Apple silicon, macOS 13+ | MLX with custom Metal kernels | Main platform |
| AMD GPU with PyTorch for ROCm | PyTorch with a fused HIP kernel | Tested on a Radeon 8060S under native Windows ROCm and an Instinct MI350P under Linux |
| Any x86 or ARM CPU | NumPy, optionally Numba | `pip install "zilver[accel]"` for the fast CPU path |

To run on an AMD GPU, install PyTorch for ROCm, then run `python -m zilver.gpu`. It simulates one circuit at each width from 20 to 32 qubits and prints the time and the norm of the final state for each.

## Performance

**Apple M1 Pro, 16 GB.** One statevector of a hardware-efficient ansatz at depth 2, complex64. Time in milliseconds until the state is a NumPy array, best of ten runs. Zilver 0.6.2 with MLX 0.32.2 against Qiskit Aer 0.17.2; the final states agree to fidelity 1.0000000 at every width.

| Qubits | Zilver | Qiskit Aer |
|-------:|-------:|-----------:|
| 12 | 1.27 | 1.42 |
| 16 | 2.73 | 5.86 |
| 20 | 18.09 | 47.15 |
| 22 | 81.32 | 208.43 |
| 24 | 353.00 | 785.39 |
| 26 | 1,586.14 | 3,085.96 |

Reproduce with `python benchmarks/statevector_vs_aer.py` (needs `[qiskit]`).

**AMD Instinct MI350P and Radeon 8060S.** The circuit from `python -m zilver.gpu`: Hadamard and Y-rotation on every qubit, then a CNOT chain. Fused HIP kernel, complex64, the same code on both. MI350P: one compute partition (gfx950, 38.65 GB), Linux, ROCm 7.1, measured October 2026. Radeon 8060S: 128 GB unified memory, native Windows ROCm, measured 2026-09-17. Time in seconds, end to end.

| Qubits | State size | MI350P | Radeon 8060S | Norm |
|-------:|-----------:|-------:|-------------:|:-----|
| 31 | 17.18 GB | 1.50 | 2.74 | 0.9999993 |
| 32 | 34.36 GB | 1.84 | 6.06 | 0.9999993 |

End to end includes allocating the state and copying it back to the host; on the Radeon the gates alone take 0.59 s at 32 qubits. The kernel is tuned for the Radeon (gfx1151), not yet for MI350P. 32 qubits is the largest state that fits on one MI350P partition. The quantum Fourier transform of a basis state matches the exact amplitudes to within 2.7e-6 on both, at every width tested up to 32 qubits.

The noise model is checked against real IBM and IQM hardware in [`benchmarks/`](https://github.com/sirius-quantum/zilver/tree/master/benchmarks).

<details>
<summary>Execution paths, memory and environment variables</summary>

`Circuit.statevector(params, method=..., precision=...)` chooses how a circuit runs:

| `method` | Runs on | Precision |
|---|---|---|
| `"auto"` (default) | `metal` when every gate is supported and precision is single; otherwise `accel` if installed, else `mlx` | as requested |
| `"metal"` | Metal kernels for RY, RZ, RX, H, X, CNOT, CZ, RZZ and U3 | complex64 |
| `"accel"` | Multithreaded Numba CPU kernels (`[accel]`) | complex64 or complex128 |
| `"mlx"` | MLX on Apple silicon, PyTorch or NumPy elsewhere; supports `vmap` | complex64 |

A complex64 statevector takes 8 × 2ⁿ bytes (1 GiB at 27 qubits, 32 GiB at 32), and most paths briefly need two to three times that. The fused HIP kernel works in place. A density matrix takes 8 × 4ⁿ bytes. On a 16 GB Mac the practical limits are about 30 qubits for a statevector, 15 for a density matrix, and 50 or more for a matrix product state, depending on entanglement.

| Variable | Effect |
|---|---|
| `ZILVER_BACKEND=torch` | Use PyTorch instead of MLX or NumPy. Set before importing `zilver`. |
| `ZILVER_DEVICE` | PyTorch device: `cuda` (also ROCm), `mps`, `cpu` or `directml`. |
| `ZILVER_HIP=0` | Turn off the fused HIP kernel. |
| `ZILVER_SHRINK=1` | Skip untouched qubits early in a run; on in `python -m zilver.gpu`. |

</details>

## Zilver network

The network is an optional, invite-only preview for running jobs on other people's Macs, or lending yours. Clients send circuits straight to a node; the registry only matches jobs to nodes. To apply as a client or a node operator, open an issue.

With a client key:

```python
from zilver.circuit import Circuit
from zilver.client import NetworkCoordinator
from zilver.node import job_from_circuit

c = Circuit(4); c.h(0); c.cnot(0, 1); c.ry(2, 0)
coord  = NetworkCoordinator("https://registry.siriusquantum.com", client_api_key="your-key")
result = coord.submit(job_from_circuit(c, params=[1.57], backend="sv"))
print(result.expectation)
```

Running a node is covered in [NODES.md](https://github.com/sirius-quantum/zilver/blob/master/NODES.md).

## Status and contributing

Zilver is alpha software: APIs and network formats can change between minor releases. Changes are listed in the [changelog](https://github.com/sirius-quantum/zilver/blob/master/CHANGELOG.md).

Issues and pull requests are welcome; see [CONTRIBUTING.md](https://github.com/sirius-quantum/zilver/blob/master/CONTRIBUTING.md). Report security problems as described in [SECURITY.md](https://github.com/sirius-quantum/zilver/blob/master/SECURITY.md). Other questions: [dev@siriusquantum.com](mailto:dev@siriusquantum.com).

[Read the Sirius Quantum Manifesto](https://github.com/sirius-quantum/zilver/blob/master/MANIFESTO.md)
