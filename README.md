# Zilver

[![PyPI version](https://img.shields.io/pypi/v/zilver.svg)](https://pypi.org/project/zilver/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![MLX](https://img.shields.io/badge/MLX-0.18%2B-orange.svg)](https://github.com/ml-explore/mlx)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1--M4-black.svg)](https://www.apple.com/mac/)

**Open quantum simulation network for Apple Silicon.**

Every Mac with Apple Silicon is a quantum computer waiting to be used. Zilver turns your MacBook, Mac Mini, Mac Studio, or Mac Pro into a node on a distributed quantum simulation network contributing GPU compute via Metal, earning rewards, and advancing open quantum science.

Built on [MLX](https://github.com/ml-explore/mlx). Statevector, density matrix, and tensor network backends. Fully open source under Apache 2.0.

---

## Join the network

Three commands to become a node operator:

```bash
pip install zilver[network]
zilver-registry start --port 7701 &        # run once on one machine as the registry host
zilver-node start --registry http://<registry-ip>:7701
```

Your node auto-detects chip, RAM, and qubit ceiling. It registers, starts serving simulation jobs, and sends heartbeats every 30 seconds. That is it.

Early node operators will receive priority allocation of Sirius Quantum network rewards when the incentive layer launches. The simulation network is live now rewards are retroactive to genesis nodes.

---

## Hardware

Every Apple Silicon chip is supported. Qubit ceiling scales with unified memory.

| Chip | RAM | SV qubits | DM qubits | TN qubits |
|---|---|---|---|---|
| M1 | 8 GB | 28 | 14 | 50 |
| M1 / M2 | 16 GB | 30 | 15 | 50 |
| M1 Pro / M2 Pro | 32 GB | 31 | 15 | 50 |
| M1 Max / M2 Max | 64 GB | 32 | 16 | 50 |
| M1 Ultra / M2 Ultra | 128 GB | 33 | 16 | 50 |
| M3 / M4 | 16–24 GB | 30–31 | 15 | 50 |
| M3 Max / M4 Max | 64–128 GB | 32–33 | 16 | 50 |

SV = statevector (exact). DM = density matrix (noise-aware). TN = tensor network / MPS (sparse circuits, scales to 50+ qubits regardless of RAM).

---

## What Zilver does

**Local simulation**

MLX-native simulator with full `mx.vmap` and `mx.compile` support. One Metal dispatch evaluates an entire loss landscape 400 parameter points in a single GPU kernel instead of 400 sequential circuit runs.

```python
from zilver.circuit import hardware_efficient
from zilver.landscape import LossLandscape

circuit = hardware_efficient(n_qubits=6, depth=3)
result = LossLandscape(circuit, sweep_params=(0, 1), resolution=20).compute()

print(f"Plateau coverage:   {result.plateau_coverage():.1%}")
print(f"Trainability score: {result.trainability_score():.3f}")
print(f"Wall time:          {result.wall_time_seconds:.3f}s")
```

**Distributed network**

Submit jobs to any registered node. The coordinator finds an eligible node, dispatches the job, and returns a verified result with a cryptographic proof.

```python
from zilver.client import NetworkCoordinator
from zilver.node import SimJob

coord = NetworkCoordinator("http://<registry-ip>:7701")

job = SimJob(
    circuit_ops=[{"type": "ry", "qubits": [0], "param_idx": 0}],
    n_qubits=4, n_params=1, params=[1.57], backend="sv",
)
result = coord.submit(job)
print(result.expectation)
print(result.verify(job))   # True cryptographic proof check
```

**Gradient computation**

Parameter-shift gradients, fully batched over parameter vectors. Plug directly into any VQA optimizer.

```python
from zilver.gradients import param_shift_gradient

f = circuit.compile(observable="sum_z")
grads = param_shift_gradient(f, params)
```

---

## Install

**Simulator only**

```bash
pip install zilver
```

**Full network node**

```bash
pip install zilver[network]
```

Requirements: Apple Silicon Mac, macOS 13 Ventura or later, Python 3.10+.

---

## Gate library

| Gate | Type | Params |
|---|---|---|
| H, X, Y, Z, S, T | Fixed single-qubit | |
| RX, RY, RZ | Rotation | 1 |
| U3 | Universal single-qubit | 3 (theta, phi, lambda) |
| CNOT, CZ, SWAP, iSWAP | Fixed two-qubit | |
| RZZ, RXX | Ising coupling | 1 |
| CRZ | Controlled rotation | 1 |

All parameterized gates are MLX-native compatible with `mx.vmap` and `mx.compile`.

---

## Development

```bash
git clone https://github.com/deryakarl/zilver
cd zilver
pip install -e ".[dev,network]"
pytest tests/   # 464 tests
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

## Manifesto

We believe quantum compute should be open, distributed, and owned by the people who run it.

[Read the Sirius Quantum Manifesto](MANIFESTO.md)
