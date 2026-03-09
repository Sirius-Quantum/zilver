# Zilver

[![PyPI version](https://img.shields.io/pypi/v/zilver.svg)](https://pypi.org/project/zilver/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![MLX](https://img.shields.io/badge/MLX-0.18%2B-orange.svg)](https://github.com/ml-explore/mlx)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1--M4-black.svg)](https://www.apple.com/mac/)

**Open quantum simulation network for Apple Silicon.**

Zilver turns your Apple Silicon Mac into a quantum simulation node.

Built on [MLX](https://github.com/ml-explore/mlx). Statevector, density matrix, and tensor network backends.

---

## Run a node

Got an Apple Silicon Mac? You can contribute compute to the network.

See [NODES.md](NODES.md) for requirements and how to apply.

---

## Submit jobs

Researchers and AI labs can submit quantum simulation jobs to the network via API. No infrastructure required — bring your circuit, we handle the compute.

To request API access, open an issue at [github.com/Sirius-Quantum/zilver](https://github.com/Sirius-Quantum/zilver) with your use case and institution.

```python
from zilver.client import NetworkCoordinator
from zilver.node import SimJob

coord = NetworkCoordinator(
    "https://registry.siriusquantum.com",
    client_api_key="your-api-key",
)

job = SimJob(
    circuit_ops=[{"type": "ry", "qubits": [0], "param_idx": 0}],
    n_qubits=4, n_params=1, params=[1.57], backend="sv",
)
result = coord.submit(job)
print(result.expectation)
print(result.verify(job))   # True
```

---

## Local simulation

Use Zilver as a standalone MLX-native quantum simulator on any Apple Silicon Mac — no network required.

```bash
pip install zilver
```

```python
from zilver.circuit import hardware_efficient
from zilver.landscape import LossLandscape

circuit = hardware_efficient(n_qubits=6, depth=3)
result = LossLandscape(circuit, sweep_params=(0, 1), resolution=20).compute()

print(f"Plateau coverage:   {result.plateau_coverage():.1%}")
print(f"Trainability score: {result.trainability_score():.3f}")
print(f"Wall time:          {result.wall_time_seconds:.3f}s")
```

Parameter-shift gradients, fully batched:

```python
from zilver.circuit import hardware_efficient
from zilver.gradients import param_shift_gradient
import mlx.core as mx

circuit = hardware_efficient(n_qubits=4, depth=2)
f = circuit.compile(observable="sum_z")
params = mx.zeros([circuit.n_params])
grads = param_shift_gradient(f, params)
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

## Manifesto

[Read the Sirius Quantum Manifesto](MANIFESTO.md)
