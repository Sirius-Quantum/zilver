# Node Operator Guide

Run a Zilver simulation node on your Apple Silicon Mac and contribute compute to the Sirius Quantum network.

---

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13 Ventura or later
- Python 3.10, 3.11, or 3.12
- At least 8 GB unified memory

---

## Apply

Node operators require approval before joining the network. Open an issue at [github.com/Sirius-Quantum/zilver](https://github.com/Sirius-Quantum/zilver) with your machine specs to apply.

---

## Install and run

```bash
pip install zilver[network]
zilver-node start --registry https://registry.siriusquantum.com --backends sv,dm
```

On first run your node identity and credentials are generated and stored in macOS Keychain. Subsequent starts reuse them automatically.

---

## Flags

| Flag | Default | Description |
|---|---|---|
| `--backends` | `sv` | Backends to enable: `sv`, `dm`, `tn`, or any combination |
| `--port` | `7700` | Port to listen on |
| `--registry` | — | Registry URL |
| `--wallet` | — | Wallet address for future reward settlement |

---

## Qubit ceilings

Auto-detected on startup. No configuration needed.

| Chip | RAM | SV | DM | TN |
|---|---|---|---|---|
| M1 | 8 GB | 28q | 14q | 50q |
| M1 / M2 | 16 GB | 30q | 15q | 50q |
| M1 Pro / M2 Pro | 32 GB | 31q | 15q | 50q |
| M1 Max / M2 Max | 64 GB | 32q | 16q | 50q |
| M1 Ultra / M2 Ultra | 128 GB | 33q | 16q | 50q |
| M3 / M4 | 16–24 GB | 30–31q | 15q | 50q |
| M3 Max / M4 Max | 64–128 GB | 32–33q | 16q | 50q |

---

## Backends

**`sv`** — Statevector. Exact simulation, up to ~33 qubits on M-Ultra hardware.

**`dm`** — Density matrix. Noise-aware simulation, roughly half the qubit count of SV for the same RAM.

**`tn`** — Tensor network. Scales to 50+ qubits for low-entanglement circuits regardless of RAM.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'fastapi'`**
```bash
pip install zilver[network]
```

**Node starts but no jobs arrive**
Check you are in the approved list and the registry is reachable:
```bash
zilver-node status --registry https://registry.siriusquantum.com
```

**Missed heartbeats / stale node**
Network interruptions are tolerated. If the node is marked stale, restart it and it will re-register automatically.
