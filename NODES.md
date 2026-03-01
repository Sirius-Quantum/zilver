# Node Operator Guide

Everything you need to run a Zilver simulation node on your Apple Silicon Mac.

---

## Prerequisites

- Apple Silicon Mac (M1, M2, M3, or M4 — any variant)
- macOS 13 Ventura or later
- Python 3.10, 3.11, or 3.12
- At least 8 GB unified memory

---

## Install

```bash
pip install zilver[network]
```

This pulls in the simulator core (MLX, NumPy) plus the network layer (FastAPI, uvicorn, httpx). If you only want the local simulator without running a node, `pip install zilver` is enough.

---

## Quick start — single machine

Run both the registry and a node on the same Mac:

```bash
# Terminal 1 — registry
zilver-registry start --port 7701

# Terminal 2 — node
zilver-node start --backends sv,dm,tn --port 7700 --registry http://localhost:7701
```

Submit a job from Python:

```python
from zilver.client import NetworkCoordinator
from zilver.node import SimJob

coord = NetworkCoordinator("http://localhost:7701")

job = SimJob(
    circuit_ops=[{"type": "h", "qubits": [0], "param_idx": None}],
    n_qubits=4, n_params=0, params=[], backend="sv",
)
result = coord.submit(job)
print(result.expectation)
print(result.verify(job))   # True
```

---

## Multi-node setup — same LAN

One machine runs the registry. All other machines run nodes pointing at it.

```
Mac A (192.168.1.10)           Mac B (192.168.1.20)
  zilver-registry :7701   ←——   zilver-node :7700 --registry http://192.168.1.10:7701
  zilver-node :7700

Mac C (192.168.1.30)
  zilver-node :7700 --registry http://192.168.1.10:7701
```

**Mac A (registry host + node)**

```bash
zilver-registry start --port 7701 &
zilver-node start \
  --backends sv,dm,tn \
  --port 7700 \
  --registry http://localhost:7701
```

**Mac B and Mac C (nodes only)**

```bash
zilver-node start \
  --backends sv,dm \
  --port 7700 \
  --registry http://192.168.1.10:7701
```

The node advertises its LAN IP automatically. The registry stores it and returns it to coordinators during matchmaking.

---

## CLI reference

### `zilver-node start`

| Flag | Default | Description |
|---|---|---|
| `--backends` | `sv` | Comma-separated list: `sv`, `dm`, `tn`, or any combination |
| `--port` | `7700` | HTTP port to listen on |
| `--registry` | required | Registry URL, e.g. `http://192.168.1.10:7701` |
| `--node-id` | auto | Custom node identifier (default: auto-generated UUID) |
| `--wallet` | — | Wallet address for future reward settlement |

### `zilver-node status`

```bash
zilver-node status --registry http://192.168.1.10:7701
```

Prints online node count, available backends, and max qubit ceiling.

### `zilver-node nodes`

```bash
zilver-node nodes --registry http://192.168.1.10:7701
```

Lists all registered nodes with their chip, RAM, and backend capabilities.

### `zilver-registry start`

| Flag | Default | Description |
|---|---|---|
| `--port` | `7701` | HTTP port to listen on |
| `--host` | `0.0.0.0` | Bind address |

---

## Qubit ceilings by chip

Your node auto-detects these on startup. No manual configuration needed.

| Chip | RAM | SV max | DM max | TN max |
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

**`sv` — Statevector (default)**
Exact simulation. Memory is `2^n × 8 bytes` (complex64). Suitable for up to ~33 qubits on M-Ultra hardware. Best for VQA gradient computation and loss landscape sweeps.

**`dm` — Density matrix**
Noise-aware simulation with Kraus operators. Memory is `4^n × 8 bytes` — roughly half the qubit count of SV for the same RAM. Use this when your circuit includes noise channels.

**`tn` — Tensor network / MPS**
Memory-efficient for sparse, low-entanglement circuits. Scales to 50+ qubits regardless of RAM. Accuracy degrades for high-entanglement circuits (large bond dimension).

Enable all three with `--backends sv,dm,tn`. The registry routes each job to a node that supports the requested backend and qubit count.

---

## Node lifecycle

1. On start, the node calls `NodeCapabilities.detect()` to read chip and RAM.
2. It registers with the registry via `POST /nodes`.
3. A background daemon thread sends `POST /nodes/{id}/heartbeat` every 30 seconds.
4. Incoming jobs arrive at `POST /execute`. Each job runs in a thread-pool executor — the event loop is never blocked.
5. On `SIGINT` (Ctrl-C), the node deregisters cleanly before exiting.

---

## Verification

Every job result includes a SHA-256 HMAC proof over `(job_id, params, expectation)`. Clients can verify locally:

```python
assert result.verify(job)
```

The network also runs redundant execution and spot-check circuits to detect faulty nodes. Nodes that return incorrect results accumulate flags and are deregistered automatically.

---

## Firewall

The node and registry use plain HTTP on the ports you specify. If your Mac's firewall is enabled, allow inbound connections on port 7700 (node) and 7701 (registry).

On macOS, go to System Settings → Network → Firewall → Options and add the Python executable, or use `pfctl` to open the ports explicitly.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'fastapi'`**
Install the network extras: `pip install zilver[network]`

**Node registers but jobs fail with 422**
The requested backend or qubit count exceeds your node's ceiling. Check with:
```bash
zilver-node status --registry http://<registry>:7701
```

**Registry not reachable from other Macs**
Ensure the registry is bound to `0.0.0.0` (default) and the port is reachable on your LAN. Test with `curl http://<registry-ip>:7701/summary`.

**Heartbeat failures in logs**
Network interruptions are tolerated — the node retries on the next 30-second cycle. If the registry marks the node stale (missed 3 heartbeats), restart the node and it will re-register.
