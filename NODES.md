# Node Operator Guide

Contribute Apple Silicon compute to the Sirius Quantum network. Nodes run the simulation workload — the more unified memory, the larger the circuits your node can handle.

## Hardware

- Apple Silicon Mac — M1 or later
- macOS 13 Ventura or later
- Python 3.10, 3.11, or 3.12
- 16 GB unified memory minimum; 32 GB or more recommended

Mac mini is the preferred operator hardware. It runs silently, draws little power, and the M4 Pro configuration with 64 GB unified memory handles circuits up to 32 qubits in statevector mode. Mac Studio with M2 Ultra or M4 Max reaches the practical ceiling of what a single node can simulate.

Zilver is macOS-only. The simulation engine is built on [MLX](https://github.com/ml-explore/mlx), which requires Apple's Metal GPU framework and unified memory architecture.

## Apply

Node registration is invite-only. Open an issue at [github.com/Sirius-Quantum/zilver](https://github.com/Sirius-Quantum/zilver) with your chip model and available RAM to apply. Include your intended uptime if you have a preference for how jobs are routed to your node.

## Install and start

```bash
pip install "zilver[network]"
zilver-node start \
  --registry https://registry.siriusquantum.com \
  --public-url https://your-public-address.com \
  --backends sv,dm
```

To run your node continuously, add the start command to macOS Login Items or create a launchd plist.

## Public URL

Your node must be reachable from the internet. If your Mac is on a home or office network, use one of the following:

- **Cloudflare Tunnel** — `cloudflared tunnel --url http://localhost:7700` — free, no open port or router access needed
- **Port forwarding** — forward port 7700 on your router to your Mac's LAN IP
- **ngrok** — `ngrok http 7700`
- **VPS proxy** — nginx or Caddy reverse proxy on a VPS

## Identity and credentials

On first run, a hardware-bound node identity is generated and stored in macOS Keychain. Subsequent starts reuse it automatically.

**Moving to a new Mac:** node identity is hardware-bound and cannot be transferred. On a new machine, `zilver-node start` generates a fresh identity. Contact [dev@siriusquantum.com](mailto:dev@siriusquantum.com) to migrate your operator approval to the new node.

**If Keychain is wiped:** the node re-registers automatically on next start. Contribution history is tied to node ID, not credentials.

## Qubit ceilings

Auto-detected at startup based on chip and RAM. No configuration needed.

| Chip | RAM | SV | DM | TN |
|---|---|---|---|---|
| M1 / M2 | 8 GB | 28q | 14q | 50q |
| M1 / M2 | 16 GB | 30q | 15q | 50q |
| M1 Pro / M2 Pro | 32 GB | 31q | 15q | 50q |
| M1 Max / M2 Max | 64 GB | 32q | 16q | 50q |
| M1 Ultra / M2 Ultra | 128 GB | 33q | 16q | 50q |
| M3 / M4 | 16–24 GB | 30–31q | 15q | 50q |
| M3 Max / M4 Max | 64–128 GB | 32–33q | 16q | 50q |
| M4 Ultra (Mac Studio) | 192 GB | 34q | 17q | 50q |

## Backends

**`sv`** — Statevector. Exact simulation, up to ~34 qubits on M4 Ultra hardware.

**`dm`** — Density matrix. Noise-aware simulation; roughly half the qubit ceiling of `sv` for equivalent RAM.

**`tn`** — Tensor network. Scales to 50+ qubits on low-entanglement circuits, largely independent of RAM.

Enable all three with `--backends sv,dm,tn`. The registry routes jobs to the appropriate backend based on circuit type and node capability.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--backends` | `sv` | Backends to enable: `sv`, `dm`, `tn`, or any combination |
| `--port` | `7700` | Port to listen on |
| `--registry` | — | Registry URL |
| `--public-url` | — | Publicly reachable URL for this node |
| `--wallet` | — | Wallet address for future reward settlement |

## Troubleshooting

**`ModuleNotFoundError: No module named 'fastapi'`**
```bash
pip install "zilver[network]"
```

**Node starts but no jobs arrive**
Confirm your node is in the approved list and your public URL is reachable from outside your network.

**Missed heartbeats / stale node**
Network interruptions are tolerated. If the node is marked stale, restart it — re-registration is automatic.
