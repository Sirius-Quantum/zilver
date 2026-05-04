# Node Operator Guide

Run a Zilver simulation node on your Apple Silicon Mac and contribute compute to the Sirius Quantum network.

---

## Requirements

- Apple Silicon Mac (M1 or later) running macOS 13 Ventura or later
- Python 3.10, 3.11, or 3.12
- At least 8 GB unified memory

Zilver runs on macOS only. The simulation engine is built on [MLX](https://github.com/ml-explore/mlx), which requires Apple's Metal GPU framework and unified memory architecture. Linux, Windows, and Intel Macs are not supported.

---

## Apply

Node operators require approval before joining the network. Open an issue at [github.com/Sirius-Quantum/zilver](https://github.com/Sirius-Quantum/zilver) with your machine specs to apply.

---

## Install and run

```bash
pip install "zilver[network]"
zilver-node start --registry https://registry.siriusquantum.com --backends sv,dm
```

The `[network]` extra installs all required dependencies (`fastapi`, `uvicorn`, `cryptography`). MLX is installed automatically as a core dependency — no separate download needed.

To start your node automatically at login, add the `zilver-node start` command to your macOS Login Items or create a launchd plist.

---

## Identity and credentials

On first run, a hardware-bound node identity is generated and your credentials are stored in macOS Keychain. Subsequent starts reuse them automatically — no manual key management needed.

**Moving to a new Mac:** node identity is hardware-bound and cannot be transferred. On a new machine, `zilver-node start` generates a fresh identity. Contact the Sirius Quantum team to migrate your operator approval to the new node pubkey.

**If Keychain is wiped:** the node will re-register with a new API key on next start. Your node's contribution history is preserved by node ID, not by credentials.

---

## Public URL

Your node must be reachable from the internet. Pass your public URL at startup:

```bash
zilver-node start \
  --registry https://registry.siriusquantum.com \
  --public-url https://your-public-address.com
```

If your Mac is on a home network, use one of these options:

- **Port forwarding** — forward port 7700 on your router to your Mac's LAN IP
- **Cloudflare Tunnel** — `cloudflared tunnel --url http://localhost:7700` (free, no open port needed)
- **ngrok** — `ngrok http 7700` (free tier available)
- **VPS proxy** — nginx or Caddy reverse proxy on a cheap VPS

---

## Flags

| Flag | Default | Description |
|---|---|---|
| `--backends` | `sv` | Backends to enable: `sv`, `dm`, `tn`, or any combination |
| `--port` | `7700` | Port to listen on |
| `--registry` | — | Registry URL |
| `--public-url` | — | Publicly reachable URL for this node |
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
