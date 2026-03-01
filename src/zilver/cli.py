"""CLI entry points."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time


# ---------------------------------------------------------------------------
# Heartbeat daemon thread
# ---------------------------------------------------------------------------

def _start_heartbeat(reg_client: "RegistryClient", node_id: str, interval: int = 30) -> None:
    """
    Send periodic heartbeats to the registry from a background daemon thread.

    The thread is marked as a daemon so it is killed automatically when the
    main process exits.  ``interval`` is the sleep duration in seconds between
    heartbeat calls.  Failures are silently swallowed — a transient registry
    outage should not crash the node.
    """
    from .client import RegistryClient  # local import to avoid circular at module level

    def _loop() -> None:
        while True:
            time.sleep(interval)
            try:
                reg_client.heartbeat(node_id)
            except Exception:
                pass  # transient failure — next tick will retry

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# zilver-node commands
# ---------------------------------------------------------------------------

def _cmd_node_start(args: argparse.Namespace) -> None:
    """
    Detect hardware, register with the registry, and serve simulation jobs.

    Flow
    ----
    1. Auto-detect chip, RAM, and qubit ceilings via ``NodeCapabilities.detect()``.
    2. Initialise a ``Node`` with the requested backends.
    3. Register capabilities and advertised URL with the registry server.
    4. Spawn a daemon thread that sends a heartbeat every 30 s.
    5. Register a SIGINT/SIGTERM handler that deregisters the node cleanly.
    6. Start uvicorn — blocks until the process is killed.
    """
    from .node import Node
    from .server import make_app
    from .client import RegistryClient

    backends = [b.strip() for b in args.backends.split(",")]
    node = Node.start(backends=backends, node_id=args.node_id, wallet=args.wallet)

    print(f"Node {node.caps.node_id[:8]} | chip: {node.caps.chip} | "
          f"RAM: {node.caps.ram_gb}GB | backends: {node.caps.backends} | "
          f"sv_max: {node.caps.sv_qubits_max}q")

    # Construct the URL this node will advertise to the registry
    advertised_host = args.host if args.host != "0.0.0.0" else _local_ip()
    node_url = f"http://{advertised_host}:{args.port}"

    reg_client: RegistryClient | None = None

    if args.registry:
        reg_client = RegistryClient(args.registry)
        try:
            reg_client.register(node.caps, node_url)
            print(f"Registered with registry at {args.registry}")
            _start_heartbeat(reg_client, node.caps.node_id)
        except Exception as exc:
            print(f"Warning: could not register with registry: {exc}", file=sys.stderr)

    def _deregister(sig: int, frame: object) -> None:
        if reg_client is not None:
            try:
                reg_client.deregister(node.caps.node_id)
                print("\nDeregistered from registry.")
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  _deregister)
    signal.signal(signal.SIGTERM, _deregister)

    print(f"Serving on {args.host}:{args.port}  (Ctrl-C to stop)")

    import uvicorn
    app = make_app(node)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def _cmd_node_status(args: argparse.Namespace) -> None:
    """Print a summary of the registry to stdout."""
    from .client import RegistryClient
    reg = RegistryClient(args.registry)
    s = reg.summary()
    print(f"Registry: {args.registry}")
    print(f"  Online nodes  : {s.get('online', 0)}")
    print(f"  Registered    : {s.get('total_registered', 0)}")
    print(f"  Backends      : {', '.join(s.get('backends', []))}")
    print(f"  Max SV qubits : {s.get('max_sv_qubits', 0)}")
    print(f"  Max DM qubits : {s.get('max_dm_qubits', 0)}")
    print(f"  Total stake   : {s.get('total_stake', 0)}")


def _cmd_node_list(args: argparse.Namespace) -> None:
    """List all online nodes in the registry."""
    from .client import RegistryClient
    reg = RegistryClient(args.registry)
    nodes = reg.nodes()
    if not nodes:
        print("No online nodes.")
        return
    print(f"{'NODE ID':36}  {'CHIP':20}  {'BACKENDS':12}  {'SV MAX':7}  URL")
    print("-" * 100)
    for n in nodes:
        nid      = n.get("node_id", "")[:36]
        chip     = n.get("chip", "")[:20]
        backends = ",".join(n.get("backends", []))
        sv_max   = n.get("sv_qubits_max", 0)
        url      = n.get("url", "")
        print(f"{nid:36}  {chip:20}  {backends:12}  {sv_max:7}  {url}")


# ---------------------------------------------------------------------------
# zilver-registry commands
# ---------------------------------------------------------------------------

def _cmd_registry_start(args: argparse.Namespace) -> None:
    """Start an in-memory capability registry server."""
    from .registry_server import serve_registry
    print(f"Registry server listening on {args.host}:{args.port}  (Ctrl-C to stop)")
    serve_registry(host=args.host, port=args.port)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_ip() -> str:
    """
    Best-effort detection of the machine's LAN IP address.

    Falls back to ``"127.0.0.1"`` if detection fails (e.g. no network).
    Used to build the node URL advertised to the registry so that other
    machines can reach this node directly.
    """
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------

def _build_node_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zilver-node",
        description="Zilver simulation node daemon.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- start --------------------------------------------------------------
    p_start = sub.add_parser("start", help="Start the node daemon.")
    p_start.add_argument(
        "--backends", default="sv",
        help="Comma-separated list of backends to enable: sv,dm,tn (default: sv).",
    )
    p_start.add_argument(
        "--port", type=int, default=7700,
        help="TCP port to listen on (default: 7700).",
    )
    p_start.add_argument(
        "--host", default="0.0.0.0",
        help="Interface to bind (default: 0.0.0.0).",
    )
    p_start.add_argument(
        "--registry", default=None,
        help="Registry server URL, e.g. http://host:7701. "
             "Omit to run standalone without registry registration.",
    )
    p_start.add_argument(
        "--node-id", dest="node_id", default=None,
        help="Explicit node identifier (auto-generated UUID if omitted).",
    )
    p_start.add_argument(
        "--wallet", default=None,
        help="Wallet address for future reward settlement (stored, not yet used).",
    )

    # --- status -------------------------------------------------------------
    p_status = sub.add_parser("status", help="Print registry summary.")
    p_status.add_argument(
        "--registry", required=True,
        help="Registry server URL.",
    )

    # --- nodes --------------------------------------------------------------
    p_nodes = sub.add_parser("nodes", help="List online nodes in the registry.")
    p_nodes.add_argument(
        "--registry", required=True,
        help="Registry server URL.",
    )

    return parser


def _build_registry_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zilver-registry",
        description="Zilver capability registry server.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start", help="Start the registry server.")
    p_start.add_argument(
        "--host", default="0.0.0.0",
        help="Interface to bind (default: 0.0.0.0).",
    )
    p_start.add_argument(
        "--port", type=int, default=7701,
        help="TCP port to listen on (default: 7701).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry points registered in pyproject.toml
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the ``zilver-node`` script."""
    parser = _build_node_parser()
    args = parser.parse_args()

    dispatch = {
        "start":  _cmd_node_start,
        "status": _cmd_node_status,
        "nodes":  _cmd_node_list,
    }
    dispatch[args.command](args)


def main_registry() -> None:
    """Entry point for the ``zilver-registry`` script."""
    parser = _build_registry_parser()
    args = parser.parse_args()

    dispatch = {
        "start": _cmd_registry_start,
    }
    dispatch[args.command](args)
