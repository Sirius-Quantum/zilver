"""CLI entry points."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path


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
# TLS certificate helpers
# ---------------------------------------------------------------------------

_ZILVER_DIR = Path.home() / ".zilver"


def _resolve_tls(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """
    Return (ssl_keyfile, ssl_certfile) to pass to uvicorn.

    If both ``--ssl-key`` and ``--ssl-cert`` are provided, use them directly.
    If neither is provided, auto-generate a self-signed cert into ``~/.zilver/``
    on first run and reuse it on subsequent runs.
    If only one is provided, raise an error.
    """
    key  = getattr(args, "ssl_key",  None)
    cert = getattr(args, "ssl_cert", None)

    if key and cert:
        return key, cert

    if (key is None) != (cert is None):
        sys.exit("Error: --ssl-key and --ssl-cert must be provided together.")

    # Auto-generate if neither is set
    auto_key  = _ZILVER_DIR / "node.key"
    auto_cert = _ZILVER_DIR / "node.crt"

    if not (auto_key.exists() and auto_cert.exists()):
        print("Generating self-signed TLS certificate …", file=sys.stderr)
        try:
            from . import _node_ops
            _node_ops.ensure_tls_cert(_ZILVER_DIR)
            print(f"Certificate written to {_ZILVER_DIR}/node.{{key,crt}}", file=sys.stderr)
        except ImportError:
            print(
                "Warning: cryptography package not installed; "
                "starting without TLS. Install with: pip install zilver[network]",
                file=sys.stderr,
            )
            return None, None

    return str(auto_key), str(auto_cert)


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
    3. Resolve TLS certificate (explicit or auto-generated self-signed).
    4. Resolve API key: explicit flag → Keychain → register and store.
    5. Register capabilities and advertised URL with the registry server.
    6. Spawn a daemon thread that sends a heartbeat every 30 s.
    7. Register a SIGINT/SIGTERM handler that deregisters the node cleanly.
    8. Start uvicorn — blocks until the process is killed.
    """
    from .node import Node
    from .server import serve
    from .client import RegistryClient

    backends = [b.strip() for b in args.backends.split(",")]

    # --- Keypair first — node_id is derived from pubkey ---------------------
    private_key_bytes: bytes | None = None
    public_key_bytes:  bytes | None = None
    derived_node_id:   str   | None = None
    _se_label:         str   | None = None
    try:
        from . import _node_ops
        private_key_bytes, public_key_bytes, _se_label, derived_node_id = _node_ops.load_identity()
    except ImportError:
        pass
    except Exception as exc:
        print(f"Warning: could not load node identity: {exc}", file=sys.stderr)

    node = Node.start(
        backends           = backends,
        node_id            = derived_node_id,
        wallet             = args.wallet,
        private_key_bytes  = private_key_bytes,
        public_key_bytes   = public_key_bytes,
        se_label           = _se_label,
    )

    print(f"Node {node.caps.node_id[:8]} | chip: {node.caps.chip} | "
          f"RAM: {node.caps.ram_gb}GB | backends: {node.caps.backends} | "
          f"sv_max: {node.caps.sv_qubits_max}q")
    if public_key_bytes is not None:
        print(f"Node pubkey: {public_key_bytes.hex()}  (add to registry allowlist)")

    # --- TLS ----------------------------------------------------------------
    ssl_key, ssl_cert = _resolve_tls(args)
    scheme = "https" if ssl_cert else "http"

    # Construct the URL this node will advertise to the registry
    advertised_host = args.host if args.host != "0.0.0.0" else _local_ip()
    node_url = f"{scheme}://{advertised_host}:{args.port}"

    # --- API key ------------------------------------------------------------
    api_key: str | None = getattr(args, "api_key", None)

    reg_client: RegistryClient | None = None

    if args.registry:
        reg_client = RegistryClient(args.registry)
        try:
            if api_key is None:
                try:
                    from . import _node_ops
                    api_key = _node_ops.load_api_key(node.caps.node_id)
                except Exception:
                    pass

            if api_key is None:
                reg_client.register(node.caps, node_url,
                                    private_key_bytes=private_key_bytes,
                                    public_key_bytes=public_key_bytes,
                                    se_label=_se_label)
                api_key = reg_client.last_api_key
                if api_key:
                    try:
                        from . import _node_ops
                        _node_ops.store_api_key(node.caps.node_id, api_key)
                        print("API key stored in macOS Keychain.")
                    except Exception as exc:
                        print(f"Warning: could not store API key in Keychain: {exc}",
                              file=sys.stderr)
            else:
                reg_client.register(node.caps, node_url,
                                    private_key_bytes=private_key_bytes,
                                    public_key_bytes=public_key_bytes,
                                    se_label=_se_label)
                # Registry always issues a fresh api_key on re-registration.
                # Update Keychain and local variable so the node serves with
                # the key the registry will return to clients via /match.
                new_key = reg_client.last_api_key
                if new_key and new_key != api_key:
                    api_key = new_key
                    try:
                        from . import _node_ops
                        _node_ops.store_api_key(node.caps.node_id, api_key)
                    except Exception:
                        pass

            # Set api_key on the registry client so that subsequent heartbeat
            # calls include the Authorization header required by the registry.
            if api_key:
                reg_client.api_key = api_key

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

    proto = "HTTPS" if ssl_cert else "HTTP (no TLS)"
    print(f"Serving {proto} on {args.host}:{args.port}  (Ctrl-C to stop)")
    if ssl_cert and not getattr(args, "ssl_cert", None):
        print("Warning: using self-signed certificate — clients need --no-verify or verify=False",
              file=sys.stderr)

    serve(
        node,
        host=args.host,
        port=args.port,
        log_level="warning",
        api_key=api_key,
        ssl_keyfile=ssl_key,
        ssl_certfile=ssl_cert,
    )


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


def _cmd_node_dashboard(args: argparse.Namespace) -> None:
    """
    Live Rich TUI dashboard showing all active nodes in the registry.

    Polls the registry every ``--interval`` seconds and re-renders a table
    with hardware info, memory capacity, jobs completed, and live status.
    Press Ctrl-C to exit.
    """
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box

    from .client import RegistryClient
    from .node import estimate_memory_bytes

    reg      = RegistryClient(args.registry)
    console  = Console()
    interval = args.interval

    def _fmt_bytes(b: int) -> str:
        if b >= 1024 ** 3:
            return f"{b / 1024**3:.0f} GB"
        if b >= 1024 ** 2:
            return f"{b / 1024**2:.0f} MB"
        return f"{b / 1024:.0f} KB"

    def _build() -> Panel:
        try:
            summary = reg.summary()
            nodes   = reg.nodes()
        except Exception as exc:
            return Panel(
                f"[red bold]Cannot reach registry:[/red bold] {exc}",
                title="[bold]Zilver Network[/bold]",
                border_style="red",
            )

        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold cyan",
            pad_edge=False,
            expand=True,
        )
        table.add_column("STATUS", width=2, no_wrap=True)
        table.add_column("NODE ID",    width=10, no_wrap=True)
        table.add_column("CHIP",       min_width=16, no_wrap=True)
        table.add_column("RAM",        width=6,  justify="right")
        table.add_column("SV Q",       width=5,  justify="right")
        table.add_column("DM Q",       width=5,  justify="right")
        table.add_column("TN Q",       width=5,  justify="right")
        table.add_column("MAX MEM",    width=8,  justify="right")
        table.add_column("BACKENDS",   width=12, no_wrap=True)
        table.add_column("JOBS",       width=6,  justify="right")
        table.add_column("URL",        min_width=20)

        for n in nodes:
            nid      = n.get("node_id", "")[:8]
            chip     = n.get("chip", "unknown")
            ram      = f"{n.get('ram_gb', 0)} GB"
            sv_q     = n.get("sv_qubits_max", 0)
            dm_q     = n.get("dm_qubits_max", 0)
            tn_q     = n.get("tn_qubits_max", 0)
            backends = ", ".join(n.get("backends", []))
            jobs     = str(n.get("jobs_completed", 0))
            url      = n.get("url", "")
            max_mem  = _fmt_bytes(estimate_memory_bytes(sv_q, "sv"))

            table.add_row(
                "[green]●[/green]",
                f"[dim]{nid}[/dim]",
                chip,
                ram,
                str(sv_q),
                str(dm_q),
                str(tn_q),
                max_mem,
                f"[cyan]{backends}[/cyan]",
                f"[yellow]{jobs}[/yellow]",
                f"[dim]{url}[/dim]",
            )

        if not nodes:
            table.add_row("", "[dim]no nodes registered[/dim]",
                          "", "", "", "", "", "", "", "", "")

        online   = summary.get("online", 0)
        backends = ", ".join(summary.get("backends", []))
        max_sv   = summary.get("max_sv_qubits", 0)
        stake    = summary.get("total_stake", 0)

        subtitle = (
            f"[green]{online}[/green] online  "
            f"│  backends: [cyan]{backends or '—'}[/cyan]  "
            f"│  max sv: [yellow]{max_sv}q[/yellow]  "
            f"│  stake: {stake}  "
            f"│  [dim]refresh {interval}s[/dim]"
        )

        return Panel(
            table,
            title="[bold white]Zilver Network[/bold white]",
            subtitle=subtitle,
            border_style="blue",
        )

    try:
        with Live(_build(), console=console, refresh_per_second=2, screen=False) as live:
            while True:
                time.sleep(interval)
                live.update(_build())
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard stopped.[/dim]")


def _cmd_node_leaderboard(args: argparse.Namespace) -> None:
    """
    Fetch and print the SQT token leaderboard from the registry.

    Displays the top nodes by balance in a Rich table with rank, node ID,
    SQT balance, jobs completed, heartbeats sent, and genesis status.
    """
    from rich.console import Console
    from rich.table import Table
    from rich import box
    import httpx

    url = args.registry.rstrip("/")
    try:
        resp = httpx.get(f"{url}/leaderboard", params={"top_n": args.top_n}, timeout=10)
        resp.raise_for_status()
        entries = resp.json()
    except Exception as exc:
        print(f"Error fetching leaderboard: {exc}", file=sys.stderr)
        sys.exit(1)

    console = Console()
    if not entries:
        console.print("[dim]Leaderboard is empty — no contributions recorded yet.[/dim]")
        return

    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
        title="[bold white]SQT Leaderboard[/bold white]",
        title_style="bold",
    )
    table.add_column("RANK",       width=5,  justify="right")
    table.add_column("NODE ID",    width=12, no_wrap=True)
    table.add_column("BALANCE",    width=12, justify="right")
    table.add_column("JOBS",       width=7,  justify="right")
    table.add_column("HEARTBEATS", width=11, justify="right")
    table.add_column("GENESIS",    width=8,  justify="center")
    table.add_column("REGISTERED", width=24, no_wrap=True)

    for e in entries:
        genesis = "[green]✓[/green]" if e.get("is_genesis") else "[dim]—[/dim]"
        table.add_row(
            str(e.get("rank", "")),
            f"[dim]{e.get('node_id', '')[:12]}[/dim]",
            f"[yellow]{e.get('balance', 0.0):.2f}[/yellow]",
            str(e.get("jobs_done", 0)),
            str(e.get("heartbeats", 0)),
            genesis,
            e.get("registered_at", "")[:19],
        )

    console.print(table)


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

    admin_key       = getattr(args, "admin_key",       None) or os.environ.get("ZILVER_REGISTRY_KEY")
    ledger_path     = getattr(args, "ledger_path",     None) or os.environ.get("ZILVER_LEDGER_PATH")
    require_signed  = getattr(args, "require_signed",  False)

    # Load node allowlist from file (one pubkey hex per line, # comments allowed)
    allowed_pubkeys: set[str] | None = None
    allowed_pubkeys_file = getattr(args, "allowed_pubkeys_file", None) or os.environ.get("ZILVER_ALLOWED_PUBKEYS_FILE")
    if allowed_pubkeys_file:
        try:
            lines = Path(allowed_pubkeys_file).read_text().splitlines()
            allowed_pubkeys = {l.strip().split("#")[0].strip() for l in lines
                               if l.strip() and not l.strip().startswith("#")}
            allowed_pubkeys.discard("")
            print(f"Node allowlist loaded: {len(allowed_pubkeys)} approved pubkey(s).")
        except Exception as exc:
            sys.exit(f"Error: could not read --allowed-pubkeys-file: {exc}")

    # Load client API keys from file (one key per line, # comments allowed)
    client_keys: set[str] | None = None
    client_keys_file = getattr(args, "client_keys_file", None) or os.environ.get("ZILVER_CLIENT_KEYS_FILE")
    if client_keys_file:
        try:
            lines = Path(client_keys_file).read_text().splitlines()
            client_keys = {l.strip().split("#")[0].strip() for l in lines
                           if l.strip() and not l.strip().startswith("#")}
            client_keys.discard("")
            print(f"Client key auth enabled: {len(client_keys)} authorized client(s).")
        except Exception as exc:
            sys.exit(f"Error: could not read --client-keys-file: {exc}")

    ssl_key, ssl_cert = _resolve_tls(args)

    if admin_key:
        print("Registry admin key is set — deregistration requires Authorization header.")
    else:
        print("Warning: no --admin-key set; deregistration endpoint is unprotected.",
              file=sys.stderr)

    if ledger_path:
        print(f"SQT ledger: {ledger_path}")

    proto = "HTTPS" if ssl_cert else "HTTP (no TLS)"
    print(f"Registry server {proto} on {args.host}:{args.port}  (Ctrl-C to stop)")

    if require_signed:
        print("Signed registration enforced — nodes must present Ed25519 signature.")

    serve_registry(
        host=args.host,
        port=args.port,
        admin_key=admin_key,
        rate_limit=True,
        ssl_keyfile=ssl_key,
        ssl_certfile=ssl_cert,
        ledger_path=ledger_path,
        require_signed=require_signed,
        allowed_pubkeys=allowed_pubkeys,
        client_keys=client_keys,
    )


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
        help="Registry server URL, e.g. https://host:7701. "
             "Omit to run standalone without registry registration.",
    )
    p_start.add_argument(
        "--wallet", default=None,
        help="Wallet address for future reward settlement (stored, not yet used).",
    )
    p_start.add_argument(
        "--ssl-cert", dest="ssl_cert", default=None,
        help="Path to TLS certificate (PEM). Auto-generates self-signed if omitted.",
    )
    p_start.add_argument(
        "--ssl-key", dest="ssl_key", default=None,
        help="Path to TLS private key (PEM). Must be paired with --ssl-cert.",
    )
    p_start.add_argument(
        "--api-key", dest="api_key", default=None,
        help="API key issued by the registry. "
             "If omitted, reads from Keychain or registers automatically.",
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

    # --- dashboard ----------------------------------------------------------
    p_dash = sub.add_parser("dashboard", help="Live Rich TUI showing active nodes.")
    p_dash.add_argument(
        "--registry", required=True,
        help="Registry server URL.",
    )
    p_dash.add_argument(
        "--interval", type=float, default=3.0,
        help="Refresh interval in seconds (default: 3).",
    )

    # --- leaderboard --------------------------------------------------------
    p_lb = sub.add_parser("leaderboard", help="Show SQT token leaderboard.")
    p_lb.add_argument(
        "--registry", default="http://127.0.0.1:7701",
        help="Registry server URL (default: http://127.0.0.1:7701).",
    )
    p_lb.add_argument(
        "--top-n", dest="top_n", type=int, default=20,
        help="Number of top nodes to display (default: 20).",
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
    p_start.add_argument(
        "--ssl-cert", dest="ssl_cert", default=None,
        help="Path to TLS certificate (PEM). Auto-generates self-signed if omitted.",
    )
    p_start.add_argument(
        "--ssl-key", dest="ssl_key", default=None,
        help="Path to TLS private key (PEM). Must be paired with --ssl-cert.",
    )
    p_start.add_argument(
        "--admin-key", dest="admin_key", default=None,
        help="Bearer token required to deregister nodes. "
             "Also read from ZILVER_REGISTRY_KEY env var.",
    )
    p_start.add_argument(
        "--ledger-path", dest="ledger_path", default=None,
        help="Path to SQT ledger JSON file (e.g. ~/.zilver/ledger.json). "
             "If omitted, rewards are not tracked.",
    )
    p_start.add_argument(
        "--require-signed", dest="require_signed", action="store_true", default=False,
        help="Require Ed25519 signed registration from all nodes. "
             "Rejects nodes that cannot prove hardware identity.",
    )
    p_start.add_argument(
        "--allowed-pubkeys-file", dest="allowed_pubkeys_file", default=None,
        help="Path to file of approved node pubkeys (one hex pubkey per line). "
             "Only these nodes can register. Also read from ZILVER_ALLOWED_PUBKEYS_FILE.",
    )
    p_start.add_argument(
        "--client-keys-file", dest="client_keys_file", default=None,
        help="Path to file of authorized client API keys (one key per line). "
             "Required for /match and job submission. Also read from ZILVER_CLIENT_KEYS_FILE.",
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
        "start":       _cmd_node_start,
        "status":      _cmd_node_status,
        "nodes":       _cmd_node_list,
        "dashboard":   _cmd_node_dashboard,
        "leaderboard": _cmd_node_leaderboard,
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
