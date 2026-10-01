"""CLI entry points."""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import RegistryClient
    from .node import NodeCapabilities


PUBLIC_REGISTRY = "https://registry.siriusquantum.com"

# ---------------------------------------------------------------------------
# Heartbeat daemon thread
# ---------------------------------------------------------------------------

_HB_BACKOFF = [5, 10, 20, 40, 80, 120]


def _start_heartbeat(
    reg_client:        "RegistryClient",
    node_id:           str,
    node_url:          str,
    caps:              "NodeCapabilities",
    interval:          int = 30,
    private_key_bytes: "bytes | None" = None,
    public_key_bytes:  "bytes | None" = None,
    se_label:          "str | None"   = None,
) -> None:
    """
    Background daemon thread: heartbeat with auto-reconnect.

    - 404 from registry  → node was dropped (e.g. registry restart); re-register
      immediately and update the stored API key.
    - Connection error   → exponential backoff before the next attempt; resets
      on the first successful heartbeat.
    """
    def _reregister() -> None:
        try:
            reg_client.register(
                caps, node_url,
                private_key_bytes=private_key_bytes,
                public_key_bytes=public_key_bytes,
                se_label=se_label,
            )
            new_key = reg_client.last_api_key
            if new_key:
                reg_client.api_key = new_key
                try:
                    from . import _node_ops
                    _node_ops.store_api_key(node_id, new_key)
                except Exception:
                    pass
        except Exception:
            pass  # will retry on next heartbeat cycle

    def _loop() -> None:
        backoff_idx = 0
        while True:
            time.sleep(interval)
            try:
                found = reg_client.heartbeat(node_id)
                backoff_idx = 0
                if not found:
                    _reregister()
            except Exception:
                extra = _HB_BACKOFF[min(backoff_idx, len(_HB_BACKOFF) - 1)]
                backoff_idx += 1
                time.sleep(extra)

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

    allow_unsigned = getattr(args, "allow_unsigned", False)
    try:
        node = Node.start(
            backends           = backends,
            node_id            = derived_node_id,
            private_key_bytes  = private_key_bytes,
            public_key_bytes   = public_key_bytes,
            se_label           = _se_label,
            allow_unsigned     = allow_unsigned,
        )
    except ValueError:
        # No signing key. Say what is missing and what the two ways forward are,
        # in CLI terms — the exception's own text is written for API callers, and
        # a user's first run should not end in a traceback either way.
        sys.exit(
            "This node has no signing key, so every result it returned would be "
            "unsigned — and a client cannot tell an unsigned result from a forged "
            "one.\n\n"
            "  * To run a node on the Zilver network, a node identity is "
            "required — see NODES.md.\n"
            "  * To run locally without signing, start with --allow-unsigned."
        )

    if allow_unsigned and public_key_bytes is None:
        print("Warning: running UNSIGNED — results are unverifiable.", file=sys.stderr)

    print(f"Node {node.caps.node_id[:8]} | chip: {node.caps.chip} | "
          f"RAM: {node.caps.ram_gb}GB | backends: {node.caps.backends} | "
          f"sv_max: {node.caps.sv_qubits_max}q")
    if public_key_bytes is not None:
        print(f"Node pubkey: {public_key_bytes.hex()}  (add to registry allowlist)")

    # --- TLS ----------------------------------------------------------------
    ssl_key, ssl_cert = _resolve_tls(args)
    scheme = "https" if ssl_cert else "http"

    # Construct the URL this node will advertise to the registry.
    # --public-url takes precedence; then a tunnel, which is the default when
    # joining a registry; otherwise the detected LAN IP.
    public_url = getattr(args, "public_url", None)
    use_tunnel = getattr(args, "tunnel", None)
    if use_tunnel is None:
        use_tunnel = bool(args.registry) and not public_url
    if args.host is None:
        # Behind a tunnel nothing needs to reach the port except cloudflared.
        args.host = "127.0.0.1" if use_tunnel else "0.0.0.0"
    if use_tunnel and not public_url:
        from .service import exit_with_tunnel, start_tunnel
        try:
            tunnel_proc, public_url = start_tunnel(args.port)
        except RuntimeError as exc:
            sys.exit(f"Could not open a tunnel: {exc}\n"
                     "Pass --public-url <url> if this node is reachable another way.")
        atexit.register(tunnel_proc.terminate)
        exit_with_tunnel(tunnel_proc)
        print(f"Tunnel: {public_url}")
    if public_url:
        node_url = public_url.rstrip("/")
    else:
        advertised_host = args.host if args.host != "0.0.0.0" else _local_ip()
        node_url = f"{scheme}://{advertised_host}:{args.port}"
        import ipaddress
        import urllib.parse as _up
        try:
            _h = _up.urlparse(node_url).hostname or ""
            _a = ipaddress.ip_address(_h)
            if _a.is_private or _a.is_loopback:
                print(
                    f"Warning: advertised URL {node_url!r} is a private address — "
                    "external clients cannot reach this node.\n"
                    "Use --public-url <url> to set a reachable address.",
                    file=sys.stderr,
                )
        except ValueError:
            pass

    # --- Credentials -------------------------------------------------------
    # With a registry, api_key is this node's own registry key (heartbeat,
    # re-register, contribute) and never leaves the node. Clients reach
    # /execute with per-match tokens signed by reg_client.execute_secret.
    # Without a registry, api_key is a static /execute key the operator chose.
    api_key: str | None = getattr(args, "api_key", None)

    reg_client: RegistryClient | None = None
    on_executed = None

    if args.registry:
        reg_client = RegistryClient(args.registry)
        try:
            if api_key is None:
                try:
                    from . import _node_ops
                    api_key = _node_ops.load_api_key(node.caps.node_id)
                except Exception:
                    pass

            # Re-registering a known node_id must present its current key.
            reg_client.api_key = api_key
            reg_client.register(node.caps, node_url,
                                private_key_bytes=private_key_bytes,
                                public_key_bytes=public_key_bytes,
                                se_label=_se_label)
            # The registry issues a fresh key on every registration.
            new_key = reg_client.last_api_key
            if new_key and new_key != api_key:
                reg_client.api_key = new_key
                try:
                    from . import _node_ops
                    _node_ops.store_api_key(node.caps.node_id, new_key)
                    print("API key stored in macOS Keychain.")
                except Exception as exc:
                    print(f"Warning: could not store API key in Keychain: {exc}",
                          file=sys.stderr)

            def on_executed(job_token: str, result) -> None:
                """Report this node's own work; clients no longer do."""
                try:
                    reg_client.contribute(
                        node_id=node.caps.node_id,
                        elapsed_ms=result.elapsed_ms,
                        memory_used_mb=result.memory_used_mb,
                        proof=result.proof,
                        job_token=job_token,
                    )
                except Exception as exc:
                    print(f"Warning: could not report job to registry: {exc}",
                          file=sys.stderr)

            print(f"Registered with registry at {args.registry}")
            _start_heartbeat(
                reg_client, node.caps.node_id, node_url, node.caps,
                private_key_bytes=private_key_bytes,
                public_key_bytes=public_key_bytes,
                se_label=_se_label,
            )
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
        api_key=None if reg_client is not None else api_key,
        ssl_keyfile=ssl_key,
        ssl_certfile=ssl_cert,
        execute_secret=reg_client.execute_secret if reg_client is not None else None,
        on_executed=on_executed,
    )


def _cmd_node_service(args: argparse.Namespace) -> None:
    """Install or remove the launchd agent that runs the node in the background."""
    from . import service
    if sys.platform != "darwin":
        sys.exit("The background service uses macOS launchd; on Linux, run "
                 "zilver-node start under systemd instead.")
    if args.command == "uninstall-service":
        removed = service.uninstall_service()
        print("Background node stopped and removed." if removed
              else "No background node was installed.")
        return
    start_args = ["--registry", args.registry, "--backends", args.backends,
                  "--port", str(args.port)]
    try:
        path = service.install_service(start_args)
    except Exception as exc:
        sys.exit(f"Could not install the background node: {exc}")
    print(f"Background node installed ({path}).")
    print(f"It runs now and at every login. Log: {service.ZILVER_DIR / 'node.log'}")
    print("Remove it with: zilver-node uninstall-service")


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

    admin_key          = getattr(args, "admin_key",          None) or os.environ.get("ZILVER_REGISTRY_KEY")
    ledger_path        = getattr(args, "ledger_path",        None) or os.environ.get("ZILVER_LEDGER_PATH")
    db_path            = getattr(args, "db_path",            None) or os.environ.get("ZILVER_DB_PATH")
    audit_log_path     = getattr(args, "audit_log_path",     None) or os.environ.get("ZILVER_AUDIT_LOG")
    require_signed     = not getattr(args, "allow_unsigned_nodes", False)
    allow_private_urls = getattr(args, "allow_private_urls", False)

    # Load node allowlist from file (one pubkey hex per line, # comments allowed)
    allowed_pubkeys: set[str] | None = None
    allowed_pubkeys_file = getattr(args, "allowed_pubkeys_file", None) or os.environ.get("ZILVER_ALLOWED_PUBKEYS_FILE")
    if allowed_pubkeys_file:
        try:
            lines = Path(allowed_pubkeys_file).read_text().splitlines()
            allowed_pubkeys = {line.strip().split("#")[0].strip() for line in lines
                               if line.strip() and not line.strip().startswith("#")}
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
            client_keys = {line.strip().split("#")[0].strip() for line in lines
                           if line.strip() and not line.strip().startswith("#")}
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
        print(f"Ledger: {ledger_path}")
    if db_path:
        print(f"Registry DB: {db_path}")
    if audit_log_path:
        print(f"Audit log: {audit_log_path}")
    if allow_private_urls:
        print("Warning: private URL registration allowed (dev mode).", file=sys.stderr)

    proto = "HTTPS" if ssl_cert else "HTTP (no TLS)"
    print(f"Registry server {proto} on {args.host}:{args.port}  (Ctrl-C to stop)")

    if require_signed:
        print("Signed registration enforced — nodes must present a valid signature.")
    else:
        print("Warning: unsigned node registration allowed (--allow-unsigned-nodes); "
              "re-registering a node_id still needs that node's key.", file=sys.stderr)

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
        db_path=db_path,
        allow_private_urls=allow_private_urls,
        audit_log_path=audit_log_path,
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
    p_start = sub.add_parser(
        "start",
        help="Start the node daemon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    grp_core = p_start.add_argument_group("core")
    grp_core.add_argument(
        "--backends", default="sv",
        help="Backends to enable: sv, dm, tn, or any comma-separated combination.",
    )
    grp_core.add_argument(
        "--port", type=int, default=7700,
        help="TCP port to listen on.",
    )
    grp_core.add_argument(
        "--host", default=None,
        help="Interface to bind (default: 127.0.0.1 behind a tunnel, else 0.0.0.0).",
    )

    grp_net = p_start.add_argument_group(
        "network mode",
        "Connect to a registry and contribute jobs to the network. "
        "Omit --registry to run in standalone mode (local simulation only).",
    )
    grp_net.add_argument(
        "--registry", default=None,
        help=f"Registry server URL, e.g. {PUBLIC_REGISTRY}.",
    )
    grp_net.add_argument(
        "--public-url", dest="public_url", default=None,
        help="Externally reachable URL for this node "
             "(e.g. https://your-tunnel.example.com). "
             "Required when behind NAT or a Cloudflare Tunnel.",
    )
    grp_net.add_argument(
        "--api-key", dest="api_key", default=None,
        help="This node's own registry key (never given to clients). "
             "If omitted, loaded from Keychain or obtained automatically on first run. "
             "Without --registry, a static key clients must send to /execute.",
    )
    grp_net.add_argument(
        "--tunnel", action=argparse.BooleanOptionalAction, default=None,
        help="Publish this node through a Cloudflare tunnel (needs cloudflared), "
             "so it is reachable behind a home router without port forwarding. "
             "Default: on with --registry unless --public-url is given.",
    )
    grp_net.add_argument(
        "--allow-unsigned", dest="allow_unsigned", action="store_true", default=False,
        help="Start without a signing key. Every result is then unsigned, and no "
             "client can tell it from a forgery — local and test use only.",
    )

    grp_tls = p_start.add_argument_group(
        "TLS",
        "Custom certificate paths. If omitted, a self-signed certificate is "
        "auto-generated to ~/.zilver/node.{key,crt} on first run.",
    )
    grp_tls.add_argument(
        "--ssl-cert", dest="ssl_cert", default=None,
        help="Path to TLS certificate (PEM).",
    )
    grp_tls.add_argument(
        "--ssl-key", dest="ssl_key", default=None,
        help="Path to TLS private key (PEM). Must be paired with --ssl-cert.",
    )

    # --- status -------------------------------------------------------------
    p_status = sub.add_parser("status", help="Print registry summary.")
    p_status.add_argument(
        "--registry", default=PUBLIC_REGISTRY,
        help=f"Registry server URL (default: {PUBLIC_REGISTRY}).",
    )

    # --- nodes --------------------------------------------------------------
    # --- install-service / uninstall-service --------------------------------
    p_svc = sub.add_parser(
        "install-service",
        help="Run the node in the background: starts at login, restarts after "
             "a crash, and keeps the Mac awake while it runs.",
    )
    p_svc.add_argument(
        "--registry", default=PUBLIC_REGISTRY,
        help=f"Registry server URL (default: {PUBLIC_REGISTRY}).",
    )
    p_svc.add_argument(
        "--backends", default="sv",
        help="Backends to enable: sv, dm, tn, or any comma-separated combination.",
    )
    p_svc.add_argument("--port", type=int, default=7700, help="TCP port to listen on.")
    sub.add_parser("uninstall-service", help="Stop and remove the background node.")

    p_nodes = sub.add_parser("nodes", help="List online nodes in the registry.")
    p_nodes.add_argument(
        "--registry", default=PUBLIC_REGISTRY,
        help=f"Registry server URL (default: {PUBLIC_REGISTRY}).",
    )

    # --- dashboard ----------------------------------------------------------
    p_dash = sub.add_parser("dashboard", help="Live Rich TUI showing active nodes.")
    p_dash.add_argument(
        "--registry", default=PUBLIC_REGISTRY,
        help=f"Registry server URL (default: {PUBLIC_REGISTRY}).",
    )
    p_dash.add_argument(
        "--interval", type=float, default=3.0,
        help="Refresh interval in seconds (default: 3).",
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
        help="Path to the job-accounting JSON file. If omitted, none is kept.",
    )
    p_start.add_argument(
        "--require-signed", dest="require_signed", action="store_true", default=True,
        help="Require signed registration from all nodes (the default; kept "
             "so existing unit files keep working).",
    )
    p_start.add_argument(
        "--allow-unsigned-nodes", dest="allow_unsigned_nodes", action="store_true",
        default=False,
        help="Accept nodes that register without a signature. Local and test use only.",
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
    p_start.add_argument(
        "--db-path", dest="db_path", default=None,
        help="Path to SQLite registry database (e.g. /var/lib/zilver/registry.db). "
             "Node registrations persist across restarts. Also read from ZILVER_DB_PATH.",
    )
    p_start.add_argument(
        "--allow-private-urls", dest="allow_private_urls",
        action="store_true", default=False,
        help="Allow nodes to register with private/loopback URLs. "
             "For local development only — do not use in production.",
    )
    p_start.add_argument(
        "--audit-log", dest="audit_log_path", default=None,
        help="Path to append-only JSONL audit log "
             "(e.g. /var/log/zilver/audit.jsonl). "
             "Also read from ZILVER_AUDIT_LOG.",
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
    }
    if args.command == "start":
        dispatch[args.command](args)
        return
    if args.command in ("install-service", "uninstall-service"):
        _cmd_node_service(args)
        return
    try:
        import httpx
    except ImportError:
        sys.exit('The network commands need the network extra: pip install "zilver[network]"')
    try:
        dispatch[args.command](args)
    except httpx.HTTPError as exc:
        sys.exit(f"Cannot reach registry at {args.registry}: {exc or type(exc).__name__}")


def main_registry() -> None:
    """Entry point for the ``zilver-registry`` script."""
    parser = _build_registry_parser()
    args = parser.parse_args()

    dispatch = {
        "start": _cmd_registry_start,
    }
    dispatch[args.command](args)
