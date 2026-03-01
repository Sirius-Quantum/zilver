"""
Registry HTTP server.

Exposes the in-process :class:`~zilver.registry.Registry` over HTTP so that
nodes on different machines can register, send heartbeats, and be discovered
by coordinators without any shared memory or direct imports.

Design
------
The registry is intentionally in-memory for v1.  Every registered node
carries its advertised URL (``http://host:port``) alongside its capabilities,
so a coordinator that calls ``GET /match`` receives a connectable address,
not just a node ID.  Persistent storage (SQLite, Redis) is deferred to a
future release.

Node URL storage
----------------
FastAPI stores the URL-to-node mapping in a module-level dict keyed by
``node_id``.  This is safe for the single-process case; the distributed
registry service is designed to run as a single process in v1.

Endpoints
---------
POST   /nodes                          Register or re-register a node.
DELETE /nodes/{node_id}                Mark a node offline.
POST   /nodes/{node_id}/heartbeat      Refresh last-seen timestamp.
GET    /nodes                          List all online nodes with URLs.
GET    /match                          Best node for a job (returns URL).
GET    /summary                        Aggregate registry statistics.

Example
-------
::

    from zilver.registry_server import make_registry_app, serve_registry

    app = make_registry_app()           # fresh in-memory registry
    serve_registry(host="0.0.0.0", port=7701)   # blocks until Ctrl-C
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from .node import NodeCapabilities
from .registry import Registry


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def make_registry_app(registry: Registry | None = None) -> FastAPI:
    """
    Build the FastAPI application for the capability registry.

    Parameters
    ----------
    registry:
        An existing :class:`~zilver.registry.Registry` instance to wrap.
        If ``None``, a fresh in-memory registry is created.  Passing an
        existing instance is useful in tests where the caller needs direct
        access to registry state.

    Returns
    -------
    FastAPI
        The application instance.  Use ``uvicorn.run`` for production or
        ``fastapi.testclient.TestClient`` for tests.
    """
    reg = registry if registry is not None else Registry()

    # Maps node_id → advertised URL so clients can connect directly.
    # Stored separately from NodeCapabilities to keep the core registry
    # transport-agnostic.
    node_urls: dict[str, str] = {}

    app = FastAPI(title="zilver-registry", version="0.1.0")

    # --- Registration -------------------------------------------------------

    @app.post("/nodes", status_code=201)
    async def register(body: dict[str, Any]) -> dict[str, Any]:
        """
        Register or re-register a node.

        Request body
        ~~~~~~~~~~~~
        A JSON object with two fields:

        - ``caps`` — ``NodeCapabilities.to_dict()``
        - ``url``  — the node's reachable HTTP base URL,
          e.g. ``"http://192.168.1.5:7700"``

        Response
        ~~~~~~~~
        ``{"registered": true, "node_id": "<id>"}``

        Idempotent: re-registering an existing node refreshes its
        capabilities and last-seen timestamp.
        """
        try:
            caps = NodeCapabilities(**body["caps"])
            url: str = body["url"]
        except (KeyError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid registration body: {exc}")

        reg.register(caps)
        node_urls[caps.node_id] = url
        return {"registered": True, "node_id": caps.node_id}

    @app.delete("/nodes/{node_id}")
    async def deregister(node_id: str) -> dict[str, Any]:
        """
        Mark a node offline.

        The node remains in the registry's history (for diagnostics) but
        will no longer be returned by ``/match`` or ``/nodes``.

        Returns ``{"deregistered": true}`` if found, ``{"deregistered": false}``
        if the node was not registered.
        """
        found = reg.deregister(node_id)
        node_urls.pop(node_id, None)
        return {"deregistered": found, "node_id": node_id}

    @app.post("/nodes/{node_id}/heartbeat")
    async def heartbeat(node_id: str) -> dict[str, Any]:
        """
        Refresh the last-seen timestamp for a node.

        Called by the node daemon on a fixed interval (default 30 s) so the
        registry can detect stale nodes.  Returns 404 if the node is unknown.
        """
        found = reg.heartbeat(node_id)
        if not found:
            raise HTTPException(status_code=404, detail=f"Node {node_id!r} not found")
        return {"status": "ok", "node_id": node_id}

    # --- Discovery ----------------------------------------------------------

    @app.get("/nodes")
    async def list_nodes() -> list[dict[str, Any]]:
        """
        Return all currently online nodes.

        Each element is ``NodeCapabilities.to_dict()`` extended with a
        ``"url"`` field containing the node's reachable HTTP address.
        """
        entries = reg.all_entries()
        result = []
        for entry in entries:
            d = entry.caps.to_dict()
            d["url"] = node_urls.get(entry.caps.node_id, "")
            result.append(d)
        return result

    @app.get("/match")
    async def match(
        backend:   str,
        n_qubits:  int,
        min_stake: int = 0,
    ) -> dict[str, Any]:
        """
        Find the best available node for a job.

        Query parameters
        ~~~~~~~~~~~~~~~~
        - ``backend``   — ``"sv"``, ``"dm"``, or ``"tn"``
        - ``n_qubits``  — qubit count required by the job
        - ``min_stake`` — minimum stake (default 0)

        Response
        ~~~~~~~~
        On success: ``NodeCapabilities.to_dict()`` plus ``"url"``.

        Raises **404** if no eligible node exists.
        """
        entry = reg.match(backend, n_qubits, min_stake=min_stake)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No eligible node for backend={backend!r} "
                    f"n_qubits={n_qubits} min_stake={min_stake}"
                ),
            )
        d = entry.caps.to_dict()
        d["url"] = node_urls.get(entry.caps.node_id, "")
        return d

    @app.get("/summary")
    async def summary() -> dict[str, Any]:
        """
        Aggregate registry statistics.

        Returns a dict with ``online``, ``total_registered``, ``backends``,
        ``max_sv_qubits``, ``max_dm_qubits``, and ``total_stake``.
        Useful for monitoring dashboards and CLI status commands.
        """
        return reg.summary()

    return app


# ---------------------------------------------------------------------------
# Blocking server entrypoint
# ---------------------------------------------------------------------------

def serve_registry(
    registry:  Registry | None = None,
    host:      str = "0.0.0.0",
    port:      int = 7701,
    log_level: str = "warning",
) -> None:
    """
    Start a uvicorn HTTP server for the capability registry and block until
    interrupted.

    This is called by the CLI (``zilver-registry start``).  For tests, use
    ``make_registry_app`` with ``fastapi.testclient.TestClient`` instead.

    Parameters
    ----------
    registry:
        Registry instance to expose.  A fresh one is created if ``None``.
    host:
        Interface to bind.  Default ``"0.0.0.0"`` (all interfaces).
    port:
        TCP port number.  Default 7701 (separate from node default 7700).
    log_level:
        uvicorn log level.  Default ``"warning"``.
    """
    import uvicorn
    app = make_registry_app(registry)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
