"""
Node HTTP server.

Exposes a Zilver simulation node over HTTP so that coordinators and clients
on other machines can submit jobs and retrieve results without any shared memory.

Architecture note
-----------------
``make_app`` returns a plain FastAPI application instance.  Separating app
construction from server startup means tests can drive the endpoints through
``fastapi.testclient.TestClient`` without binding a real port.

Thread-pool execution
---------------------
MLX statevector simulation is CPU-bound (Metal dispatch + result copy).
Running it directly inside an ``async def`` endpoint would block the event
loop and stall concurrent health/heartbeat requests during a long computation.
Every ``/execute`` call is therefore dispatched to
``asyncio.get_event_loop().run_in_executor(None, node.execute, job)``, which
hands the work to a thread from the default ``ThreadPoolExecutor`` and
suspends the coroutine until it completes.

Endpoints
---------
POST /execute      SimJob JSON  →  JobResult JSON (422 on capacity/backend error)
GET  /caps         NodeCapabilities JSON
POST /heartbeat    {"status": "ok"}
GET  /health       {"status": "ok", "node_id": "<id>"}

Example
-------
::

    from zilver.node import Node
    from zilver.server import make_app, serve

    node = Node.start(backends=["sv", "dm"])
    serve(node, host="0.0.0.0", port=7700)   # blocks until Ctrl-C
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException

from .node import Node, SimJob, JobResult


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def make_app(node: Node) -> FastAPI:
    """
    Build the FastAPI application for *node*.

    Parameters
    ----------
    node:
        A fully initialised :class:`~zilver.node.Node` instance.  The same
        node object is reused across all requests; ``Node.execute`` is
        thread-safe because it holds no mutable state beyond the
        ``jobs_completed`` counter (incremented atomically in CPython due to
        the GIL).

    Returns
    -------
    FastAPI
        The application instance.  Pass it to ``uvicorn.run`` or wrap it with
        ``fastapi.testclient.TestClient`` for testing.
    """
    app = FastAPI(title="zilver-node", version="0.1.0")

    @app.post("/execute")
    async def execute(body: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a simulation job on this node.

        Request body
        ~~~~~~~~~~~~
        A JSON object produced by ``SimJob.to_dict()``.  Required fields:
        ``circuit_ops``, ``n_qubits``, ``n_params``, ``params``, ``backend``.

        Response
        ~~~~~~~~
        A JSON object matching ``JobResult.to_dict()``, including the SHA-256
        proof that can be verified by the caller with ``JobResult.verify(job)``.

        Errors
        ~~~~~~
        - **422** if the request body cannot be parsed as a ``SimJob``.
        - **422** if the node does not support the requested backend or the
          qubit count exceeds the node's ceiling.
        """
        try:
            job = SimJob.from_dict(body)
        except (KeyError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid SimJob: {exc}")

        loop = asyncio.get_event_loop()
        try:
            result: JobResult = await loop.run_in_executor(None, node.execute, job)
        except ValueError as exc:
            # node.execute raises ValueError for backend/capacity mismatches
            raise HTTPException(status_code=422, detail=str(exc))

        return result.to_dict()

    @app.get("/caps")
    async def caps() -> dict[str, Any]:
        """
        Return the node's hardware capabilities.

        Response matches ``NodeCapabilities.to_dict()``:
        ``node_id``, ``chip``, ``ram_gb``, ``sv_qubits_max``,
        ``dm_qubits_max``, ``tn_qubits_max``, ``backends``,
        ``jobs_completed``, ``stake``.
        """
        return node.caps.to_dict()

    @app.post("/heartbeat")
    async def heartbeat() -> dict[str, str]:
        """
        Liveness signal called by the registry on a fixed interval.

        Returns ``{"status": "ok"}`` as long as the process is alive.
        The registry marks a node stale if this endpoint stops responding
        within the configured TTL (default 60 s).
        """
        return {"status": "ok"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        """
        Quick health check for load balancers and monitoring tools.

        Returns ``{"status": "ok", "node_id": "<id>"}`` without touching the
        simulator, so it never blocks on compute.
        """
        return {"status": "ok", "node_id": node.caps.node_id}

    return app


# ---------------------------------------------------------------------------
# Blocking server entrypoint
# ---------------------------------------------------------------------------

def serve(
    node: Node,
    host: str = "0.0.0.0",
    port: int = 7700,
    log_level: str = "warning",
) -> None:
    """
    Start a uvicorn HTTP server for *node* and block until interrupted.

    This is called by the CLI (``zilver-node start``).  For tests, use
    ``make_app`` with ``fastapi.testclient.TestClient`` instead.

    Parameters
    ----------
    node:
        Initialised node to serve.
    host:
        Interface to bind.  ``"0.0.0.0"`` listens on all interfaces;
        use ``"127.0.0.1"`` to restrict to localhost only.
    port:
        TCP port number.  Default 7700.
    log_level:
        uvicorn log level (``"debug"``, ``"info"``, ``"warning"``, etc.).
        Default is ``"warning"`` to keep stdout clean in production.
    """
    import uvicorn
    app = make_app(node)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
