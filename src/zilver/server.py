"""Node HTTP server."""

from __future__ import annotations

import asyncio
import secrets
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request

from .node import Node, SimJob, JobResult

# ---------------------------------------------------------------------------
# Input size limits
# ---------------------------------------------------------------------------

_MAX_CIRCUIT_OPS  = 10_000
_MAX_PARAMS       = 10_000
_MAX_QUBITS       = 50
_MAX_BODY_BYTES   = 2 * 1024 * 1024   # 2 MB — large enough for 10k ops + 10k params


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def make_app(node: Node, api_key: str | None = None) -> FastAPI:
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
    api_key:
        Bearer token required on ``POST /execute``.  When ``None`` (the
        default) authentication is skipped — suitable for local development
        and tests.  In production this should be the key issued by the
        registry on node registration.

    Returns
    -------
    FastAPI
        The application instance.  Pass it to ``uvicorn.run`` or wrap it with
        ``fastapi.testclient.TestClient`` for testing.
    """
    app = FastAPI(title="zilver-node", version="0.1.0")

    # --- Shared dependencies ------------------------------------------------

    async def _check_body_size(request: Request) -> None:
        """Reject requests whose Content-Length exceeds _MAX_BODY_BYTES."""
        length = request.headers.get("content-length")
        if length and int(length) > _MAX_BODY_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Request body too large (limit {_MAX_BODY_BYTES // 1024} KB)",
            )

    async def _require_auth(request: Request) -> None:
        """Verify Bearer token.  No-op when api_key is None (test/dev mode)."""
        if api_key is None:
            return
        header = request.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else ""
        if not secrets.compare_digest(token, api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # --- Routes -------------------------------------------------------------

    @app.post("/execute", dependencies=[Depends(_check_body_size), Depends(_require_auth)])
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
        - **401** if authentication is configured and the ``Authorization``
          header is missing or wrong.
        - **422** if the request body cannot be parsed as a ``SimJob``.
        - **422** if any input exceeds the allowed size limits
          (circuit_ops > 10 000, params > 10 000, n_qubits > 50).
        - **422** if the node does not support the requested backend or the
          qubit count exceeds the node's ceiling.
        """
        # Input size limits
        if len(body.get("circuit_ops", [])) > _MAX_CIRCUIT_OPS:
            raise HTTPException(
                status_code=422,
                detail=f"circuit_ops length exceeds limit of {_MAX_CIRCUIT_OPS}",
            )
        if len(body.get("params", [])) > _MAX_PARAMS:
            raise HTTPException(
                status_code=422,
                detail=f"params length exceeds limit of {_MAX_PARAMS}",
            )
        if body.get("n_qubits", 0) > _MAX_QUBITS:
            raise HTTPException(
                status_code=422,
                detail=f"n_qubits exceeds limit of {_MAX_QUBITS}",
            )

        try:
            job = SimJob.from_dict(body)
        except (KeyError, TypeError):
            raise HTTPException(status_code=422, detail="Invalid job body")

        loop = asyncio.get_running_loop()
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
    node:         Node,
    host:         str = "0.0.0.0",
    port:         int = 7700,
    log_level:    str = "warning",
    api_key:      str | None = None,
    ssl_keyfile:  str | None = None,
    ssl_certfile: str | None = None,
) -> None:
    """
    Start a uvicorn HTTP(S) server for *node* and block until interrupted.

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
    api_key:
        Bearer token required on ``POST /execute``.  ``None`` disables auth.
    ssl_keyfile:
        Path to the TLS private key (PEM).  When set, the server uses HTTPS.
    ssl_certfile:
        Path to the TLS certificate (PEM).
    """
    import uvicorn
    app = make_app(node, api_key=api_key)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )
