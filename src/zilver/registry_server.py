"""Registry HTTP server."""

from __future__ import annotations

import secrets
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Request

from .ledger import Ledger
from .node_types import NodeCapabilities
from .registry import Registry

try:
    from . import _registry_ops as _reg_ops
except ImportError:
    _reg_ops = None  # type: ignore[assignment]

_MAX_BODY_BYTES    = 64 * 1024          # 64 KB — sufficient for any registration payload
_JOB_TOKEN_TTL     = 3600.0             # job tokens expire after 1 hour


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

def _make_rate_limiter(max_calls: int, window_secs: int = 60) -> Callable:
    """
    Return a FastAPI dependency that enforces a per-IP sliding-window rate limit.

    When *max_calls* is 0 the returned callable is a no-op (disabled).
    """
    if max_calls == 0:
        async def _noop(request: Request) -> None:
            return
        return _noop

    hits: dict[str, list[float]] = defaultdict(list)

    async def _limit(request: Request) -> None:
        ip = request.client.host if request.client else "local"
        now = time.monotonic()
        cutoff = now - window_secs
        valid = [t for t in hits[ip] if t > cutoff]
        if len(valid) >= max_calls:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
            )
        valid.append(now)
        hits[ip] = valid

    return _limit


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def make_registry_app(
    registry:        Registry | None = None,
    admin_key:       str | None = None,
    rate_limit:      bool = False,
    ledger:          Ledger | None = None,
    require_signed:  bool = False,
    allowed_pubkeys: set[str] | None = None,
    client_keys:     set[str] | None = None,
) -> FastAPI:
    """
    Build the FastAPI application for the capability registry.

    Parameters
    ----------
    registry:
        An existing :class:`~zilver.registry.Registry` instance to wrap.
        If ``None``, a fresh in-memory registry is created.  Passing an
        existing instance is useful in tests where the caller needs direct
        access to registry state.
    admin_key:
        Bearer token required to deregister nodes via
        ``DELETE /nodes/{node_id}``.  When ``None`` (the default) the
        endpoint is unprotected — suitable for local development and tests.
    rate_limit:
        When ``True``, apply per-IP sliding-window rate limits:
        ``POST /nodes`` → 5/min, ``GET /match`` → 60/min.
        Default ``False`` for test and dev compatibility.
    ledger:
        :class:`~zilver.ledger.Ledger` instance to use for SQT reward
        tracking.  When ``None`` (the default) reward endpoints are
        available but return zero balances.
    require_signed:
        When ``True``, every ``POST /nodes`` registration must include a
        valid Ed25519 ``pubkey`` + ``signature`` + ``timestamp``.  Nodes
        without a valid signature are rejected with HTTP 403.  Default
        ``False`` for test and dev compatibility.
    allowed_pubkeys:
        When set, only nodes whose ``pubkey`` hex is in this set may
        register.  Any other pubkey is rejected with HTTP 403, even if
        the signature is otherwise valid.  Requires ``require_signed=True``
        to have effect.  ``None`` means no allowlist.
    client_keys:
        When set, ``GET /match``, ``POST /jobs/estimate``, and
        ``POST /jobs/estimate`` require ``Authorization: Bearer <key>``
        matching one of these keys.  ``None`` (default) means open access.

    Returns
    -------
    FastAPI
        The application instance.  Use ``uvicorn.run`` for production or
        ``fastapi.testclient.TestClient`` for tests.
    """
    reg = registry if registry is not None else Registry()

    # Maps node_id → advertised URL so clients can connect directly.
    node_urls:      dict[str, str] = {}
    # Maps node_id → issued API key for identity verification.
    node_keys:      dict[str, str] = {}
    # Maps node_id → registered Ed25519 public key (hex).
    node_pubkeys:   dict[str, str] = {}
    # Maps pubkey_hex → node_id — enforces one slot per keypair.
    pubkey_node_ids: dict[str, str] = {}
    # Maps job_token → (node_id, issued_at) for single-use contribute authorisation.
    # Tokens are deleted on first use or when they exceed _JOB_TOKEN_TTL.
    job_tokens:     dict[str, tuple[str, float]] = {}

    app = FastAPI(title="zilver-registry", version="0.1.0")
    app.state.job_tokens = job_tokens  # exposed for testing

    # --- Rate limiters (per-endpoint) ----------------------------------------

    _register_limit        = _make_rate_limiter(5  if rate_limit else 0, window_secs=60)
    _register_limit_hourly = _make_rate_limiter(3  if rate_limit else 0, window_secs=3600)
    _match_limit           = _make_rate_limiter(60 if rate_limit else 0)
    # Heartbeat: nodes send one every 30 s → 2/min is the honest ceiling
    _heartbeat_limit  = _make_rate_limiter(2   if rate_limit else 0)
    # Contribute: one per completed job; generous ceiling to absorb bursts
    _contribute_limit = _make_rate_limiter(120 if rate_limit else 0)

    # --- Auth dependencies ---------------------------------------------------

    async def _check_body_size(request: Request) -> None:
        """Reject requests whose Content-Length exceeds _MAX_BODY_BYTES."""
        length = request.headers.get("content-length")
        if length and int(length) > _MAX_BODY_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Request body too large (limit {_MAX_BODY_BYTES // 1024} KB)",
            )

    async def _require_admin(request: Request) -> None:
        """Verify admin Bearer token.  No-op when admin_key is None."""
        if admin_key is None:
            return
        header = request.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else ""
        if not secrets.compare_digest(token, admin_key):
            raise HTTPException(status_code=401, detail="Invalid or missing admin key")

    async def _require_client(request: Request) -> None:
        """Verify client Bearer token against client_keys set.  No-op when client_keys is None."""
        if client_keys is None:
            return
        header = request.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else ""
        if not token or not any(secrets.compare_digest(token, k) for k in client_keys):
            raise HTTPException(status_code=401, detail="Invalid or missing client API key")

    # --- Registration -------------------------------------------------------

    @app.post("/nodes", status_code=201,
              dependencies=[Depends(_check_body_size), Depends(_register_limit),
                            Depends(_register_limit_hourly)])
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
        ``{"registered": true, "node_id": "<id>", "api_key": "<key>"}``

        The ``api_key`` is a 32-byte random hex string issued by the registry.
        The node should store it securely (e.g. macOS Keychain) and present it
        in ``Authorization: Bearer <key>`` on subsequent requests.

        Idempotent: re-registering an existing node refreshes its capabilities
        and last-seen timestamp, and returns a new API key.

        When the registry was started with ``require_signed=True``, the body
        must also include:

        - ``pubkey``    — hex Ed25519 public key (64 chars)
        - ``timestamp`` — ISO 8601 UTC, within ±5 minutes of server time
        - ``signature`` — hex Ed25519 signature over the canonical JSON of
          ``{caps, url, pubkey, timestamp}`` (sorted keys, no spaces)
        """
        from datetime import timedelta
        try:
            caps = NodeCapabilities(**body["caps"])
            url: str = body["url"]
        except (KeyError, TypeError):
            raise HTTPException(status_code=422, detail="Invalid registration body")

        if require_signed:
            pubkey_hex    = body.get("pubkey", "")
            timestamp_str = body.get("timestamp", "")
            signature_hex = body.get("signature", "")

            # Detect key type from pubkey length:
            #   64 hex chars  = Ed25519 (32 bytes)
            #  130 hex chars  = P-256 Secure Enclave (65 bytes uncompressed)
            pubkey_len = len(pubkey_hex)
            if pubkey_len == 64:
                key_type = "ed25519"
            elif pubkey_len == 130:
                key_type = "p256"
            else:
                raise HTTPException(status_code=403, detail="Missing or invalid pubkey")

            # Validate timestamp freshness (±5 minutes)
            try:
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                age = abs((datetime.now(tz=timezone.utc) - ts).total_seconds())
                if age > 300:
                    raise HTTPException(status_code=403, detail="Timestamp expired")
            except (ValueError, TypeError):
                raise HTTPException(status_code=403, detail="Invalid timestamp")

            # Verify signature over payload (everything except signature field)
            signed_payload = {"caps": body["caps"], "url": url,
                              "pubkey": pubkey_hex, "timestamp": timestamp_str}
            if _reg_ops is None:
                raise HTTPException(status_code=503, detail="Signature verification unavailable")
            if not _reg_ops.verify_registration(pubkey_hex, key_type, signed_payload, signature_hex):
                raise HTTPException(status_code=403, detail="Invalid signature")

            # Allowlist check — reject if operator has configured an approved set
            if allowed_pubkeys is not None and pubkey_hex not in allowed_pubkeys:
                raise HTTPException(status_code=403, detail="Public key not in allowlist")

            # node_id must be derived from pubkey — not self-reported
            expected_node_id = _reg_ops.node_id_from_pubkey(pubkey_hex)
            if caps.node_id != expected_node_id:
                raise HTTPException(status_code=403,
                                    detail="node_id does not match pubkey derivation")

            # One keypair = one registry slot — reject if pubkey already
            # registered under a different node_id
            existing = pubkey_node_ids.get(pubkey_hex)
            if existing and existing != caps.node_id:
                raise HTTPException(status_code=403,
                                    detail="pubkey already registered under a different node_id")
            pubkey_node_ids[pubkey_hex] = caps.node_id

            # Re-registration: pubkey must match stored one (prevents takeover)
            stored_pubkey = node_pubkeys.get(caps.node_id)
            if stored_pubkey and stored_pubkey != pubkey_hex:
                raise HTTPException(status_code=403,
                                    detail="Public key mismatch — node_id already registered "
                                           "with a different key")

            node_pubkeys[caps.node_id] = pubkey_hex

        # Capture registration_index before mutating the registry.
        # This is safe because all async handlers run in the same event loop
        # thread (single-worker uvicorn) — no true concurrency between the
        # len() read and register() call.
        registration_index = len(reg._entries)
        reg.register(caps)
        node_urls[caps.node_id] = url

        key = _reg_ops.new_api_key() if _reg_ops is not None else secrets.token_hex(32)
        node_keys[caps.node_id] = key

        if ledger is not None:
            registered_at = datetime.now(tz=timezone.utc).isoformat()
            ledger.ensure_account(caps.node_id, registered_at, registration_index)

        return {"registered": True, "node_id": caps.node_id, "api_key": key}

    @app.delete("/nodes/{node_id}", dependencies=[Depends(_require_admin)])
    async def deregister(node_id: str) -> dict[str, Any]:
        """
        Mark a node offline.

        Requires ``Authorization: Bearer <admin_key>`` when the registry was
        started with ``--admin-key`` / ``ZILVER_REGISTRY_KEY``.

        The node remains in the registry's history (for diagnostics) but
        will no longer be returned by ``/match`` or ``/nodes``.

        Returns ``{"deregistered": true}`` if found, ``{"deregistered": false}``
        if the node was not registered.
        """
        found = reg.deregister(node_id)
        node_urls.pop(node_id, None)
        node_keys.pop(node_id, None)
        return {"deregistered": found, "node_id": node_id}

    @app.post("/nodes/{node_id}/heartbeat",
              dependencies=[Depends(_heartbeat_limit)])
    async def heartbeat(node_id: str, request: Request) -> dict[str, Any]:
        """
        Refresh the last-seen timestamp for a node.

        Called by the node daemon on a fixed interval (default 30 s) so the
        registry can detect stale nodes.

        Requires ``Authorization: Bearer <node_api_key>`` — the key issued to
        this specific node at registration time.  Returns 403 if the key is
        missing or wrong, 404 if the node_id is unknown.
        """
        stored_key = node_keys.get(node_id)
        if stored_key is not None:
            header = request.headers.get("Authorization", "")
            token  = header[7:] if header.startswith("Bearer ") else ""
            if not secrets.compare_digest(token, stored_key):
                raise HTTPException(status_code=403, detail="Invalid node API key")

        found = reg.heartbeat(node_id)
        if not found:
            raise HTTPException(status_code=404, detail="Node not found")
        sqt_earned = ledger.reward_heartbeat(node_id) if ledger is not None else 0.0
        return {"status": "ok", "node_id": node_id, "sqt_earned": sqt_earned}

    # --- Discovery ----------------------------------------------------------

    @app.get("/nodes", dependencies=[Depends(_require_client)])
    async def list_nodes() -> list[dict[str, Any]]:
        """
        Return all currently online nodes.

        Each element is ``NodeCapabilities.to_dict()`` extended with:
        - ``"url"`` — the node's reachable HTTP address
        - ``"node_execute_key"`` — bearer token for ``POST /execute`` on that node

        Requires client authorization (``Authorization: Bearer <client_key>``).
        """
        entries = reg.all_entries()
        result = []
        for entry in entries:
            d = entry.caps.to_dict()
            d["url"] = node_urls.get(entry.caps.node_id, "")
            d["node_execute_key"] = node_keys.get(entry.caps.node_id, "")
            result.append(d)
        return result

    @app.get("/match", dependencies=[Depends(_match_limit), Depends(_require_client)])
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
        On success: ``NodeCapabilities.to_dict()`` plus ``"url"`` and a
        single-use ``"job_token"`` (32 hex chars).  The token must be
        included in the subsequent ``POST /nodes/{id}/contribute`` call to
        authorise SQT reward crediting.  Tokens expire after one hour.

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
        d["node_execute_key"] = node_keys.get(entry.caps.node_id, "")

        # Issue a single-use job token so only the coordinator that matched
        # this job can later call /contribute for it.
        job_token = secrets.token_hex(16)
        job_tokens[job_token] = (entry.caps.node_id, time.monotonic())
        d["job_token"] = job_token
        return d

    # --- Incentive layer -----------------------------------------------------

    @app.post("/nodes/{node_id}/contribute",
              dependencies=[Depends(_check_body_size), Depends(_contribute_limit)])
    async def contribute(node_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """
        Report a completed job contribution and credit SQT rewards.

        Called by :class:`~zilver.client.NetworkCoordinator` after a
        successful job execution.

        The ``job_token`` field must be the single-use token returned by the
        ``GET /match`` call that preceded this job.  The token binds the
        contribute request to a specific coordinator–node pair and is deleted
        on first use.  Expired or reused tokens are rejected with HTTP 403.

        The ``proof`` field must be a valid SHA-256 hex digest (64 hex
        characters) from :class:`~zilver.node.JobResult`.

        Request body
        ~~~~~~~~~~~~
        - ``job_token``      — single-use token from ``GET /match``
        - ``elapsed_ms``     — wall-clock time of the job in milliseconds
        - ``memory_used_mb`` — peak Metal memory used by the job
        - ``proof``          — SHA-256 hex digest from :class:`~zilver.node.JobResult`

        Response
        ~~~~~~~~
        ``{"sqt_earned": float, "balance": float}``
        """
        # Validate body fields first — a malformed request must not consume a token
        try:
            elapsed_ms     = float(body["elapsed_ms"])
            memory_used_mb = float(body["memory_used_mb"])
            proof          = str(body["proof"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(status_code=422, detail="Invalid contribute body")

        if len(proof) != 64 or not all(c in "0123456789abcdefABCDEF" for c in proof):
            raise HTTPException(status_code=422, detail="proof must be a 64-char SHA-256 hex string")

        # Now validate and consume the single-use job token
        job_token = body.get("job_token", "")
        if not job_token:
            raise HTTPException(status_code=403, detail="Missing job_token")
        token_data = job_tokens.pop(job_token, None)
        if token_data is None:
            raise HTTPException(status_code=403, detail="Invalid or already-used job_token")
        token_node_id, token_issued_at = token_data
        if time.monotonic() - token_issued_at > _JOB_TOKEN_TTL:
            raise HTTPException(status_code=403, detail="job_token has expired")
        if token_node_id != node_id:
            raise HTTPException(status_code=403, detail="job_token was not issued for this node")

        _entry = reg._entries.get(node_id)
        if _entry is None or not _entry.online:
            raise HTTPException(status_code=404, detail="Node not found")

        sqt_earned = ledger.reward_job(node_id, elapsed_ms, memory_used_mb) if ledger is not None else 0.0
        balance    = ledger.balance(node_id) if ledger is not None else 0.0
        return {"sqt_earned": sqt_earned, "balance": balance}

    @app.get("/leaderboard")
    async def leaderboard(top_n: int = 20) -> list[dict[str, Any]]:
        """
        Return the top nodes ranked by SQT balance.

        Query parameters
        ~~~~~~~~~~~~~~~~
        - ``top_n`` — number of entries to return (default 20, max 100)

        Response
        ~~~~~~~~
        List of account dicts with fields: ``rank``, ``node_id``, ``balance``,
        ``jobs_done``, ``heartbeats``, ``is_genesis``, ``registered_at``.
        """
        top_n = min(max(top_n, 1), 100)
        if ledger is None:
            return []
        return ledger.leaderboard(top_n=top_n)

    @app.get("/summary")
    async def summary() -> dict[str, Any]:
        """
        Aggregate registry statistics.

        Returns a dict with ``online``, ``total_registered``, ``backends``,
        ``max_sv_qubits``, ``max_dm_qubits``, and ``total_stake``.
        Useful for monitoring dashboards and CLI status commands.
        """
        return reg.summary()

    # --- Credit estimation --------------------------------------------------

    @app.post("/jobs/estimate", dependencies=[Depends(_check_body_size), Depends(_require_client)])
    async def estimate_job(body: dict[str, Any]) -> dict[str, Any]:
        """
        Estimate the credit cost of a job before execution.

        Request body
        ~~~~~~~~~~~~
        - ``backend``  — ``"sv"``, ``"dm"``, or ``"tn"``
        - ``n_qubits`` — qubit count
        - ``shots``    — number of measurement shots (omit for expectation-only)

        Response
        ~~~~~~~~
        JSON dict with ``estimated_credits`` and ``breakdown`` fields.
        """
        from .pricing import estimate_credits, DEFAULT_CONFIG
        from dataclasses import asdict
        try:
            backend  = str(body.get("backend", "sv"))
            n_qubits = int(body["n_qubits"])
            shots    = int(body["shots"]) if body.get("shots") is not None else None
        except (KeyError, TypeError, ValueError):
            raise HTTPException(status_code=422, detail="Invalid estimate body")
        est = estimate_credits(DEFAULT_CONFIG, backend, n_qubits, shots)
        return asdict(est)

    return app


# ---------------------------------------------------------------------------
# Blocking server entrypoint
# ---------------------------------------------------------------------------

def serve_registry(
    registry:           Registry | None = None,
    host:               str = "0.0.0.0",
    port:               int = 7701,
    log_level:          str = "warning",
    admin_key:          str | None = None,
    rate_limit:         bool = False,
    ssl_keyfile:        str | None = None,
    ssl_certfile:       str | None = None,
    ledger_path:        str | None = None,
    require_signed:     bool = False,
    allowed_pubkeys:    set[str] | None = None,
    client_keys:        set[str] | None = None,
) -> None:
    """
    Start a uvicorn HTTP(S) server for the capability registry and block until
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
    admin_key:
        Bearer token required for node deregistration.  ``None`` disables.
    rate_limit:
        Enable per-IP rate limiting on ``POST /nodes`` and ``GET /match``.
    ssl_keyfile:
        Path to the TLS private key (PEM).  When set, the server uses HTTPS.
    ssl_certfile:
        Path to the TLS certificate (PEM).
    """
    import uvicorn
    from pathlib import Path
    _ledger = Ledger(Path(ledger_path)) if ledger_path else None
    app = make_registry_app(
        registry,
        admin_key=admin_key,
        rate_limit=rate_limit,
        ledger=_ledger,
        require_signed=require_signed,
        allowed_pubkeys=allowed_pubkeys,
        client_keys=client_keys,
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )
