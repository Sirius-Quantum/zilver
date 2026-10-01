"""Registry HTTP server."""

from __future__ import annotations

import asyncio
import json
import math
import secrets
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Request

from .node_types import NodeCapabilities, issue_execute_token
from .registry import Registry

try:
    from . import _registry_ops as _reg_ops
except ImportError:
    _reg_ops = None  # type: ignore[assignment]

try:
    from .registry_store import RegistryStore as _RS
except ImportError:
    _RS = None  # type: ignore[assignment,misc]

try:
    from . import _canary as _cn
except ImportError:
    _cn = None  # type: ignore[assignment]

try:
    from .ledger import Ledger as _Ledger
except ImportError:
    _Ledger = None  # type: ignore[assignment,misc]

_MAX_BODY_BYTES    = 64 * 1024
_JOB_TOKEN_TTL     = 3600.0


def _is_private_url(url: str) -> bool:
    """Return True unless every address *url*'s host resolves to is public.

    Hostnames are resolved, so a public DNS name that points at 10.x or at
    169.254.169.254 is refused, and so is a host that does not resolve. The
    registry resolves the name again when it calls the node, so this does not
    stop DNS rebinding; the instance must enforce IMDSv2 as well.
    """
    import ipaddress
    import socket
    import urllib.parse
    try:
        host = urllib.parse.urlparse(url).hostname or ""
        infos = socket.getaddrinfo(host, None) if host else []
    except (ValueError, UnicodeError, OSError):
        return True
    if not infos:
        return True
    for *_, sockaddr in infos:
        addr = ipaddress.ip_address(str(sockaddr[0]).split("%")[0])
        if not addr.is_global or addr.is_multicast:
            return True
    return False


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
    ledger:          Any = None,
    require_signed:  bool = False,
    allowed_pubkeys: set[str] | None = None,
    client_keys:        set[str] | None = None,
    db_path:            Path | str | None = None,
    allow_private_urls: bool = False,
    audit_log_path:     Path | str | None = None,
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
        Optional job-accounting hook, notified on registration, heartbeat
        and job report.  When ``None`` (the default) those endpoints return
        only their own fields.
    require_signed:
        When ``True``, every ``POST /nodes`` registration must include a
        valid ``pubkey`` + ``signature`` + ``timestamp``.  Nodes
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
    db_path:
        Path to the SQLite persistence file.  When set, node registrations
        survive registry restarts.  When ``None`` (default) state is
        in-memory only.
    allow_private_urls:
        When ``False`` (default), ``POST /nodes`` rejects registrations
        whose URL resolves to a private, loopback, or link-local address.
        Set to ``True`` for local development and tests.

    Returns
    -------
    FastAPI
        The application instance.  Use ``uvicorn.run`` for production or
        ``fastapi.testclient.TestClient`` for tests.
    """
    reg = registry if registry is not None else Registry()

    _store: Any = _RS(Path(db_path)) if (db_path is not None and _RS is not None) else None

    # Maps node_id → advertised URL so clients can connect directly.
    node_urls:      dict[str, str] = {}
    # Maps node_id → the node's own key: heartbeat, contribute, re-register.
    # Never returned to anyone but that node.
    node_keys:      dict[str, str] = {}
    # Maps node_id → the secret the node sent at registration. The registry
    # signs execute tokens with it; it is never returned to anyone.
    node_execute_secrets: dict[str, str] = {}
    # Maps node_id → registered public key (hex).
    node_pubkeys:   dict[str, str] = {}
    # Maps pubkey_hex → node_id — enforces one slot per keypair.
    pubkey_node_ids: dict[str, str] = {}
    # Maps job_token → (node_id, client_key, issued_at).
    # client_key is "" when client_keys auth is disabled.
    # Tokens are deleted on first use, and by the 60 s loop once they exceed
    # _JOB_TOKEN_TTL.
    job_tokens:     dict[str, tuple[str, str, float]] = {}

    # Async job queue: job_id → {"status", "result"?, "error"?}
    _async_jobs:    dict[str, dict[str, Any]] = {}
    # Idempotency cache: idem_key → job_id (1h TTL)
    _idem_cache:    dict[str, tuple[str, float]] = {}

    # Restore persisted state from DB (online nodes only).
    if _store is not None:
        for _row in _store.load_all():
            if not _row.get("online", 1) or not _row.get("execute_secret"):
                continue   # a row without a secret re-registers on its next heartbeat
            try:
                _caps = NodeCapabilities(**json.loads(_row["caps_json"]))
            except Exception:
                continue
            reg.register(_caps)
            node_urls[_row["node_id"]] = _row["url"]
            node_keys[_row["node_id"]] = _row["api_key"]
            node_execute_secrets[_row["node_id"]] = _row["execute_secret"]
            _pk = _row.get("pubkey_hex", "")
            if _pk:
                node_pubkeys[_row["node_id"]] = _pk
                pubkey_node_ids[_pk] = _row["node_id"]

    # Per-node canary failure counter.
    _canary_fails: dict[str, int] = {}

    # --- Audit log -----------------------------------------------------------

    _alog = Path(audit_log_path) if audit_log_path else None

    def _audit(event: str, **kw: Any) -> None:
        if _alog is None:
            return
        import json as _j
        line = _j.dumps({"ts": datetime.now(tz=timezone.utc).isoformat(),
                         "event": event, **kw})
        try:
            with _alog.open("a") as _f:
                _f.write(line + "\n")
        except OSError:
            pass

    # --- Canary check coroutine ----------------------------------------------

    async def _run_canary(node_id: str, node_url: str) -> None:
        secret = node_execute_secrets.get(node_id)
        if _cn is None or not secret:
            return
        # A token of its own, never entered in job_tokens, so a canary job
        # earns nothing when the node reports it.
        token = issue_execute_token(secret, node_id, secrets.token_hex(16))

        async def _on_fail() -> None:
            _canary_fails[node_id] = _canary_fails.get(node_id, 0) + 1
            _audit("canary_fail", node_id=node_id,
                   fails=_canary_fails[node_id])
            if _canary_fails[node_id] >= _cn.MAX_FAILS:
                reg.deregister(node_id)
                if _store is not None:
                    _store.set_online(node_id, False)
                _audit("evict", node_id=node_id, reason="canary")

        await _cn.maybe_check(node_url, token, on_fail=_on_fail)

    @asynccontextmanager
    async def _lifespan(application: FastAPI):  # type: ignore[type-arg]
        async def _evict_loop() -> None:
            while True:
                await asyncio.sleep(60)
                _now = time.monotonic()
                for _tok, (_, _, _issued) in list(job_tokens.items()):
                    if _now - _issued > _JOB_TOKEN_TTL:
                        job_tokens.pop(_tok, None)
                evicted = reg.prune_stale()
                for _nid in evicted:
                    if _store is not None:
                        _store.set_online(_nid, False)
                    _audit("evict", node_id=_nid, reason="stale")
        task = asyncio.create_task(_evict_loop())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="zilver-registry", version="0.1.0", lifespan=_lifespan)
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

    def _extract_client_key(request: Request) -> str:
        """Return the Bearer token from Authorization header, or '' if absent."""
        h = request.headers.get("Authorization", "")
        return h[7:] if h.startswith("Bearer ") else ""

    def _check_node_key(node_id: str, request: Request) -> None:
        """403 unless the request carries *node_id*'s own key."""
        stored_key = node_keys.get(node_id)
        token = _extract_client_key(request)
        if stored_key is None or not secrets.compare_digest(token, stored_key):
            raise HTTPException(status_code=403, detail="Invalid node API key")

    def _issue_match(entry: Any, client_key: str) -> dict[str, Any]:
        """Record a single-use job token and wrap it in an execute token for
        the client. The client gets the node's URL and that token, never a
        node credential."""
        node_id = entry.caps.node_id
        job_token = secrets.token_hex(16)
        job_tokens[job_token] = (node_id, client_key, time.monotonic())
        return {
            "url":           node_urls.get(node_id, ""),
            "job_token":     job_token,
            "execute_token": issue_execute_token(
                node_execute_secrets[node_id], node_id, job_token),
        }

    async def _require_client(request: Request) -> None:
        """Verify client Bearer token against client_keys set.  No-op when client_keys is None."""
        if client_keys is None:
            return
        token = _extract_client_key(request)
        if not token or not any(secrets.compare_digest(token, k) for k in client_keys):
            raise HTTPException(status_code=401, detail="Invalid or missing client API key")

    # --- Registration -------------------------------------------------------

    @app.post("/nodes", status_code=201,
              dependencies=[Depends(_check_body_size), Depends(_register_limit),
                            Depends(_register_limit_hourly)])
    async def register(body: dict[str, Any], request: Request) -> dict[str, Any]:
        """
        Register or re-register a node.

        Request body
        ~~~~~~~~~~~~
        A JSON object with two fields:

        - ``caps`` — ``NodeCapabilities.to_dict()``
        - ``url``  — the node's reachable HTTP base URL,
          e.g. ``"http://192.168.1.5:7700"``

        - ``execute_secret`` — at least 32 random hex characters the node
          generated. The registry signs the node's execute tokens with it and
          never returns it.

        Re-registering a node_id that is already registered requires that
        node's current key in ``Authorization: Bearer <key>``, or, when the
        registry requires signatures, a signature from the same public key.

        ``caps.stake`` is not taken from the node; the registry sets it.

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

        - ``pubkey``    — hex-encoded public key
        - ``timestamp`` — ISO 8601 UTC, within ±5 minutes of server time
        - ``signature`` — hex-encoded signature over the canonical JSON of
          ``{caps, url, pubkey, timestamp}`` (sorted keys, no spaces)
        """
        try:
            caps = NodeCapabilities(**body["caps"])
            url: str = body["url"]
            execute_secret = str(body["execute_secret"])
        except (KeyError, TypeError):
            raise HTTPException(status_code=422, detail="Invalid registration body")
        if len(execute_secret) < 32 or not all(c in "0123456789abcdef" for c in execute_secret):
            raise HTTPException(status_code=422,
                                detail="execute_secret must be at least 32 lowercase hex characters")
        caps.stake = 0   # self-reported stake would decide /match; nothing backs it yet

        if not allow_private_urls and _is_private_url(url):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Private or loopback URLs are not allowed. "
                    "Use --public-url to set an externally reachable address "
                    "(e.g. a Cloudflare Tunnel or VPS public IP)."
                ),
            )

        if require_signed:
            pubkey_hex    = body.get("pubkey", "")
            timestamp_str = body.get("timestamp", "")
            signature_hex = body.get("signature", "")

            # Detect key type from pubkey length.
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

        elif caps.node_id in node_keys:
            # Unsigned mode: taking over a registered node_id needs its key.
            _check_node_key(caps.node_id, request)

        # Capture registration_index before mutating the registry.
        # This is safe because all async handlers run in the same event loop
        # thread (single-worker uvicorn) — no true concurrency between the
        # len() read and register() call.
        registration_index = len(reg._entries)
        reg.register(caps)
        node_urls[caps.node_id] = url

        key = _reg_ops.new_api_key() if _reg_ops is not None else secrets.token_hex(32)
        node_keys[caps.node_id] = key
        node_execute_secrets[caps.node_id] = execute_secret

        registered_at = datetime.now(tz=timezone.utc).isoformat()

        if ledger is not None:
            ledger.on_register(caps.node_id, registered_at, registration_index)

        if _store is not None:
            _pk = node_pubkeys.get(caps.node_id, "")
            _store.save_node(
                caps.node_id,
                json.dumps(caps.to_dict()),
                url,
                key,
                _pk,
                datetime.now(tz=timezone.utc).timestamp(),
                execute_secret,
            )

        _audit("register", node_id=caps.node_id, url=url)
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
        node_execute_secrets.pop(node_id, None)
        if _store is not None:
            _store.delete_node(node_id)
        _audit("deregister", node_id=node_id)
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
        if node_id in node_keys:
            _check_node_key(node_id, request)

        found = reg.heartbeat(node_id)
        if not found:
            raise HTTPException(status_code=404, detail="Node not found")

        if _store is not None:
            _store.update_heartbeat(node_id, time.time())

        extra = ledger.on_heartbeat(node_id) if ledger is not None else {}
        return {"status": "ok", "node_id": node_id, **extra}

    # --- Discovery ----------------------------------------------------------

    @app.get("/nodes", dependencies=[Depends(_require_client)])
    async def list_nodes() -> list[dict[str, Any]]:
        """
        Return all currently online nodes.

        Each element is ``NodeCapabilities.to_dict()`` extended with
        ``"url"``. To run a job, get an execute token from ``GET /match``.

        Requires client authorization (``Authorization: Bearer <client_key>``).
        """
        entries = reg.all_entries()
        result = []
        for entry in entries:
            d = entry.caps.to_dict()
            d["url"] = node_urls.get(entry.caps.node_id, "")
            result.append(d)
        return result

    @app.get("/match", dependencies=[Depends(_match_limit), Depends(_require_client)])
    async def match(
        request:   Request,
        backend:   str,
        n_qubits:  int,
        min_stake: int = 0,
        node_id:   str | None = None,
    ) -> dict[str, Any]:
        """
        Find the best available node for a job.

        Query parameters
        ~~~~~~~~~~~~~~~~
        - ``backend``   — ``"sv"``, ``"dm"``, or ``"tn"``
        - ``n_qubits``  — qubit count required by the job
        - ``min_stake`` — minimum stake (default 0)
        - ``node_id``   — optional: match only this node (batch dispatch)

        Response
        ~~~~~~~~
        On success: ``NodeCapabilities.to_dict()`` plus ``"url"``, a
        single-use ``"job_token"`` (32 hex chars) and an ``"execute_token"``
        to send as ``Authorization: Bearer`` on the node's ``POST /execute``.
        The execute token expires after ten minutes; the node reports the
        job_token back in its own ``/contribute`` call.

        Raises **404** if no eligible node exists.
        """
        entry = reg.match(backend, n_qubits, min_stake=min_stake, node_id=node_id)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No eligible node for backend={backend!r} "
                    f"n_qubits={n_qubits} min_stake={min_stake}"
                ),
            )
        ck = _extract_client_key(request) if client_keys is not None else ""
        return {**entry.caps.to_dict(), **_issue_match(entry, ck)}

    # --- Job reports ---------------------------------------------------------

    @app.post("/nodes/{node_id}/contribute",
              dependencies=[Depends(_check_body_size), Depends(_contribute_limit)])
    async def contribute(node_id: str, request: Request, body: dict[str, Any]) -> dict[str, Any]:
        """
        Report a completed job to the registry.

        Called by the node that ran the job, with its own key in
        ``Authorization: Bearer <node_api_key>``. Clients never report work.

        The ``job_token`` field must be the single-use token from the
        ``GET /match`` that the node's execute token carried. It must have
        been issued for this node and is deleted on first use. Expired or
        reused tokens are rejected with HTTP 403.

        ``elapsed_ms`` must be finite and non-negative, and is capped at the
        time since the match. ``memory_used_mb`` must be finite and
        non-negative, and is capped at the node's RAM.

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
        ``{"status": "ok"}``
        """
        _entry = reg._entries.get(node_id)
        if _entry is None or not _entry.online:
            raise HTTPException(status_code=404, detail="Node not found")
        _check_node_key(node_id, request)

        # Validate body fields first — a malformed request must not consume a token
        try:
            elapsed_ms     = float(body["elapsed_ms"])
            memory_used_mb = float(body["memory_used_mb"])
            proof          = str(body["proof"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(status_code=422, detail="Invalid contribute body")
        if not (math.isfinite(elapsed_ms) and elapsed_ms >= 0
                and math.isfinite(memory_used_mb) and memory_used_mb >= 0):
            raise HTTPException(status_code=422,
                                detail="elapsed_ms and memory_used_mb must be finite and >= 0")

        if len(proof) != 64 or not all(c in "0123456789abcdefABCDEF" for c in proof):
            raise HTTPException(status_code=422, detail="proof must be a 64-char SHA-256 hex string")

        # Validate and consume the single-use job token
        job_token = body.get("job_token", "")
        if not job_token:
            raise HTTPException(status_code=403, detail="Missing job_token")
        token_data = job_tokens.pop(job_token, None)
        if token_data is None:
            raise HTTPException(status_code=403, detail="Invalid or already-used job_token")
        token_node_id, _, token_issued_at = token_data
        since_match_s = time.monotonic() - token_issued_at
        if since_match_s > _JOB_TOKEN_TTL:
            raise HTTPException(status_code=403, detail="job_token has expired")
        if token_node_id != node_id:
            raise HTTPException(status_code=403, detail="job_token was not issued for this node")

        # The node cannot have worked longer than since the match, or used
        # more memory than it has.
        elapsed_ms     = min(elapsed_ms, since_match_s * 1000.0)
        memory_used_mb = min(memory_used_mb, _entry.caps.ram_gb * 1024.0)

        extra = ledger.on_job(node_id, elapsed_ms, memory_used_mb) if ledger is not None else {}

        _url = node_urls.get(node_id, "")
        if _url and _cn is not None:
            asyncio.create_task(_run_canary(node_id, _url))

        return {"status": "ok", **extra}

    # --- Async job API -------------------------------------------------------

    @app.post("/jobs", status_code=202,
              dependencies=[Depends(_check_body_size), Depends(_require_client)])
    async def submit_job(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        """
        Submit a job for async execution.

        Accepts an optional ``X-Idempotency-Key`` header; repeating the same
        key within one hour returns the original job_id without creating a
        duplicate.

        Returns ``{"job_id": "<uuid>", "status": "queued"}``.
        """
        ck = _extract_client_key(request) if client_keys is not None else ""
        idem_key = request.headers.get("X-Idempotency-Key", "")
        now = time.monotonic()
        if idem_key:
            cached = _idem_cache.get(idem_key)
            if cached:
                jid, issued = cached
                if (now - issued < _JOB_TOKEN_TTL and jid in _async_jobs
                        and _async_jobs[jid].get("client_key") == ck):
                    return {"job_id": jid, "status": _async_jobs[jid]["status"]}

        job_id = str(uuid.uuid4())
        _async_jobs[job_id] = {"status": "queued", "body": body}
        if idem_key:
            _idem_cache[idem_key] = (job_id, now)

        # Dispatch: try to match and execute immediately; mark result inline.
        # Full async worker pool is a Phase 3 concern — this gives clients the
        # non-blocking interface now without a separate worker process.
        try:
            backend  = str(body.get("backend", "sv"))
            n_qubits = int(body["n_qubits"])
        except (KeyError, TypeError, ValueError):
            _async_jobs[job_id] = {
                "status": "failed",
                "error": {"code": "invalid_circuit", "detail": "Missing or invalid n_qubits/backend"},
                "client_key": ck,
            }
            return {"job_id": job_id, "status": "failed"}

        entry = reg.match(backend, n_qubits)
        if entry is None:
            _async_jobs[job_id] = {
                "status": "failed",
                "error": {"code": "node_unreachable", "detail": "No eligible node available"},
            }
        else:
            _async_jobs[job_id] = {
                "status": "matched",
                "node_id": entry.caps.node_id,
                **_issue_match(entry, ck),
            }
        _async_jobs[job_id]["client_key"] = ck

        return {"job_id": job_id, "status": _async_jobs[job_id]["status"]}

    @app.get("/jobs/{job_id}", dependencies=[Depends(_require_client)])
    async def get_job(job_id: str, request: Request) -> dict[str, Any]:
        """
        Poll the status of an async job.

        Returns ``{"job_id", "status"}`` plus ``"node_id"``, ``"url"``,
        ``"job_token"`` and ``"execute_token"`` when status is ``"matched"``,
        or ``"error"`` when status is ``"failed"``. Only the client that
        submitted the job can read it.
        """
        job = _async_jobs.get(job_id)
        ck = _extract_client_key(request) if client_keys is not None else ""
        if job is None or job.get("client_key", "") != ck:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job_id,
                **{k: v for k, v in job.items() if k not in ("client_key", "body")}}

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

    if ledger is not None and hasattr(ledger, "install"):
        ledger.install(app)

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
    db_path:            str | None = None,
    allow_private_urls: bool = False,
    audit_log_path:     str | None = None,
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
    if ledger_path and _Ledger is None:
        raise SystemExit("--ledger-path: no ledger is available")
    _ledger = _Ledger(Path(ledger_path)) if ledger_path else None
    app = make_registry_app(
        registry,
        admin_key=admin_key,
        rate_limit=rate_limit,
        ledger=_ledger,
        require_signed=require_signed,
        allowed_pubkeys=allowed_pubkeys,
        client_keys=client_keys,
        db_path=Path(db_path) if db_path else None,
        allow_private_urls=allow_private_urls,
        audit_log_path=Path(audit_log_path) if audit_log_path else None,
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )
