"""Off-chain ledger for Zilver node contribution tracking."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_GENESIS_CUTOFF = datetime(2026, 5, 1, tzinfo=timezone.utc)
_GENESIS_MAX_INDEX = 100

_BASE_JOB_REWARD    = 0.001
_job_time_rate      = 0.000_001   # per ms
_job_memory_rate    = 0.000_005   # per MB
_HEARTBEAT_REWARD   = 0.000_1
_GENESIS_MULTIPLIER = 2.0

# Public aliases used by tests and external callers
_ELAPSED_MS_RATE = _job_time_rate
_MEMORY_MB_RATE  = _job_memory_rate


def _configure(
    base_job: float,
    time_rate: float,
    memory_rate: float,
    heartbeat: float,
    genesis_multiplier: float,
) -> None:
    """Override reward parameters. Called once at server startup."""
    global _BASE_JOB_REWARD, _job_time_rate, _job_memory_rate
    global _HEARTBEAT_REWARD, _GENESIS_MULTIPLIER
    global _ELAPSED_MS_RATE, _MEMORY_MB_RATE
    _BASE_JOB_REWARD    = base_job
    _job_time_rate      = time_rate
    _job_memory_rate    = memory_rate
    _HEARTBEAT_REWARD   = heartbeat
    _GENESIS_MULTIPLIER = genesis_multiplier
    _ELAPSED_MS_RATE    = time_rate
    _MEMORY_MB_RATE     = memory_rate


@dataclass
class LedgerAccount:
    node_id:       str
    balance:       float = 0.0
    jobs_done:     int   = 0
    heartbeats:    int   = 0
    is_genesis:    bool  = False
    registered_at: str   = ""   # ISO 8601 UTC


class Ledger:
    """
    Persistent off-chain SQT token ledger.

    Stores one :class:`LedgerAccount` per node.  All mutating operations are
    thread-safe.  The ledger is atomically persisted to a JSON file after
    every mutation (write to ``.tmp``, then ``os.replace``).

    Parameters
    ----------
    path:
        Absolute path to the ledger JSON file, e.g. ``~/.zilver/ledger.json``.
        Parent directory is created automatically if it does not exist.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._accounts: dict[str, LedgerAccount] = {}
        self._registration_order: list[str] = []  # node_ids in insertion order
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_account(
        self,
        node_id:            str,
        registered_at:      str,
        registration_index: int,
    ) -> LedgerAccount:
        """
        Create an account for *node_id* if it does not already exist.

        Determines genesis status: node must be in the first
        :data:`_GENESIS_MAX_INDEX` registrations **and** registered before
        :data:`_GENESIS_CUTOFF`.

        Parameters
        ----------
        node_id:
            Unique node identifier.
        registered_at:
            ISO 8601 UTC timestamp string of registration time.
        registration_index:
            Zero-based count of nodes registered before this one.
            Pass ``len(existing_nodes)`` from the registry.

        Returns
        -------
        LedgerAccount
            The (possibly newly created) account.
        """
        with self._lock:
            if node_id in self._accounts:
                return self._accounts[node_id]

            try:
                ts = datetime.fromisoformat(registered_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ts = datetime.now(tz=timezone.utc)

            is_genesis = (
                registration_index < _GENESIS_MAX_INDEX
                and ts < _GENESIS_CUTOFF
            )

            acct = LedgerAccount(
                node_id=node_id,
                registered_at=registered_at,
                is_genesis=is_genesis,
            )
            self._accounts[node_id] = acct
            self._registration_order.append(node_id)
            self._save()
            return acct

    def reward_job(
        self,
        node_id:        str,
        elapsed_ms:     float,
        memory_used_mb: float,
    ) -> float:
        """
        Credit a node for completing a simulation job.

        Returns
        -------
        float
            Tokens credited.
        """
        with self._lock:
            acct = self._accounts.get(node_id)
            if acct is None:
                return 0.0
            sqt = _BASE_JOB_REWARD + _job_time_rate * elapsed_ms + _job_memory_rate * memory_used_mb
            if acct.is_genesis:
                sqt *= _GENESIS_MULTIPLIER
            acct.balance   += sqt
            acct.jobs_done += 1
            self._save()
            return sqt

    def reward_heartbeat(self, node_id: str) -> float:
        """
        Credit a node for sending a heartbeat (uptime signal).

        Returns
        -------
        float
            Tokens credited (0.0 if node not found).
        """
        with self._lock:
            acct = self._accounts.get(node_id)
            if acct is None:
                return 0.0
            sqt = _HEARTBEAT_REWARD * (_GENESIS_MULTIPLIER if acct.is_genesis else 1.0)
            acct.balance    += sqt
            acct.heartbeats += 1
            self._save()
            return sqt

    def leaderboard(self, top_n: int = 20) -> list[dict[str, Any]]:
        """
        Return the top *top_n* accounts sorted by SQT balance (descending).

        Returns
        -------
        list[dict]
            Each dict is a :class:`LedgerAccount` serialized with an added
            ``"rank"`` key (1-based).
        """
        with self._lock:
            ranked = sorted(
                self._accounts.values(),
                key=lambda a: a.balance,
                reverse=True,
            )[:top_n]
            return [
                {**asdict(acct), "rank": i + 1}
                for i, acct in enumerate(ranked)
            ]

    def balance(self, node_id: str) -> float:
        """Return current SQT balance for *node_id*, or 0.0 if unknown."""
        with self._lock:
            acct = self._accounts.get(node_id)
            return acct.balance if acct else 0.0

    def get_account(self, node_id: str) -> LedgerAccount | None:
        """Return the account for *node_id*, or None if not found."""
        with self._lock:
            return self._accounts.get(node_id)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Atomically write ledger to disk (caller must hold self._lock)."""
        data = {
            "accounts": {
                nid: asdict(acct)
                for nid, acct in self._accounts.items()
            },
            "registration_order": self._registration_order,
        }
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self._path)

    def _load(self) -> None:
        """Load ledger from disk (called once at init, no lock needed)."""
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for nid, raw in data.get("accounts", {}).items():
                self._accounts[nid] = LedgerAccount(**raw)
            self._registration_order = data.get("registration_order", list(self._accounts.keys()))
        except (json.JSONDecodeError, TypeError, KeyError):
            # Corrupt file — start fresh
            self._accounts = {}
            self._registration_order = []
