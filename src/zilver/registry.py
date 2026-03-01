"""Node registry."""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Iterator

from .node import NodeCapabilities, SimJob


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

@dataclass
class RegistryEntry:
    """
    Live record for one registered node.

    caps:           capabilities advertised at registration
    registered_at:  epoch timestamp of first registration
    last_seen:      epoch timestamp of most recent heartbeat
    jobs_in_flight: number of jobs currently executing (updated by coordinator)
    online:         False if the node has been explicitly deregistered
    """
    caps:           NodeCapabilities
    registered_at:  float = field(default_factory=time.time)
    last_seen:      float = field(default_factory=time.time)
    jobs_in_flight: int   = 0
    online:         bool  = True

    def heartbeat(self) -> None:
        self.last_seen = time.time()

    def is_stale(self, ttl_seconds: float = 60.0) -> bool:
        return (time.time() - self.last_seen) > ttl_seconds


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class Registry:
    """
    In-memory capability registry.

    Thread-safety: not thread-safe by default. For concurrent access,
    wrap in a lock or use the future async registry layer.

    Usage:
        registry = Registry()
        registry.register(node.caps)

        entry = registry.match(backend="sv", n_qubits=20)
        if entry:
            result = node.execute(job)
            registry.complete(entry.caps.node_id)
    """

    def __init__(self, stale_ttl: float = 60.0):
        self._entries: dict[str, RegistryEntry] = {}
        self.stale_ttl = stale_ttl

    # --- Registration --------------------------------------------------------

    def register(self, caps: NodeCapabilities) -> RegistryEntry:
        """
        Register or re-register a node.

        If the node_id already exists, updates capabilities and resets
        last_seen (equivalent to a heartbeat + capability refresh).
        """
        if caps.node_id in self._entries:
            entry = self._entries[caps.node_id]
            entry.caps     = caps
            entry.online   = True
            entry.last_seen = time.time()
        else:
            entry = RegistryEntry(caps=caps)
            self._entries[caps.node_id] = entry
        return entry

    def deregister(self, node_id: str) -> bool:
        """Mark a node offline. Returns True if the node was found."""
        if node_id in self._entries:
            self._entries[node_id].online = False
            return True
        return False

    def heartbeat(self, node_id: str) -> bool:
        """Refresh last_seen for a node. Returns True if found."""
        if node_id in self._entries:
            self._entries[node_id].heartbeat()
            return True
        return False

    # --- Querying ------------------------------------------------------------

    def get(self, node_id: str) -> RegistryEntry | None:
        return self._entries.get(node_id)

    def all_entries(self, include_offline: bool = False) -> list[RegistryEntry]:
        entries = list(self._entries.values())
        if not include_offline:
            entries = [e for e in entries if e.online]
        return entries

    def online_count(self) -> int:
        return sum(1 for e in self._entries.values() if e.online)

    def prune_stale(self) -> list[str]:
        """
        Mark stale nodes (no heartbeat within stale_ttl) as offline.
        Returns list of node_ids that were pruned.
        """
        pruned = []
        for node_id, entry in self._entries.items():
            if entry.online and entry.is_stale(self.stale_ttl):
                entry.online = False
                pruned.append(node_id)
        return pruned

    # --- Matchmaking ---------------------------------------------------------

    def match(
        self,
        backend:  str,
        n_qubits: int,
        min_stake: int = 0,
    ) -> RegistryEntry | None:
        """
        Find the best available node for a job.

        Selection criteria:
          - online and not stale
          - supports the requested backend
          - qubit ceiling >= n_qubits
          - stake >= min_stake
          - sorted by (jobs_in_flight ASC, stake DESC)

        Returns None if no eligible node exists.
        """
        candidates = [
            e for e in self._entries.values()
            if e.online
            and not e.is_stale(self.stale_ttl)
            and e.caps.supports(backend, n_qubits)
            and e.caps.stake >= min_stake
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda e: (e.jobs_in_flight, -e.caps.stake))

    def match_pair(
        self,
        backend_left:   str,
        n_qubits_left:  int,
        backend_right:  str,
        n_qubits_right: int,
    ) -> tuple[RegistryEntry, RegistryEntry] | None:
        """
        Find two nodes for a subcircuit pair (circuit cutting).

        Tries to use distinct nodes for left and right; falls back to
        the same node if only one eligible node is available.
        """
        left = self.match(backend_left, n_qubits_left)
        if left is None:
            return None
        right = self.match(backend_right, n_qubits_right)
        if right is None:
            return None
        return left, right

    def match_all(
        self,
        backend:  str,
        n_qubits: int,
        count:    int,
    ) -> list[RegistryEntry]:
        """
        Return up to `count` eligible nodes for parallel batch dispatch.
        Used by the batch distributor for vmap grid slicing.
        """
        candidates = [
            e for e in self._entries.values()
            if e.online
            and not e.is_stale(self.stale_ttl)
            and e.caps.supports(backend, n_qubits)
        ]
        candidates.sort(key=lambda e: (e.jobs_in_flight, -e.caps.stake))
        return candidates[:count]

    # --- Load tracking -------------------------------------------------------

    def assign(self, node_id: str) -> None:
        """Increment jobs_in_flight when a job is dispatched."""
        if node_id in self._entries:
            self._entries[node_id].jobs_in_flight += 1

    def complete(self, node_id: str) -> None:
        """Decrement jobs_in_flight when a job finishes."""
        if node_id in self._entries:
            entry = self._entries[node_id]
            entry.jobs_in_flight = max(0, entry.jobs_in_flight - 1)
            entry.caps.jobs_completed += 1

    # --- Routing a full job --------------------------------------------------

    def route(self, job: SimJob, min_stake: int = 0) -> RegistryEntry | None:
        """
        Match a SimJob to an eligible node.

        Convenience wrapper over match() that reads backend and n_qubits
        directly from the job.
        """
        return self.match(job.backend, job.n_qubits, min_stake=min_stake)

    # --- Diagnostics ---------------------------------------------------------

    def summary(self) -> dict:
        entries = self.all_entries()
        return {
            "online":          len(entries),
            "total_registered": len(self._entries),
            "backends":        sorted({b for e in entries for b in e.caps.backends}),
            "max_sv_qubits":   max((e.caps.sv_qubits_max for e in entries), default=0),
            "max_dm_qubits":   max((e.caps.dm_qubits_max for e in entries), default=0),
            "total_stake":     sum(e.caps.stake for e in entries),
        }

    def __len__(self) -> int:
        return self.online_count()

    def __iter__(self) -> Iterator[RegistryEntry]:
        return iter(self.all_entries())

    def __repr__(self) -> str:
        return f"Registry(online={self.online_count()}, total={len(self._entries)})"
