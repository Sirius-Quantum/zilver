"""
Tests for the capability registry.

Correctness checks:
  - RegistryEntry fields and defaults
  - RegistryEntry.heartbeat() updates last_seen
  - RegistryEntry.is_stale() respects TTL
  - Registry.register() creates and updates entries
  - Registry.deregister() marks offline
  - Registry.heartbeat() refreshes last_seen
  - Registry.get() returns entry or None
  - Registry.all_entries() filters offline
  - Registry.online_count()
  - Registry.prune_stale() marks stale nodes offline
  - Registry.match() returns best eligible node
  - Registry.match() rejects offline/stale/unsupported/over-capacity nodes
  - Registry.match() sorts by (jobs_in_flight ASC, stake DESC)
  - Registry.match_pair() returns two entries
  - Registry.match_all() returns up to count entries sorted
  - Registry.assign() / Registry.complete() update jobs_in_flight
  - Registry.complete() increments jobs_completed
  - Registry.complete() clamps at zero
  - Registry.route() delegates to match()
  - Registry.summary() has correct keys
  - __len__, __iter__, __repr__
"""

import time
import pytest

import sys; sys.path.insert(0, "src")
from zilver.node import NodeCapabilities, SimJob
from zilver.registry import Registry, RegistryEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _caps(
    node_id: str = "node-0",
    backends: list[str] | None = None,
    sv_max: int = 20,
    stake: int = 0,
) -> NodeCapabilities:
    return NodeCapabilities(
        node_id=node_id,
        chip="Apple M4",
        ram_gb=16,
        sv_qubits_max=sv_max,
        dm_qubits_max=10,
        tn_qubits_max=50,
        backends=backends or ["sv"],
        stake=stake,
    )


def _simple_job(backend: str = "sv", n_qubits: int = 4) -> SimJob:
    return SimJob(
        circuit_ops=[{"type": "h", "qubits": [0], "param_idx": None}],
        n_qubits=n_qubits,
        n_params=0,
        params=[],
        backend=backend,
    )


# ---------------------------------------------------------------------------
# RegistryEntry
# ---------------------------------------------------------------------------

class TestRegistryEntry:
    def test_defaults(self):
        caps = _caps()
        entry = RegistryEntry(caps=caps)
        assert entry.jobs_in_flight == 0
        assert entry.online is True
        assert entry.registered_at > 0
        assert entry.last_seen > 0

    def test_heartbeat_updates_last_seen(self):
        caps = _caps()
        entry = RegistryEntry(caps=caps)
        old = entry.last_seen
        time.sleep(0.01)
        entry.heartbeat()
        assert entry.last_seen > old

    def test_is_stale_false_immediately(self):
        entry = RegistryEntry(caps=_caps())
        assert entry.is_stale(ttl_seconds=60.0) is False

    def test_is_stale_true_after_ttl(self):
        entry = RegistryEntry(caps=_caps())
        entry.last_seen = time.time() - 120.0
        assert entry.is_stale(ttl_seconds=60.0) is True

    def test_is_stale_boundary(self):
        entry = RegistryEntry(caps=_caps())
        entry.last_seen = time.time() - 59.9
        assert entry.is_stale(ttl_seconds=60.0) is False
        entry.last_seen = time.time() - 60.1
        assert entry.is_stale(ttl_seconds=60.0) is True


# ---------------------------------------------------------------------------
# Registry.register / deregister / heartbeat / get
# ---------------------------------------------------------------------------

class TestRegistryRegistration:
    def test_register_returns_entry(self):
        reg = Registry()
        entry = reg.register(_caps("n0"))
        assert isinstance(entry, RegistryEntry)

    def test_register_stores_node(self):
        reg = Registry()
        reg.register(_caps("n0"))
        assert reg.get("n0") is not None

    def test_register_idempotent(self):
        reg = Registry()
        reg.register(_caps("n0"))
        reg.register(_caps("n0"))
        assert len(reg._entries) == 1

    def test_register_updates_caps(self):
        reg = Registry()
        reg.register(_caps("n0", sv_max=10))
        reg.register(_caps("n0", sv_max=30))
        assert reg.get("n0").caps.sv_qubits_max == 30

    def test_register_reactivates_offline_node(self):
        reg = Registry()
        reg.register(_caps("n0"))
        reg.deregister("n0")
        assert reg.get("n0").online is False
        reg.register(_caps("n0"))
        assert reg.get("n0").online is True

    def test_deregister_marks_offline(self):
        reg = Registry()
        reg.register(_caps("n0"))
        result = reg.deregister("n0")
        assert result is True
        assert reg.get("n0").online is False

    def test_deregister_unknown_returns_false(self):
        reg = Registry()
        assert reg.deregister("nonexistent") is False

    def test_heartbeat_updates_last_seen(self):
        reg = Registry()
        reg.register(_caps("n0"))
        entry = reg.get("n0")
        old = entry.last_seen
        time.sleep(0.01)
        result = reg.heartbeat("n0")
        assert result is True
        assert entry.last_seen > old

    def test_heartbeat_unknown_returns_false(self):
        reg = Registry()
        assert reg.heartbeat("nonexistent") is False

    def test_get_unknown_returns_none(self):
        reg = Registry()
        assert reg.get("nonexistent") is None


# ---------------------------------------------------------------------------
# Registry.all_entries / online_count
# ---------------------------------------------------------------------------

class TestRegistryQuerying:
    def test_all_entries_excludes_offline_by_default(self):
        reg = Registry()
        reg.register(_caps("n0"))
        reg.register(_caps("n1"))
        reg.deregister("n1")
        entries = reg.all_entries()
        assert len(entries) == 1
        assert entries[0].caps.node_id == "n0"

    def test_all_entries_include_offline(self):
        reg = Registry()
        reg.register(_caps("n0"))
        reg.register(_caps("n1"))
        reg.deregister("n1")
        entries = reg.all_entries(include_offline=True)
        assert len(entries) == 2

    def test_online_count(self):
        reg = Registry()
        reg.register(_caps("n0"))
        reg.register(_caps("n1"))
        reg.register(_caps("n2"))
        reg.deregister("n2")
        assert reg.online_count() == 2

    def test_len_equals_online_count(self):
        reg = Registry()
        reg.register(_caps("n0"))
        reg.register(_caps("n1"))
        reg.deregister("n1")
        assert len(reg) == reg.online_count() == 1

    def test_iter_yields_online_entries(self):
        reg = Registry()
        reg.register(_caps("n0"))
        reg.register(_caps("n1"))
        reg.deregister("n1")
        ids = {e.caps.node_id for e in reg}
        assert ids == {"n0"}

    def test_repr(self):
        reg = Registry()
        reg.register(_caps("n0"))
        r = repr(reg)
        assert "Registry(" in r
        assert "online=1" in r


# ---------------------------------------------------------------------------
# Registry.prune_stale
# ---------------------------------------------------------------------------

class TestPruneStale:
    def test_prune_stale_marks_offline(self):
        reg = Registry(stale_ttl=60.0)
        reg.register(_caps("n0"))
        reg.get("n0").last_seen = time.time() - 120.0
        pruned = reg.prune_stale()
        assert "n0" in pruned
        assert reg.get("n0").online is False

    def test_prune_stale_skips_fresh_nodes(self):
        reg = Registry(stale_ttl=60.0)
        reg.register(_caps("n0"))
        pruned = reg.prune_stale()
        assert pruned == []
        assert reg.get("n0").online is True

    def test_prune_stale_skips_already_offline(self):
        reg = Registry(stale_ttl=60.0)
        reg.register(_caps("n0"))
        reg.deregister("n0")
        reg.get("n0").last_seen = time.time() - 120.0
        pruned = reg.prune_stale()
        assert "n0" not in pruned

    def test_prune_stale_returns_list_of_ids(self):
        reg = Registry(stale_ttl=60.0)
        reg.register(_caps("n0"))
        reg.register(_caps("n1"))
        reg.get("n0").last_seen = time.time() - 120.0
        reg.get("n1").last_seen = time.time() - 120.0
        pruned = reg.prune_stale()
        assert set(pruned) == {"n0", "n1"}


# ---------------------------------------------------------------------------
# Registry.match
# ---------------------------------------------------------------------------

class TestMatch:
    def test_match_returns_entry(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv"]))
        entry = reg.match("sv", 4)
        assert entry is not None
        assert entry.caps.node_id == "n0"

    def test_match_returns_none_when_empty(self):
        reg = Registry()
        assert reg.match("sv", 4) is None

    def test_match_rejects_wrong_backend(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv"]))
        assert reg.match("dm", 4) is None

    def test_match_rejects_over_capacity(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv"], sv_max=8))
        assert reg.match("sv", 16) is None

    def test_match_rejects_offline_node(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv"]))
        reg.deregister("n0")
        assert reg.match("sv", 4) is None

    def test_match_rejects_stale_node(self):
        reg = Registry(stale_ttl=60.0)
        reg.register(_caps("n0", backends=["sv"]))
        reg.get("n0").last_seen = time.time() - 120.0
        assert reg.match("sv", 4) is None

    def test_match_rejects_insufficient_stake(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv"], stake=5))
        assert reg.match("sv", 4, min_stake=10) is None

    def test_match_accepts_sufficient_stake(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv"], stake=10))
        entry = reg.match("sv", 4, min_stake=10)
        assert entry is not None

    def test_match_prefers_lower_load(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv"]))
        reg.register(_caps("n1", backends=["sv"]))
        reg.get("n0").jobs_in_flight = 5
        reg.get("n1").jobs_in_flight = 1
        entry = reg.match("sv", 4)
        assert entry.caps.node_id == "n1"

    def test_match_breaks_tie_by_stake(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv"], stake=10))
        reg.register(_caps("n1", backends=["sv"], stake=50))
        # both have jobs_in_flight=0; higher stake wins
        entry = reg.match("sv", 4)
        assert entry.caps.node_id == "n1"

    def test_match_at_capacity_boundary(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv"], sv_max=8))
        assert reg.match("sv", 8) is not None
        assert reg.match("sv", 9) is None


# ---------------------------------------------------------------------------
# Registry.match_pair
# ---------------------------------------------------------------------------

class TestMatchPair:
    def test_match_pair_returns_two_entries(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv"]))
        reg.register(_caps("n1", backends=["sv"]))
        result = reg.match_pair("sv", 4, "sv", 4)
        assert result is not None
        left, right = result
        assert isinstance(left, RegistryEntry)
        assert isinstance(right, RegistryEntry)

    def test_match_pair_returns_none_when_no_left(self):
        reg = Registry()
        assert reg.match_pair("sv", 4, "sv", 4) is None

    def test_match_pair_single_node_fallback(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv"]))
        result = reg.match_pair("sv", 4, "sv", 4)
        assert result is not None
        left, right = result
        assert left.caps.node_id == right.caps.node_id == "n0"

    def test_match_pair_returns_none_when_no_right(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv"]))
        # right requires dm which n0 doesn't support
        result = reg.match_pair("sv", 4, "dm", 4)
        assert result is None


# ---------------------------------------------------------------------------
# Registry.match_all
# ---------------------------------------------------------------------------

class TestMatchAll:
    def test_match_all_returns_up_to_count(self):
        reg = Registry()
        for i in range(5):
            reg.register(_caps(f"n{i}", backends=["sv"]))
        result = reg.match_all("sv", 4, count=3)
        assert len(result) == 3

    def test_match_all_returns_fewer_when_not_enough(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv"]))
        result = reg.match_all("sv", 4, count=5)
        assert len(result) == 1

    def test_match_all_empty_when_none_eligible(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv"]))
        result = reg.match_all("dm", 4, count=3)
        assert result == []

    def test_match_all_sorted_by_load(self):
        reg = Registry()
        for i in range(3):
            reg.register(_caps(f"n{i}", backends=["sv"]))
        reg.get("n0").jobs_in_flight = 10
        reg.get("n1").jobs_in_flight = 2
        reg.get("n2").jobs_in_flight = 5
        result = reg.match_all("sv", 4, count=3)
        loads = [e.jobs_in_flight for e in result]
        assert loads == sorted(loads)


# ---------------------------------------------------------------------------
# Registry.assign / complete (load tracking)
# ---------------------------------------------------------------------------

class TestLoadTracking:
    def test_assign_increments_jobs_in_flight(self):
        reg = Registry()
        reg.register(_caps("n0"))
        reg.assign("n0")
        assert reg.get("n0").jobs_in_flight == 1

    def test_assign_multiple(self):
        reg = Registry()
        reg.register(_caps("n0"))
        reg.assign("n0")
        reg.assign("n0")
        assert reg.get("n0").jobs_in_flight == 2

    def test_complete_decrements_jobs_in_flight(self):
        reg = Registry()
        reg.register(_caps("n0"))
        reg.assign("n0")
        reg.assign("n0")
        reg.complete("n0")
        assert reg.get("n0").jobs_in_flight == 1

    def test_complete_clamps_at_zero(self):
        reg = Registry()
        reg.register(_caps("n0"))
        reg.complete("n0")
        assert reg.get("n0").jobs_in_flight == 0

    def test_complete_increments_jobs_completed(self):
        reg = Registry()
        reg.register(_caps("n0"))
        assert reg.get("n0").caps.jobs_completed == 0
        reg.assign("n0")
        reg.complete("n0")
        assert reg.get("n0").caps.jobs_completed == 1

    def test_assign_unknown_node_is_noop(self):
        reg = Registry()
        reg.assign("nonexistent")  # must not raise

    def test_complete_unknown_node_is_noop(self):
        reg = Registry()
        reg.complete("nonexistent")  # must not raise


# ---------------------------------------------------------------------------
# Registry.route
# ---------------------------------------------------------------------------

class TestRoute:
    def test_route_delegates_to_match(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv"]))
        job = _simple_job(backend="sv", n_qubits=4)
        entry = reg.route(job)
        assert entry is not None
        assert entry.caps.node_id == "n0"

    def test_route_returns_none_when_no_match(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv"]))
        job = _simple_job(backend="dm", n_qubits=4)
        assert reg.route(job) is None


# ---------------------------------------------------------------------------
# Registry.summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_has_required_keys(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv", "dm"]))
        s = reg.summary()
        for key in ["online", "total_registered", "backends", "max_sv_qubits", "max_dm_qubits", "total_stake"]:
            assert key in s

    def test_summary_online_count(self):
        reg = Registry()
        reg.register(_caps("n0"))
        reg.register(_caps("n1"))
        reg.deregister("n1")
        assert reg.summary()["online"] == 1

    def test_summary_backends_union(self):
        reg = Registry()
        reg.register(_caps("n0", backends=["sv"]))
        reg.register(_caps("n1", backends=["dm"]))
        assert set(reg.summary()["backends"]) == {"sv", "dm"}

    def test_summary_max_sv_qubits(self):
        reg = Registry()
        reg.register(_caps("n0", sv_max=20))
        reg.register(_caps("n1", sv_max=30))
        assert reg.summary()["max_sv_qubits"] == 30

    def test_summary_total_stake(self):
        reg = Registry()
        reg.register(_caps("n0", stake=100))
        reg.register(_caps("n1", stake=200))
        assert reg.summary()["total_stake"] == 300

    def test_summary_empty_registry(self):
        reg = Registry()
        s = reg.summary()
        assert s["online"] == 0
        assert s["max_sv_qubits"] == 0
        assert s["total_stake"] == 0
