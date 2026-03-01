"""Verification protocol tests.

  - known_zero_state: expected = n_qubits, circuit has no ops
  - known_all_x: expected = -n_qubits
  - known_all_h: expected = 0
  - known_ry_pi: expected = -n_qubits (parametrized path)
  - Known circuits produce correct results when executed
  - default_library returns 4 circuits
  - SpotCheckScheduler.sample returns KnownCircuit
  - SpotCheckScheduler.should_inject rate=1.0 always injects
  - SpotCheckScheduler.should_inject rate=0.0 never injects
  - SpotCheckScheduler.add extends pool
  - SpotCheckScheduler len
  - VerificationResult.agreed true when delta <= tolerance
  - VerificationResult.agreed false when delta > tolerance
  - VerificationResult.job_id matches result_a.job_id
  - VerificationResult repr
  - SpotCheckResult repr
  - SlashEvent repr
  - Verifier.check_results agrees on matching results
  - Verifier.check_results disagrees on different results
  - Verifier.run_redundant returns VerificationResult
  - Verifier.run_redundant agrees when nodes match
  - Verifier.run_redundant flags both nodes on disagreement
  - Verifier.run_redundant slashes both on disagreement
  - Verifier.spot_check passes for honest node
  - Verifier.spot_check fails for faulty node
  - Verifier.spot_check flags faulty node
  - Verifier.spot_check does not flag honest node
  - Verifier.slash reduces stake
  - Verifier.slash floors at zero
  - Verifier.slash logs event
  - Verifier.slash returns None for unknown node
  - Verifier.flag increments count
  - Verifier.flag slashes on each flag
  - Verifier.flag deregisters after threshold
  - Verifier.flag_count returns correct value
  - Verifier.slash_log is snapshot
  - Verifier.summary has required keys
  - Verifier repr
"""

import math
import pytest

import sys; sys.path.insert(0, "src")
from zilver.node import Node, NodeCapabilities, SimJob, JobResult, _compute_proof
from zilver.registry import Registry
from zilver.verification import (
    VerificationResult,
    SpotCheckResult,
    SlashEvent,
    KnownCircuit,
    known_zero_state,
    known_all_x,
    known_all_h,
    known_ry_pi,
    default_library,
    SpotCheckScheduler,
    Verifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(node_id: str = "n0", sv_max: int = 20) -> Node:
    caps = NodeCapabilities(
        node_id=node_id,
        chip="Apple M4",
        ram_gb=16,
        sv_qubits_max=sv_max,
        dm_qubits_max=10,
        tn_qubits_max=50,
        backends=["sv", "dm", "tn"],
        stake=100,
    )
    return Node(caps)


def _registry_with_node(node_id: str = "n0", stake: int = 100) -> Registry:
    reg = Registry()
    caps = NodeCapabilities(
        node_id=node_id,
        chip="Apple M4",
        ram_gb=16,
        sv_qubits_max=20,
        dm_qubits_max=10,
        tn_qubits_max=50,
        backends=["sv"],
        stake=stake,
    )
    reg.register(caps)
    return reg


def _job_result(
    expectation: float,
    node_id: str = "n0",
    job_id: str = "job-0",
) -> JobResult:
    proof = _compute_proof(job_id, [], expectation)
    return JobResult(
        expectation=expectation,
        job_id=job_id,
        node_id=node_id,
        elapsed_ms=1.0,
        proof=proof,
    )


class _FaultyNode(Node):
    """Node that returns a deliberately wrong expectation value."""
    def __init__(self, base_node: Node, offset: float = 10.0):
        super().__init__(base_node.caps)
        self._offset = offset

    def execute(self, job: SimJob) -> JobResult:
        result = super().execute(job)
        bad_exp   = result.expectation + self._offset
        bad_proof = _compute_proof(job.job_id, job.params, bad_exp)
        return JobResult(
            expectation=bad_exp,
            job_id=result.job_id,
            node_id=result.node_id,
            elapsed_ms=result.elapsed_ms,
            proof=bad_proof,
        )


# ---------------------------------------------------------------------------
# Known-output circuits
# ---------------------------------------------------------------------------

class TestKnownCircuits:
    def test_zero_state_expected(self):
        kc = known_zero_state(4)
        assert kc.expected == 4.0

    def test_zero_state_name(self):
        kc = known_zero_state(4)
        assert "zero" in kc.name

    def test_all_x_expected(self):
        kc = known_all_x(4)
        assert kc.expected == -4.0

    def test_all_h_expected(self):
        kc = known_all_h(4)
        assert kc.expected == 0.0

    def test_ry_pi_expected(self):
        kc = known_ry_pi(4)
        assert kc.expected == -4.0

    def test_zero_state_executes_correctly(self):
        kc = known_zero_state(4)
        result = _node().execute(kc.job)
        assert abs(result.expectation - kc.expected) < 1e-4

    def test_all_x_executes_correctly(self):
        kc = known_all_x(4)
        result = _node().execute(kc.job)
        assert abs(result.expectation - kc.expected) < 1e-4

    def test_all_h_executes_correctly(self):
        kc = known_all_h(4)
        result = _node().execute(kc.job)
        assert abs(result.expectation - kc.expected) < 1e-4

    def test_ry_pi_executes_correctly(self):
        kc = known_ry_pi(4)
        result = _node().execute(kc.job)
        assert abs(result.expectation - kc.expected) < 1e-4

    def test_default_library_length(self):
        lib = default_library(4)
        assert len(lib) == 4

    def test_default_library_all_known_circuits(self):
        lib = default_library(4)
        for kc in lib:
            assert isinstance(kc, KnownCircuit)


# ---------------------------------------------------------------------------
# SpotCheckScheduler
# ---------------------------------------------------------------------------

class TestSpotCheckScheduler:
    def test_sample_returns_known_circuit(self):
        sched = SpotCheckScheduler(n_qubits=4)
        kc = sched.sample()
        assert isinstance(kc, KnownCircuit)

    def test_always_injects_at_rate_one(self):
        sched = SpotCheckScheduler(n_qubits=4, inject_rate=1.0)
        for _ in range(20):
            assert sched.should_inject() is True

    def test_never_injects_at_rate_zero(self):
        sched = SpotCheckScheduler(n_qubits=4, inject_rate=0.0)
        for _ in range(20):
            assert sched.should_inject() is False

    def test_add_extends_pool(self):
        sched = SpotCheckScheduler(n_qubits=4)
        before = len(sched)
        sched.add(known_zero_state(2))
        assert len(sched) == before + 1

    def test_len(self):
        sched = SpotCheckScheduler(n_qubits=4)
        assert len(sched) == 4  # default_library has 4


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------

class TestVerificationResult:
    def test_agreed_when_within_tolerance(self):
        ra = _job_result(1.0)
        rb = _job_result(1.00005)
        vr = VerificationResult(agreed=True, delta=0.00005, tolerance=1e-4,
                                result_a=ra, result_b=rb)
        assert vr.agreed is True

    def test_disagreed_when_over_tolerance(self):
        ra = _job_result(1.0)
        rb = _job_result(2.0)
        vr = VerificationResult(agreed=False, delta=1.0, tolerance=1e-4,
                                result_a=ra, result_b=rb)
        assert vr.agreed is False

    def test_job_id_property(self):
        ra = _job_result(1.0, job_id="abc")
        rb = _job_result(1.0, job_id="abc")
        vr = VerificationResult(agreed=True, delta=0.0, tolerance=1e-4,
                                result_a=ra, result_b=rb)
        assert vr.job_id == "abc"

    def test_repr_pass(self):
        ra = _job_result(1.0)
        rb = _job_result(1.0)
        vr = VerificationResult(agreed=True, delta=0.0, tolerance=1e-4,
                                result_a=ra, result_b=rb)
        assert "PASS" in repr(vr)

    def test_repr_fail(self):
        ra = _job_result(1.0)
        rb = _job_result(2.0)
        vr = VerificationResult(agreed=False, delta=1.0, tolerance=1e-4,
                                result_a=ra, result_b=rb)
        assert "FAIL" in repr(vr)


# ---------------------------------------------------------------------------
# Verifier.check_results
# ---------------------------------------------------------------------------

class TestVerifierCheckResults:
    def _verifier(self) -> Verifier:
        return Verifier(Registry(), tolerance=1e-4)

    def test_agrees_on_identical(self):
        v = self._verifier()
        ra = _job_result(1.5)
        rb = _job_result(1.5)
        vr = v.check_results(ra, rb)
        assert vr.agreed is True
        assert vr.delta == 0.0

    def test_agrees_within_tolerance(self):
        v = self._verifier()
        ra = _job_result(1.0)
        rb = _job_result(1.00005)
        vr = v.check_results(ra, rb)
        assert vr.agreed is True

    def test_disagrees_over_tolerance(self):
        v = self._verifier()
        ra = _job_result(1.0)
        rb = _job_result(1.001)
        vr = v.check_results(ra, rb)
        assert vr.agreed is False

    def test_delta_computed_correctly(self):
        v = self._verifier()
        ra = _job_result(1.0)
        rb = _job_result(2.5)
        vr = v.check_results(ra, rb)
        assert abs(vr.delta - 1.5) < 1e-9

    def test_no_side_effects_on_agree(self):
        reg = _registry_with_node("n0", stake=100)
        v = Verifier(reg)
        ra = _job_result(1.0, node_id="n0")
        rb = _job_result(1.0, node_id="n0")
        v.check_results(ra, rb)
        assert reg.get("n0").caps.stake == 100  # unchanged


# ---------------------------------------------------------------------------
# Verifier.run_redundant
# ---------------------------------------------------------------------------

class TestVerifierRunRedundant:
    def _simple_job(self) -> SimJob:
        return SimJob(
            circuit_ops=[{"type": "h", "qubits": [0], "param_idx": None}],
            n_qubits=4,
            n_params=0,
            params=[],
            backend="sv",
        )

    def test_returns_verification_result(self):
        reg = _registry_with_node("n0")
        v = Verifier(reg)
        na = _node("n0")
        nb = _node("n1")
        vr = v.run_redundant(self._simple_job(), na, nb)
        assert isinstance(vr, VerificationResult)

    def test_agrees_for_honest_nodes(self):
        reg = _registry_with_node("n0")
        v = Verifier(reg)
        na = _node("n0")
        nb = _node("n1")
        vr = v.run_redundant(self._simple_job(), na, nb)
        assert vr.agreed is True

    def test_no_flags_when_agree(self):
        reg = _registry_with_node("n0")
        v = Verifier(reg)
        na = _node("n0")
        nb = _node("n1")
        v.run_redundant(self._simple_job(), na, nb)
        assert v.flag_count("n0") == 0
        assert v.flag_count("n1") == 0

    def test_flags_both_on_disagreement(self):
        reg = _registry_with_node("n0", stake=200)
        reg.register(NodeCapabilities(
            node_id="n1", chip="M4", ram_gb=16,
            sv_qubits_max=20, dm_qubits_max=10, tn_qubits_max=50,
            backends=["sv"], stake=200,
        ))
        v = Verifier(reg, tolerance=1e-4, slash_amount=10)
        na = _node("n0")
        nb = _FaultyNode(_node("n1"), offset=10.0)
        vr = v.run_redundant(self._simple_job(), na, nb)
        assert vr.agreed is False
        assert v.flag_count("n0") == 1
        assert v.flag_count("n1") == 1

    def test_slashes_both_on_disagreement(self):
        reg = _registry_with_node("n0", stake=200)
        reg.register(NodeCapabilities(
            node_id="n1", chip="M4", ram_gb=16,
            sv_qubits_max=20, dm_qubits_max=10, tn_qubits_max=50,
            backends=["sv"], stake=200,
        ))
        v = Verifier(reg, tolerance=1e-4, slash_amount=30)
        na = _node("n0")
        nb = _FaultyNode(_node("n1"), offset=10.0)
        v.run_redundant(self._simple_job(), na, nb)
        assert reg.get("n0").caps.stake == 170
        assert reg.get("n1").caps.stake == 170

    def test_result_a_proof_verifies(self):
        reg = _registry_with_node("n0")
        v = Verifier(reg)
        na = _node("n0")
        nb = _node("n1")
        job = self._simple_job()
        vr = v.run_redundant(job, na, nb)
        assert vr.result_a.verify(job) is True

    def test_result_b_proof_verifies(self):
        reg = _registry_with_node("n0")
        v = Verifier(reg)
        na = _node("n0")
        nb = _node("n1")
        job = self._simple_job()
        vr = v.run_redundant(job, na, nb)
        assert vr.result_b.verify(job) is True


# ---------------------------------------------------------------------------
# Verifier.spot_check
# ---------------------------------------------------------------------------

class TestVerifierSpotCheck:
    def test_returns_spot_check_result(self):
        reg = _registry_with_node("n0")
        v = Verifier(reg)
        node = _node("n0")
        sr = v.spot_check(node, known_zero_state(4))
        assert isinstance(sr, SpotCheckResult)

    def test_passes_for_honest_node(self):
        reg = _registry_with_node("n0")
        v = Verifier(reg)
        node = _node("n0")
        sr = v.spot_check(node, known_zero_state(4))
        assert sr.passed is True

    def test_does_not_flag_honest_node(self):
        reg = _registry_with_node("n0")
        v = Verifier(reg)
        node = _node("n0")
        v.spot_check(node, known_zero_state(4))
        assert v.flag_count("n0") == 0

    def test_passes_all_known_circuits(self):
        reg = _registry_with_node("n0")
        v = Verifier(reg)
        node = _node("n0")
        for kc in default_library(4):
            sr = v.spot_check(node, kc)
            assert sr.passed, f"Failed on {kc.name}: expected={kc.expected} got={sr.actual}"

    def test_fails_for_faulty_node(self):
        reg = _registry_with_node("n0", stake=200)
        v = Verifier(reg)
        faulty = _FaultyNode(_node("n0"), offset=5.0)
        sr = v.spot_check(faulty, known_zero_state(4))
        assert sr.passed is False

    def test_flags_faulty_node(self):
        reg = _registry_with_node("n0", stake=200)
        v = Verifier(reg)
        faulty = _FaultyNode(_node("n0"), offset=5.0)
        v.spot_check(faulty, known_zero_state(4))
        assert v.flag_count("n0") == 1

    def test_slashes_faulty_node(self):
        reg = _registry_with_node("n0", stake=100)
        v = Verifier(reg, slash_amount=20)
        faulty = _FaultyNode(_node("n0"), offset=5.0)
        v.spot_check(faulty, known_zero_state(4))
        assert reg.get("n0").caps.stake == 80

    def test_spot_check_result_fields(self):
        reg = _registry_with_node("n0")
        v = Verifier(reg)
        node = _node("n0")
        kc = known_zero_state(4)
        sr = v.spot_check(node, kc)
        assert sr.expected == 4.0
        assert math.isfinite(sr.actual)
        assert sr.node_id == "n0"
        assert sr.name == kc.name

    def test_spot_check_repr(self):
        reg = _registry_with_node("n0")
        v = Verifier(reg)
        node = _node("n0")
        sr = v.spot_check(node, known_zero_state(4))
        assert "SpotCheckResult" in repr(sr)
        assert "PASS" in repr(sr)


# ---------------------------------------------------------------------------
# Verifier.slash
# ---------------------------------------------------------------------------

class TestVerifierSlash:
    def test_reduces_stake(self):
        reg = _registry_with_node("n0", stake=100)
        v = Verifier(reg, slash_amount=30)
        v.slash("n0")
        assert reg.get("n0").caps.stake == 70

    def test_custom_amount(self):
        reg = _registry_with_node("n0", stake=100)
        v = Verifier(reg)
        v.slash("n0", amount=10)
        assert reg.get("n0").caps.stake == 90

    def test_floors_at_zero(self):
        reg = _registry_with_node("n0", stake=10)
        v = Verifier(reg)
        v.slash("n0", amount=100)
        assert reg.get("n0").caps.stake == 0

    def test_logs_event(self):
        reg = _registry_with_node("n0", stake=100)
        v = Verifier(reg, slash_amount=20)
        v.slash("n0", reason="test")
        assert len(v.slash_log) == 1
        event = v.slash_log[0]
        assert event.node_id == "n0"
        assert event.amount == 20
        assert event.stake_before == 100
        assert event.stake_after == 80

    def test_returns_none_for_unknown_node(self):
        reg = Registry()
        v = Verifier(reg)
        result = v.slash("nonexistent")
        assert result is None

    def test_slash_log_is_snapshot(self):
        reg = _registry_with_node("n0", stake=100)
        v = Verifier(reg, slash_amount=10)
        v.slash("n0")
        log1 = v.slash_log
        v.slash("n0")
        log2 = v.slash_log
        assert len(log1) == 1
        assert len(log2) == 2

    def test_slash_event_repr(self):
        reg = _registry_with_node("n0", stake=100)
        v = Verifier(reg, slash_amount=10)
        event = v.slash("n0", reason="test reason")
        r = repr(event)
        assert "SlashEvent" in r
        assert "n0" in r


# ---------------------------------------------------------------------------
# Verifier.flag
# ---------------------------------------------------------------------------

class TestVerifierFlag:
    def test_increments_count(self):
        reg = _registry_with_node("n0", stake=500)
        v = Verifier(reg, flag_threshold=5)
        v.flag("n0")
        v.flag("n0")
        assert v.flag_count("n0") == 2

    def test_slashes_on_each_flag(self):
        reg = _registry_with_node("n0", stake=500)
        v = Verifier(reg, slash_amount=50, flag_threshold=10)
        v.flag("n0")
        v.flag("n0")
        assert reg.get("n0").caps.stake == 400

    def test_deregisters_at_threshold(self):
        reg = _registry_with_node("n0", stake=500)
        v = Verifier(reg, flag_threshold=3, slash_amount=10)
        v.flag("n0")
        v.flag("n0")
        assert reg.get("n0").online is True  # not yet
        v.flag("n0")
        assert reg.get("n0").online is False  # deregistered

    def test_flag_count_zero_for_unknown(self):
        v = Verifier(Registry())
        assert v.flag_count("nonexistent") == 0

    def test_returns_updated_count(self):
        reg = _registry_with_node("n0", stake=500)
        v = Verifier(reg, flag_threshold=10)
        assert v.flag("n0") == 1
        assert v.flag("n0") == 2


# ---------------------------------------------------------------------------
# Verifier.summary / repr
# ---------------------------------------------------------------------------

class TestVerifierSummary:
    def test_summary_has_required_keys(self):
        reg = _registry_with_node("n0")
        v = Verifier(reg)
        s = v.summary()
        for key in ["flagged_nodes", "slash_events", "total_slashed", "deregistered"]:
            assert key in s

    def test_summary_total_slashed(self):
        reg = _registry_with_node("n0", stake=500)
        v = Verifier(reg, slash_amount=30, flag_threshold=10)
        v.flag("n0")
        v.flag("n0")
        assert v.summary()["total_slashed"] == 60

    def test_summary_deregistered_list(self):
        reg = _registry_with_node("n0", stake=500)
        v = Verifier(reg, flag_threshold=2, slash_amount=10)
        v.flag("n0")
        v.flag("n0")
        assert "n0" in v.summary()["deregistered"]

    def test_repr(self):
        reg = _registry_with_node("n0")
        v = Verifier(reg)
        r = repr(v)
        assert "Verifier(" in r
