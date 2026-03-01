"""
Verification protocol for distributed simulation results.

Two strategies:
  1. Redundant execution — same job on 2 nodes, compare within tolerance.
     Statevector simulation is deterministic: |E_A - E_B| should be < 1e-6.
     A delta > tolerance signals a faulty node; both are flagged.

  2. Spot-check — inject circuits with analytically known outputs into the
     job stream at a configurable rate. A node that fails a spot-check is
     flagged and its stake is slashed. Accumulated flags trigger deregistration.

Stake slashing:
  Reduces registry entry's stake by slash_amount per infraction.
  Stake floors at 0. Nodes that accumulate flag_threshold flags are
  automatically deregistered from the registry.

Known-output circuit library:
  zero_state, all_x, all_h — analytically verifiable for any qubit count.
  SpotCheckScheduler manages a pool and controls injection rate.
"""

from __future__ import annotations
import time
from collections import defaultdict
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from .node import Node, NodeCapabilities, SimJob, JobResult
from .registry import Registry


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """
    Outcome of comparing two node executions of the same job.

    agreed:    True if |expectation_a - expectation_b| <= tolerance
    delta:     absolute difference between the two expectations
    result_a:  JobResult from node A
    result_b:  JobResult from node B
    """
    agreed:    bool
    delta:     float
    tolerance: float
    result_a:  JobResult
    result_b:  JobResult

    @property
    def job_id(self) -> str:
        return self.result_a.job_id

    def __repr__(self) -> str:
        status = "PASS" if self.agreed else "FAIL"
        return (
            f"VerificationResult({status}, "
            f"delta={self.delta:.2e}, "
            f"tol={self.tolerance:.2e})"
        )


@dataclass
class SpotCheckResult:
    """
    Outcome of a spot-check against a known-output circuit.

    passed:   True if |actual - expected| <= tolerance
    expected: analytically known expectation value
    actual:   expectation value returned by the node
    delta:    absolute difference
    node_id:  node under test
    """
    passed:    bool
    expected:  float
    actual:    float
    delta:     float
    tolerance: float
    node_id:   str
    job_id:    str
    name:      str

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"SpotCheckResult({status}, "
            f"circuit={self.name!r}, "
            f"expected={self.expected:.4f}, "
            f"actual={self.actual:.4f}, "
            f"delta={self.delta:.2e})"
        )


@dataclass
class SlashEvent:
    """Record of a stake-slash action."""
    node_id:      str
    amount:       int
    stake_before: int
    stake_after:  int
    reason:       str
    timestamp:    float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"SlashEvent(node={self.node_id!r}, "
            f"amount={self.amount}, "
            f"stake={self.stake_before}→{self.stake_after}, "
            f"reason={self.reason!r})"
        )


# ---------------------------------------------------------------------------
# Known-output circuit library
# ---------------------------------------------------------------------------

@dataclass
class KnownCircuit:
    """
    A pre-serialized SimJob with an analytically known sum_Z expectation value.

    Stores a SimJob directly rather than a Circuit object to avoid the
    job_from_circuit gate-kind inference heuristic. This guarantees exact
    round-trip through Node.execute() regardless of gate type.
    """
    job:      SimJob
    expected: float
    name:     str


def known_zero_state(n_qubits: int) -> KnownCircuit:
    """No gates: |0...0>, Z eigenvalue +1 on every qubit. sum_Z = n_qubits."""
    job = SimJob(
        circuit_ops=[],
        n_qubits=n_qubits,
        n_params=0,
        params=[],
    )
    return KnownCircuit(job=job, expected=float(n_qubits), name=f"zero_state_{n_qubits}q")


def known_all_x(n_qubits: int) -> KnownCircuit:
    """X on every qubit: |1...1>, Z eigenvalue -1. sum_Z = -n_qubits."""
    ops = [{"type": "x", "qubits": [q], "param_idx": None} for q in range(n_qubits)]
    job = SimJob(circuit_ops=ops, n_qubits=n_qubits, n_params=0, params=[])
    return KnownCircuit(job=job, expected=float(-n_qubits), name=f"all_x_{n_qubits}q")


def known_all_h(n_qubits: int) -> KnownCircuit:
    """H on every qubit: equal superposition, <Z> = 0. sum_Z = 0."""
    ops = [{"type": "h", "qubits": [q], "param_idx": None} for q in range(n_qubits)]
    job = SimJob(circuit_ops=ops, n_qubits=n_qubits, n_params=0, params=[])
    return KnownCircuit(job=job, expected=0.0, name=f"all_h_{n_qubits}q")


def known_ry_pi(n_qubits: int) -> KnownCircuit:
    """RY(π) on every qubit: same as X, sum_Z = -n_qubits. Exercises parametrized path."""
    ops = [{"type": "ry", "qubits": [q], "param_idx": q} for q in range(n_qubits)]
    params = [float(np.pi)] * n_qubits
    job = SimJob(circuit_ops=ops, n_qubits=n_qubits, n_params=n_qubits, params=params)
    return KnownCircuit(job=job, expected=float(-n_qubits), name=f"ry_pi_{n_qubits}q")


def default_library(n_qubits: int = 4) -> list[KnownCircuit]:
    """Standard spot-check pool for an n-qubit node."""
    return [
        known_zero_state(n_qubits),
        known_all_x(n_qubits),
        known_all_h(n_qubits),
        known_ry_pi(n_qubits),
    ]


# ---------------------------------------------------------------------------
# Spot-check scheduler
# ---------------------------------------------------------------------------

class SpotCheckScheduler:
    """
    Manages a pool of known-output circuits for random spot-check injection.

    Usage:
        sched = SpotCheckScheduler(n_qubits=4, inject_rate=0.1)
        if sched.should_inject():
            known = sched.sample()
            result = verifier.spot_check(node, known)
    """

    def __init__(
        self,
        n_qubits:    int   = 4,
        inject_rate: float = 0.1,
        seed:        int   = 42,
    ):
        self._rng = np.random.default_rng(seed)
        self._pool: list[KnownCircuit] = default_library(n_qubits)
        self.inject_rate = inject_rate

    def sample(self) -> KnownCircuit:
        """Return a random circuit from the pool."""
        idx = int(self._rng.integers(len(self._pool)))
        return self._pool[idx]

    def should_inject(self) -> bool:
        """Return True with probability inject_rate."""
        return bool(self._rng.random() < self.inject_rate)

    def add(self, known: KnownCircuit) -> None:
        """Add a custom known circuit to the pool."""
        self._pool.append(known)

    def __len__(self) -> int:
        return len(self._pool)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class Verifier:
    """
    Verification protocol for distributed simulation results.

    Handles both redundant-execution verification and spot-checks.
    Tracks per-node flag counts and applies stake slashing.

    Args:
        registry:        Registry instance (for stake slashing and deregistration)
        tolerance:       max |E_A - E_B| for agreement (default 1e-4)
        flag_threshold:  flags before automatic deregistration (default 3)
        slash_amount:    stake units deducted per infraction (default 50)

    Usage:
        verifier = Verifier(registry)

        # Redundant execution
        vr = verifier.run_redundant(job, node_a, node_b)
        if not vr.agreed:
            print("Nodes disagree — both flagged")

        # Spot-check
        sr = verifier.spot_check(node, known_zero_state(4))
        if not sr.passed:
            print(f"Spot-check failed: {sr}")
    """

    def __init__(
        self,
        registry:        Registry,
        tolerance:       float = 1e-4,
        flag_threshold:  int   = 3,
        slash_amount:    int   = 50,
    ):
        self.registry       = registry
        self.tolerance      = tolerance
        self.flag_threshold = flag_threshold
        self.slash_amount   = slash_amount
        self._flags:     dict[str, int]     = defaultdict(int)
        self._slash_log: list[SlashEvent]   = []

    # --- Core comparison -----------------------------------------------------

    def check_results(
        self,
        result_a: JobResult,
        result_b: JobResult,
    ) -> VerificationResult:
        """
        Compare two JobResults for the same job.

        Does not trigger any flagging or slashing — call flag() explicitly
        when acting on the result.
        """
        delta = abs(result_a.expectation - result_b.expectation)
        return VerificationResult(
            agreed=delta <= self.tolerance,
            delta=delta,
            tolerance=self.tolerance,
            result_a=result_a,
            result_b=result_b,
        )

    # --- Redundant execution -------------------------------------------------

    def run_redundant(
        self,
        job:    SimJob,
        node_a: Node,
        node_b: Node,
    ) -> VerificationResult:
        """
        Execute the same job on two nodes and verify agreement.

        If the results disagree (delta > tolerance), both nodes are flagged
        and their stakes are slashed. The caller can inspect VerificationResult
        to decide further action.

        Returns:
            VerificationResult containing both JobResults and agreement status.
        """
        result_a = node_a.execute(job)
        result_b = node_b.execute(job)
        vr = self.check_results(result_a, result_b)

        if not vr.agreed:
            reason = f"redundant disagreement delta={vr.delta:.2e}"
            self.flag(node_a.caps.node_id, reason=reason)
            self.flag(node_b.caps.node_id, reason=reason)

        return vr

    # --- Spot-check ----------------------------------------------------------

    def spot_check(
        self,
        node:  Node,
        known: KnownCircuit,
    ) -> SpotCheckResult:
        """
        Execute a known-output circuit on a node and compare to expected value.

        On failure, the node is flagged and its stake is slashed.

        Args:
            node:  Node to test
            known: KnownCircuit with analytically known expected value

        Returns:
            SpotCheckResult with pass/fail status and diagnostics.
        """
        job    = known.job
        result = node.execute(job)
        delta  = abs(result.expectation - known.expected)
        passed = delta <= self.tolerance

        if not passed:
            self.flag(
                node.caps.node_id,
                reason=(
                    f"spot-check {known.name!r} failed: "
                    f"expected={known.expected:.4f} got={result.expectation:.4f} "
                    f"delta={delta:.2e}"
                ),
            )

        return SpotCheckResult(
            passed=passed,
            expected=known.expected,
            actual=result.expectation,
            delta=delta,
            tolerance=self.tolerance,
            node_id=node.caps.node_id,
            job_id=result.job_id,
            name=known.name,
        )

    # --- Stake management ----------------------------------------------------

    def slash(
        self,
        node_id: str,
        amount:  int | None = None,
        reason:  str = "",
    ) -> SlashEvent | None:
        """
        Reduce a node's stake by amount (default: self.slash_amount).

        Stake floors at 0. Returns None if the node is not in the registry.
        The event is appended to self.slash_log for auditing.
        """
        entry = self.registry.get(node_id)
        if entry is None:
            return None

        amount = amount if amount is not None else self.slash_amount
        before = entry.caps.stake
        entry.caps.stake = max(0, before - amount)

        event = SlashEvent(
            node_id=node_id,
            amount=amount,
            stake_before=before,
            stake_after=entry.caps.stake,
            reason=reason,
        )
        self._slash_log.append(event)
        return event

    def flag(self, node_id: str, reason: str = "") -> int:
        """
        Increment a node's flag count, slash its stake, and deregister
        if flag_threshold is reached.

        Returns the updated flag count.
        """
        self._flags[node_id] += 1
        count = self._flags[node_id]
        self.slash(node_id, reason=f"flag #{count}: {reason}")

        if count >= self.flag_threshold:
            self.registry.deregister(node_id)

        return count

    def flag_count(self, node_id: str) -> int:
        """Return current flag count for a node."""
        return self._flags.get(node_id, 0)

    # --- Audit ---------------------------------------------------------------

    @property
    def slash_log(self) -> list[SlashEvent]:
        return list(self._slash_log)

    def summary(self) -> dict:
        """Return a snapshot of flagged nodes and total slashed stake."""
        return {
            "flagged_nodes":    dict(self._flags),
            "slash_events":     len(self._slash_log),
            "total_slashed":    sum(e.amount for e in self._slash_log),
            "deregistered":     [
                nid for nid, count in self._flags.items()
                if count >= self.flag_threshold
            ],
        }

    def __repr__(self) -> str:
        return (
            f"Verifier("
            f"tolerance={self.tolerance:.1e}, "
            f"flags={dict(self._flags)}, "
            f"slashes={len(self._slash_log)})"
        )
