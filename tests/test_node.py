"""Node daemon tests.

  - NodeCapabilities.detect() returns sane values for the current machine
  - NodeCapabilities.supports() enforces backend and qubit limits
  - Node.execute() runs a job and returns a JobResult
  - JobResult.verify() validates the proof hash
  - Node rejects jobs exceeding qubit capacity
  - Node rejects unsupported backends
  - sv / dm / tn backends all execute
  - jobs_completed counter increments
  - job_from_circuit serializes a Circuit into a SimJob
  - SimJob round-trips through to_dict / from_dict
"""

import math
import numpy as np
import mlx.core as mx
import pytest

import sys; sys.path.insert(0, "src")
from zilver.node import (
    Node,
    NodeCapabilities,
    SimJob,
    JobResult,
    job_from_circuit,
    _compute_proof,
    _sv_qubit_ceiling,
    _dm_qubit_ceiling,
)
from zilver.circuit import Circuit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_job(n=4, backend="sv") -> SimJob:
    ops = [
        {"type": "h",    "qubits": [0], "param_idx": None},
        {"type": "cnot", "qubits": [0, 1], "param_idx": None},
    ]
    return SimJob(
        circuit_ops=ops,
        n_qubits=n,
        n_params=0,
        params=[],
        backend=backend,
    )


def _ry_job(n=4, backend="sv") -> SimJob:
    ops = [{"type": "ry", "qubits": [q], "param_idx": q} for q in range(n)]
    rng = np.random.default_rng(0)
    params = rng.uniform(-math.pi, math.pi, n).tolist()
    return SimJob(
        circuit_ops=ops,
        n_qubits=n,
        n_params=n,
        params=params,
        backend=backend,
    )


# ---------------------------------------------------------------------------
# NodeCapabilities
# ---------------------------------------------------------------------------

class TestNodeCapabilities:
    def test_detect_returns_capabilities(self):
        caps = NodeCapabilities.detect()
        assert isinstance(caps, NodeCapabilities)

    def test_node_id_assigned(self):
        caps = NodeCapabilities.detect()
        assert len(caps.node_id) > 0

    def test_explicit_node_id(self):
        caps = NodeCapabilities.detect(node_id="test-node-0")
        assert caps.node_id == "test-node-0"

    def test_ram_positive(self):
        caps = NodeCapabilities.detect()
        assert caps.ram_gb > 0

    def test_sv_qubit_ceiling_positive(self):
        caps = NodeCapabilities.detect()
        assert caps.sv_qubits_max > 0

    def test_dm_qubit_ceiling_less_than_sv(self):
        caps = NodeCapabilities.detect()
        assert caps.dm_qubits_max < caps.sv_qubits_max

    def test_tn_qubits_max_is_50(self):
        caps = NodeCapabilities.detect()
        assert caps.tn_qubits_max == 50

    def test_backends_default_sv(self):
        caps = NodeCapabilities.detect()
        assert "sv" in caps.backends

    def test_backends_explicit(self):
        caps = NodeCapabilities.detect(backends=["sv", "dm"])
        assert "sv" in caps.backends
        assert "dm" in caps.backends

    def test_supports_sv_small(self):
        caps = NodeCapabilities.detect(backends=["sv"])
        assert caps.supports("sv", 4) is True

    def test_supports_rejects_unsupported_backend(self):
        caps = NodeCapabilities.detect(backends=["sv"])
        assert caps.supports("dm", 4) is False

    def test_supports_rejects_over_capacity(self):
        caps = NodeCapabilities.detect(backends=["sv"])
        assert caps.supports("sv", caps.sv_qubits_max + 1) is False

    def test_to_dict_has_required_keys(self):
        caps = NodeCapabilities.detect()
        d = caps.to_dict()
        for key in ["node_id", "chip", "ram_gb", "sv_qubits_max", "backends"]:
            assert key in d


# ---------------------------------------------------------------------------
# Qubit ceiling helpers
# ---------------------------------------------------------------------------

class TestQubitCeilings:
    def test_sv_ceiling_8gb(self):
        # 8 GB * 0.8 = 6.4 GB, 2^n * 8 bytes: 2^29 * 8 = 4 GB < 6.4 GB, 2^30 * 8 = 8 GB > 6.4 GB
        assert _sv_qubit_ceiling(8) == 29

    def test_dm_ceiling_8gb(self):
        # 8 GB * 0.8 = 6.4 GB, 4^n * 8 bytes: 4^13 * 8 = 536 MB, 4^14 * 8 = 2 GB, 4^15 * 8 = 8 GB
        assert _dm_qubit_ceiling(8) == 14

    def test_sv_ceiling_48gb(self):
        # 48 GB * 0.8 = 38.4 GB, 2^32 * 8 = 32 GB < 38.4 GB, so n=32
        assert _sv_qubit_ceiling(48) == 32

    def test_dm_ceiling_48gb(self):
        # 48 GB * 0.8 = 38.4 GB, 4^16 * 8 = 32 GB fits, 4^17 * 8 = 128 GB > 38.4 GB
        assert _dm_qubit_ceiling(48) == 16

    def test_sv_ceiling_never_exceeds_34(self):
        assert _sv_qubit_ceiling(10000) <= 34

    def test_dm_ceiling_never_exceeds_17(self):
        assert _dm_qubit_ceiling(10000) <= 17


# ---------------------------------------------------------------------------
# SimJob
# ---------------------------------------------------------------------------

class TestSimJob:
    def test_job_id_auto_assigned(self):
        job = _simple_job()
        assert len(job.job_id) > 0

    def test_to_dict_round_trip(self):
        job = _simple_job()
        d   = job.to_dict()
        job2 = SimJob.from_dict(d)
        assert job2.job_id    == job.job_id
        assert job2.n_qubits  == job.n_qubits
        assert job2.backend   == job.backend

    def test_from_dict_preserves_ops(self):
        job = _ry_job(4)
        job2 = SimJob.from_dict(job.to_dict())
        assert len(job2.circuit_ops) == len(job.circuit_ops)


# ---------------------------------------------------------------------------
# Node.execute()
# ---------------------------------------------------------------------------

class TestNodeExecute:
    def _node(self, backends=None):
        return Node.start(backends=backends or ["sv", "dm", "tn"], node_id="test-0")

    def test_execute_returns_job_result(self):
        node = self._node()
        result = node.execute(_simple_job())
        assert isinstance(result, JobResult)

    def test_execute_finite_expectation(self):
        node = self._node()
        result = node.execute(_simple_job())
        assert math.isfinite(result.expectation)

    def test_execute_bounded_expectation(self):
        n = 4
        node = self._node()
        result = node.execute(_simple_job(n=n))
        assert abs(result.expectation) <= n + 1e-4

    def test_execute_increments_counter(self):
        node = self._node()
        assert node.caps.jobs_completed == 0
        node.execute(_simple_job())
        assert node.caps.jobs_completed == 1
        node.execute(_simple_job())
        assert node.caps.jobs_completed == 2

    def test_execute_job_id_in_result(self):
        node = self._node()
        job  = _simple_job()
        result = node.execute(job)
        assert result.job_id == job.job_id

    def test_execute_node_id_in_result(self):
        node = self._node()
        result = node.execute(_simple_job())
        assert result.node_id == "test-0"

    def test_execute_elapsed_positive(self):
        node = self._node()
        result = node.execute(_simple_job())
        assert result.elapsed_ms >= 0

    def test_execute_parametrized(self):
        node = self._node()
        result = node.execute(_ry_job(4))
        assert math.isfinite(result.expectation)

    def test_execute_sv_backend(self):
        node = self._node(["sv"])
        result = node.execute(_simple_job(backend="sv"))
        assert math.isfinite(result.expectation)

    def test_execute_dm_backend(self):
        node = self._node(["dm"])
        result = node.execute(_simple_job(backend="dm"))
        assert math.isfinite(result.expectation)

    def test_execute_tn_backend(self):
        node = self._node(["tn"])
        result = node.execute(_ry_job(6, backend="tn"))
        assert math.isfinite(result.expectation)

    def test_rejects_unsupported_backend(self):
        node = Node.start(backends=["sv"], node_id="sv-only")
        with pytest.raises(ValueError, match="cannot handle"):
            node.execute(_simple_job(backend="dm"))

    def test_rejects_over_capacity(self):
        caps = NodeCapabilities(
            node_id="tiny",
            chip="test",
            ram_gb=1,
            sv_qubits_max=2,
            dm_qubits_max=1,
            tn_qubits_max=4,
            backends=["sv"],
        )
        node = Node(caps)
        with pytest.raises(ValueError, match="cannot handle"):
            node.execute(_simple_job(n=8, backend="sv"))


# ---------------------------------------------------------------------------
# JobResult.verify()
# ---------------------------------------------------------------------------

class TestJobResultVerify:
    def test_proof_verifies(self):
        node = Node.start(backends=["sv"], node_id="v-node")
        job  = _simple_job()
        result = node.execute(job)
        assert result.verify(job) is True

    def test_tampered_expectation_fails_verify(self):
        node = Node.start(backends=["sv"], node_id="v-node")
        job  = _simple_job()
        result = node.execute(job)
        tampered = JobResult(
            expectation = result.expectation + 1.0,
            job_id      = result.job_id,
            node_id     = result.node_id,
            elapsed_ms  = result.elapsed_ms,
            proof       = result.proof,
        )
        assert tampered.verify(job) is False

    def test_tampered_params_fails_verify(self):
        node = Node.start(backends=["sv"], node_id="v-node")
        job  = _ry_job(4)
        result = node.execute(job)
        tampered_job = SimJob.from_dict({**job.to_dict(), "params": [0.0]*4})
        assert result.verify(tampered_job) is False

    def test_proof_is_sha256_hex(self):
        node = Node.start(backends=["sv"], node_id="v-node")
        result = node.execute(_simple_job())
        assert len(result.proof) == 64
        assert all(c in "0123456789abcdef" for c in result.proof)


# ---------------------------------------------------------------------------
# job_from_circuit
# ---------------------------------------------------------------------------

class TestJobFromCircuit:
    def test_serializes_circuit(self):
        c = Circuit(4)
        c.h(0); c.cnot(0, 1)
        params = mx.array([], dtype=mx.float32)
        job = job_from_circuit(c, params)
        assert isinstance(job, SimJob)
        assert job.n_qubits == 4

    def test_serialized_job_executes(self):
        c = Circuit(4)
        for q in range(4):
            c.ry(q, q)
        c.n_params = 4
        rng = np.random.default_rng(5)
        params = mx.array(rng.uniform(-math.pi, math.pi, 4).astype(np.float32))
        job  = job_from_circuit(c, params, backend="sv")
        node = Node.start(backends=["sv"], node_id="ser-node")
        result = node.execute(job)
        assert math.isfinite(result.expectation)

    def test_node_repr(self):
        node = Node.start(backends=["sv"], node_id="repr-node")
        r = repr(node)
        assert "Node(" in r
