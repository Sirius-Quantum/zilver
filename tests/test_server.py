"""
Tests for the node HTTP server.

All tests use ``fastapi.testclient.TestClient``, which drives the ASGI app
synchronously without binding a real port.  No network access is required.

Coverage:
  - GET  /health — returns 200 with node_id
  - GET  /caps   — returns required capability fields
  - POST /heartbeat — returns 200 with status ok
  - POST /execute — valid job returns finite expectation and valid proof
  - POST /execute — invalid body returns 422
  - POST /execute — unsupported backend returns 422
  - POST /execute — qubit count over capacity returns 422
  - POST /execute — sv, dm, tn backends all succeed
  - POST /execute — jobs_completed increments after each call
  - JobResult proof verifies against original job
"""

import math

import pytest
from fastapi.testclient import TestClient

import sys; sys.path.insert(0, "src")
from zilver.node import Node, NodeCapabilities, SimJob
from zilver.server import make_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _node(backends=None, sv_max=20, node_id="test-node") -> Node:
    caps = NodeCapabilities(
        node_id=node_id,
        chip="Apple M4",
        ram_gb=16,
        sv_qubits_max=sv_max,
        dm_qubits_max=10,
        tn_qubits_max=50,
        backends=backends or ["sv", "dm", "tn"],
    )
    return Node(caps)


def _client(backends=None, sv_max=20, node_id="test-node") -> TestClient:
    node = _node(backends=backends, sv_max=sv_max, node_id=node_id)
    return TestClient(make_app(node))


def _h_job(n=4, backend="sv") -> dict:
    return SimJob(
        circuit_ops=[{"type": "h", "qubits": [0], "param_idx": None}],
        n_qubits=n,
        n_params=0,
        params=[],
        backend=backend,
    ).to_dict()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_status_200(self):
        c = _client()
        assert c.get("/health").status_code == 200

    def test_status_ok(self):
        c = _client()
        assert c.get("/health").json()["status"] == "ok"

    def test_node_id_present(self):
        c = _client(node_id="my-node")
        assert c.get("/health").json()["node_id"] == "my-node"


# ---------------------------------------------------------------------------
# GET /caps
# ---------------------------------------------------------------------------

class TestCaps:
    def test_status_200(self):
        c = _client()
        assert c.get("/caps").status_code == 200

    def test_required_fields(self):
        resp = _client().get("/caps").json()
        for key in ["node_id", "chip", "ram_gb", "sv_qubits_max", "backends"]:
            assert key in resp, f"Missing key: {key}"

    def test_backends_match(self):
        c = _client(backends=["sv", "dm"])
        resp = c.get("/caps").json()
        assert set(resp["backends"]) == {"sv", "dm"}


# ---------------------------------------------------------------------------
# POST /heartbeat
# ---------------------------------------------------------------------------

class TestHeartbeat:
    def test_status_200(self):
        c = _client()
        assert c.post("/heartbeat").status_code == 200

    def test_status_ok(self):
        c = _client()
        assert c.post("/heartbeat").json()["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /execute
# ---------------------------------------------------------------------------

class TestExecute:
    def test_status_200_valid_job(self):
        c = _client()
        assert c.post("/execute", json=_h_job()).status_code == 200

    def test_expectation_is_finite(self):
        c = _client()
        resp = c.post("/execute", json=_h_job()).json()
        assert math.isfinite(resp["expectation"])

    def test_expectation_bounded(self):
        n = 4
        c = _client()
        resp = c.post("/execute", json=_h_job(n=n)).json()
        assert abs(resp["expectation"]) <= n + 1e-4

    def test_job_id_preserved(self):
        c = _client()
        job_dict = _h_job()
        resp = c.post("/execute", json=job_dict).json()
        assert resp["job_id"] == job_dict["job_id"]

    def test_node_id_in_response(self):
        c = _client(node_id="srv-node")
        resp = c.post("/execute", json=_h_job()).json()
        assert resp["node_id"] == "srv-node"

    def test_proof_is_sha256_hex(self):
        c = _client()
        resp = c.post("/execute", json=_h_job()).json()
        assert len(resp["proof"]) == 64
        assert all(ch in "0123456789abcdef" for ch in resp["proof"])

    def test_proof_verifies(self):
        c = _client()
        job = SimJob(**{k: v for k, v in _h_job().items()})
        resp_json = c.post("/execute", json=job.to_dict()).json()
        from zilver.node import JobResult
        result = JobResult(**resp_json)
        assert result.verify(job) is True

    def test_elapsed_ms_present(self):
        c = _client()
        resp = c.post("/execute", json=_h_job()).json()
        assert resp["elapsed_ms"] >= 0

    def test_invalid_body_returns_422(self):
        c = _client()
        assert c.post("/execute", json={"bad": "body"}).status_code == 422

    def test_unsupported_backend_returns_422(self):
        c = _client(backends=["sv"])
        job = _h_job(backend="dm")
        assert c.post("/execute", json=job).status_code == 422

    def test_over_capacity_returns_422(self):
        c = _client(sv_max=2)
        job = _h_job(n=8, backend="sv")
        assert c.post("/execute", json=job).status_code == 422

    def test_sv_backend_executes(self):
        c = _client(backends=["sv"])
        resp = c.post("/execute", json=_h_job(backend="sv"))
        assert resp.status_code == 200
        assert math.isfinite(resp.json()["expectation"])

    def test_dm_backend_executes(self):
        c = _client(backends=["dm"])
        resp = c.post("/execute", json=_h_job(backend="dm"))
        assert resp.status_code == 200
        assert math.isfinite(resp.json()["expectation"])

    def test_tn_backend_executes(self):
        import math as _math
        import numpy as np
        c = _client(backends=["tn"])
        ops = [{"type": "ry", "qubits": [q], "param_idx": q} for q in range(4)]
        job = SimJob(
            circuit_ops=ops, n_qubits=4, n_params=4,
            params=[0.3, 0.7, 1.1, 1.5], backend="tn",
        ).to_dict()
        resp = c.post("/execute", json=job)
        assert resp.status_code == 200
        assert _math.isfinite(resp.json()["expectation"])

    def test_jobs_completed_increments(self):
        node = _node()
        c = TestClient(make_app(node))
        assert node.caps.jobs_completed == 0
        c.post("/execute", json=_h_job())
        assert node.caps.jobs_completed == 1
        c.post("/execute", json=_h_job())
        assert node.caps.jobs_completed == 2
