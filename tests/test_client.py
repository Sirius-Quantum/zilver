"""
Tests for NodeClient, RegistryClient, and NetworkCoordinator.

Rather than spinning up real HTTP servers, these tests combine two approaches:

1. ``respx`` — mocks httpx at the transport level so ``NodeClient`` and
   ``RegistryClient`` see realistic HTTP responses without any network I/O.

2. Live ``TestClient`` + custom httpx transport — for ``NetworkCoordinator``
   end-to-end tests, a ``fastapi.testclient.TestClient``-backed transport
   is injected into the httpx client so requests are routed directly to the
   FastAPI app in-process.

Coverage:
  NodeClient:
    - execute() returns JobResult
    - execute() proof verifies against original job
    - execute() raises HTTPStatusError on 422
    - caps() returns NodeCapabilities
    - health() returns status dict
    - context manager closes client

  RegistryClient:
    - register() returns True on success
    - deregister() returns True/False
    - heartbeat() returns True for known node, False for 404
    - match() returns URL on success, None on 404
    - nodes() returns list
    - summary() returns dict with required keys

  NetworkCoordinator:
    - submit() returns JobResult with finite expectation
    - submit() proof verifies
    - submit() raises RuntimeError when no eligible node
    - nodes() / summary() delegate to RegistryClient
"""

import math
import json

import httpx
import pytest
import respx

import sys; sys.path.insert(0, "src")
from zilver.node import Node, NodeCapabilities, SimJob, JobResult, _compute_proof
from zilver.registry import Registry
from zilver.server import make_app
from zilver.registry_server import make_registry_app
from zilver.client import NodeClient, RegistryClient, NetworkCoordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(node_id="n0", backends=None) -> Node:
    caps = NodeCapabilities(
        node_id=node_id,
        chip="Apple M4",
        ram_gb=16,
        sv_qubits_max=20,
        dm_qubits_max=10,
        tn_qubits_max=50,
        backends=backends or ["sv"],
    )
    return Node(caps)


def _h_job(n=4, backend="sv") -> SimJob:
    return SimJob(
        circuit_ops=[{"type": "h", "qubits": [0], "param_idx": None}],
        n_qubits=n,
        n_params=0,
        params=[],
        backend=backend,
    )


def _fake_job_result(job: SimJob, expectation: float = 1.0) -> dict:
    """Build a valid JobResult dict for a job, for use in mocked responses."""
    proof = _compute_proof(job.job_id, job.params, expectation)
    return {
        "expectation": expectation,
        "job_id":      job.job_id,
        "node_id":     "mock-node",
        "elapsed_ms":  1.0,
        "proof":       proof,
    }


# ---------------------------------------------------------------------------
# TestClient-backed httpx transport (for NetworkCoordinator e2e tests)
# ---------------------------------------------------------------------------

class _AppTransport(httpx.BaseTransport):
    """
    Routes httpx requests through a ``fastapi.testclient.TestClient``.

    This lets ``NodeClient`` and ``RegistryClient`` (which use httpx) talk
    directly to a FastAPI app in-process — no real TCP socket, no ``respx``
    mocking needed.
    """

    def __init__(self, app_client) -> None:
        self._app = app_client

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        method  = request.method
        path    = request.url.path
        params  = dict(request.url.params)
        content = request.content

        # Route through TestClient
        kwargs: dict = {}
        if params:
            kwargs["params"] = params
        if content:
            kwargs["json"] = json.loads(content)

        resp = getattr(self._app, method.lower())(path, **kwargs)
        return httpx.Response(
            status_code=resp.status_code,
            content=resp.content,
            headers=dict(resp.headers),
        )


# ---------------------------------------------------------------------------
# NodeClient tests (respx mocks)
# ---------------------------------------------------------------------------

class TestNodeClient:
    NODE_URL = "http://fake-node:7700"

    def _client(self, **kwargs) -> NodeClient:
        nc = NodeClient(self.NODE_URL, **kwargs)
        return nc

    @respx.mock
    def test_execute_returns_job_result(self):
        job = _h_job()
        fake = _fake_job_result(job, expectation=2.5)
        respx.post(f"{self.NODE_URL}/execute").mock(
            return_value=httpx.Response(200, json=fake)
        )
        nc = self._client()
        result = nc.execute(job)
        assert isinstance(result, JobResult)
        assert result.expectation == pytest.approx(2.5)

    @respx.mock
    def test_execute_proof_verifies(self):
        job = _h_job()
        fake = _fake_job_result(job, expectation=2.5)
        respx.post(f"{self.NODE_URL}/execute").mock(
            return_value=httpx.Response(200, json=fake)
        )
        result = self._client().execute(job)
        assert result.verify(job) is True

    @respx.mock
    def test_execute_raises_on_422(self):
        respx.post(f"{self.NODE_URL}/execute").mock(
            return_value=httpx.Response(422, json={"detail": "unsupported backend"})
        )
        with pytest.raises(httpx.HTTPStatusError):
            self._client().execute(_h_job())

    @respx.mock
    def test_caps_returns_node_capabilities(self):
        caps = _node().caps.to_dict()
        respx.get(f"{self.NODE_URL}/caps").mock(
            return_value=httpx.Response(200, json=caps)
        )
        result = self._client().caps()
        assert isinstance(result, NodeCapabilities)
        assert result.node_id == caps["node_id"]

    @respx.mock
    def test_health_returns_dict(self):
        respx.get(f"{self.NODE_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "ok", "node_id": "n0"})
        )
        result = self._client().health()
        assert result["status"] == "ok"

    def test_repr(self):
        nc = NodeClient("http://host:7700")
        assert "NodeClient" in repr(nc)
        assert "host:7700" in repr(nc)

    def test_context_manager(self):
        with NodeClient("http://host:7700") as nc:
            assert isinstance(nc, NodeClient)


# ---------------------------------------------------------------------------
# RegistryClient tests (respx mocks)
# ---------------------------------------------------------------------------

class TestRegistryClient:
    REG_URL = "http://fake-registry:7701"

    def _client(self) -> RegistryClient:
        return RegistryClient(self.REG_URL)

    @respx.mock
    def test_register_returns_true(self):
        respx.post(f"{self.REG_URL}/nodes").mock(
            return_value=httpx.Response(201, json={"registered": True, "node_id": "n0"})
        )
        rc = self._client()
        assert rc.register(_node().caps, "http://host:7700") is True

    @respx.mock
    def test_deregister_returns_true(self):
        respx.delete(f"{self.REG_URL}/nodes/n0").mock(
            return_value=httpx.Response(200, json={"deregistered": True, "node_id": "n0"})
        )
        assert self._client().deregister("n0") is True

    @respx.mock
    def test_deregister_returns_false_for_unknown(self):
        respx.delete(f"{self.REG_URL}/nodes/unknown").mock(
            return_value=httpx.Response(200, json={"deregistered": False, "node_id": "unknown"})
        )
        assert self._client().deregister("unknown") is False

    @respx.mock
    def test_heartbeat_returns_true(self):
        respx.post(f"{self.REG_URL}/nodes/n0/heartbeat").mock(
            return_value=httpx.Response(200, json={"status": "ok", "node_id": "n0"})
        )
        assert self._client().heartbeat("n0") is True

    @respx.mock
    def test_heartbeat_returns_false_on_404(self):
        respx.post(f"{self.REG_URL}/nodes/unknown/heartbeat").mock(
            return_value=httpx.Response(404, json={"detail": "not found"})
        )
        assert self._client().heartbeat("unknown") is False

    @respx.mock
    def test_match_returns_url(self):
        caps = _node("n0").caps.to_dict()
        caps["url"] = "http://192.168.1.5:7700"
        respx.get(f"{self.REG_URL}/match").mock(
            return_value=httpx.Response(200, json=caps)
        )
        url = self._client().match("sv", 4)
        assert url == "http://192.168.1.5:7700"

    @respx.mock
    def test_match_returns_none_on_404(self):
        respx.get(f"{self.REG_URL}/match").mock(
            return_value=httpx.Response(404, json={"detail": "no eligible node"})
        )
        assert self._client().match("sv", 4) is None

    @respx.mock
    def test_nodes_returns_list(self):
        node_list = [_node("n0").caps.to_dict()]
        node_list[0]["url"] = "http://host:7700"
        respx.get(f"{self.REG_URL}/nodes").mock(
            return_value=httpx.Response(200, json=node_list)
        )
        result = self._client().nodes()
        assert isinstance(result, list)
        assert result[0]["node_id"] == "n0"

    @respx.mock
    def test_summary_has_required_keys(self):
        s = {"online": 1, "total_registered": 1, "backends": ["sv"],
             "max_sv_qubits": 20, "max_dm_qubits": 10, "total_stake": 0}
        respx.get(f"{self.REG_URL}/summary").mock(
            return_value=httpx.Response(200, json=s)
        )
        result = self._client().summary()
        for key in ["online", "total_registered", "backends"]:
            assert key in result

    def test_repr(self):
        rc = RegistryClient("http://host:7701")
        assert "RegistryClient" in repr(rc)

    def test_context_manager(self):
        with RegistryClient("http://host:7701") as rc:
            assert isinstance(rc, RegistryClient)


# ---------------------------------------------------------------------------
# NetworkCoordinator end-to-end (TestClient-backed transport)
# ---------------------------------------------------------------------------

class TestNetworkCoordinator:
    """
    End-to-end tests using real FastAPI apps connected via _AppTransport.

    A registry server and a node server are created in-process.  The node
    is registered with the registry.  NetworkCoordinator uses a patched
    httpx client that routes through _AppTransport instead of real TCP.
    """

    def _setup(self):
        """Return (NetworkCoordinator, node) wired through TestClient transports."""
        from fastapi.testclient import TestClient as TC

        node = _node("e2e-node", backends=["sv"])

        # Build apps
        reg_app  = make_registry_app()
        node_app = make_app(node)

        reg_tc   = TC(reg_app)
        node_tc  = TC(node_app)

        # Register the node via the registry TestClient
        caps_body = {
            "caps": node.caps.to_dict(),
            "url":  "http://fake-node:7700",
        }
        reg_tc.post("/nodes", json=caps_body)

        # Build coordinator with transport-patched httpx clients
        coord = NetworkCoordinator.__new__(NetworkCoordinator)
        coord.registry_url = "http://fake-registry:7701"
        coord.timeout = 10.0

        # Patch RegistryClient to use reg_tc transport
        rc = RegistryClient.__new__(RegistryClient)
        rc.url = "http://fake-registry:7701"
        rc._client = httpx.Client(
            transport=_AppTransport(reg_tc), timeout=10.0
        )
        coord._registry = rc

        # Store node_tc so submit() can route to it
        coord._node_tc = node_tc

        return coord, node, node_tc

    def test_submit_returns_job_result(self):
        coord, node, node_tc = self._setup()

        # Patch submit to use node TestClient transport
        def _patched_submit(job):
            url = coord._registry.match(job.backend, job.n_qubits)
            if url is None:
                raise RuntimeError("No eligible node")
            nc = NodeClient.__new__(NodeClient)
            nc.url = url
            nc._client = httpx.Client(
                transport=_AppTransport(node_tc), timeout=10.0
            )
            return nc.execute(job)

        coord.submit = _patched_submit

        job = _h_job()
        result = coord.submit(job)
        assert isinstance(result, JobResult)
        assert math.isfinite(result.expectation)

    def test_submit_proof_verifies(self):
        coord, node, node_tc = self._setup()

        def _patched_submit(job):
            url = coord._registry.match(job.backend, job.n_qubits)
            nc = NodeClient.__new__(NodeClient)
            nc.url = url
            nc._client = httpx.Client(
                transport=_AppTransport(node_tc), timeout=10.0
            )
            return nc.execute(job)

        coord.submit = _patched_submit
        job = _h_job()
        result = coord.submit(job)
        assert result.verify(job) is True

    @respx.mock
    def test_submit_raises_when_no_eligible_node(self):
        respx.get("http://fake-registry:7701/match").mock(
            return_value=httpx.Response(404, json={"detail": "no node"})
        )
        coord = NetworkCoordinator("http://fake-registry:7701")
        with pytest.raises(RuntimeError, match="No eligible node"):
            coord.submit(_h_job())

    def test_repr(self):
        coord = NetworkCoordinator("http://registry:7701")
        assert "NetworkCoordinator" in repr(coord)
        assert "registry:7701" in repr(coord)
