"""
Tests for the registry HTTP server.

All tests use ``fastapi.testclient.TestClient`` — no real network required.

Coverage:
  - POST   /nodes        — register a node (201), idempotent re-register
  - POST   /nodes        — missing fields returns 422
  - DELETE /nodes/{id}   — deregisters a known node
  - DELETE /nodes/{id}   — unknown node returns deregistered=false
  - POST   /nodes/{id}/heartbeat — refreshes last-seen (200)
  - POST   /nodes/{id}/heartbeat — unknown node returns 404
  - GET    /nodes        — lists online nodes; excludes deregistered
  - GET    /nodes        — each entry contains a url field
  - GET    /match        — returns best node matching backend+n_qubits
  - GET    /match        — returns 404 when no eligible node
  - GET    /match        — respects min_stake parameter
  - GET    /summary      — has required stat keys
  - GET    /summary      — counts are correct after register/deregister
"""

import pytest
from fastapi.testclient import TestClient

import sys; sys.path.insert(0, "src")
from zilver.registry import Registry
from zilver.registry_server import make_registry_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _app_and_reg() -> tuple:
    """Return (TestClient, Registry) sharing the same in-memory state."""
    reg = Registry()
    app = make_registry_app(reg)
    return TestClient(app), reg


def _caps_body(
    node_id: str = "n0",
    backends: list[str] | None = None,
    sv_max: int = 20,
    stake: int = 0,
    url: str = "http://localhost:7700",
) -> dict:
    return {
        "caps": {
            "node_id":       node_id,
            "chip":          "Apple M4",
            "ram_gb":        16,
            "sv_qubits_max": sv_max,
            "dm_qubits_max": 10,
            "tn_qubits_max": 50,
            "backends":      backends or ["sv"],
            "jobs_completed": 0,
            "stake":         stake,
        },
        "url": url,
    }


# ---------------------------------------------------------------------------
# POST /nodes
# ---------------------------------------------------------------------------

class TestRegister:
    def test_returns_201(self):
        c, _ = _app_and_reg()
        resp = c.post("/nodes", json=_caps_body())
        assert resp.status_code == 201

    def test_registered_true(self):
        c, _ = _app_and_reg()
        resp = c.post("/nodes", json=_caps_body()).json()
        assert resp["registered"] is True

    def test_node_id_in_response(self):
        c, _ = _app_and_reg()
        resp = c.post("/nodes", json=_caps_body("my-node")).json()
        assert resp["node_id"] == "my-node"

    def test_idempotent_reregister(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0"))
        resp = c.post("/nodes", json=_caps_body("n0"))
        assert resp.status_code == 201

    def test_missing_caps_returns_422(self):
        c, _ = _app_and_reg()
        resp = c.post("/nodes", json={"url": "http://localhost:7700"})
        assert resp.status_code == 422

    def test_missing_url_returns_422(self):
        c, _ = _app_and_reg()
        body = _caps_body()
        del body["url"]
        resp = c.post("/nodes", json=body)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# DELETE /nodes/{node_id}
# ---------------------------------------------------------------------------

class TestDeregister:
    def test_returns_200(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0"))
        resp = c.delete("/nodes/n0")
        assert resp.status_code == 200

    def test_deregistered_true_for_known_node(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0"))
        resp = c.delete("/nodes/n0").json()
        assert resp["deregistered"] is True

    def test_deregistered_false_for_unknown_node(self):
        c, _ = _app_and_reg()
        resp = c.delete("/nodes/nonexistent").json()
        assert resp["deregistered"] is False

    def test_node_absent_from_list_after_deregister(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0"))
        c.delete("/nodes/n0")
        nodes = c.get("/nodes").json()
        ids = [n["node_id"] for n in nodes]
        assert "n0" not in ids


# ---------------------------------------------------------------------------
# POST /nodes/{node_id}/heartbeat
# ---------------------------------------------------------------------------

class TestHeartbeat:
    def test_returns_200_for_known_node(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0"))
        assert c.post("/nodes/n0/heartbeat").status_code == 200

    def test_status_ok(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0"))
        resp = c.post("/nodes/n0/heartbeat").json()
        assert resp["status"] == "ok"

    def test_returns_404_for_unknown_node(self):
        c, _ = _app_and_reg()
        assert c.post("/nodes/nonexistent/heartbeat").status_code == 404


# ---------------------------------------------------------------------------
# GET /nodes
# ---------------------------------------------------------------------------

class TestListNodes:
    def test_empty_on_fresh_registry(self):
        c, _ = _app_and_reg()
        assert c.get("/nodes").json() == []

    def test_registered_node_appears(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0"))
        nodes = c.get("/nodes").json()
        assert any(n["node_id"] == "n0" for n in nodes)

    def test_deregistered_node_excluded(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0"))
        c.post("/nodes", json=_caps_body("n1"))
        c.delete("/nodes/n1")
        nodes = c.get("/nodes").json()
        ids = [n["node_id"] for n in nodes]
        assert "n0" in ids
        assert "n1" not in ids

    def test_url_field_present(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0", url="http://1.2.3.4:7700"))
        nodes = c.get("/nodes").json()
        assert nodes[0]["url"] == "http://1.2.3.4:7700"

    def test_multiple_nodes(self):
        c, _ = _app_and_reg()
        for i in range(3):
            c.post("/nodes", json=_caps_body(f"n{i}"))
        assert len(c.get("/nodes").json()) == 3


# ---------------------------------------------------------------------------
# GET /match
# ---------------------------------------------------------------------------

class TestMatch:
    def test_returns_matching_node(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0", backends=["sv"]))
        resp = c.get("/match", params={"backend": "sv", "n_qubits": 4})
        assert resp.status_code == 200
        assert resp.json()["node_id"] == "n0"

    def test_url_in_response(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0", url="http://1.2.3.4:7700"))
        resp = c.get("/match", params={"backend": "sv", "n_qubits": 4})
        assert resp.json()["url"] == "http://1.2.3.4:7700"

    def test_returns_404_when_no_eligible_node(self):
        c, _ = _app_and_reg()
        resp = c.get("/match", params={"backend": "sv", "n_qubits": 4})
        assert resp.status_code == 404

    def test_rejects_wrong_backend(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0", backends=["sv"]))
        resp = c.get("/match", params={"backend": "dm", "n_qubits": 4})
        assert resp.status_code == 404

    def test_rejects_over_capacity(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0", sv_max=4))
        resp = c.get("/match", params={"backend": "sv", "n_qubits": 8})
        assert resp.status_code == 404

    def test_min_stake_filter(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0", stake=10))
        resp = c.get("/match", params={"backend": "sv", "n_qubits": 4, "min_stake": 50})
        assert resp.status_code == 404

    def test_min_stake_passes(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0", stake=100))
        resp = c.get("/match", params={"backend": "sv", "n_qubits": 4, "min_stake": 50})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_required_keys(self):
        c, _ = _app_and_reg()
        s = c.get("/summary").json()
        for key in ["online", "total_registered", "backends", "max_sv_qubits", "total_stake"]:
            assert key in s, f"Missing key: {key}"

    def test_online_count_after_register(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0"))
        c.post("/nodes", json=_caps_body("n1"))
        assert c.get("/summary").json()["online"] == 2

    def test_online_count_after_deregister(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0"))
        c.post("/nodes", json=_caps_body("n1"))
        c.delete("/nodes/n1")
        assert c.get("/summary").json()["online"] == 1

    def test_backends_union(self):
        c, _ = _app_and_reg()
        c.post("/nodes", json=_caps_body("n0", backends=["sv"]))
        c.post("/nodes", json=_caps_body("n1", backends=["dm"]))
        s = c.get("/summary").json()
        assert set(s["backends"]) == {"sv", "dm"}
