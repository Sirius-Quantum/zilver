"""
Node answers across backends and result types, and what the proof binds.
"""

import numpy as np
import pytest

import sys; sys.path.insert(0, "src")
from zilver.circuit import Circuit
from zilver.node import Node, job_from_circuit
from zilver import node_types


@pytest.fixture(scope="module")
def node():
    return Node.start(backends=["sv", "dm", "tn"], node_id="t", allow_unsigned=True)


def _ansatz() -> tuple[Circuit, list[float]]:
    c = Circuit(3)
    c.h(0); c.ry(1, 0); c.rx(2, 1); c.cz(0, 1); c.cnot(1, 2)
    c.rzz(0, 2, 2); c.u3(1, 3, 4, 5); c.x(2)
    return c, [0.3, -1.1, 0.7, 0.4, 1.9, -0.6]


# --- every backend agrees, none skips a gate --------------------------------

@pytest.mark.parametrize("observable", ["sum_z", "z0"])
def test_backends_agree(node, observable):
    c, params = _ansatz()
    c_no_u3 = Circuit(3)
    for op in c._ops:
        if op.kind != "u3":
            c_no_u3.add(op)
    c_no_u3.n_params = c.n_params
    values = {
        b: node.execute(job_from_circuit(c_no_u3, params, observable, backend=b)).expectation
        for b in ("sv", "dm", "tn")
    }
    assert values["dm"] == pytest.approx(values["sv"], abs=1e-4)
    assert values["tn"] == pytest.approx(values["sv"], abs=1e-4)


def test_tn_does_not_drop_cz(node):
    c = Circuit(2); c.h(0); c.h(1); c.cz(0, 1); c.h(1)
    assert node.execute(job_from_circuit(c, [], backend="tn")).expectation == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("backend", ["dm", "tn"])
def test_unsupported_gate_is_refused_not_skipped(node, backend):
    c, params = _ansatz()          # contains u3, which dm/tn cannot build
    with pytest.raises(ValueError, match="u3"):
        node.execute(job_from_circuit(c, params, backend=backend))


@pytest.mark.parametrize("bad_op", [
    {"type": "cnot", "qubits": [0, 0], "param_indices": []},
    {"type": "h", "qubits": [5], "param_indices": []},
    {"type": "ry", "qubits": [0], "param_indices": [9]},
    {"type": "ry", "qubits": [0], "param_indices": []},
    {"type": "swap", "qubits": [0, 1], "param_indices": []},
])
def test_malformed_ops_raise_value_error(node, bad_op):
    job = node_types.SimJob(circuit_ops=[bad_op], n_qubits=2, n_params=1, params=[0.1])
    with pytest.raises(ValueError):
        node.execute(job)


def test_dm_applies_noise(node):
    c = Circuit(2); c.x(0); c.cnot(0, 1)
    clean = job_from_circuit(c, [], backend="dm")
    noisy = job_from_circuit(c, [], backend="dm")
    noisy.noise = {"depolarizing": {"p1": 0.1, "p2": 0.1}}
    assert node.execute(clean).expectation == pytest.approx(-2.0, abs=1e-5)
    assert node.execute(noisy).expectation > -1.9


def test_bad_noise_spec_is_refused(node):
    c = Circuit(1); c.x(0)
    job = job_from_circuit(c, [], backend="dm")
    job.noise = {"cosmic_rays": {}}
    with pytest.raises(ValueError, match="noise model"):
        node.execute(job)


def test_non_expectation_needs_sv(node):
    c = Circuit(2); c.x(0)
    job = job_from_circuit(c, [], backend="tn")
    job.result_type = "samples"; job.shots = 10
    with pytest.raises(ValueError, match="statevector"):
        node.execute(job)


# --- sampled expectation ----------------------------------------------------

@pytest.mark.parametrize("observable,exact", [("sum_z", 1.0), ("z0", -1.0)])
def test_samples_expectation_matches_exact(node, observable, exact):
    c = Circuit(3); c.x(0)                  # |100>: Z0=-1, Z1=Z2=+1
    job = job_from_circuit(c, [], observable)
    job.result_type = "samples"; job.shots = 500
    assert node.execute(job).expectation == pytest.approx(exact)


def test_samples_are_reproducible_per_job(node):
    c = Circuit(2); c.h(0); c.h(1)
    job = job_from_circuit(c, [])
    job.result_type = "samples"; job.shots = 200
    assert node.execute(job).samples == node.execute(job).samples


# --- the proof binds the whole job ------------------------------------------

@pytest.mark.parametrize("result_type", ["expectation", "samples", "statevector", "pauli"])
def test_verify_true_for_every_result_type(node, result_type):
    c = Circuit(2); c.h(0); c.cnot(0, 1)
    job = job_from_circuit(c, [])
    job.result_type = result_type
    job.shots = 50
    job.hamiltonian = [{"coeff": 1.0, "pauli": "ZZ"}]
    assert node.execute(job).verify(job) is True


@pytest.mark.parametrize("tamper", [
    lambda j: setattr(j, "circuit_ops", []),
    lambda j: setattr(j, "n_qubits", 30),
    lambda j: setattr(j, "backend", "dm"),
    lambda j: setattr(j, "observable", "z0"),
])
def test_verify_false_when_job_changed(node, tamper):
    c = Circuit(3); c.x(0)
    job = job_from_circuit(c, [])
    result = node.execute(job)
    tamper(job)
    assert result.verify(job) is False


def test_verify_false_when_samples_changed(node):
    c = Circuit(2); c.h(0)
    job = job_from_circuit(c, [])
    job.result_type = "samples"; job.shots = 20
    result = node.execute(job)
    result.samples = ["11"] * 20
    assert result.verify(job) is False


# --- memory probe -----------------------------------------------------------

def test_available_memory_is_a_sane_fraction_of_ram():
    ram = node_types._detect_ram_gb() * 2**30
    avail = node_types._available_memory_bytes()
    assert 0 < avail <= ram * 1.05
    assert avail > 0.02 * ram   # free pages alone read ~0.5% on an idle Mac


def test_meminfo_parsing(tmp_path, monkeypatch):
    f = tmp_path / "meminfo"
    f.write_text("MemTotal:       16384000 kB\nMemAvailable:    8192000 kB\n")
    real_open = open
    monkeypatch.setattr("builtins.open",
                        lambda p, *a, **k: real_open(f, *a, **k) if p == "/proc/meminfo"
                        else real_open(p, *a, **k))
    assert node_types._available_memory_bytes() == 8192000 * 1024
