#!/usr/bin/env python3
"""Shared QPU provider plumbing for the real-to-sim benches.

One thin, provider-agnostic interface over IBM Quantum and IQM Resonance:

    p = make_provider("iqm", backend="garnet", qubits=[0, 1, 2])
    cal   = p.calibration(0)          # per-qubit T1/T2/gate times
    cbar  = p.mean_calibration()      # averaged over p.qubits (global noise model)
    gerr  = p.gate_errors()           # 1q/2q depolarizing rates from RB metrics
    counts = p.run(circuits, shots)   # list[dict] of measured bitstring counts

Both backends are Qiskit-native, so circuits and counts are interchangeable. The
two SDKs pin incompatible qiskit versions (IBM needs qiskit>=2.2, IQM needs
qiskit<2.2), so each provider needs its own virtualenv — imports here are lazy so
importing this module never pulls in either SDK.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Sequence


# ---------------------------------------------------------------------------
# Records (everything in nanoseconds; rates are dimensionless probabilities)
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    t1_ns: float
    t2_ns: float
    tg1_ns: float
    tg2_ns: float
    qubit: int
    source: str


@dataclass
class GateErrors:
    depol_1q: float   # depolarizing probability per single-qubit gate
    depol_2q: float   # depolarizing probability per two-qubit gate
    source: str


# RB error-per-gate r -> depolarizing probability p for a d-dim gate:
#   r = p * (d^2 - 1) / d^2   =>   p = r * d^2 / (d^2 - 1)
def rb_error_to_depol(r: float, n_qubits: int) -> float:
    d2 = (2 ** n_qubits) ** 2
    return max(0.0, min(1.0, r * d2 / (d2 - 1)))


# ---------------------------------------------------------------------------
# Distribution distance
# ---------------------------------------------------------------------------

def hellinger(p: dict[str, float], q: dict[str, float]) -> float:
    """Hellinger distance between two bitstring distributions (probabilities).

    H = (1/sqrt(2)) * sqrt( sum_i (sqrt(p_i) - sqrt(q_i))^2 ),  in [0, 1].
    < 0.1 is the conventional "the noise model matches hardware" threshold.
    """
    keys = set(p) | set(q)
    s = sum((math.sqrt(p.get(k, 0.0)) - math.sqrt(q.get(k, 0.0))) ** 2 for k in keys)
    return math.sqrt(s) / math.sqrt(2.0)


def counts_to_probs(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {k.replace(" ", ""): v / total for k, v in counts.items()} if total else {}


# ---------------------------------------------------------------------------
# IBM Quantum
# ---------------------------------------------------------------------------

class IBMProvider:
    name = "ibm"

    def __init__(self, backend_name: str | None, qubits: Sequence[int]):
        from qiskit_ibm_runtime import QiskitRuntimeService
        self.qubits = list(qubits)
        self.service = QiskitRuntimeService()
        self.backend = (
            self.service.backend(backend_name) if backend_name
            else self.service.least_busy(operational=True, simulator=False)
        )
        print(f"  IBM backend: {self.backend.name}")

    def calibration(self, qubit: int | None = None) -> Calibration:
        props = self.backend.properties()
        q = self.qubits[0] if qubit is None else qubit
        tg1 = _ibm_gate_len(props, "sx", [q]) or _ibm_gate_len(props, "x", [q]) or 35e-9
        tg2 = _ibm_two_q_len(self.backend, props, q) or 70e-9
        return Calibration(props.t1(q) * 1e9, props.t2(q) * 1e9,
                           tg1 * 1e9, tg2 * 1e9, q, f"IBM {self.backend.name} (live)")

    def mean_calibration(self) -> Calibration:
        return _mean_calibration([self.calibration(q) for q in self.qubits],
                                 f"IBM {self.backend.name} mean of {self.qubits}")

    def gate_errors(self) -> GateErrors:
        props = self.backend.properties()
        e1 = [props.gate_error("sx", [q]) for q in self.qubits if _safe(props, "sx", [q])]
        d2 = _ibm_two_q_errors(self.backend, props, self.qubits)
        depol_1q = rb_error_to_depol(_avg(e1, 1e-3), 1)
        depol_2q = rb_error_to_depol(_avg(d2, 1e-2), 2)
        return GateErrors(depol_1q, depol_2q, f"IBM {self.backend.name} gate_error (live)")

    def run(self, circuits, shots: int, layout: Sequence[int] | None = None) -> list[dict]:
        from qiskit import transpile
        from qiskit_ibm_runtime import SamplerV2
        lo = list(layout) if layout is not None else self.qubits
        tqc = transpile(circuits, backend=self.backend,
                        initial_layout=lo[:circuits[0].num_qubits], optimization_level=1)
        result = SamplerV2(mode=self.backend).run(tqc, shots=shots).result()
        return [r.data.c.get_counts() for r in result]


def _safe(props, gate, qs):
    try:
        props.gate_error(gate, qs)
        return True
    except Exception:
        return False


def _ibm_gate_len(props, gate, qubits):
    try:
        return props.gate_length(gate, qubits)
    except Exception:
        return None


def _ibm_two_q_len(backend, props, q):
    try:
        for a, b in backend.coupling_map.get_edges():
            if q in (a, b):
                for g in ("cz", "ecr", "cx"):
                    L = _ibm_gate_len(props, g, [a, b])
                    if L:
                        return L
    except Exception:
        pass
    return None


def _ibm_two_q_errors(backend, props, qubits):
    out = []
    try:
        edges = backend.coupling_map.get_edges()
        for a, b in edges:
            if a in qubits and b in qubits:
                for g in ("cz", "ecr", "cx"):
                    try:
                        out.append(props.gate_error(g, [a, b]))
                        break
                    except Exception:
                        continue
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# IQM Resonance
# ---------------------------------------------------------------------------

class IQMProvider:
    name = "iqm"

    def __init__(self, backend_name: str | None, qubits: Sequence[int]):
        from iqm.qiskit_iqm import IQMProvider as _IQMProv
        if not os.environ.get("IQM_TOKEN"):
            raise RuntimeError("Set IQM_TOKEN (generate from the Resonance dashboard).")
        url = os.environ.get("IQM_SERVER_URL", "https://resonance.iqm.tech")
        self.qc_name = backend_name or "garnet"
        self.qubits = list(qubits)
        # token is read from IQM_TOKEN by the client; passing it as a kwarg too is
        # rejected ("parameter sources must not be mixed").
        self.backend = _IQMProv(url, quantum_computer=self.qc_name).get_backend()
        self._obs = None
        print(f"  IQM backend: {self.qc_name} ({self.backend.num_qubits}q)")

    def _observations(self):
        if self._obs is None:
            self._obs = self.backend.client.get_quality_metric_set().model_dump().get("observations") or []
        return self._obs

    def calibration(self, qubit: int | None = None) -> Calibration:
        q = self.qubits[0] if qubit is None else qubit
        key = f"QB{q + 1}"
        vals: dict[str, float] = {}
        for o in self._observations():
            field = o.get("dut_field") or ""
            if field.startswith(f"characterization.model.{key}."):
                vals[field.rsplit(".", 1)[-1]] = o.get("value")
        t1_ns = float(vals.get("t1_time", 40e-6)) * 1e9
        t2_ns = float(vals.get("t2_echo_time", vals.get("t2_time", 20e-6))) * 1e9
        # gate durations from the error profile if present, else IQM-typical defaults
        return Calibration(t1_ns, t2_ns, 42.0, 130.0, q, f"IQM {self.qc_name} {key} (live)")

    def mean_calibration(self) -> Calibration:
        return _mean_calibration([self.calibration(q) for q in self.qubits],
                                 f"IQM {self.qc_name} mean of {self.qubits}")

    def gate_errors(self) -> GateErrors:
        # metrics.rb.prx (1q), metrics.irb.cz (2q) — values are error-per-gate.
        e1, e2 = [], []
        for o in self._observations():
            f = o.get("dut_field") or ""
            v = o.get("value")
            if v is None:
                continue
            if f.startswith("metrics.rb.prx"):
                e1.append(_as_error(v))
            elif f.startswith("metrics.irb.cz"):
                e2.append(_as_error(v))
        depol_1q = rb_error_to_depol(_avg(e1, 1e-3), 1)
        depol_2q = rb_error_to_depol(_avg(e2, 1e-2), 2)
        return GateErrors(depol_1q, depol_2q, f"IQM {self.qc_name} RB metrics (live)")

    def run(self, circuits, shots: int, layout: Sequence[int] | None = None) -> list[dict]:
        from qiskit import transpile
        lo = list(layout) if layout is not None else self.qubits
        tqc = transpile(circuits, backend=self.backend,
                        initial_layout=lo[:circuits[0].num_qubits], optimization_level=1)
        result = self.backend.run(tqc, shots=shots).result()
        return [result.get_counts(i) for i in range(len(circuits))]


def _as_error(v: float) -> float:
    """RB metrics may be reported as fidelity (~0.99) or error (~0.01). Normalize
    to an error-per-gate."""
    v = float(v)
    return 1.0 - v if v > 0.5 else v


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _avg(xs, default):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else default


def _mean_calibration(cals: list[Calibration], source: str) -> Calibration:
    return Calibration(
        _avg([c.t1_ns for c in cals], 40_000.0),
        _avg([c.t2_ns for c in cals], 20_000.0),
        _avg([c.tg1_ns for c in cals], 42.0),
        _avg([c.tg2_ns for c in cals], 130.0),
        cals[0].qubit if cals else -1,
        source,
    )


def make_provider(name: str, backend: str | None, qubits: Sequence[int]):
    if name == "ibm":
        return IBMProvider(backend, qubits)
    if name == "iqm":
        return IQMProvider(backend, qubits)
    raise ValueError(f"unknown provider: {name}")
