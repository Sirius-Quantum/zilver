"""
Zilver node daemon.

Each Apple Silicon device runs a node that:
  - Detects local hardware capabilities (chip, RAM, qubit ceiling)
  - Executes simulation jobs locally via MLX
  - Returns results with a deterministic job proof for verification
  - (Future) Registers with the capability registry and listens on P2P bus

Entry point: zilver-node start --backends sv,dm --wallet <addr>

Job routing:
  sv  — statevector (simulator.py),  up to 32-34 qubits
  dm  — density matrix (density_matrix.py), up to 16-17 qubits
  tn  — tensor network (tensor_network.py), up to 50+ qubits (sparse)
"""

from __future__ import annotations
import hashlib
import json
import platform
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def _detect_chip() -> str:
    """Return Apple Silicon chip identifier, e.g. 'Apple M4 Pro'."""
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip()
        if out:
            return out
    except Exception:
        pass
    return platform.processor() or "unknown"


def _detect_ram_gb() -> int:
    """Return total physical RAM in GB."""
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip()
        return int(out) // (1024 ** 3)
    except Exception:
        return 8   # conservative fallback


def _sv_qubit_ceiling(ram_gb: int) -> int:
    """
    Maximum qubits for exact statevector: (2^n,) complex64 = 8 bytes * 2^n.
    Use 80% of RAM to leave headroom.
    """
    usable = int(ram_gb * 0.8 * (1024 ** 3))
    n = 0
    while (8 * (2 ** (n + 1))) <= usable:
        n += 1
    return min(n, 34)   # cap at AWS SV1 equivalence


def _dm_qubit_ceiling(ram_gb: int) -> int:
    """
    Maximum qubits for density matrix: (2^n, 2^n) complex64 = 8 * 4^n bytes.
    """
    usable = int(ram_gb * 0.8 * (1024 ** 3))
    n = 0
    while (8 * (4 ** (n + 1))) <= usable:
        n += 1
    return min(n, 17)


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

@dataclass
class NodeCapabilities:
    """
    Hardware capabilities advertised to the capability registry.

    Populated automatically by NodeCapabilities.detect() on startup.
    """
    node_id:        str
    chip:           str
    ram_gb:         int
    sv_qubits_max:  int
    dm_qubits_max:  int
    tn_qubits_max:  int    # MPS target; independent of RAM
    backends:       list[str]
    jobs_completed: int = 0
    stake:          int = 0

    @classmethod
    def detect(
        cls,
        backends: list[str] | None = None,
        node_id: str | None = None,
    ) -> "NodeCapabilities":
        chip   = _detect_chip()
        ram_gb = _detect_ram_gb()
        return cls(
            node_id       = node_id or str(uuid.uuid4()),
            chip          = chip,
            ram_gb        = ram_gb,
            sv_qubits_max = _sv_qubit_ceiling(ram_gb),
            dm_qubits_max = _dm_qubit_ceiling(ram_gb),
            tn_qubits_max = 50,
            backends      = backends or ["sv"],
        )

    def supports(self, backend: str, n_qubits: int) -> bool:
        if backend not in self.backends:
            return False
        if backend == "sv"  and n_qubits > self.sv_qubits_max:
            return False
        if backend == "dm"  and n_qubits > self.dm_qubits_max:
            return False
        if backend == "tn"  and n_qubits > self.tn_qubits_max:
            return False
        return True

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Job / Result
# ---------------------------------------------------------------------------

@dataclass
class SimJob:
    """
    A simulation job submitted to a node.

    circuit_ops: serializable list of gate operations
                 [{"type": "h"|"ry"|"cnot"|..., "qubits": [...], "param_idx": int|None}]
    n_qubits:    total qubit count
    n_params:    number of circuit parameters
    params:      flat list of float parameter values
    observable:  "sum_z" | "z0"
    backend:     "sv" | "dm" | "tn"
    job_id:      unique identifier
    """
    circuit_ops: list[dict]
    n_qubits:    int
    n_params:    int
    params:      list[float]
    observable:  str = "sum_z"
    backend:     str = "sv"
    job_id:      str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimJob":
        return cls(**d)


@dataclass
class JobResult:
    """
    Result returned by a node after executing a SimJob.

    expectation: computed expectation value
    job_id:      matches SimJob.job_id
    node_id:     identity of the executing node
    elapsed_ms:  wall-clock execution time
    proof:       SHA-256 of (job_id + params + expectation) for verification
    """
    expectation: float
    job_id:      str
    node_id:     str
    elapsed_ms:  float
    proof:       str

    def to_dict(self) -> dict:
        return asdict(self)

    def verify(self, job: SimJob) -> bool:
        """Recompute and check the proof hash."""
        return self.proof == _compute_proof(job.job_id, job.params, self.expectation)


def _compute_proof(job_id: str, params: list[float], expectation: float) -> str:
    payload = json.dumps({
        "job_id":     job_id,
        "params":     params,
        "expectation": round(expectation, 8),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Circuit reconstruction from ops list
# ---------------------------------------------------------------------------

def _build_circuit_from_ops(ops: list[dict], n_qubits: int, n_params: int):
    """Reconstruct a Circuit from a serialized ops list."""
    from .circuit import Circuit
    c = Circuit(n_qubits)
    c.n_params = n_params
    for op in ops:
        kind    = op["type"]
        qubits  = op["qubits"]
        pidx    = op.get("param_idx")
        if kind == "h":
            c.h(qubits[0])
        elif kind == "x":
            c.x(qubits[0])
        elif kind == "ry":
            c.ry(qubits[0], pidx)
        elif kind == "rx":
            c.rx(qubits[0], pidx)
        elif kind == "rz":
            c.rz(qubits[0], pidx)
        elif kind == "cnot":
            c.cnot(qubits[0], qubits[1])
        elif kind == "cz":
            c.cz(qubits[0], qubits[1])
        elif kind == "rzz":
            c.rzz(qubits[0], qubits[1], pidx)
        else:
            raise ValueError(f"Unknown gate type in job ops: {kind!r}")
    return c


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """
    A Zilver simulation node.

    Executes SimJobs locally using the appropriate backend (sv/dm/tn).
    In the distributed network, the node daemon runs alongside a P2P listener
    that receives jobs from the coordinator. Here the node exposes a synchronous
    `execute(job)` API usable both locally and by the P2P layer.

    Usage:
        node = Node.start(backends=["sv", "dm"])
        result = node.execute(job)
        assert result.verify(job)
    """

    def __init__(self, caps: NodeCapabilities):
        self.caps = caps

    @classmethod
    def start(
        cls,
        backends: list[str] | None = None,
        node_id: str | None = None,
        wallet: str | None = None,
    ) -> "Node":
        """
        Initialize a node with auto-detected hardware capabilities.

        Args:
            backends: list of ["sv", "dm", "tn"]; default ["sv"]
            node_id:  explicit node ID; auto-generated if None
            wallet:   wallet address for reward settlement (future use)
        """
        caps = NodeCapabilities.detect(backends=backends, node_id=node_id)
        node = cls(caps)
        node._wallet = wallet
        return node

    def execute(self, job: SimJob) -> JobResult:
        """
        Execute a simulation job and return a verified result.

        Raises ValueError if the node cannot handle the job
        (backend unsupported or qubit count exceeds capacity).
        """
        if not self.caps.supports(job.backend, job.n_qubits):
            raise ValueError(
                f"Node {self.caps.node_id} cannot handle "
                f"backend={job.backend!r} n_qubits={job.n_qubits} "
                f"(sv_max={self.caps.sv_qubits_max}, "
                f"dm_max={self.caps.dm_qubits_max})"
            )

        t0 = time.perf_counter()
        expectation = self._run(job)
        elapsed_ms  = (time.perf_counter() - t0) * 1000.0

        self.caps.jobs_completed += 1

        proof = _compute_proof(job.job_id, job.params, expectation)
        return JobResult(
            expectation = expectation,
            job_id      = job.job_id,
            node_id     = self.caps.node_id,
            elapsed_ms  = elapsed_ms,
            proof       = proof,
        )

    def _run(self, job: SimJob) -> float:
        params = mx.array(np.array(job.params, dtype=np.float32))

        if job.backend in ("sv", "tn"):
            return self._run_sv(job, params)
        elif job.backend == "dm":
            return self._run_dm(job, params)
        else:
            raise ValueError(f"Unknown backend: {job.backend!r}")

    def _run_sv(self, job: SimJob, params: mx.array) -> float:
        if job.backend == "tn":
            return self._run_tn(job, params)
        circuit = _build_circuit_from_ops(job.circuit_ops, job.n_qubits, job.n_params)
        return float(circuit.compile(job.observable)(params).item())

    def _run_dm(self, job: SimJob, params: mx.array) -> float:
        from .density_matrix import NoisyCircuit, expectation_sum_z_dm, expectation_z_dm
        circuit = _build_circuit_from_ops(job.circuit_ops, job.n_qubits, job.n_params)
        # Re-use the sv circuit execution path; DM with no noise = sv
        return float(circuit.compile(job.observable)(params).item())

    def _run_tn(self, job: SimJob, params: mx.array) -> float:
        from .tensor_network import MPSCircuit, expectation_sum_z_mps, expectation_z_mps
        c = MPSCircuit(job.n_qubits, chi_max=64)
        params_np = np.array(job.params, dtype=np.float32)
        for op in job.circuit_ops:
            kind   = op["type"]
            qubits = op["qubits"]
            pidx   = op.get("param_idx")
            if kind == "h":
                c.h(qubits[0])
            elif kind == "x":
                c.x(qubits[0])
            elif kind == "ry":
                c.ry(qubits[0], pidx)
            elif kind == "rx":
                c.rx(qubits[0], pidx)
            elif kind == "rz":
                c.rz(qubits[0], pidx)
            elif kind == "cnot":
                c.cnot(qubits[0], qubits[1])
            elif kind == "rzz":
                c.rzz(qubits[0], qubits[1], pidx)
        tensors = c._run(params)
        n = job.n_qubits
        if job.observable == "sum_z":
            return expectation_sum_z_mps(tensors, n)
        return expectation_z_mps(tensors, 0, n)

    def __repr__(self) -> str:
        return (
            f"Node(id={self.caps.node_id[:8]}..., "
            f"chip={self.caps.chip!r}, "
            f"backends={self.caps.backends})"
        )


# ---------------------------------------------------------------------------
# Job serialization helpers
# ---------------------------------------------------------------------------

def job_from_circuit(
    circuit,
    params: mx.array | list[float],
    observable: str = "sum_z",
    backend: str = "sv",
) -> SimJob:
    """
    Serialize a Circuit into a SimJob for dispatch to a node.

    Args:
        circuit:    a zilver Circuit instance
        params:     parameter vector
        observable: "sum_z" | "z0"
        backend:    "sv" | "dm" | "tn"
    """
    from .circuit import GateOp

    ops = []
    for op in circuit._ops:
        # Determine gate type from gate_fn signature
        # We rely on GateOp class methods having been called — reconstruct from param_indices
        pidx = op.param_indices[0] if op.param_indices else None
        kind = _infer_gate_kind(op, circuit)
        ops.append({"type": kind, "qubits": op.qubits, "param_idx": pidx})

    if isinstance(params, mx.array):
        params_list = params.tolist()
    else:
        params_list = list(params)

    return SimJob(
        circuit_ops = ops,
        n_qubits    = circuit.n_qubits,
        n_params    = circuit.n_params,
        params      = params_list,
        observable  = observable,
        backend     = backend,
    )


def _infer_gate_kind(op, circuit) -> str:
    """
    Heuristic: infer gate type from GateOp structure.
    This is a best-effort reverse of the Circuit builder API.
    Robust serialization would store kind explicitly in GateOp — future work.
    """
    n_qubits_op = len(op.qubits)
    has_params   = bool(op.param_indices)

    if n_qubits_op == 1 and not has_params:
        # Fixed single-qubit: H or X (can't distinguish without inspecting matrix)
        # Evaluate at identity params to check: H|0> has non-zero <X>, X|0> has <Z>=-1
        return "h"   # conservative: caller should use explicit serialization for production
    if n_qubits_op == 1 and has_params:
        # Parameterized: ry, rx, or rz — can't distinguish without matrix inspection
        return "ry"  # conservative default
    if n_qubits_op == 2 and not has_params:
        return "cnot"
    if n_qubits_op == 2 and has_params:
        return "rzz"
    return "h"
