"""Simulation node."""

from __future__ import annotations

import time
from typing import Any

import mlx.core as mx
import numpy as np

# Pure-Python types shared with the registry (no MLX dependency there)
from .node_types import (
    NodeCapabilities,
    SimJob,
    JobResult,
    estimate_memory_bytes,
    _available_memory_bytes,
    _compute_proof,
    _compute_proof_v2,
    _detect_hardware_uuid,
    _detect_chip,
    _detect_ram_gb,
    _sv_qubit_ceiling,
    _dm_qubit_ceiling,
)

__all__ = [
    # Re-export so existing `from zilver.node import NodeCapabilities` keeps working
    "NodeCapabilities",
    "SimJob",
    "JobResult",
    "estimate_memory_bytes",
    "_compute_proof",
    "Node",
    "job_from_circuit",
]


# ---------------------------------------------------------------------------
# Circuit reconstruction from ops list
# ---------------------------------------------------------------------------

def _build_circuit_from_ops(ops: list[dict], n_qubits: int, n_params: int):
    """
    Reconstruct a Circuit from a serialized ops list.

    Supports both the current format (``param_indices`` list) and the legacy
    format (``param_idx`` single value) for backward compatibility.
    """
    from .circuit import Circuit
    c = Circuit(n_qubits)
    c.n_params = n_params
    for op in ops:
        kind   = op["type"]
        qubits = op["qubits"]

        # Resolve parameter indices — support both serialization formats
        raw_indices = op.get("param_indices")
        if raw_indices is not None:
            param_indices = raw_indices
        else:
            legacy = op.get("param_idx")
            param_indices = [legacy] if legacy is not None else []
        pidx = param_indices[0] if param_indices else None

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
        elif kind == "u3":
            if len(param_indices) < 3:
                raise ValueError(
                    "u3 gate requires 3 param_indices (theta, phi, lambda); "
                    f"got {param_indices!r}"
                )
            c.u3(qubits[0], param_indices[0], param_indices[1], param_indices[2])
        elif kind == "toffoli":
            c.toffoli(qubits[0], qubits[1], qubits[2])
        elif kind == "fredkin":
            c.fredkin(qubits[0], qubits[1], qubits[2])
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
        private_key_bytes: bytes | None = None,
        public_key_bytes:  bytes | None = None,
        se_label:          str   | None = None,
    ) -> "Node":
        """
        Initialize a node with auto-detected hardware capabilities.

        Args:
            backends:          list of ["sv", "dm", "tn"]; default ["sv"]
            node_id:           explicit node ID; auto-generated if None
            wallet:            wallet address for reward settlement (future use)
            private_key_bytes: raw Ed25519 private key for result signing (None for SE)
            public_key_bytes:  raw public key bytes (Ed25519 or P-256 uncompressed)
            se_label:          Secure Enclave key label (takes priority over private_key_bytes)
        """
        caps = NodeCapabilities.detect(backends=backends, node_id=node_id)
        node = cls(caps)
        node._wallet            = wallet
        node._private_key_bytes = private_key_bytes
        node._public_key_bytes  = public_key_bytes
        node._se_label          = se_label
        return node

    def execute(self, job: SimJob) -> JobResult:
        """
        Execute a simulation job and return a signed result.

        Raises ValueError if the node cannot handle the job
        (backend unsupported, qubit count exceeds capacity, or
        insufficient free system memory for the job).
        """
        if not self.caps.supports(job.backend, job.n_qubits):
            raise ValueError(
                f"Node {self.caps.node_id} cannot handle "
                f"backend={job.backend!r} n_qubits={job.n_qubits} "
                f"(sv_max={self.caps.sv_qubits_max}, "
                f"dm_max={self.caps.dm_qubits_max})"
            )

        needed    = estimate_memory_bytes(job.n_qubits, job.backend)
        available = _available_memory_bytes()
        if needed > available:
            raise ValueError(
                f"Insufficient memory for job: needs {needed / 2**20:.0f} MB, "
                f"only {available / 2**20:.0f} MB free"
            )

        # Reset Metal peak counter before the run
        try:
            if hasattr(mx, "reset_peak_memory"):
                mx.reset_peak_memory()
            elif hasattr(mx.metal, "reset_peak_memory"):
                mx.metal.reset_peak_memory()
        except Exception:
            pass

        t0 = time.perf_counter()
        result_data = self._run_typed(job)
        elapsed_ms  = (time.perf_counter() - t0) * 1000.0

        # Read peak Metal memory consumed by this job
        memory_used_mb = 0.0
        try:
            if hasattr(mx, "get_peak_memory"):
                memory_used_mb = mx.get_peak_memory() / (1024 ** 2)
            elif hasattr(mx.metal, "get_peak_memory"):
                memory_used_mb = mx.metal.get_peak_memory() / (1024 ** 2)
        except Exception:
            pass

        self.caps.jobs_completed += 1

        expectation = result_data.get("expectation", 0.0)

        # Compute proof — use v2 format for non-expectation result types
        result_type = getattr(job, "result_type", "expectation")
        if result_type == "expectation":
            proof = _compute_proof(job.job_id, job.params, expectation)
        elif result_type == "samples":
            samples = result_data.get("samples") or []
            proof = _compute_proof_v2(
                job.job_id, job.params, "samples", {"samples": sorted(samples)}
            )
        elif result_type == "statevector":
            import json as _json
            sv = result_data.get("statevector") or []
            sv_bytes = _json.dumps(sv, sort_keys=True).encode()
            import hashlib as _hashlib
            sv_hash = _hashlib.sha256(sv_bytes).hexdigest()
            proof = _compute_proof_v2(
                job.job_id, job.params, "statevector", {"statevector_sha256": sv_hash}
            )
        elif result_type == "pauli":
            pe = result_data.get("pauli_expectations") or {}
            proof = _compute_proof_v2(
                job.job_id, job.params, "pauli",
                {k: round(v, 8) for k, v in sorted(pe.items())}
            )
        else:
            proof = _compute_proof(job.job_id, job.params, expectation)

        # Sign the proof if we have a key
        node_signature = ""
        node_pubkey    = ""
        private_key_bytes = getattr(self, "_private_key_bytes", None)
        public_key_bytes  = getattr(self, "_public_key_bytes",  None)
        se_label          = getattr(self, "_se_label",          None)
        if public_key_bytes is not None and (
            private_key_bytes is not None or se_label is not None
        ):
            try:
                from .security import sign_result
                node_signature = sign_result(proof, private_key_bytes, se_label)
                node_pubkey    = public_key_bytes.hex()
            except Exception:
                pass

        return JobResult(
            expectation        = expectation,
            job_id             = job.job_id,
            node_id            = self.caps.node_id,
            elapsed_ms         = elapsed_ms,
            proof              = proof,
            memory_used_mb     = memory_used_mb,
            node_signature     = node_signature,
            node_pubkey        = node_pubkey,
            samples            = result_data.get("samples"),
            sample_counts      = result_data.get("sample_counts"),
            statevector        = result_data.get("statevector"),
            pauli_expectations = result_data.get("pauli_expectations"),
        )

    def _run_typed(self, job: SimJob) -> dict:
        """Execute job and return a result dict covering all result_type variants."""
        result_type = getattr(job, "result_type", "expectation")
        params_mx   = mx.array(np.array(job.params, dtype=np.float32))

        if result_type == "samples":
            return self._run_samples(job, params_mx)
        if result_type == "statevector":
            return self._run_statevector(job, params_mx)
        if result_type == "pauli":
            return self._run_pauli(job, params_mx)
        # Default: expectation
        expectation = self._run(job)
        return {"expectation": float(expectation)}

    def _run_samples(self, job: SimJob, params: mx.array) -> dict:
        """Sample bitstrings from the statevector probability distribution."""
        shots = getattr(job, "shots", None) or 1024
        circuit = _build_circuit_from_ops(job.circuit_ops, job.n_qubits, job.n_params)
        state   = circuit._run(params)
        mx.eval(state)
        probs = np.array((mx.abs(state) ** 2).tolist(), dtype=np.float64)
        probs = np.abs(probs)
        probs /= probs.sum()  # renormalize for numerical safety
        n = job.n_qubits
        indices   = np.random.choice(len(probs), size=shots, p=probs)
        bitstrings = [format(int(idx), f"0{n}b") for idx in indices]
        counts: dict[str, int] = {}
        for s in bitstrings:
            counts[s] = counts.get(s, 0) + 1
        # Expectation from sample mean of Z eigenvalues
        expval = float(
            sum((-1) ** b.count("1") / n * v for b, v in counts.items()) / shots
        ) if n > 0 else 0.0
        return {"samples": bitstrings, "sample_counts": counts, "expectation": expval}

    def _run_statevector(self, job: SimJob, params: mx.array) -> dict:
        """Return the full state vector as [[real, imag], …]."""
        from .simulator import expectation_pauli_sum, expectation_z
        circuit = _build_circuit_from_ops(job.circuit_ops, job.n_qubits, job.n_params)
        state   = circuit._run(params)
        mx.eval(state)
        sv_np    = np.array(state.tolist(), dtype=np.complex64)
        sv_pairs = [[float(v.real), float(v.imag)] for v in sv_np]
        if job.observable == "sum_z":
            expval = float(expectation_pauli_sum(state, job.n_qubits))
        else:
            expval = float(expectation_z(state, 0, job.n_qubits))
        return {"statevector": sv_pairs, "expectation": expval}

    def _run_pauli(self, job: SimJob, params: mx.array) -> dict:
        """Compute expectation values for a Pauli Hamiltonian."""
        hamiltonian = getattr(job, "hamiltonian", None) or []
        circuit = _build_circuit_from_ops(job.circuit_ops, job.n_qubits, job.n_params)
        state   = circuit._run(params)
        mx.eval(state)

        pauli_expectations: dict[str, float] = {}
        total_expectation = 0.0

        for term in hamiltonian:
            coeff      = float(term.get("coeff", 1.0))
            pauli_str  = str(term.get("pauli", "Z" * job.n_qubits))
            term_exp   = _expectation_pauli_term(state, pauli_str, job.n_qubits)
            weighted   = coeff * term_exp
            pauli_expectations[pauli_str] = term_exp
            total_expectation += weighted

        return {
            "pauli_expectations": pauli_expectations,
            "expectation": total_expectation,
        }

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

# ---------------------------------------------------------------------------
# Pauli expectation helper
# ---------------------------------------------------------------------------

def _expectation_pauli_term(state: "mx.array", pauli: str, n: int) -> float:
    """Compute ⟨ψ|P|ψ⟩ for a tensor-product Pauli string.

    Parameters
    ----------
    state:
        (2**n,) complex64 MLX statevector.
    pauli:
        Pauli string of length *n*, e.g. ``"ZXI"``.  Characters: I, X, Y, Z
        (case-insensitive).
    n:
        Number of qubits.

    Returns
    -------
    float
        Real-valued expectation value.
    """
    from .simulator import apply_gate

    _PAULI_X = mx.array([[0, 1], [1, 0]], dtype=mx.complex64)
    _PAULI_Y = mx.array([[0, -1j], [1j, 0]], dtype=mx.complex64)
    _PAULI_Z = mx.array([[1, 0], [0, -1]], dtype=mx.complex64)
    _PAULI_MATS = {"X": _PAULI_X, "Y": _PAULI_Y, "Z": _PAULI_Z}

    phi = state
    for q, p in enumerate(pauli.upper()):
        if p == "I":
            continue
        mat = _PAULI_MATS[p]
        phi = apply_gate(phi, mat, [q], n)

    mx.eval(state, phi)
    psi_np = np.array(state.tolist(), dtype=np.complex64)
    phi_np = np.array(phi.tolist(), dtype=np.complex64)
    return float(np.dot(psi_np.conj(), phi_np).real)


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

    Every gate in the circuit must have been constructed via a Circuit builder
    method (c.h(), c.ry(), c.cnot(), etc.) so that GateOp.kind is set.
    Circuits produced by circuit.fuse() cannot be serialized — call
    job_from_circuit() before fuse().

    Args:
        circuit:    a zilver Circuit instance
        params:     parameter vector
        observable: "sum_z" | "z0"
        backend:    "sv" | "dm" | "tn"
    """
    ops = []
    for op in circuit._ops:
        if not op.kind:
            raise ValueError(
                "GateOp has no kind — rebuild the circuit using Circuit builder "
                "methods (c.h(), c.ry(), c.cnot(), …) instead of raw GateOp()."
            )
        if op.kind == "fused":
            raise ValueError(
                "Cannot serialize a fused circuit for remote execution. "
                "Call job_from_circuit() before circuit.fuse()."
            )
        ops.append({
            "type":          op.kind,
            "qubits":        op.qubits,
            "param_indices": op.param_indices,
        })

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
