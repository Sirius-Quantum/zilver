"""Pure-Python node types — no MLX dependency.

These dataclasses and hardware-detection helpers are shared between the
simulation node (which needs MLX) and the capability registry (which runs
on Linux x86 servers that have no MLX).  Keeping them separate lets the
registry import only this module.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def estimate_memory_bytes(n_qubits: int, backend: str, chi_max: int = 64) -> int:
    """
    Estimate peak memory required to simulate a job.

    Parameters
    ----------
    n_qubits:
        Number of qubits in the circuit.
    backend:
        ``"sv"`` (statevector), ``"dm"`` (density matrix), or ``"tn"`` (MPS).
    chi_max:
        Bond dimension for MPS / tensor-network backend.

    Returns
    -------
    int
        Estimated bytes.  Exact for sv/dm (complex64 arrays);
        approximate for tn (scales with bond dimension, not exponentially).
    """
    if backend in ("sv", "tn") and backend != "tn":
        # Statevector: (2^n,) complex64 = 8 bytes per element
        return 8 * (2 ** n_qubits)
    if backend == "dm":
        # Density matrix: (2^n, 2^n) complex64
        return 8 * (4 ** n_qubits)
    if backend == "tn":
        # MPS tensors: n tensors each of shape (chi, 2, chi) complex64
        return n_qubits * 2 * (chi_max ** 2) * 8
    return 8 * (2 ** n_qubits)


def _available_memory_bytes() -> int:
    """
    Return immediately available system memory in bytes (macOS).

    Uses ``sysctl vm.page_free_count`` and ``hw.pagesize``.
    Falls back to 8 GB on failure so callers never hard-block on errors.
    """
    try:
        page_size = int(subprocess.check_output(
            ["sysctl", "-n", "hw.pagesize"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip())
        free_pages = int(subprocess.check_output(
            ["sysctl", "-n", "vm.page_free_count"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip())
        return page_size * free_pages
    except Exception:
        return 8 * (1024 ** 3)  # 8 GB fallback


def _detect_hardware_uuid() -> str | None:
    """
    Return the IOPlatformUUID of this Apple Silicon Mac.

    Reads the hardware-unique device identifier via ``ioreg``.  Returns
    ``None`` on non-macOS systems or if the command fails.
    """
    try:
        out = subprocess.check_output(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode()
        for line in out.splitlines():
            if "IOPlatformUUID" in line:
                # Line format: "IOPlatformUUID" = "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
                parts = line.split('"')
                if len(parts) >= 4:
                    return parts[-2]
    except Exception:
        pass
    return None


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
    return min(n, 34)


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
            node_id       = node_id or _detect_hardware_uuid() or str(uuid.uuid4()),
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
    result_type: "expectation" | "samples" | "statevector" | "pauli"
    shots:       number of measurement shots (for result_type="samples")
    hamiltonian: list of {"coeff": float, "pauli": str} dicts (for result_type="pauli")
    """
    circuit_ops: list[dict]
    n_qubits:    int
    n_params:    int
    params:      list[float]
    observable:  str              = "sum_z"
    backend:     str              = "sv"
    job_id:      str              = field(default_factory=lambda: str(uuid.uuid4()))
    result_type: str              = "expectation"
    shots:       int | None       = None
    hamiltonian: list[dict] | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimJob":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class JobResult:
    """
    Result returned by a node after executing a SimJob.

    expectation:        computed expectation value
    job_id:             matches SimJob.job_id
    node_id:            identity of the executing node
    elapsed_ms:         wall-clock execution time
    proof:              SHA-256 of (job_id + params + primary_result)
    node_signature:     hex Ed25519 or P-256 ECDSA signature over proof
    node_pubkey:        hex public key — enables offline verification
    samples:            measurement bitstrings (result_type="samples")
    statevector:        complex amplitudes as [[real, imag], …] (result_type="statevector")
    pauli_expectations: {pauli_string: expectation_value} (result_type="pauli")
    credits_charged:    credits deducted from client for this job
    node_revenue:       credits earned by the executing node
    """
    expectation:        float
    job_id:             str
    node_id:            str
    elapsed_ms:         float
    proof:              str
    memory_used_mb:     float = 0.0
    node_signature:     str   = ""
    node_pubkey:        str   = ""
    samples:            list[str] | None          = None
    sample_counts:      dict[str, int] | None     = None
    statevector:        list[list[float]] | None  = None
    pauli_expectations: dict[str, float] | None   = None
    credits_charged:    float = 0.0
    node_revenue:       float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def verify(self, job: SimJob) -> bool:
        """Recompute and check the proof hash (expectation-type proof)."""
        return self.proof == _compute_proof(job.job_id, job.params, self.expectation)

    def verify_signature(self) -> bool:
        """Verify the node's cryptographic signature over the proof.

        Returns ``False`` if any of proof, node_signature, or node_pubkey is absent.
        """
        if not self.proof or not self.node_signature or not self.node_pubkey:
            return False
        return verify_result_signature(self.proof, self.node_pubkey, self.node_signature)


def _compute_proof(job_id: str, params: list[float], expectation: float) -> str:
    """Compute a SHA-256 proof for a job result.

    Serialises job_id, params, and expectation (rounded to 8 d.p.) as a
    deterministic JSON string and returns its hex digest. Used by
    JobResult.verify() to confirm the node computed the correct result.
    """
    payload = json.dumps({
        "job_id":     job_id,
        "params":     params,
        "expectation": round(expectation, 8),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_result_signature(
    proof: str,
    node_pubkey_hex: str,
    signature_hex: str,
) -> bool:
    """Verify a node's cryptographic signature over a job result proof.

    Detects key type from the pubkey length:

    - 64 hex chars  (32 bytes) → Ed25519 public key
    - 130 hex chars (65 bytes) → P-256 uncompressed public key (``04 || x || y``)
    """
    if not proof or not node_pubkey_hex or not signature_hex:
        return False
    message = proof.encode()
    try:
        pubkey_len = len(node_pubkey_hex)

        if pubkey_len == 64:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            pub_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(node_pubkey_hex))
            pub_key.verify(bytes.fromhex(signature_hex), message)
            return True

        if pubkey_len == 130:
            from cryptography.hazmat.primitives.asymmetric.ec import (
                ECDSA, EllipticCurvePublicNumbers, SECP256R1,
            )
            from cryptography.hazmat.primitives.hashes import SHA256
            from cryptography.hazmat.backends import default_backend
            pub_bytes = bytes.fromhex(node_pubkey_hex)
            if pub_bytes[0] != 0x04 or len(pub_bytes) != 65:
                return False
            x = int.from_bytes(pub_bytes[1:33], "big")
            y = int.from_bytes(pub_bytes[33:65], "big")
            pub_key = EllipticCurvePublicNumbers(x, y, SECP256R1()).public_key(
                default_backend()
            )
            pub_key.verify(bytes.fromhex(signature_hex), message, ECDSA(SHA256()))
            return True

    except Exception:
        pass
    return False


def _compute_proof_v2(
    job_id: str,
    params: list[float],
    result_type: str,
    primary: dict,
) -> str:
    """Extended proof covering non-expectation result types.

    Parameters
    ----------
    job_id:
        Job identifier.
    params:
        Circuit parameter vector.
    result_type:
        One of ``"samples"``, ``"statevector"``, or ``"pauli"``.
    primary:
        Dict holding the primary result data to include in the proof.
        For samples: ``{"samples": sorted_list}``.
        For statevector: ``{"statevector_sha256": hex_digest}``.
        For pauli: ``{pauli_str: round(val, 8), …}``.
    """
    payload = json.dumps(
        {"job_id": job_id, "params": params, "result_type": result_type, **primary},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
