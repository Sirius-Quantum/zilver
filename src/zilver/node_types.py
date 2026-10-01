"""Pure-Python node types — no MLX dependency.

These dataclasses and hardware-detection helpers are shared between the
simulation node (which needs MLX) and the capability registry (which runs
on Linux x86 servers that have no MLX).  Keeping them separate lets the
registry import only this module.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import platform
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict


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
    if backend == "sv":
        # Statevector: (2^n,) complex64 = 8 bytes per element
        return 8 * (2 ** n_qubits)
    if backend == "dm":
        # Density matrix: (2^n, 2^n) complex64
        return 8 * (4 ** n_qubits)
    if backend == "tn":
        # MPS tensors: n tensors each of shape (chi, 2, chi) complex64
        return n_qubits * 2 * (chi_max ** 2) * 8
    return 8 * (2 ** n_qubits)


def _meminfo_bytes(key: str) -> int | None:
    """Read one ``/proc/meminfo`` field (Linux), in bytes. None elsewhere."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _available_memory_bytes() -> int:
    """
    Return memory a new job can use without swapping, in bytes.

    Linux: ``MemAvailable``. macOS: free + inactive + speculative + purgeable
    pages from ``vm_stat``, which the kernel hands out on demand. Free pages
    alone read 0.08 GB on an idle 16 GB M1 Pro.
    Falls back to 8 GB on failure so callers never hard-block on errors.
    """
    available = _meminfo_bytes("MemAvailable")
    if available is not None:
        return available
    try:
        out = subprocess.check_output(
            ["vm_stat"], stderr=subprocess.DEVNULL, timeout=2,
        ).decode()
        page_size = int(re.search(r"page size of (\d+) bytes", out).group(1))
        pages = re.findall(
            r"^Pages (?:free|inactive|speculative|purgeable):\s+(\d+)\.", out, re.M,
        )
        if pages:
            return page_size * sum(int(p) for p in pages)
    except Exception:
        pass
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
        pass
    total = _meminfo_bytes("MemTotal")
    if total is not None:
        return total // (1024 ** 3)
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
    noise:       noise model for backend="dm", e.g.
                 {"depolarizing": {"p1": 0.001, "p2": 0.01}} or
                 {"thermal_relaxation": {"t1": ..., "t2": ..., "gate_time_1q": ...}};
                 None runs the density matrix noiselessly
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
    noise:       dict | None       = None

    def to_dict(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        """SHA-256 over every field that determines the answer, the circuit included."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

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
    proof:              SHA-256 of (SimJob.digest() + primary result), see _compute_proof
    node_signature:     hex signature over proof
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
        """Check that the proof binds this result to exactly this job.

        A consistency check, not a correctness check: it fails if the circuit,
        params, backend or result were changed after the node produced the
        proof, but a node can still compute a wrong answer and prove it. The
        signature (:meth:`verify_signature`) covers the same proof.
        """
        return self.proof == _compute_proof(job, self)

    def verify_signature(self) -> bool:
        """Verify the node's cryptographic signature over the proof.

        Returns ``False`` if any of proof, node_signature, or node_pubkey is absent.
        """
        if not self.proof or not self.node_signature or not self.node_pubkey:
            return False
        return verify_result_signature(self.proof, self.node_pubkey, self.node_signature)


def _compute_proof(job: SimJob, result: JobResult) -> str:
    """SHA-256 over the job digest and the primary result for its result_type.

    The node computes this before signing and :meth:`JobResult.verify`
    recomputes it, so both sides go through this one function.
    """
    primary: dict = {"expectation": round(result.expectation, 8)}
    if job.result_type == "samples":
        primary["samples"] = sorted(result.samples or [])
    elif job.result_type == "statevector":
        sv_bytes = json.dumps(result.statevector or [], sort_keys=True).encode()
        primary["statevector_sha256"] = hashlib.sha256(sv_bytes).hexdigest()
    elif job.result_type == "pauli":
        primary["pauli"] = {
            k: round(v, 8) for k, v in sorted((result.pauli_expectations or {}).items())
        }
    payload = json.dumps(
        {"job": job.digest(), "result_type": job.result_type, **primary},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_result_signature(
    proof: str,
    node_pubkey_hex: str,
    signature_hex: str,
) -> bool:
    """Verify a node's cryptographic signature over a job result proof.

    Detects key type from the pubkey length:

    - 64 hex chars  (32 bytes) → public key
    - 130 hex chars (65 bytes) → hardware public key
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


# ---------------------------------------------------------------------------
# Execute tokens
# ---------------------------------------------------------------------------
# A node sends the registry a random execute secret at registration, and the
# registry never returns it. For each match the registry hands the client a
# token that lets it call POST /execute on that one node until it expires:
#
#     "<job_token>.<exp>.<hmac_sha256(secret, node_id.job_token.exp)>"
#
# The node checks the MAC with the same secret and reports the job_token back
# with its own contribution, so a client never holds a node's credential.

EXECUTE_TOKEN_TTL = 600.0


def _execute_mac(secret: str, node_id: str, job_token: str, exp: int) -> str:
    return hmac.new(
        secret.encode(), f"{node_id}.{job_token}.{exp}".encode(), hashlib.sha256,
    ).hexdigest()


def issue_execute_token(
    secret: str, node_id: str, job_token: str, ttl: float = EXECUTE_TOKEN_TTL,
) -> str:
    """Mint a token for POST /execute on *node_id*, valid for *ttl* seconds."""
    exp = int(time.time() + ttl)
    return f"{job_token}.{exp}.{_execute_mac(secret, node_id, job_token, exp)}"


def check_execute_token(secret: str, node_id: str, token: str) -> str | None:
    """Return the token's job_token if it is valid and unexpired, else None."""
    try:
        job_token, exp_str, mac = token.split(".")
        exp = int(exp_str)
    except ValueError:
        return None
    if exp < time.time():
        return None
    if not hmac.compare_digest(mac, _execute_mac(secret, node_id, job_token, exp)):
        return None
    return job_token
