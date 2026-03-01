"""
QASM bridge: convert Qiskit QASM 2.0 circuits into Zilver Circuit objects.

Parses Qiskit-generated QASM circuits
into Zilver for MLX-accelerated landscape computation without rewriting
circuit generation code.
"""

from __future__ import annotations
import re
from .circuit import Circuit
from . import gates as G
import mlx.core as mx


# Gates we know how to map; extend as needed.
_SINGLE_QUBIT_FIXED = {
    "h":  G.H,
    "x":  G.X,
    "y":  G.Y,
    "z":  G.Z,
    "s":  G.S,
    "t":  G.T,
    "id": G.I,
}

_TWO_QUBIT_FIXED = {
    "cx":   G.CNOT,
    "cnot": G.CNOT,
    "cz":   G.CZ,
    "swap": G.SWAP,
}

_SINGLE_QUBIT_PARAM = {
    "rx": G.RX,
    "ry": G.RY,
    "rz": G.RZ,
}


def circuit_from_qasm(qasm_str: str) -> Circuit:
    """
    Parse a QASM 2.0 string and return a Zilver Circuit.

    Parameter handling: each unique QASM parameter name is assigned a
    sequential index in the output parameter vector.

    Limitations: does not support custom gate definitions or classical control.
    Intended for hardware-efficient / RealAmplitudes / QAOA circuits as
    
    """
    lines = [l.strip() for l in qasm_str.splitlines()]
    lines = [l for l in lines if l and not l.startswith("//")]

    n_qubits = _parse_n_qubits(lines)
    circuit = Circuit(n_qubits)

    param_registry: dict[str, int] = {}

    for line in lines:
        if line.startswith("OPENQASM") or line.startswith("include") or line.startswith("qreg") or line.startswith("creg"):
            continue
        if line.startswith("measure") or line.startswith("barrier"):
            continue

        _parse_gate_line(line, circuit, param_registry, n_qubits)

    circuit.n_params = len(param_registry)
    return circuit


def _parse_n_qubits(lines: list[str]) -> int:
    """Extract the qubit count from a ``qreg`` declaration line."""
    for line in lines:
        m = re.match(r"qreg\s+\w+\[(\d+)\]", line)
        if m:
            return int(m.group(1))
    raise ValueError("No qreg declaration found in QASM string")


def _parse_gate_line(
    line: str,
    circuit: Circuit,
    param_registry: dict[str, int],
    n_qubits: int,
) -> None:
    """Parse one QASM gate statement and append the corresponding op to circuit.

    Handles fixed single-qubit gates, fixed two-qubit gates, and parameterized
    single-qubit rotations (rx/ry/rz). Unrecognised gate names are silently
    skipped, matching standard QASM parser behaviour for custom gate defs.
    """
    # Match: gate_name[(param,...)] qubit[i](, qubit[j])*;
    m = re.match(r"(\w+)(?:\(([^)]*)\))?\s+(.+);", line)
    if not m:
        return

    gate_name = m.group(1).lower()
    params_str = m.group(2) or ""
    qubits_str = m.group(3)

    qubits = [_parse_qubit(q.strip()) for q in qubits_str.split(",")]

    if gate_name in _SINGLE_QUBIT_FIXED:
        gate_matrix = _SINGLE_QUBIT_FIXED[gate_name]()
        from .circuit import GateOp
        circuit.add(GateOp.fixed(gate_matrix, qubits))

    elif gate_name in _TWO_QUBIT_FIXED:
        gate_matrix = _TWO_QUBIT_FIXED[gate_name]()
        from .circuit import GateOp
        circuit.add(GateOp.fixed(gate_matrix, qubits))

    elif gate_name in _SINGLE_QUBIT_PARAM:
        param_name = params_str.strip()
        if param_name not in param_registry:
            param_registry[param_name] = len(param_registry)
        param_idx = param_registry[param_name]
        gate_fn_cls = _SINGLE_QUBIT_PARAM[gate_name]
        from .circuit import GateOp
        op = GateOp(
            gate_fn=lambda p, fn=gate_fn_cls: fn(float(p[0])),
            qubits=qubits,
            param_indices=[param_idx],
        )
        circuit.add(op)

    # Silently skip unrecognized gates (barrier, custom defs, etc.)


def _parse_qubit(token: str) -> int:
    m = re.search(r"\[(\d+)\]", token)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse qubit index from: {token}")
