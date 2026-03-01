"""Gate matrices."""

import math
import numpy as np
import mlx.core as mx


# Fixed single-qubit gates

def I() -> mx.array:
    """Identity gate."""
    return mx.array([[1, 0], [0, 1]], dtype=mx.complex64)

def X() -> mx.array:
    """Pauli-X (bit-flip) gate."""
    return mx.array([[0, 1], [1, 0]], dtype=mx.complex64)

def Y() -> mx.array:
    """Pauli-Y gate."""
    return mx.array([[0, -1j], [1j, 0]], dtype=mx.complex64)

def Z() -> mx.array:
    """Pauli-Z (phase-flip) gate."""
    return mx.array([[1, 0], [0, -1]], dtype=mx.complex64)

def H() -> mx.array:
    """Hadamard gate. Maps |0> -> |+>, |1> -> |->."""
    s = 1.0 / math.sqrt(2)
    return mx.array([[s, s], [s, -s]], dtype=mx.complex64)

def S() -> mx.array:
    """Phase gate (sqrt of Z). Applies a pi/2 phase to |1>."""
    return mx.array([[1, 0], [0, 1j]], dtype=mx.complex64)

def T() -> mx.array:
    """T gate (fourth root of Z). Applies a pi/4 phase to |1>."""
    return mx.array([[1, 0], [0, complex(math.cos(math.pi/4), math.sin(math.pi/4))]], dtype=mx.complex64)


# Parameterized single-qubit rotations

def RX(theta: float) -> mx.array:
    """Rotation about the X axis by angle theta: exp(-i theta/2 X)."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return mx.array([[c, -1j * s], [-1j * s, c]], dtype=mx.complex64)

def RY(theta: float) -> mx.array:
    """Rotation about the Y axis by angle theta: exp(-i theta/2 Y)."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return mx.array([[c, -s], [s, c]], dtype=mx.complex64)

def RZ(theta: float) -> mx.array:
    """Rotation about the Z axis by angle theta: exp(-i theta/2 Z)."""
    e_neg = complex(math.cos(theta / 2), -math.sin(theta / 2))
    e_pos = complex(math.cos(theta / 2),  math.sin(theta / 2))
    return mx.array([[e_neg, 0], [0, e_pos]], dtype=mx.complex64)

def P(phi: float) -> mx.array:
    """Phase gate."""
    return mx.array([[1, 0], [0, complex(math.cos(phi), math.sin(phi))]], dtype=mx.complex64)

def U(theta: float, phi: float, lam: float) -> mx.array:
    """General single-qubit unitary (IBM U gate)."""
    ct, st = math.cos(theta / 2), math.sin(theta / 2)
    return mx.array([
        [ct, -complex(math.cos(lam), math.sin(lam)) * st],
        [complex(math.cos(phi), math.sin(phi)) * st,
         complex(math.cos(phi + lam), math.sin(phi + lam)) * ct],
    ], dtype=mx.complex64)


# Fixed two-qubit gates

def CNOT() -> mx.array:
    """Controlled-NOT gate. Flips target qubit when control is |1>."""
    return mx.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=mx.complex64)

def CZ() -> mx.array:
    """Controlled-Z gate. Applies a Z phase to target when control is |1>."""
    return mx.array([
        [1, 0, 0,  0],
        [0, 1, 0,  0],
        [0, 0, 1,  0],
        [0, 0, 0, -1],
    ], dtype=mx.complex64)

def SWAP() -> mx.array:
    """SWAP gate. Exchanges the states of two qubits."""
    return mx.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=mx.complex64)

def iSWAP() -> mx.array:
    """iSWAP gate. Swaps two qubits and applies an i phase to the swapped amplitudes."""
    return mx.array([
        [1,  0,  0, 0],
        [0,  0, 1j, 0],
        [0, 1j,  0, 0],
        [0,  0,  0, 1],
    ], dtype=mx.complex64)


# Parameterized two-qubit gates

def CRZ(theta: float) -> mx.array:
    e_neg = complex(math.cos(theta / 2), -math.sin(theta / 2))
    e_pos = complex(math.cos(theta / 2),  math.sin(theta / 2))
    return mx.array([
        [1, 0, 0,     0],
        [0, 1, 0,     0],
        [0, 0, e_neg, 0],
        [0, 0, 0,  e_pos],
    ], dtype=mx.complex64)

def RZZ(theta: float) -> mx.array:
    """Ising ZZ coupling gate, native to many hardware platforms."""
    e_neg = complex(math.cos(theta / 2), -math.sin(theta / 2))
    e_pos = complex(math.cos(theta / 2),  math.sin(theta / 2))
    return mx.array([
        [e_neg, 0,     0,    0],
        [0,     e_pos, 0,    0],
        [0,     0,     e_pos, 0],
        [0,     0,     0,  e_neg],
    ], dtype=mx.complex64)

def RXX(theta: float) -> mx.array:
    """Ising XX coupling gate."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return mx.array([
        [c,     0,     0,  -1j*s],
        [0,     c,  -1j*s,    0],
        [0,  -1j*s,    c,     0],
        [-1j*s,  0,    0,     c],
    ], dtype=mx.complex64)

# Three-qubit gates
# Qubit convention: qubit 0 = most significant bit
# Basis: |q0 q1 q2> -> index q0*4 + q1*2 + q2

def Toffoli() -> mx.array:
    """
    Toffoli (CCX) gate: flips target qubit (q2) when both controls (q0, q1) are |1>.
    Standard building block for quantum error correction and fault-tolerant circuits.
    |110> <-> |111>  (indices 6 <-> 7)
    """
    mat = np.eye(8, dtype=np.complex64)
    mat[6, 6] = 0; mat[6, 7] = 1
    mat[7, 7] = 0; mat[7, 6] = 1
    return mx.array(mat)

def Fredkin() -> mx.array:
    """
    Fredkin (CSWAP) gate: swaps q1 and q2 when control q0 is |1>.
    Used in quantum error correction and reversible computing.
    |101> <-> |110>  (indices 5 <-> 6)
    """
    mat = np.eye(8, dtype=np.complex64)
    mat[5, 5] = 0; mat[5, 6] = 1
    mat[6, 6] = 0; mat[6, 5] = 1
    return mx.array(mat)
