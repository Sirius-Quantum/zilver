"""Gate matrices."""

import math
import numpy as np
from ._array import mx, HAS_COMPLEX


# Fixed single-qubit gates


def _const(m):
    """A gate constant, on the device only if the device can hold it.

    Gate matrices are 2x2 or 4x4 constants; building them on a GPU buys
    nothing. It also breaks on DirectML, which has no ComplexFloat and answers
    the cast by aborting the process rather than raising -- so a plain
    `c.h(0)` would kill the interpreter at circuit-construction time, long
    before any state existed.

    With complex support, behaviour is exactly as before. Without it, the
    constant stays a numpy complex array on the host and the caller lifts it
    to its real block form.
    """
    a = np.asarray(m, dtype=np.complex64)
    return mx.array(a, dtype=mx.complex64) if HAS_COMPLEX else a


def I() -> mx.array:  # noqa: E743
    """Identity gate."""
    return _const([[1, 0], [0, 1]])

def X() -> mx.array:
    """Pauli-X (bit-flip) gate."""
    return _const([[0, 1], [1, 0]])

def Y() -> mx.array:
    """Pauli-Y gate."""
    return _const([[0, -1j], [1j, 0]])

def Z() -> mx.array:
    """Pauli-Z (phase-flip) gate."""
    return _const([[1, 0], [0, -1]])

def H() -> mx.array:
    """Hadamard gate. Maps |0> -> |+>, |1> -> |->."""
    s = 1.0 / math.sqrt(2)
    return _const([[s, s], [s, -s]])

def S() -> mx.array:
    """Phase gate (sqrt of Z). Applies a pi/2 phase to |1>."""
    return _const([[1, 0], [0, 1j]])

def T() -> mx.array:
    """T gate (fourth root of Z). Applies a pi/4 phase to |1>."""
    return _const([[1, 0], [0, complex(math.cos(math.pi/4), math.sin(math.pi/4))]])


# Parameterized single-qubit rotations

def RX(theta: float) -> mx.array:
    """Rotation about the X axis by angle theta: exp(-i theta/2 X)."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return _const([[c, -1j * s], [-1j * s, c]])

def RY(theta: float) -> mx.array:
    """Rotation about the Y axis by angle theta: exp(-i theta/2 Y)."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return _const([[c, -s], [s, c]])

def RZ(theta: float) -> mx.array:
    """Rotation about the Z axis by angle theta: exp(-i theta/2 Z)."""
    e_neg = complex(math.cos(theta / 2), -math.sin(theta / 2))
    e_pos = complex(math.cos(theta / 2),  math.sin(theta / 2))
    return _const([[e_neg, 0], [0, e_pos]])

def P(phi: float) -> mx.array:
    """Phase gate."""
    return _const([[1, 0], [0, complex(math.cos(phi), math.sin(phi))]])

def U(theta: float, phi: float, lam: float) -> mx.array:
    """General single-qubit unitary (IBM U gate)."""
    ct, st = math.cos(theta / 2), math.sin(theta / 2)
    return _const([
        [ct, -complex(math.cos(lam), math.sin(lam)) * st],
        [complex(math.cos(phi), math.sin(phi)) * st,
         complex(math.cos(phi + lam), math.sin(phi + lam)) * ct],
    ])


# Fixed two-qubit gates

def CNOT() -> mx.array:
    """Controlled-NOT gate. Flips target qubit when control is |1>."""
    return _const([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ])

def CZ() -> mx.array:
    """Controlled-Z gate. Applies a Z phase to target when control is |1>."""
    return _const([
        [1, 0, 0,  0],
        [0, 1, 0,  0],
        [0, 0, 1,  0],
        [0, 0, 0, -1],
    ])

def SWAP() -> mx.array:
    """SWAP gate. Exchanges the states of two qubits."""
    return _const([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])

def iSWAP() -> mx.array:
    """iSWAP gate. Swaps two qubits and applies an i phase to the swapped amplitudes."""
    return _const([
        [1,  0,  0, 0],
        [0,  0, 1j, 0],
        [0, 1j,  0, 0],
        [0,  0,  0, 1],
    ])


# Parameterized two-qubit gates

def CRZ(theta: float) -> mx.array:
    e_neg = complex(math.cos(theta / 2), -math.sin(theta / 2))
    e_pos = complex(math.cos(theta / 2),  math.sin(theta / 2))
    return _const([
        [1, 0, 0,     0],
        [0, 1, 0,     0],
        [0, 0, e_neg, 0],
        [0, 0, 0,  e_pos],
    ])

def RZZ(theta: float) -> mx.array:
    """Ising ZZ coupling gate, native to many hardware platforms."""
    e_neg = complex(math.cos(theta / 2), -math.sin(theta / 2))
    e_pos = complex(math.cos(theta / 2),  math.sin(theta / 2))
    return _const([
        [e_neg, 0,     0,    0],
        [0,     e_pos, 0,    0],
        [0,     0,     e_pos, 0],
        [0,     0,     0,  e_neg],
    ])

def RXX(theta: float) -> mx.array:
    """Ising XX coupling gate."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return _const([
        [c,     0,     0,  -1j*s],
        [0,     c,  -1j*s,    0],
        [0,  -1j*s,    c,     0],
        [-1j*s,  0,    0,     c],
    ])

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
    mat[6, 6] = 0
    mat[6, 7] = 1
    mat[7, 7] = 0
    mat[7, 6] = 1
    return mx.array(mat)

def Fredkin() -> mx.array:
    """
    Fredkin (CSWAP) gate: swaps q1 and q2 when control q0 is |1>.
    Used in quantum error correction and reversible computing.
    |101> <-> |110>  (indices 5 <-> 6)
    """
    mat = np.eye(8, dtype=np.complex64)
    mat[5, 5] = 0
    mat[5, 6] = 1
    mat[6, 6] = 0
    mat[6, 5] = 1
    return mx.array(mat)
